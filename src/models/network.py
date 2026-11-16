"""
主网络结构：ClsNetwork

弱监督语义分割（WSSS）的分类网络，整体设计：
1. UNI 编码器提取图像特征，再经特征金字塔（FPN）得到 4 个尺度的特征图；
2. 用「类别原型」（视觉原型 / 文本原型 / 二者混合）与特征图做余弦相似度，
   得到每个像素属于各子类别的得分（CAM），并池化为分类 logits；
3. 支持 LoRA 微调编码器、文本原型混合、以及背景提示词抑制等扩展。
"""
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_

from .encoder import UNIEncoder
from .fpn import SimpleFeaturePyramid
from .text_anchor import TextAnchoredPrototype
from .text_encoder import (
    get_text_prompts,
    get_background_prompts,
    encode_text_prompts,
)
from src.utils.hierarchical import merge_to_parent_predictions


class AdaptiveLayer(nn.Module):
    """两层 MLP（fc -> ReLU -> fc），用于把 512 维标签原型投影到各 FPN 层级的通道维度。"""

    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        # 隐层维度 = 输入维度 * n_ratio
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """参数初始化：Linear 用截断正态，LayerNorm 置为 bias=0 / weight=1。"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(
        self,
        hf_token=None,                  # HuggingFace token，用于加载受权限保护的基础模型
        cls_num_classes=4,              # 父类别数（不含背景的语义类别数）
        fpn_out_dims=(64, 128, 320, 512),  # FPN 四个层级的输出通道数
        n_ratio=0.5,                    # AdaptiveLayer 隐层维度缩放比例
        l_fea_path=None,                # 标签特征文件路径（含视觉原型 visual_proto）
        use_lora=True,                  # 是否用 LoRA 微调编码器
        use_text_proto=False,           # 是否使用文本原型（text prototype）
        dataset_name="bcss",            # 数据集名称，决定提示词模板
        text_prompt_mode="pathology",   # 文本提示词模板类型
        text_encoder_mode="auto",       # 文本编码器选择方式
        use_hybrid_proto_blend=False,   # 是否在 forward 中混合文本/视觉原型
        hybrid_alpha_mode="learnable",  # 混合权重模式：learnable / fixed
        hybrid_alpha_fixed=0.5,         # 固定模式下的混合权重
        hybrid_alpha_parent_values=None,  # 固定模式下每个父类别的混合权重列表
        use_bg_prompt_suppression=False,  # 是否启用背景提示词抑制
        bg_prompt_mode="pathology",     # 背景提示词模板类型
        use_text_anchored_mask=False,   # 是否用文本锚定的通道掩码构建原型（替换 hybrid blend）
        text_anchor_tau=0.1,            # 文本锚定软掩码温度
        text_anchor_map_bias=False,     # 文本→视觉映射层是否带 bias
        text_anchor_sparsity_weight=0.0,  # 掩码 L1 稀疏正则权重（抗退化）
    ):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.in_channels = list(fpn_out_dims)

        # UNI 编码器 + 特征金字塔（FPN）
        self.encoder = UNIEncoder(hf_token=hf_token, img_size=224, use_lora=use_lora)
        self.sfp = SimpleFeaturePyramid(in_dim=1024, out_dims=fpn_out_dims)
        self.pooling = F.adaptive_avg_pool2d  # 全局平均池化，将 CAM 空间维压缩为 1x1

        # 4 个投影层，把 512 维标签原型分别映射到 FPN 各层级的通道维度
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])

        # 加载标签特征文件：visual_proto 是视觉原型，k_list 记录每个父类别包含的子类别数
        with open(l_fea_path, "rb") as lf:
            info = pkl.load(lf)
            self.register_buffer("visual_proto", info["features"].cpu())
            self.k_list = info["k_list"]
            self.cumsum_k = info["cumsum_k"]

        # 子类别总数（所有父类别的子类别数之和）
        self.total_classes = sum(self.k_list)

        self.use_text_proto = use_text_proto
        self.use_hybrid_proto_blend = use_hybrid_proto_blend
        self.hybrid_alpha_mode = hybrid_alpha_mode.lower()
        self.hybrid_alpha_fixed = float(hybrid_alpha_fixed)
        self.use_bg_prompt_suppression = use_bg_prompt_suppression

        # 构造「子类别 -> 父类别」映射表，例如 k_list=[3,3,3,3] 时映射为 [0,0,0,1,1,1,2,2,2,3,3,3]
        subclass_to_parent = []
        for parent_idx, k in enumerate(self.k_list):
            subclass_to_parent.extend([parent_idx] * k)
        self.register_buffer("subclass_to_parent", torch.tensor(subclass_to_parent, dtype=torch.long))

        # 若启用文本原型：生成文本提示词并编码，得到 text_proto
        if self.use_text_proto:
            # 得到文本原型提示词
            text_prompts = get_text_prompts(dataset_name, prompt_mode=text_prompt_mode)
            # 编码文本原型提示词
            text_emb = encode_text_prompts(text_prompts, encoder_mode=text_encoder_mode)
            self.register_buffer("text_proto", text_emb)
            num_parents = len(self.k_list)
            # 混合权重 alpha 的两种模式：可学习（sigmoid 输出）或固定值
            if self.hybrid_alpha_mode == "learnable":
                # learnable 模式：self.proto_alpha_raw 是一个可训练参数 nn.Parameter(torch.zeros(num_parents))，初值是 0。
                # 用 sigmoid 把它压到 (0,1) 区间——因为 sigmoid(0)=0.5，所以初始 α = 0.5，
                # 即文本和视觉各占一半，训练时模型自己学每个父类别该偏向哪边。
                self.proto_alpha_raw = nn.Parameter(torch.zeros(num_parents))
                alpha_desc = "per-parent learnable alpha, init=0.5"
            elif self.hybrid_alpha_mode == "fixed":
                fixed_alpha = torch.full((num_parents,), self.hybrid_alpha_fixed, dtype=torch.float32)
                if hybrid_alpha_parent_values is not None:
                    fixed_alpha = torch.tensor(list(hybrid_alpha_parent_values), dtype=torch.float32)
                    if fixed_alpha.numel() != num_parents:
                        raise ValueError(
                            f"hybrid_alpha_parent_values must have {num_parents} values, "
                            f"got {fixed_alpha.numel()}."
                        )
                fixed_alpha = fixed_alpha.clamp(0.0, 1.0)
                self.register_buffer("proto_alpha_fixed", fixed_alpha)
                alpha_desc = f"fixed alpha={fixed_alpha.tolist()}"
            else:
                raise ValueError(f"Unsupported hybrid_alpha_mode={hybrid_alpha_mode}")
            print(f"Hybrid text-visual prototypes enabled ({alpha_desc})")
            print(f"  Text prompts: {len(text_prompts)}, Visual protos: {self.visual_proto.shape[0]}")
            print(f"  Alpha shape: [{num_parents}] (one per parent class)")
            print(f"  Alpha mode: {self.hybrid_alpha_mode}")
            print(f"  Prompt mode: {text_prompt_mode} | Text encoder mode: {text_encoder_mode}")
            print(f"  Use hybrid prototype blending in forward: {self.use_hybrid_proto_blend}")
            # 打印文本与视觉原型之间的跨模态对齐质量（对角线为同类别相似度）
            with torch.no_grad():
                text_emb_cpu = text_emb.cpu().float()
                vp_cpu = self.visual_proto.cpu().float()
                vp_norm = vp_cpu / (vp_cpu.norm(dim=-1, keepdim=True) + 1e-8)
                cross_sim = text_emb_cpu @ vp_norm.t()
                diag = cross_sim.diag()
                class_names = ["TUM"] * 3 + ["STR"] * 3 + ["LYM"] * 3 + ["NEC"] * 3 if dataset_name == "bcss" else None
                print("  -- Cross-modal alignment (text<->visual cosine sim) --")
                for i, sim_val in enumerate(diag):
                    flag = "OK" if sim_val > 0.3 else "WARN" if sim_val > 0.1 else "BAD"
                    prefix = class_names[i] if class_names is not None else f"C{i}"
                    print(f"    [{flag}] {prefix}[{i%3}]: {sim_val:.4f}")
                print(f"  Mean same-subclass sim: {diag.mean():.4f}")
                # 文本提示词之间的相似度：过高说明提示词存在近似重复
                text_sim = text_emb_cpu @ text_emb_cpu.t()
                off_diag = text_sim[~torch.eye(text_sim.shape[0], dtype=bool)]
                print(f"  Text-text off-diag sim: mean={off_diag.mean():.4f}, max={off_diag.max():.4f}")
                if off_diag.max() > 0.85:
                    print(f"  WARNING: Some text prompts are near-duplicates (max={off_diag.max():.4f})")

        # 若启用背景提示词抑制：每个父类别一个背景提示词，编码后作为背景原型的初始化锚点
        if self.use_bg_prompt_suppression:
            bg_prompts = get_background_prompts(dataset_name, prompt_mode=bg_prompt_mode)
            if len(bg_prompts) != len(self.k_list):
                raise ValueError(
                    f"Background prompts must match number of parent classes. "
                    f"Got {len(bg_prompts)} prompts for {len(self.k_list)} parent classes."
                )
            bg_emb = encode_text_prompts(bg_prompts, encoder_mode=text_encoder_mode)
            self.register_buffer("bg_prompt_init", bg_emb)          # 背景原型初始锚点（固定）
            self.bg_visual_proto = nn.Parameter(bg_emb.clone())     # 可学习的背景视觉原型
            print("Class-associated background prompt suppression enabled")
            print(f"  Background prompts: {len(bg_prompts)} (one per parent class)")
            print(f"  Background prompt mode: {bg_prompt_mode} | Text encoder mode: {text_encoder_mode}")
            print("  Background branch uses text-guided visual prototypes (text anchor + learnable visual proto)")
            # 打印背景提示词与前景父类别文本原型之间的相似度
            with torch.no_grad():
                bg_norm = bg_emb.float() / (bg_emb.float().norm(dim=-1, keepdim=True) + 1e-8)
                if getattr(self, "use_text_proto", False):
                    fg_parent = []
                    start = 0
                    for k in self.k_list:
                        fg_parent.append(self.text_proto[start:start + k].float().mean(dim=0))
                        start += k
                    fg_parent = torch.stack(fg_parent, dim=0)
                    fg_parent = fg_parent / (fg_parent.norm(dim=-1, keepdim=True) + 1e-8)
                    sim = bg_norm @ fg_parent.t()
                    print("  -- Background prompt vs parent foreground text similarity --")
                    for idx in range(sim.shape[0]):
                        row = " ".join([f"{v:.3f}" for v in sim[idx]])
                        print(f"    BG[{idx}] -> {row}")

        # 文本锚定的通道掩码原型：用文本原型驱动通道选择，掩蔽视觉原型得到类别原型
        self.use_text_anchored_mask = use_text_anchored_mask
        self.text_anchor_sparsity_weight = float(text_anchor_sparsity_weight)
        if self.use_text_anchored_mask:
            if not self.use_text_proto:
                raise ValueError("use_text_anchored_mask requires use_text_proto=True")
            self.text_anchor = TextAnchoredPrototype(
                dim=512,
                num_subclasses=self.total_classes,
                tau=float(text_anchor_tau),
                map_bias=bool(text_anchor_map_bias),
            )
            print(f"Text-anchored prototype enabled "
                  f"(tau={text_anchor_tau}, sparsity_weight={text_anchor_sparsity_weight})")

        # 标签特征 = 视觉原型（后续 forward 中可能被混合原型替换）
        self.l_fea = self.visual_proto

        # 4 个可学习的温度系数（logit scale），初始化为 1/0.07
        self.logit_scale1 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.Parameter(torch.ones([1]) * 1 / 0.07)

    def get_param_groups(self):
        """返回优化器的参数分组：bias 和 1 维参数不做 weight decay，其余参数做。"""
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

    def aggregate_parent_predictions(self, predictions, method="max"):
        """把子类别预测聚合为父类别预测（默认取子类别最大值）。"""
        return merge_to_parent_predictions(predictions, self.k_list, method=method)

    def _get_parent_alpha(self, device):
        """获取每个父类别的混合权重 alpha（可学习模式用 sigmoid 约束到 (0,1)）。"""
        if self.hybrid_alpha_mode == "learnable":
            return torch.sigmoid(self.proto_alpha_raw).to(device)
        return self.proto_alpha_fixed.to(device)

    def _build_hybrid_prototypes(self, device):
        """构建混合原型：alpha * 文本原型 + (1-alpha) * 视觉原型，再归一化。"""
        visual = F.normalize(self.visual_proto.to(device), dim=-1)
        text = F.normalize(self.text_proto.to(device), dim=-1)

        parent_alpha = self._get_parent_alpha(device)
        # 把父类别 alpha 扩展到每个子类别（4个类别，每类3个子类原型）
        subclass_alpha = []
        for parent_idx, k in enumerate(self.k_list):
            subclass_alpha.extend([parent_alpha[parent_idx]] * k)
        subclass_alpha = torch.stack(subclass_alpha).unsqueeze(-1)

        hybrid = subclass_alpha * text + (1.0 - subclass_alpha) * visual
        return F.normalize(hybrid, dim=-1)

    def get_hybrid_alpha_analysis(self):
        """返回混合权重 alpha 的分析信息（各父类别取值、均值、最值），用于日志。"""
        if not getattr(self, "use_text_proto", False):
            return None
        with torch.no_grad():
            alpha = self._get_parent_alpha(device=self.visual_proto.device).detach().float().cpu()
            if len(self.k_list) == 4:
                names = ["TUM", "STR", "LYM", "NEC"]
            else:
                names = [f"C{i}" for i in range(len(self.k_list))]
            return {
                "mode": self.hybrid_alpha_mode,
                "values": {name: alpha[i].item() for i, name in enumerate(names)},
                "mean": alpha.mean().item(),
                "min": alpha.min().item(),
                "max": alpha.max().item(),
            }

    def get_background_prompt_bank(self, device=None):
        """返回归一化后的可学习背景原型（background prompt bank）。"""
        if not getattr(self, "use_bg_prompt_suppression", False):
            raise RuntimeError("Background prompt bank requested, but use_bg_prompt_suppression is disabled.")
        proto = self.bg_visual_proto if device is None else self.bg_visual_proto.to(device)
        return F.normalize(proto, dim=-1)

    def get_background_prompt_analysis(self):
        """返回背景原型的漂移量（相对初始锚点）与类间相似度，用于监控训练。"""
        if not getattr(self, "use_bg_prompt_suppression", False):
            return None
        with torch.no_grad():
            init = F.normalize(self.bg_prompt_init.detach().float().cpu(), dim=-1)
            cur = F.normalize(self.bg_visual_proto.detach().float().cpu(), dim=-1)
            # drift = 1 - 余弦相似度，衡量当前原型偏离初始锚点的程度
            drift = 1.0 - F.cosine_similarity(cur, init, dim=-1)
            inter = cur @ cur.t()
            off_diag = inter[~torch.eye(inter.shape[0], dtype=torch.bool)]
            return {
                "drift_mean": drift.mean().item(),
                "drift_max": drift.max().item(),
                "inter_mean": off_diag.mean().item() if off_diag.numel() > 0 else 0.0,
                "inter_max": off_diag.max().item() if off_diag.numel() > 0 else 0.0,
            }

    def get_text_anchor_analysis(self):
        """返回文本锚定掩码的统计信息（每子类均值、阈值、整体均值/最值），用于监控掩码收敛。"""
        if not getattr(self, "use_text_anchored_mask", False):
            return None
        with torch.no_grad():
            dev = self.text_anchor.threshold_raw.device
            _, mask = self.text_anchor(
                self.text_proto.to(dev).float(), self.visual_proto.to(dev).float())
            mask = mask.float()
            per_subclass = mask.mean(dim=1)
            return {
                "per_subclass_mean": per_subclass.cpu().numpy(),
                "threshold": self.text_anchor.threshold_raw.detach().float().cpu().numpy(),
                "mean": mask.mean().item(),
                "min": mask.min().item(),
                "max": mask.max().item(),
            }

    def _compute_scale_logits(self, feats, proto, logit_scale):
        """计算特征图每个像素与原型之间的余弦相似度，得到子类别得分图。"""
        bsz, ch, h, w = feats.shape
        # 将空间维展平为 token 序列，并对特征做 L2 归一化
        tokens = feats.permute(0, 2, 3, 1).reshape(bsz, h * w, ch)
        tokens = F.normalize(tokens, dim=-1)
        # logits = scale * (tokens · proto)，再还原为 (B, C, H, W)
        logits = logit_scale * (tokens @ proto.t().float())
        return logits.view(bsz, h, w, -1).permute(0, 3, 1, 2)

    def forward(self, x):
        # 编码器提取特征，再经 FPN 得到 4 个尺度的特征图
        uni_feats = self.encoder(x)  # 获取4个尺度的特征图
        _x = self.sfp(uni_feats)  # 特征金字塔

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        # 确定最终使用的标签原型：文本锚定掩码 / 混合原型 / 纯视觉原型
        aux_losses = {}
        if getattr(self, "use_text_anchored_mask", False):
            base_l_fea, proto_mask = self.text_anchor(
                self.text_proto.to(x.device), self.visual_proto.to(x.device))
            if self.text_anchor_sparsity_weight > 0:
                aux_losses["text_anchor_sparse"] = (
                    self.text_anchor_sparsity_weight * proto_mask.mean())
        elif getattr(self, "use_text_proto", False) and getattr(self, "use_hybrid_proto_blend", False):
            base_l_fea = self._build_hybrid_prototypes(x.device)
        else:
            base_l_fea = F.normalize(self.visual_proto.to(x.device), dim=-1)
        l_fea = base_l_fea

        # 将 512 维标签原型分别投影到 4 个 FPN 层级的通道维度
        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)

        # 逐尺度计算得分图（CAM），并全局平均池化得到分类 logits
        out1 = self._compute_scale_logits(_x[0], l_fea1, logit_scale1)
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, self.total_classes)

        out2 = self._compute_scale_logits(_x[1], l_fea2, logit_scale2)
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, self.total_classes)

        out3 = self._compute_scale_logits(_x[2], l_fea3, logit_scale3)
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, self.total_classes)

        out4 = self._compute_scale_logits(_x[3], l_fea4, logit_scale4)
        cam4 = out4.clone()
        cls4 = self.pooling(out4, (1, 1)).view(-1, self.total_classes)

        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, l_fea, self.k_list, aux_losses
