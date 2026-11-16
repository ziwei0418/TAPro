"""
组织原型（tissue prototype）的文本编码器 —— 支持 BCSS 和 LUAD 两个数据集

把病理组织描述文本编码成归一化的文本原型向量 [N, 512]，
优先使用病理专用的 CONCH 文本编码器，不可用时回退到 MedCLIP。
"""
import torch

# 可选依赖的全局标志
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor


# BCSS 数据集的正类提示词 —— 每类 3 条，与 k_list=[3,3,3,3] 对齐
# BCSS_TEXT_PROMPTS = [
#     # TUM（肿瘤，3 个子类）
#     "dense cluster of tumor epithelial cells in breast tissue",
#     "scattered invasive tumor cells with high nuclear grade",
#     "tumor cells forming glandular or cribriform structures",
#     # STR（基质，3 个子类）
#     "dense fibrous stromal tissue with abundant collagen bundles",
#     "loose reactive stroma with fibroblasts surrounding tumor",
#     "myxoid stromal tissue with sparse cellularity",
#     # LYM（淋巴细胞，3 个子类）
#     "dense lymphocytic aggregate in tumor microenvironment",
#     "scattered tumor-infiltrating lymphocytes between cancer cells",
#     "lymphocyte-rich inflammatory region with plasma cells",
#     # NEC（坏死，3 个子类）
#     "coagulative necrotic tissue with cell debris and ghost outlines",
#     "early necrosis with pyknotic and karyorrhectic nuclei",
#     "necrotic region with hemorrhagic extravasation",
# ]


BCSS_TEXT_PROMPTS = [
    # TUM — Tumor（肿瘤上皮）
    # TUM-1：密集肿瘤细胞
    "densely packed malignant breast epithelial cells with enlarged pleomorphic and hyperchromatic nuclei.",

    # TUM-2：浸润性肿瘤细胞
    "invasive malignant breast epithelial cells forming irregular nests and cords within surrounding stroma.",

    # TUM-3：腺管及筛状结构
    "malignant breast epithelial cells forming irregular glandular, tubular, or cribriform structures.",

    # STR — Stroma（基质）
    # STR-1：致密纤维化基质
    "dense fibrous breast stroma with abundant eosinophilic collagen bundles and spindle-shaped fibroblasts.",

    # STR-2：反应性肿瘤基质
    "reactive tumor-associated stroma containing fibroblasts and collagen fibers surrounding malignant epithelial nests.",

    # STR-3：疏松或黏液样基质
    "loose myxoid stromal tissue with sparse spindle-shaped cells and abundant extracellular matrix.",

    # LYM — Lymphocytes（淋巴细胞）
    # LYM-1：密集淋巴细胞聚集
    "dense aggregates of small lymphocytes with round hyperchromatic nuclei and scant cytoplasm.",

    # LYM-2：肿瘤浸润性淋巴细胞
    "small dark-staining lymphocytes infiltrating between malignant breast epithelial cells.",

    # LYM-3：基质内淋巴细胞浸润
    "clusters of small lymphocytes distributed within the tumor-associated stroma.",

    # NEC — Necrosis（坏死组织）
    # NEC-1：凝固性坏死
    "coagulative tumor necrosis with eosinophilic ghost cells and loss of nuclear detail.",

    # NEC-2：核固缩及核碎裂
    "necrotic tumor tissue containing pyknotic nuclei, karyorrhectic fragments, and cellular debris.",

    # NEC-3：大片无结构坏死
    "extensive amorphous eosinophilic necrotic regions with loss of viable cellular architecture."
]


# LUAD 数据集的正类提示词 —— 每类 3 条：TE, NEC, LYM, TAS
# LUAD_TEXT_PROMPTS = [
#     # TE（肿瘤上皮，3 个子类）
#     "dense cluster of tumor epithelial cells in lung adenocarcinoma",
#     "invasive lung tumor cells forming acinar or solid patterns",
#     "tumor epithelial cells with prominent nucleoli in lung tissue",
#     # NEC（坏死，3 个子类）
#     "coagulative necrotic lung tissue with cell debris",
#     "necrotic tumor region with pyknotic nuclei in lung adenocarcinoma",
#     "extensive necrosis within tumor bed of lung tissue",
#     # LYM（淋巴细胞，3 个子类）
#     "dense lymphocytic aggregate in lung tumor microenvironment",
#     "scattered tumor-infiltrating lymphocytes between lung cancer cells",
#     "lymphocyte-rich inflammatory region in lung tissue",
#     # TAS（肿瘤相关基质，3 个子类）
#     "dense fibrous tumor-associated stroma surrounding lung cancer",
#     "loose reactive stroma with myofibroblasts in lung adenocarcinoma",
#     "stromal tissue supporting tumor epithelial growth in lung",
# ]

LUAD_TEXT_PROMPTS = [
    # TE（肿瘤上皮，3 个子类）
    "malignant lung epithelial cells forming irregular glandular and acinar structures with crowded hyperchromatic nuclei",
    "densely packed malignant epithelial cells forming solid sheets or nests with enlarged pleomorphic nuclei",
    "malignant lung epithelial cells lining papillary structures with fibrovascular cores and atypical nuclei",
    # NEC（坏死，3 个子类）
    "coagulative tumor necrosis with eosinophilic ghost cells, faded nuclei, and partially preserved tissue architecture",
    "necrotic tumor tissue containing fragmented nuclear debris, karyorrhexis, and disintegrated cellular structures",
    "extensive amorphous eosinophilic necrotic regions with loss of viable cellular architecture and nuclear detail",
    # LYM（淋巴细胞，3 个子类）
    "dense aggregates of small lymphocytes with round hyperchromatic nuclei and scant cytoplasm",
    "small dark-staining lymphocytes infiltrating between malignant epithelial cells and within tumor nests",
    "clusters of small lymphocytes distributed within the peritumoral stroma, forming inflammatory infiltrates",
    # TAS（肿瘤相关基质，3 个子类）
    "dense desmoplastic stroma with abundant eosinophilic collagen bundles and elongated spindle-shaped fibroblast nuclei",
    "loose reactive tumor stroma containing scattered spindle-shaped fibroblasts and delicate collagen fibers",
    "fibrocollagenous stromal tissue surrounding invasive malignant glands, with spindle cells and extracellular matrix",
]

# BCSS 类别关联的背景提示词 —— 每个父类别 1 条
BCSS_BACKGROUND_PROMPTS = [
    # TUM 背景
    "stromal and inflammatory tissue surrounding tumor nests in breast histology",
    # STR 背景
    "adjacent tumor epithelium and inflammatory infiltrates within stromal breast tissue",
    # LYM 背景
    "surrounding tumor and stromal tissue around lymphocytic aggregates in breast histology",
    # NEC 背景
    "adjacent viable tumor and stromal tissue bordering necrotic breast regions",
]

# LUAD 类别关联的背景提示词 —— 每个父类别 1 条
LUAD_BACKGROUND_PROMPTS = [
    # TE 背景
    "surrounding stroma and inflammatory tissue around tumor epithelium in lung adenocarcinoma",
    # NEC 背景
    "adjacent viable tumor and stromal tissue surrounding necrotic lung regions",
    # LYM 背景
    "surrounding tumor epithelium and stromal tissue around lymphocytic infiltrates in lung tissue",
    # TAS 背景
    "adjacent tumor epithelium and inflammatory regions within tumor-associated stroma",
]

# 向后兼容别名
TISSUE_TEXT_PROMPTS = BCSS_TEXT_PROMPTS


def _repeat_class_names(class_names, repeats=3):
    """把每个类别名重复 repeats 次，用于 class_name 模式下生成提示词。"""
    prompts = []
    for class_name in class_names:
        prompts.extend([class_name] * repeats)
    return prompts


def get_text_prompts(dataset_name="bcss", prompt_mode="pathology"):
    """按数据集和提示词模式返回文本提示词列表。"""
    prompt_mode = prompt_mode.lower()  # 转换为小写

    if dataset_name == "luad":
        # class_name 模式：仅用类别名重复 3 次；否则用预定义的病理描述
        if prompt_mode == "class_name":
            return _repeat_class_names(["tumor epithelium", "necrosis", "lymphocyte", "tumor-associated stroma"])
        return LUAD_TEXT_PROMPTS

    if prompt_mode == "class_name":
        return _repeat_class_names(["tumor", "stroma", "lymphocyte", "necrosis"])
    return BCSS_TEXT_PROMPTS


def get_background_prompts(dataset_name="bcss", prompt_mode="pathology"):
    """返回类别关联的背景提示词，每个父类别一条。"""
    prompt_mode = prompt_mode.lower()

    if dataset_name == "luad":
        if prompt_mode == "class_name":
            return [
                "background around tumor epithelium",
                "background around necrosis",
                "background around lymphocyte regions",
                "background around tumor-associated stroma",
            ]
        return LUAD_BACKGROUND_PROMPTS

    if prompt_mode == "class_name":
        return [
            "background around tumor",
            "background around stroma",
            "background around lymphocyte regions",
            "background around necrosis",
        ]
    return BCSS_BACKGROUND_PROMPTS


@torch.no_grad()
def encode_text_prompts(prompts, encoder_mode="auto"):
    """用 CONCH 文本编码器把提示词编码为 [N, 512] 的冻结嵌入（不参与梯度）。

    若 CONCH 不可用，则回退到 MedCLIP 编码器。
    """
    encoder_mode = encoder_mode.lower()
    # auto 或显式指定 conch 且 CONCH 可用时，优先用 CONCH
    use_conch = encoder_mode in {"auto", "conch"} and CONCH_AVAILABLE

    if use_conch:
        print("  Using CONCH text encoder (histopathology-specialized)")
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
        )
        conch_model.eval()
        if torch.cuda.is_available():
            conch_model = conch_model.cuda()
        tokenizer = get_tokenizer()
        # CONCH 的 get_tokenizer() 可能返回 HuggingFace 或 open_clip 两种 tokenizer
        try:
            # 先尝试 open_clip 风格的调用方式
            tokens = tokenizer(prompts, context_length=77)
        except TypeError:
            # HuggingFace 的 PreTrainedTokenizerFast
            inputs = tokenizer(prompts, max_length=77, truncation=True,
                               padding=True, return_tensors="pt")
            tokens = inputs['input_ids']
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        text_features = conch_model.encode_text(tokens)
        # 断言输出维度为 512（下游 AdaptiveLayer 依赖此维度）
        assert text_features.shape[-1] == 512, \
            f"CONCH output dim={text_features.shape[-1]}, expected 512. Update AdaptiveLayer!"
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del conch_model, tokenizer
        print(f"  Encoded {len(prompts)} text prompts via CONCH → shape {text_features.shape}")
        return text_features.float().cpu()
    else:
        # 回退到 MedCLIP
        if encoder_mode == "conch" and not CONCH_AVAILABLE:
            print("  Requested CONCH text encoder but CONCH is unavailable. Falling back to MedCLIP.")
        print("  Falling back to MedCLIP text encoder")
        processor = MedCLIPProcessor()
        clip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        clip_model.eval()
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True)
        text_features = clip_model.encode_text(inputs['input_ids'], inputs['attention_mask'])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del clip_model, processor
        print(f"  Encoded {len(prompts)} text prompts via MedCLIP → shape {text_features.shape}")
        return text_features.float().cpu()
