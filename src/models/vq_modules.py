"""
Vector Quantization modules for discrete semantic prototypes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQCodebook(nn.Module):
    """Semantic Vector Quantization codebook with EMA updates.
    Uses cosine similarity for codebook matching.
    Supports text-guided initialization: codebook entries are seeded from
    text prototype projections (macro→micro knowledge transfer).
    """
    def __init__(self, num_codes=64, code_dim=256, commitment_weight=0.25,
                 ema_decay=0.99, dead_code_threshold=2):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook embeddings (NOT trained by gradient — updated via EMA)
        codebook = torch.randn(num_codes, code_dim)
        codebook = F.normalize(codebook, dim=-1)  # init normalized
        self.register_buffer('codebook', codebook)
        self.register_buffer('ema_count', torch.zeros(num_codes))
        self.register_buffer('ema_weight', self.codebook.clone())
        self.register_buffer('usage_count', torch.zeros(num_codes))
        self._initialized = False

    def init_from_prototypes(self, proto_codes):
        """Text-guided init: seed codebook from projected text prototypes.
        proto_codes: [num_protos, code_dim] — text protos projected to code space.
        Each prototype spawns num_codes/num_protos entries with small noise.
        """
        num_protos = proto_codes.shape[0]
        codes_per_proto = self.num_codes // num_protos
        remainder = self.num_codes - codes_per_proto * num_protos
        all_codes = []
        for i in range(num_protos):
            n_copies = codes_per_proto + (1 if i < remainder else 0)
            base = proto_codes[i:i+1].expand(n_copies, -1)  # [n, D]
            noise = torch.randn_like(base) * 0.05  # small perturbation
            all_codes.append(base + noise)
        all_codes = F.normalize(torch.cat(all_codes, dim=0), dim=-1)  # [K, D]
        self.codebook.copy_(all_codes)
        self.ema_weight.copy_(all_codes)
        self._initialized = True
        print(f"    VQ codebook initialized from {num_protos} text prototypes "
              f"({codes_per_proto}+ codes each)")

    def _init_from_data(self, z_c):
        """Fallback: initialize from first batch of encoder outputs (if no text init)."""
        if self._initialized:
            return  # already initialized from text prototypes
        n = min(z_c.shape[0], self.num_codes)
        idx = torch.randperm(z_c.shape[0])[:n]
        init_data = F.normalize(z_c[idx].detach().float(), dim=-1)
        self.codebook[:n] = init_data
        self.ema_weight[:n] = init_data
        self._initialized = True

    def forward(self, z_c):
        # z_c: [N, D] continuous features
        if not self._initialized and self.training:
            self._init_from_data(z_c)

        # Cosine distance: normalize both, then max similarity
        z_norm = F.normalize(z_c, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)
        cosine_sim = z_norm @ cb_norm.t()          # [N, K]
        indices = torch.argmax(cosine_sim, dim=1)  # [N]
        z_q = self.codebook[indices]                # [N, D]

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.num_codes).float()  # [N, K]
                batch_count = one_hot.sum(dim=0)                      # [K]
                batch_sum = one_hot.t() @ z_c.detach().float()        # [K, D]

                self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * batch_count
                self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * batch_sum

                # Laplace smoothing + normalize codebook
                n = self.ema_count.sum()
                count_smoothed = (self.ema_count + 1e-5) / (n + self.num_codes * 1e-5) * n
                self.codebook = self.ema_weight / count_smoothed.unsqueeze(1)
                self.codebook = F.normalize(self.codebook, dim=-1)  # keep normalized

                # Track usage for dead code revival
                self.usage_count += batch_count

            # Dead code revival: reset unused codes to random encoder outputs
            if self.usage_count.sum() > 0:
                dead_mask = self.usage_count < self.dead_code_threshold
                num_dead = dead_mask.sum().item()
                if num_dead > 0 and z_c.shape[0] > 0:
                    num_replace = min(num_dead, z_c.shape[0])
                    replace_idx = torch.randperm(z_c.shape[0])[:num_replace]
                    dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_replace]
                    with torch.no_grad():
                        new_codes = F.normalize(z_c[replace_idx].detach().float(), dim=-1)
                        self.codebook[dead_indices] = new_codes
                        self.ema_weight[dead_indices] = new_codes
                        self.ema_count[dead_indices] = 1.0
                        self.usage_count[dead_indices] = 0.0

        # Commitment loss on normalized space (cosine-friendly)
        commitment_loss = F.mse_loss(z_norm, F.normalize(z_q, dim=-1).detach())
        vq_loss = self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q_st = z_c + (z_q - z_c).detach()
        return z_q_st, vq_loss


class VQCAMHead(nn.Module):
    """Produces discrete CAM from VQ codebook for one scale.
    Includes semantic regularization and text-guided codebook initialization."""
    def __init__(self, in_dim, num_classes, num_codes=128, code_dim=256,
                 commitment_weight=0.25, proto_dim=512):
        super().__init__()
        self.code_dim = code_dim
        self.proto_dim = proto_dim
        self.project = nn.Sequential(
            nn.Linear(in_dim, code_dim),
            nn.ReLU(),
            nn.Linear(code_dim, code_dim),
        )
        self.vq = VQCodebook(num_codes, code_dim, commitment_weight)
        self.cam_head = nn.Conv2d(code_dim, num_classes, 1)
        # Semantic regularization: project VQ codes → prototype space
        self.sem_reg_proj = nn.Linear(code_dim, proto_dim, bias=False)
        # Reverse projection: prototype space → code space (for text-guided init)
        self.proto_to_code = nn.Linear(proto_dim, code_dim, bias=False)

    def init_codebook_from_prototypes(self, text_prototypes):
        """Initialize VQ codebook from text prototypes (macro→micro transfer).
        text_prototypes: [num_protos, proto_dim] (e.g. [12, 512])
        Projects text protos to code_dim, then seeds codebook.
        """
        with torch.no_grad():
            proto_codes = self.proto_to_code(text_prototypes.float())  # [12, 256]
            proto_codes = F.normalize(proto_codes, dim=-1)
            self.vq.init_from_prototypes(proto_codes)

    def forward(self, features, H, W):
        # features: [B*H*W, in_dim]
        B = features.shape[0] // (H * W)
        z = self.project(features)                          # [B*H*W, code_dim]
        z_q, vq_loss = self.vq(z)                           # [B*H*W, code_dim]
        z_q = z_q.view(B, H, W, self.code_dim).permute(0, 3, 1, 2)  # [B, D, H, W]
        cam_vq = self.cam_head(z_q)                         # [B, num_classes, H, W]
        return cam_vq, vq_loss

    def compute_sem_reg_loss(self, prototypes, tau=0.1):
        """Semantic regularization: pull VQ codes toward nearest prototypes.
        prototypes: [num_protos, proto_dim] (l_fea, the hybrid prototypes)
        Returns scalar loss.
        """
        # Project VQ codes to prototype space
        vq_codes = self.vq.codebook.detach()  # [128, 256] — detach to not backprop through EMA codebook
        vq_proj = self.sem_reg_proj(vq_codes)  # [128, 512]
        vq_proj = F.normalize(vq_proj, dim=-1)
        proto_norm = F.normalize(prototypes.detach(), dim=-1)  # [12, 512]

        # Soft assignment: each VQ code → which prototype?
        sim = vq_proj @ proto_norm.t()  # [128, 12]
        assign = F.softmax(sim / tau, dim=1)  # [128, 12]

        # Target: weighted combination of prototypes
        target = assign @ proto_norm  # [128, 512]
        target = F.normalize(target, dim=-1)

        # Loss: maximize cosine similarity between projected VQ codes and targets
        loss = 1.0 - (vq_proj * target).sum(dim=-1).mean()
        return loss
