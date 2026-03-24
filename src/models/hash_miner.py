"""
Hash-guided hard negative mining modules
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HashProjectionHead(nn.Module):
    """Projects features to binary hash codes with straight-through estimator."""
    def __init__(self, in_dim=512, hash_dim=128):
        super().__init__()
        self.fc = nn.Linear(in_dim, hash_dim)
        self.bn = nn.BatchNorm1d(hash_dim)

    def forward(self, x):
        h = self.bn(self.fc(x))
        # sign() with straight-through estimator for gradient
        binary = h + (torch.sign(h) - h).detach()
        return binary, h


class HashMemoryBank:
    """FIFO memory bank storing patch features + hashes + labels for hard negative mining.
    Stores features from previous batches so HashContrastiveLoss can mine
    hard negatives across thousands of patches instead of just 3 text prototypes.
    """
    def __init__(self, capacity=4096, feat_dim=512, hash_dim=128):
        self.capacity = capacity
        self.feat_dim = feat_dim
        self.hash_dim = hash_dim
        self.features = None   # [capacity, feat_dim]
        self.hashes = None     # [capacity, hash_dim]
        self.labels = None     # [capacity] — class index per entry
        self.ptr = 0
        self.full = False

    def push(self, features, hashes, labels):
        """Add batch of features to memory bank (FIFO)."""
        features = features.detach()
        hashes = hashes.detach()
        labels = labels.detach()
        n = features.shape[0]
        device = features.device

        if self.features is None:
            self.features = torch.zeros(self.capacity, self.feat_dim, device=device)
            self.hashes = torch.zeros(self.capacity, self.hash_dim, device=device)
            self.labels = torch.zeros(self.capacity, dtype=torch.long, device=device)

        if n >= self.capacity:
            self.features = features[-self.capacity:].clone()
            self.hashes = hashes[-self.capacity:].clone()
            self.labels = labels[-self.capacity:].clone()
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + n
        if end <= self.capacity:
            self.features[self.ptr:end] = features
            self.hashes[self.ptr:end] = hashes
            self.labels[self.ptr:end] = labels
        else:
            first = self.capacity - self.ptr
            self.features[self.ptr:] = features[:first]
            self.hashes[self.ptr:] = hashes[:first]
            self.labels[self.ptr:] = labels[:first]
            rest = n - first
            self.features[:rest] = features[first:]
            self.hashes[:rest] = hashes[first:]
            self.labels[:rest] = labels[first:]
            self.full = True
        self.ptr = end % self.capacity

    @property
    def size(self):
        return self.capacity if self.full else self.ptr

    def get_all(self):
        """Return all valid entries."""
        s = self.size
        return self.features[:s], self.hashes[:s], self.labels[:s]


class HashHardNegativeMiner(nn.Module):
    """Hash-guided hard negative mining for InfoNCE.
    Instead of replacing InfoNCE, this module ENHANCES it by mining
    hard negatives from a memory bank using binary hash codes.
    Returns hard negative features that are fed back to InfoNCE.
    """
    def __init__(self, in_dim=512, hash_dim=128, top_k=16,
                 temperature=0.07, quant_weight=0.01, bank_capacity=4096):
        super().__init__()
        self.hash_head = HashProjectionHead(in_dim, hash_dim)
        self.top_k = top_k
        self.temperature = temperature
        self.quant_weight = quant_weight
        self.hash_dim = hash_dim
        self.memory_bank = HashMemoryBank(capacity=bank_capacity,
                                          feat_dim=in_dim, hash_dim=hash_dim)
        print(f"HashHardNegativeMiner: bank_capacity={bank_capacity}, top_k={top_k}")

    def forward(self, fg_features, fg_labels=None):
        """Mine hard negatives and compute auxiliary hash loss."""
        device = fg_features.device
        batch_size = fg_features.shape[0]

        fg_hash, fg_h = self.hash_head(fg_features)
        fg_norm = F.normalize(fg_features, dim=-1)

        quant_loss = self.quant_weight * (1 - fg_h.abs()).mean()

        bank_size = self.memory_bank.size
        use_bank = bank_size >= self.top_k

        hard_neg_loss = torch.tensor(0.0, device=device)

        if use_bank and fg_labels is not None:
            bank_feats, bank_hashes, bank_labels = self.memory_bank.get_all()
            bank_feats_norm = F.normalize(bank_feats, dim=-1)

            for i in range(batch_size):
                curr_fg = fg_norm[i:i+1]
                curr_label = fg_labels[i]

                with torch.no_grad():
                    curr_hash_det = fg_hash[i:i+1].detach()
                    hamming_dist = (self.hash_dim - curr_hash_det @ bank_hashes.t()) / 2
                    hamming_dist = hamming_dist.squeeze(0)

                    same_class_mask = (bank_labels == curr_label)
                    hamming_dist[same_class_mask] = float('inf')
                    valid_k = min(self.top_k, (~same_class_mask).sum().item())

                if valid_k > 0:
                    _, top_k_idx = hamming_dist.topk(valid_k, largest=False)
                    hard_negs = bank_feats_norm[top_k_idx]

                    pos_mask = (bank_labels == curr_label)
                    if pos_mask.sum() > 0:
                        pos_feats = bank_feats_norm[pos_mask][:self.top_k]
                        pos_sim = torch.exp(curr_fg @ pos_feats.t() / self.temperature).sum()
                    else:
                        pos_sim = torch.exp(torch.tensor(1.0 / self.temperature, device=device))

                    neg_sim = torch.exp(curr_fg @ hard_negs.t() / self.temperature).sum()
                    hard_neg_loss = hard_neg_loss - torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))

            hard_neg_loss = hard_neg_loss / max(batch_size, 1)

        if self.training and fg_labels is not None:
            self.memory_bank.push(fg_features.detach(), fg_hash.detach(), fg_labels.detach())

        return hard_neg_loss + quant_loss
