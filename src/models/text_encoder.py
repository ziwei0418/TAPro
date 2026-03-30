"""
Text encoder for tissue prototypes — supports BCSS and LUAD
"""
import torch

# Global flags for optional dependencies
try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor


# BCSS text prompts — 3 per class to match k_list=[3,3,3,3]
BCSS_TEXT_PROMPTS = [
    # TUM (3 subclasses)
    "dense cluster of tumor epithelial cells in breast tissue",
    "scattered invasive tumor cells with high nuclear grade",
    "tumor cells forming glandular or cribriform structures",
    # STR (3 subclasses)
    "dense fibrous stromal tissue with abundant collagen bundles",
    "loose reactive stroma with fibroblasts surrounding tumor",
    "myxoid stromal tissue with sparse cellularity",
    # LYM (3 subclasses)
    "dense lymphocytic aggregate in tumor microenvironment",
    "scattered tumor-infiltrating lymphocytes between cancer cells",
    "lymphocyte-rich inflammatory region with plasma cells",
    # NEC (3 subclasses)
    "coagulative necrotic tissue with cell debris and ghost outlines",
    "early necrosis with pyknotic and karyorrhectic nuclei",
    "necrotic region with hemorrhagic extravasation",
]

# LUAD text prompts — 3 per class: TE, NEC, LYM, TAS
LUAD_TEXT_PROMPTS = [
    # TE (3 subclasses)
    "dense cluster of tumor epithelial cells in lung adenocarcinoma",
    "invasive lung tumor cells forming acinar or solid patterns",
    "tumor epithelial cells with prominent nucleoli in lung tissue",
    # NEC (3 subclasses)
    "coagulative necrotic lung tissue with cell debris",
    "necrotic tumor region with pyknotic nuclei in lung adenocarcinoma",
    "extensive necrosis within tumor bed of lung tissue",
    # LYM (3 subclasses)
    "dense lymphocytic aggregate in lung tumor microenvironment",
    "scattered tumor-infiltrating lymphocytes between lung cancer cells",
    "lymphocyte-rich inflammatory region in lung tissue",
    # TAS (3 subclasses)
    "dense fibrous tumor-associated stroma surrounding lung cancer",
    "loose reactive stroma with myofibroblasts in lung adenocarcinoma",
    "stromal tissue supporting tumor epithelial growth in lung",
]

# Backward compat alias
TISSUE_TEXT_PROMPTS = BCSS_TEXT_PROMPTS


def get_text_prompts(dataset_name="bcss"):
    """Get text prompts for the given dataset."""
    if dataset_name == "luad":
        return LUAD_TEXT_PROMPTS
    return BCSS_TEXT_PROMPTS


@torch.no_grad()
def encode_text_prompts(prompts):
    """Encode text prompts using CONCH text encoder → [N, 512] frozen embeddings.
    Falls back to MedCLIP if CONCH is not available."""
    if CONCH_AVAILABLE:
        print("  Using CONCH text encoder (histopathology-specialized)")
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
        )
        conch_model.eval()
        if torch.cuda.is_available():
            conch_model = conch_model.cuda()
        tokenizer = get_tokenizer()
        # CONCH get_tokenizer() may return HuggingFace or open_clip tokenizer
        try:
            # Try open_clip style first
            tokens = tokenizer(prompts, context_length=77)
        except TypeError:
            # HuggingFace PreTrainedTokenizerFast
            inputs = tokenizer(prompts, max_length=77, truncation=True,
                               padding=True, return_tensors="pt")
            tokens = inputs['input_ids']
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        text_features = conch_model.encode_text(tokens)
        assert text_features.shape[-1] == 512, \
            f"CONCH output dim={text_features.shape[-1]}, expected 512. Update AdaptiveLayer!"
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        del conch_model, tokenizer
        print(f"  Encoded {len(prompts)} text prompts via CONCH → shape {text_features.shape}")
        return text_features.float().cpu()
    else:
        # Fallback to MedCLIP
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
