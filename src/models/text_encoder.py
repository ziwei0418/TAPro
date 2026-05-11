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


# BCSS single prompts — 3 per class to match k_list=[3,3,3,3]
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

# LUAD single prompts — 3 per class: TE, NEC, LYM, TAS
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

# BCSS class-associated background prompts — 1 per parent class
BCSS_BACKGROUND_PROMPTS = [
    # TUM background
    "stromal and inflammatory tissue surrounding tumor nests in breast histology",
    # STR background
    "adjacent tumor epithelium and inflammatory infiltrates within stromal breast tissue",
    # LYM background
    "surrounding tumor and stromal tissue around lymphocytic aggregates in breast histology",
    # NEC background
    "adjacent viable tumor and stromal tissue bordering necrotic breast regions",
]

# LUAD class-associated background prompts — 1 per parent class
LUAD_BACKGROUND_PROMPTS = [
    # TE background
    "surrounding stroma and inflammatory tissue around tumor epithelium in lung adenocarcinoma",
    # NEC background
    "adjacent viable tumor and stromal tissue surrounding necrotic lung regions",
    # LYM background
    "surrounding tumor epithelium and stromal tissue around lymphocytic infiltrates in lung tissue",
    # TAS background
    "adjacent tumor epithelium and inflammatory regions within tumor-associated stroma",
]

# Backward compat alias
TISSUE_TEXT_PROMPTS = BCSS_TEXT_PROMPTS


def _repeat_class_names(class_names, repeats=3):
    prompts = []
    for class_name in class_names:
        prompts.extend([class_name] * repeats)
    return prompts


def get_text_prompts(dataset_name="bcss", prompt_mode="pathology"):
    """Get text prompts for the given dataset and prompt mode."""
    prompt_mode = prompt_mode.lower()

    if dataset_name == "luad":
        if prompt_mode == "class_name":
            return _repeat_class_names(["tumor epithelium", "necrosis", "lymphocyte", "tumor-associated stroma"])
        return LUAD_TEXT_PROMPTS

    if prompt_mode == "class_name":
        return _repeat_class_names(["tumor", "stroma", "lymphocyte", "necrosis"])
    return BCSS_TEXT_PROMPTS


def get_background_prompts(dataset_name="bcss", prompt_mode="pathology"):
    """Get class-associated background prompts, one per parent class."""
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
    """Encode text prompts using CONCH text encoder → [N, 512] frozen embeddings.
    Falls back to MedCLIP if CONCH is not available."""
    encoder_mode = encoder_mode.lower()
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
