import torch
from src.models.network import ClsNetwork

# Initialize network
net = ClsNetwork(
    hf_token="test_token", 
    use_vq=True, 
    use_text_proto=False,
    l_fea_path='/home/s12gbn5/Desktop/WSSS_2026_RESEARCH/scripts/logs/vis_protos.pkl' # Assuming exist
)

# Test forward pass
B = 2
C, H, W = 3, 224, 224
dummy_img = torch.randn(B, C, H, W)
print("Forward pass...")
with torch.no_grad():
    out = net(dummy_img)

print("Output shape 0:", out[0].shape)
print("vq_loss:", out[-1]['vq_loss'])
print("Success!")
