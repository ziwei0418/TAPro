import numpy as np
import cv2
import torch

from PIL import Image

cam4_p = torch.load('checkpoints/2026-03-10-18-29/best_cam.pth', map_location='cpu')['model']
# Actually I'll just change the visualization script to use thresh=0.2 and see if it looks like what the user expects.
