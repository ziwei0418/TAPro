# ============================================================
# CRF Post-processing for Segmentation
# ============================================================

import numpy as np

try:
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as crf_utils
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    print("Warning: pydensecrf not available, CRF disabled")


def apply_crf(img_rgb, prob_map, iter_max=10):
    """Apply DenseCRF post-processing to probability map."""
    if not CRF_AVAILABLE:
        return prob_map.argmax(0)
    C, H, W = prob_map.shape
    U = crf_utils.unary_from_softmax(prob_map)
    U = np.ascontiguousarray(U)
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=1, compat=3)
    d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=np.ascontiguousarray(img_rgb), compat=4)
    Q = np.array(d.inference(iter_max)).reshape(C, H, W)
    return Q.argmax(0)
