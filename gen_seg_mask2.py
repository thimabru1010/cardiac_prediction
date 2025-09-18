import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths provided by the user prompt
mask_path = Path('data/mask_resized.png')
ct_path = Path('data/ct_slice.png')

# Read images
mask_rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)   # BGR by default
ct_rgb = cv2.imread(str(ct_path), cv2.IMREAD_COLOR)

if mask_rgb is None or ct_rgb is None:
    raise FileNotFoundError("One of the images could not be loaded. Check the paths.")

# Convert both to grayscale for alignment
mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
ct_gray = cv2.cvtColor(ct_rgb, cv2.COLOR_BGR2GRAY)

# Normalize to float32 [0,1]
mask_float = mask_gray.astype(np.float32) / 255.0
ct_float = ct_gray.astype(np.float32) / 255.0

# Ensure images are the same size
if mask_float.shape != ct_float.shape:
    raise ValueError(f"Image shapes differ. mask: {mask_float.shape}, ct: {ct_float.shape}")

# Initialize warp matrix (translation + rotation allowed here)
warp_mode = cv2.MOTION_AFFINE  # allows translation + rotation + scale
warp_matrix = np.eye(2, 3, dtype=np.float32)

# Define the stopping criteria for ECC
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)

# Run the ECC algorithm
(cc, warp_matrix) = cv2.findTransformECC(
    templateImage=ct_float,
    inputImage=mask_float,
    warpMatrix=warp_matrix,
    motionType=warp_mode,
    criteria=criteria,
    inputMask=None, # type: ignore
    gaussFiltSize=5
) # type: ignore

# Warp the mask image (color) to CT space
aligned_mask_rgb = cv2.warpAffine(
    mask_rgb,
    warp_matrix,
    (ct_rgb.shape[1], ct_rgb.shape[0]),
    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_TRANSPARENT
)

# Create an overlay for visualization
overlay = cv2.addWeighted(ct_rgb, 0.7, aligned_mask_rgb, 0.3, 0)

# Save the aligned mask for user download
aligned_path = Path("data/aligned_mask.png")
cv2.imwrite(str(aligned_path), aligned_mask_rgb)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original CT")
plt.axis('off')
plt.imshow(cv2.cvtColor(ct_rgb, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Original Mask (Unaligned)")
plt.axis('off')
plt.imshow(cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Overlay after Alignment")
plt.axis('off')
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()

# aligned_path
