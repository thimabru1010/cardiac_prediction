import numpy as np
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk

# ------------------------------------------------------------------
# 0.  LOAD THE SAME DATA  (unchanged)
# ------------------------------------------------------------------
ct_img  = sitk.ReadImage('data/EXAMES/310176-20250505T180932Z-1-001/310176/G14.dcm')
mask_img = sitk.ReadImage('data/EXAMES/310176-20250505T180932Z-1-001/310176/GC3.dcm')

ct_np  = sitk.GetArrayFromImage(ct_img)[0].astype(np.float32)   # (512,512)
ps_rgb = sitk.GetArrayFromImage(mask_img)[0]                    # (H,W,3)

zoom = 1.57     # Vitrea zoom

# ------------------------------------------------------------------
# 1.  LATERAL + VERTICAL  CROP  BASED  ON  ZOOM
# ------------------------------------------------------------------
#   • Vitrea shows the *zoomed* CT (512×512) centred in its window.
#   • Width of that region in the screenshot = 512 * zoom
#   • Same for height (isotropic zoom).
#
H, W = ps_rgb.shape[:2]
crop_w = int(round(512 * zoom))
crop_h = int(round(512 * zoom))

# Find rough centre of CT in the screenshot (non‑black bbox)
mask = ps_rgb.mean(-1) > 10
ys, xs = np.where(mask)
cy = int(round(ys.mean()))
cx = int(round(xs.mean()))

# Make symmetric crop inside image bounds
x0 = max(0, cx - crop_w // 2)
x1 = min(W, x0 + crop_w)
x0 = x1 - crop_w                                  # adjust if hit right edge

y0 = max(0, cy - crop_h // 2)
y1 = min(H, y0 + crop_h)
y0 = y1 - crop_h                                  # adjust if hit bottom edge

ps_crop = ps_rgb[y0:y1, x0:x1]                    # shape ≈ (crop_h, crop_w)

# ------------------------------------------------------------------
# 2.  RESIZE  TO  512×512  (undo the zoom)
# ------------------------------------------------------------------
ps_resized = cv2.resize(ps_crop, (512, 512), interpolation=cv2.INTER_LINEAR)

# ------------------------------------------------------------------
# 3.  QUICK  OVERLAY  CHECK
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(ct_np, cmap='gray')
axs[0].set_title("CT image")
axs[0].axis('off')

axs[1].imshow(ps_resized)
axs[1].set_title("Cropped + resized screenshot")
axs[1].axis('off')

axs[2].imshow(ct_np, cmap='gray')
axs[2].imshow(ps_resized, alpha=0.4)
axs[2].set_title("Overlay α=0.4")
axs[2].axis('off')

plt.tight_layout(); plt.show()
