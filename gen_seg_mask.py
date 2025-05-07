import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift  # or use cv2.warpAffine for integer shift

def apply_window(img, center, width):
    """Apply DICOM windowing."""
    img = img.astype(np.float32)
    min_val = center - width / 2
    max_val = center + width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255.0
    return img.astype(np.uint8)

def window_level(arr, center=40, width=350):
    lo, hi = center - width/2, center + width/2
    arr = np.clip(arr, lo, hi)
    # return arr, (arr - lo) / (hi - lo)             # 0‑1 for imshow()
    return arr

# Manual crop and resize (zoom effect)
def crop_and_resize(img, zoom_factor):
    # img: (y, x) or (y, x, c)
    y, x = img.shape[-3], img.shape[-2]
    crop_y = int(y / zoom_factor)
    crop_x = int(x / zoom_factor)
    start_y = (y - crop_y) // 2
    start_x = (x - crop_x) // 2
    if img.ndim == 2:
        cropped = img[start_y:start_y+crop_y, start_x:start_x+crop_x]
        resized = cv2.resize(cropped, (x, y), interpolation=cv2.INTER_LINEAR)
    elif img.ndim == 3:
        cropped = img[:, start_y:start_y+crop_y, start_x:start_x+crop_x]
        resized = np.stack([
            cv2.resize(cropped[z], (x, y), interpolation=cv2.INTER_LINEAR)
            for z in range(cropped.shape[0])
        ])
    elif img.ndim == 4:
        cropped = img[:, start_y:start_y+crop_y, start_x:start_x+crop_x, :]
        resized = np.stack([
            cv2.resize(cropped[z], (x, y), interpolation=cv2.INTER_LINEAR)
            for z in range(cropped.shape[0])
        ])
    else:
        raise RuntimeError("Unexpected mask image shape")
    return resized

# Manual crop only (no resize)
def crop_only(img, zoom_factor):
    # img: (z, y, x) or (z, y, x, c) or (y, x)
    y, x = img.shape[0], img.shape[1]
    print(x, y)
    crop_x = int(x / zoom_factor)
    start_x = (x - crop_x) // 2
    end_x = start_x + crop_x
    if img.ndim in [2, 3]:
        return img[:, start_x:end_x]
    elif img.ndim == 4:
        return img[:, :, start_x:end_x, :]
    else:
        raise RuntimeError("Unexpected mask image shape")

def tight_crop(arr, thr=10):
    if arr.ndim == 2:
        print('here')
        mask = arr
    else:
        mask = arr.mean(-1)
    ys, xs = np.where(mask > thr)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return arr[y0:y1+1, x0:x1+1, ...]

def hue_mask(hsv_array, hue_range, sat_thresh=30, val_thresh=30):
    h, s, v = cv2.split(hsv_array)
    lo, hi = hue_range
    if lo <= hi:
        mask_h = cv2.inRange(h, lo, hi)
    else:                              # wrap-around (red)
        mask_h = cv2.bitwise_or(cv2.inRange(h, 0, hi),
                                cv2.inRange(h, lo, 179))
    mask_s = cv2.inRange(s, sat_thresh, 255)   # suppress grey text
    mask_v = cv2.inRange(v, val_thresh, 255)   # suppress dark bg
    return cv2.bitwise_and(mask_h, mask_s, mask_v)   # binary 0/255

if __name__ == "__main__":
    # Read the exam_img from a DICOM file
    dicom_path = 'data/EXAMES/310176-20250505T180932Z-1-001/310176/G14.dcm'
    ct_img = sitk.ReadImage(dicom_path)
    
    dicom_path = 'data/EXAMES/310176-20250505T180932Z-1-001/310176/GC3.dcm'
    mask_img = sitk.ReadImage(dicom_path)

    mask_np = sitk.GetArrayFromImage(mask_img)[0]  # (z, y, x) or (z, y, x, c)
    ct_np = sitk.GetArrayFromImage(ct_img)[0]  # (z, y, x)
    ct_slice = window_level(ct_np)       # Use first slice, apply window/level
    # mask_np = window_level(mask_np)  # Apply window/level to mask
    zoom_factor = 1.57

    # Apply manual crop (zoom effect, no resize)
    print(mask_np.shape)
    if mask_np.ndim in [3, 4]:
        cropped_mask_np = crop_only(mask_np, zoom_factor)
    else:
        raise RuntimeError("Unexpected mask image shape")
    
    # Remove the letters written from top to allow tight_crop
    cropped_mask_np = cropped_mask_np[50:, :]
    
    cropped_mask_np = tight_crop(cropped_mask_np, thr=10)
    cropped_mask_np = cropped_mask_np[:, :-40]

    print(ct_slice.shape)
    print(ct_slice.min(), ct_slice.max())
    ct_slice = tight_crop(ct_slice, thr=10)
    
    # Resize cropped mask to 512×512 for plotting
    target_size = (512, 512)
    
    # kernel = np.ones((3,3), np.uint8)
    # cropped_mask_np = cv2.morphologyEx(cropped_mask_np, cv2.MORPH_OPEN, kernel)
    
    mask_resized = cv2.resize(cropped_mask_np, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    ct_slice = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    kernel = np.ones((3,3), np.uint8)
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
    
    # Plot CT, cropped mask, and overlay
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # first row
    calc_candidates = ct_slice.copy()
    calc_candidates[ct_slice < 130] = 0
    calc_candidates[ct_slice > 130] = 1
    ct_slice2 = ct_slice.copy()
    ct_slice2 = (ct_slice2 - ct_slice2.min()) / (ct_slice2.max() - ct_slice2.min())
    axs[0,0].imshow(ct_slice2, cmap='gray')
    axs[0,0].set_title("CT image (window/level)")
    axs[0,0].axis('off')

    # Show resized mask
    # if mask_resized.ndim == 2:
    #     axs[1].imshow(mask_resized, cmap='gray')
    # else:
    print(mask_resized.shape)
    calc_candidates = np.stack([
        calc_candidates,
        calc_candidates,
        calc_candidates
    ], axis=-1)
    # mask_resized = mask_resized * calc_candidates
    # lower_r_bound = 200
    # upper_r_bound = 255
    # lower_g_bound = 200
    # upper_g_bound = 255
    # lower_b_bound = 200
    # upper_b_bound = 255
    # Apply color map
    print(mask_resized.shape, mask_resized.min(), mask_resized.max())
    mask_resized2 = mask_resized.copy()
    # mask_resized2 = (mask_resized2 - mask_resized2.min()) / (mask_resized2.max() - mask_resized2.min())
    axs[0,1].imshow(mask_resized2)
        
    axs[0,1].set_title(f"Cropped mask_img (zoom={zoom_factor})")
    axs[0,1].axis('off')

    # Overlay CT with mask at 50% opacity
    axs[0,2].imshow(ct_slice, cmap='gray')
    axs[0,2].imshow(mask_resized, cmap='jet', alpha=0.5)
    axs[0,2].set_title("Overlay (alpha=0.5)")
    axs[0,2].axis('off')

    # second row (repeat or replace with other images)
    axs[1,0].imshow(calc_candidates, cmap='gray')
    axs[1,0].set_title("CT image (window/level) [row 2]")
    axs[1,0].axis('off')
    
    # mask_segs = mask_resized * calc_candidates
    RED1  = (0,   5)      # 0°-10°
    RED2  = (170, 179)    # 340°-360°
    GREEN = (50,  80)     # 100°-160°  (LAD in your screenshots)
    BLUE  = (100,130)     # 200°-260°  (CX)
    PINK  = (145,175)     # 290°-350°  (calcification contour)

    mask_segs = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2HSV)
    
    mask_segs = hue_mask(mask_segs, GREEN)
    # mask_segs  = hue_mask(mask_segs, BLUE)
    # mask_pink  = hue_mask(PINK)
    # mask_segs   = hue_mask(mask_segs, RED1) | hue_mask(mask_segs, RED2)   # combine two red intervals

    # optional clean-up: remove isolated edge noise
    # kernel = np.ones((3,3), np.uint8)
    # mask_segs = cv2.morphologyEx(mask_segs, cv2.MORPH_OPEN, kernel)

    # mask_segs = mask_resized[:, :, 1]
    print(mask_segs.shape, mask_segs.min(), mask_segs.max())
    # mask_segs[mask_segs < 150] = 0
    # mask_segs[mask_segs >= 150] = 255
    # mask_segs[:, :] = 255
    
    mask_segs2 = mask_segs * calc_candidates[:, :, 0]
    # mask_segs2 = ct_slice - 
    
    axs[1,1].imshow(mask_segs2, cmap='gray')
    axs[1,1].set_title("Mask Blue channel")
    axs[1,1].axis('off')
    
    axs[1,2].imshow(ct_slice, cmap='gray')
    axs[1,2].imshow(mask_segs2, cmap='jet', alpha=0.2)
    axs[1,2].set_title("Blue Channel and Calcification")
    axs[1,2].axis('off')

    plt.tight_layout()
    fig.savefig('data/overlay.png', dpi=300)
    plt.show()


