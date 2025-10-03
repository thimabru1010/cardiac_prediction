import matplotlib.pyplot as plt
import numpy as np
import cv2
# from gen_seg_mask import hue_mask

# mask_segs = mask_resized * calc_candidates
RED1  = (0,   5)      # 0°-10°
RED2  = (170, 179)    # 340°-360°
GREEN = (50,  80)     # 100°-160°  (LAD in your screenshots)
BLUE  = (100,130)     # 200°-260°  (CX)
PINK  = (145,175)     # 290°-350°  (calcification contour)

def convert_rgb_to_binary_mask(mask_rgb, calc_candidates):
    mask_hsv = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2HSV)

    green_mask = hue_mask(mask_hsv, GREEN) * calc_candidates[:, :, 0]
    blue_mask = hue_mask(mask_hsv, BLUE) * calc_candidates[:, :, 0]
    red_mask1 = hue_mask(mask_hsv, RED1) * calc_candidates[:, :, 0]
    red_mask2 = hue_mask(mask_hsv, RED2) * calc_candidates[:, :, 0]
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    pink_mask = hue_mask(mask_hsv, PINK) * calc_candidates[:, :, 0]

    calc_mask = 1 * green_mask + 2 * blue_mask + 3 * red_mask + 4 * pink_mask
    return calc_mask
    
def hue_mask(hsv_array, hue_range, sat_thresh=30, val_thresh=30):
    h, s, v = cv2.split(hsv_array)
    lo, hi = hue_range
    if lo <= hi:
        mask_h = cv2.inRange(h, lo, hi)
    else:                              # wrap-around (red)
        mask_h = cv2.bitwise_or(cv2.inRange(h, 0, hi), # type: ignore
                                cv2.inRange(h, lo, 179)) # type: ignore
    mask_s = cv2.inRange(s, sat_thresh, 255)   # type: ignore # suppress grey text
    mask_v = cv2.inRange(v, val_thresh, 255)   # type: ignore # suppress dark bg
    return cv2.bitwise_and(mask_h, mask_s, mask_v)   # binary 0/255

def plot_masks_and_exams_overlay(ct_slice, aligned_mask, calc_candidates):
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs[0,0].imshow(ct_slice, cmap='gray')
    axs[0,0].set_title("CT image")
    axs[0,0].axis('off')

    print(aligned_mask.shape)
    calc_candidates = np.stack([
        calc_candidates,
        calc_candidates,
        calc_candidates
    ], axis=-1)
    # Apply color map
    print(aligned_mask.shape, aligned_mask.min(), aligned_mask.max())
    aligned_mask2 = aligned_mask.copy()
    # aligned_mask2 = (aligned_mask2 - aligned_mask2.min()) / (aligned_mask2.max() - aligned_mask2.min())
    axs[0,1].imshow(aligned_mask2)

    axs[0,1].set_title(f"Cropped mask_img")
    axs[0,1].axis('off')

    # Overlay CT with mask at 50% opacity
    axs[0,2].imshow(ct_slice, cmap='gray')
    axs[0,2].imshow(aligned_mask, cmap='jet', alpha=0.5)

    axs[0,2].set_title("Overlay Mask + CT image")
    axs[0,2].axis('off')
    
    cv2.imwrite('data/ct_slice.png', ct_slice)
    cv2.imwrite('data/aligned_mask.png', aligned_mask)

    # second row (repeat or replace with other images)
    axs[1,0].imshow(calc_candidates, cmap='gray')
    axs[1,0].set_title("Calcifications")
    axs[1,0].axis('off')

    calc_mask = convert_rgb_to_binary_mask(aligned_mask, calc_candidates)
    # mask_segs = cv2.cvtColor(aligned_mask, cv2.COLOR_RGB2HSV)
    
    # green_mask = hue_mask(mask_segs, GREEN) * calc_candidates[:, :, 0]
    # blue_mask = hue_mask(mask_segs, BLUE) * calc_candidates[:, :, 0]
    # red_mask1 = hue_mask(mask_segs, RED1) * calc_candidates[:, :, 0]
    # red_mask2 = hue_mask(mask_segs, RED2) * calc_candidates[:, :, 0]
    # red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # pink_mask = hue_mask(mask_segs, PINK) * calc_candidates[:, :, 0]

    # calc_mask = 1 * green_mask + 2 * blue_mask + 3 * red_mask + 4 * pink_mask

    # print(mask_segs.shape, mask_segs.min(), mask_segs.max())
    
    mask_segs2 = calc_mask * calc_candidates[:, :, 0]
    
    axs[1,1].imshow(mask_segs2, cmap='gray')
    axs[1,1].set_title("Mask")
    axs[1,1].axis('off')
    
    axs[1,2].imshow(ct_slice, cmap='gray')
    axs[1,2].imshow(mask_segs2, cmap='jet', alpha=0.2)
    axs[1,2].set_title("Overlay Calcifications Mask + CT image")
    axs[1,2].axis('off')

    plt.tight_layout()
    fig.savefig('data/overlay1.png', dpi=300)
    plt.show()
    
def plot_compare_alignment(ct_slice, orig_mask, mask_aligned, calc_candidates):
    ''' Plot overlayed images before and after alignment '''
    fig, axs = plt.subplots(1, 2, figsize=(13, 8))
    
    calc_candidates = np.stack([
        calc_candidates,
        calc_candidates,
        calc_candidates
    ], axis=-1)
    
    # orig_mask = convert_rgb_to_binary_mask(orig_mask, calc_candidates)
    # mask_aligned = convert_rgb_to_binary_mask(mask_aligned, calc_candidates)
    
    axs[0].imshow(ct_slice, cmap='gray')
    axs[0].imshow(orig_mask, cmap='jet', alpha=0.5)
    axs[0].set_title("Before alignment")
    axs[0].axis('off')

    axs[1].imshow(ct_slice, cmap='gray')
    axs[1].imshow(mask_aligned, cmap='jet', alpha=0.5)
    axs[1].set_title("After alignment")
    axs[1].axis('off')

    plt.tight_layout()
    fig.savefig('data/compare_alignment.png', dpi=300)
    plt.show()