import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy import ndimage as ndi
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

    pink_mask[(pink_mask == 1) & (red_mask == 1)] = 0  # remove pink where is red
    pink_mask[(pink_mask == 1) & (blue_mask == 1)] = 0  # remove pink where is blue
    pink_mask[(pink_mask == 1) & (green_mask == 1)] = 0  # remove pink where is green
            
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
    # mask = cv2.bitwise_and(mask_h, mask_s, mask_v)   # binary 0/255
    mask = cv2.bitwise_and(cv2.bitwise_and(mask_h, mask_s), mask_v)
    return (mask // 255).astype(np.uint8)  # convert to 0/1

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
    
def plot_lesion_classes(ct_slice, aligned_mask, calc_candidates):
    ''' Plot each class separately
    1 - Green - LAD
    2 - Blue  - CX
    3 - Red   - RCA
    4 - Pink  - Contour
    Always plot the segmentations overlaid with the CT image slice
    '''
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    
    axs[0].imshow(ct_slice, cmap='gray')
    axs[0].set_title("CT image")
    axs[0].axis('off')
    
    calc_candidates = np.stack([
        calc_candidates,
        calc_candidates,
        calc_candidates
    ], axis=-1)
    
    calc_mask = convert_rgb_to_binary_mask(aligned_mask, calc_candidates)
    
    class_names = ['LAD', 'CX', 'RCA', 'Contour']
    class_colors = ['Greens', 'Blues', 'Reds', 'Purples']
    for i in range(1, 5):
        class_mask = (calc_mask == i).astype(np.uint8)
        axs[i].imshow(ct_slice, cmap='gray')
        axs[i].imshow(class_mask, cmap=class_colors[i-1], alpha=0.5)
        axs[i].set_title(f"Class {i}: {class_names[i-1]}")
        axs[i].axis('off')
        
    plt.tight_layout()
    fig.savefig('data/Debug/lesion_classes.png', dpi=300)
    plt.show()
    
# def filter_noisy_calcifications(mask_3d, min_voxels=3, min_slices=3):
#     """
#     mask_3d: binary np.ndarray of shape (Z, Y, X)
#     Returns: filtered binary mask (same shape) with small/noisy components removed.
#     """
#     # Label 3D components with 26-connectivity
#     struct = ndi.generate_binary_structure(3, 2)  # connectivity = 2 → 26 conn
#     print(f"Structure shape: {struct.shape}, structure:\n{struct}")
#     labeled, num = ndi.label(mask_3d, structure=struct)
#     sizes = ndi.sum(mask_3d, labeled, index=range(1, num+1))  # voxel counts
    
#     # Create an output mask
#     out = np.zeros_like(mask_3d, dtype=bool)
    
#     # Loop through components
#     for comp_id, size in enumerate(sizes, start=1):
#         if size < min_voxels:
#             continue
#         # z-range of this component
#         coords = np.where(labeled == comp_id)
#         z_coords = coords[0]
#         if (z_coords.max() - z_coords.min() + 1) < min_slices:
#             # component too “thin” in z
#             continue
#         # keep
#         out[labeled == comp_id] = True
    
#     return out

# def filter3d_noisy_calcifications(mask_3d, min_voxels=3, min_slices=2):
#     struct = ndi.generate_binary_structure(3, 2)
#     labeled, num = ndi.label(mask_3d, structure=struct)
#     sizes = ndi.sum(mask_3d, labeled, index=np.arange(1, num+1))
    
#     keep_mask = np.zeros_like(mask_3d, dtype=bool)
#     remove_mask = np.zeros_like(mask_3d, dtype=bool)
    
#     for comp_id, size in enumerate(sizes, start=1):
#         z_coords = np.where(labeled == comp_id)[0]
#         if size < min_voxels or (z_coords.max() - z_coords.min() + 1) < min_slices:
#             remove_mask[labeled == comp_id] = True
#         else:
#             keep_mask[labeled == comp_id] = True
    
#     return keep_mask, remove_mask

# def filter2d_noisy_calcifications(mask_3d_bool: np.ndarray, min_pixels: int = 3) -> np.ndarray:
#     """
#     Filtra, slice a slice (2D), componentes conectados usando 8-conectividade.
    
#     Parâmetros
#     ----------
#     mask_3d_bool : np.ndarray (Z, H, W) dtype=bool
#         Máscara binária 3D (True=voxel positivo).
#     min_pixels : int
#         Tamanho mínimo (em pixels) por componente 2D para ser mantido (por slice).
#         Componentes menores são removidos.
    
#     Retorno
#     -------
#     out : np.ndarray (Z, H, W) dtype=bool
#         Máscara binária filtrada, mesmo shape de entrada.
#     """
#     assert mask_3d_bool.ndim == 3, f"Esperado volume 3D (Z,H,W); recebido {mask_3d_bool.ndim}D."
#     Z, H, W = mask_3d_bool.shape
#     out = np.zeros_like(mask_3d_bool, dtype=bool)

#     struct_2d = ndi.generate_binary_structure(2, 2)  # 8-conectividade (2D)

#     # Checa se o slice está vazio
#     for z in range(Z):
#         sl = mask_3d_bool[z].astype(bool)
#         if not sl.any():
#             continue

#         labeled, num = ndi.label(sl, structure=struct_2d)
#         if num == 0:
#             continue

#         sizes = ndi.sum(sl, labeled, index=np.arange(1, num + 1))
#         keep = np.zeros_like(sl, dtype=bool)

#         for comp_id, size in enumerate(sizes, start=1):
#             if size >= min_pixels:
#                 keep[labeled == comp_id] = True

#         out[z] = keep

#     return out

# def filter_small_lesions(mask_2d, min_pixels=3):
#     # contours, _ = cv2.findContours(mask_2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # polygons = [cnt.squeeze() for cnt in contours if cnt.shape[0] >= 3]  # keep only valid polygons
#     # return polygons
#     conected_lesions = np.zeros(mask_2d.shape)
#     _, lesions = cv2.connectedComponents(mask_2d.astype(np.uint8), connectivity=4)
#     conected_lesions[lesions == 1] = 1
    
#     small_lesions = mask_2d - conected_lesions
#     print(np.unique(small_lesions, return_counts=True))
    
#     return 1 - small_lesions
    

