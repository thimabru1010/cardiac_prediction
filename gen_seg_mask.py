import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom, affine_transform
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift  # or use cv2.warpAffine for integer shift
import os
from extract_text_from_image import extract_text_from_image
from remove_text_from_image import remove_text_from_image
import google.generativeai as genai # type: ignore
from PIL import Image
import re

# --- Configuração da API Key ---
# RECOMENDADO: Use uma variável de ambiente chamada GOOGLE_API_KEY
try:
    api_key = "AIzaSyDZfAbiG6zTY49udgvsd56GvkmmEowEEm8"
    genai.configure(api_key=api_key)
    print("API Key configurada via variável de ambiente GOOGLE_API_KEY.")
except KeyError:
    print("ERRO: A variável de ambiente GOOGLE_API_KEY não está definida.")
    print("Por favor, defina GOOGLE_API_KEY com sua chave API.")
    exit() # Saia se a chave não estiver configurada
    
def align_mask_to_ct(
        mask_img: np.ndarray,
        ct_img:   np.ndarray,
        motion: int = cv2.MOTION_AFFINE,
        n_iter: int = 5_000,
        eps: float = 1e-7,
        gauss_filt_size: int = 5,
        return_warp: bool = False,
):
    """
    Usa o ECC do OpenCV para alinhar `mask_img` (moving) a `ct_img` (fixed).

    Parameters
    ----------
    mask_img, ct_img : np.ndarray
        Imagens RGB ou escala-de-cinza, ambas com shape (H, W [,3]).
    motion : int
        cv2.MOTION_TRANSLATION | MOTION_EUCLIDEAN | MOTION_AFFINE | MOTION_HOMOGRAPHY.
    n_iter : int
        Máximo de iterações do ECC.
    eps : float
        Critério de convergência (ΔECC mínimo).
    gauss_filt_size : int
        Suavização opcional do ECC (ajuda em ruído).
    return_warp : bool
        Se True, devolve também a matriz de deformação estimada.

    Returns
    -------
    aligned_mask : np.ndarray
        Máscara registrada no espaço de `ct_img`.
    warp_matrix  : np.ndarray  (opcional)
        Matriz de transformação estimada (2×3 ou 3×3).
    """
    if mask_img.ndim == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img.copy()

    if ct_img.ndim == 3:
        ct_gray = cv2.cvtColor(ct_img, cv2.COLOR_BGR2GRAY)
    else:
        ct_gray = ct_img.copy()

    mask_f = mask_gray.astype(np.float32) / 255.0
    ct_f   = ct_gray.astype(np.float32)   / 255.0

    # matriz inicial (identidade)
    warp_matrix = (
        np.eye(3, 3, dtype=np.float32)
        if motion == cv2.MOTION_HOMOGRAPHY
        else np.eye(2, 3, dtype=np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)

    _ = cv2.findTransformECC(
        templateImage=ct_f,
        inputImage=mask_f,
        warpMatrix=warp_matrix,
        motionType=motion,
        criteria=criteria,
        inputMask=None, # type: ignore
        gaussFiltSize=gauss_filt_size,
    ) # type: ignore

    # aplica a deformação na máscara original (RGB ou cinza)
    if motion == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(
            mask_img,
            warp_matrix,
            (ct_img.shape[1], ct_img.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
    else:
        aligned = cv2.warpAffine(
            mask_img,
            warp_matrix[:2],
            (ct_img.shape[1], ct_img.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_TRANSPARENT,
        )

    return (aligned, warp_matrix) if return_warp else aligned

def window_level(arr, center=40, width=350):
    lo, hi = center - width/2, center + width/2
    arr = np.clip(arr, lo, hi)
    # return arr, (arr - lo) / (hi - lo)             # 0‑1 for imshow()
    return arr

def artifficial_zoom_crop(img: np.ndarray, zoom_factor: float) -> np.ndarray:
    """
    Crop the central region of an image (or volume) without resizing to simulate a zoom‐in.

    Parameters
    ----------
    img : np.ndarray
        Input image array. Supported shapes:
          - 2D grayscale: (H, W)
          - 3D volume or multi‐slice: (Z, H, W)
          - 4D volume with channels:   (Z, H, W, C)
    zoom_factor : float
        Zoom factor > 1. The crop width is computed as W / zoom_factor
        and then centered. E.g. zoom_factor=2 will take the middle half
        of the width.

    Returns
    -------
    np.ndarray
        Cropped image (or volume) of identical dimensionality but
        with the width reduced to int(W / zoom_factor).
    """
    y, x = img.shape[0], img.shape[1]
    crop_x = int(x / zoom_factor)
    start_x = (x - crop_x) // 2
    end_x = start_x + crop_x

    if img.ndim in [2, 3]:
        # for shapes (H,W) or (Z,H,W)
        return img[:, start_x:end_x]
    elif img.ndim == 4:
        # for shapes (Z,H,W,C)
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
        mask_h = cv2.bitwise_or(cv2.inRange(h, 0, hi), # type: ignore
                                cv2.inRange(h, lo, 179)) # type: ignore
    mask_s = cv2.inRange(s, sat_thresh, 255)   # type: ignore # suppress grey text
    mask_v = cv2.inRange(v, val_thresh, 255)   # type: ignore # suppress dark bg
    return cv2.bitwise_and(mask_h, mask_s, mask_v)   # binary 0/255

def load_dicoms(dicom_folder):
    """
    Load DICOM files from a folder and return gated and mask exams.
    """
    files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not files:
        raise ValueError(f"No DICOM files found in folder: {dicom_folder}")
    # Exclude files containing 'SG'. Then, Filter series with files containing 'IA' or 'G{Number}'
    files = [f for f in files if 'SG' not in f and ('IA' in f or 'G' in f)]
    print(f"Filtered DICOM files: {files}")
    mask_files = [f for f in files if 'IA' in f]
    gated_files = [f for f in files if 'G' in f]
    # sort G-files by their trailing number: G1 → G2 → … → G10 → G11 → …
    gated_files.sort(key=lambda fn: int(re.search(r'G(\d+)', os.path.basename(fn)).group(1))) # type: ignore
    mask_files.sort(key=lambda fn: int(re.search(r'IA(\d+)', os.path.basename(fn)).group(1))) # type: ignore
    # Build full paths
    gated_files = [os.path.join(dicom_folder, f) for f in gated_files]
    mask_files = [os.path.join(dicom_folder, f) for f in mask_files]
    print("Sorted G-files:", gated_files)
    print("Sorted IA-files:", mask_files)
    reader = sitk.ImageSeriesReader()
    print(pydicom.dcmread(mask_files[0]).SeriesDescription)
    reader.SetFileNames(mask_files)
    mask_img = reader.Execute()
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(gated_files)
    ct_img = reader.Execute()
    print("CT image shape:", ct_img.GetSize())
    print("Mask image shape:", mask_img.GetSize())
    return ct_img, mask_img


if __name__ == "__main__":
    dicom_folder = 'data/EXAMES/Patients_AutomatedMask/311180'
    ct_img, mask_img = load_dicoms(dicom_folder)

    mask_np = sitk.GetArrayFromImage(mask_img)[1]  # (z, y, x) or (z, y, x, c)
    
    zoom_factor, slice_position = extract_text_from_image(
        model_name='gemini-2.0-flash-thinking-exp-01-21',
        prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
        image_pil=Image.fromarray(mask_np)) # type: ignore
    
    
    print(f"Zoom factor: {zoom_factor} - Slice position: {slice_position}")
    mask_np, _ = remove_text_from_image(mask_np)  # Remove text from mask

    ct_np = sitk.GetArrayFromImage(ct_img)
    print("CT image shape:", ct_np.shape)
    # slice_coord = ct_np.shape[0] + 1 - slice_position + 1
    print(f"Slice position: {slice_position}")
    ct_np = ct_np[slice_position-1]  #! O indice 0 é o final da série
    # ct_np = ct_np[slice_position - 3]  #! O indice 0 é o final da série
    ct_slice = window_level(ct_np)       # Use first slice, apply window/level
    
    # Apply manual crop (zoom effect, no resize)
    print(mask_np.shape)
    cropped_mask_np = artifficial_zoom_crop(mask_np, zoom_factor)
    
    cropped_mask_np = tight_crop(cropped_mask_np, thr=10)

    print(ct_slice.shape)
    print(ct_slice.min(), ct_slice.max())
    ct_slice = tight_crop(ct_slice, thr=0)
    
    # Resize cropped mask to 512×512 for plotting
    target_size = (512, 512)
    
    mask_resized = cv2.resize(cropped_mask_np, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    ct_slice = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    kernel = np.ones((3,3), np.uint8)
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, kernel)
    # mask_resized = cropped_mask_np.copy()
    
    mask_resized, W = align_mask_to_ct(mask_resized, ct_slice, return_warp=True)

    print(mask_resized.shape, mask_resized.min(), mask_resized.max())
    print(ct_slice.shape, ct_slice.min(), ct_slice.max())
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
    axs[0,2].set_title("Overlay")
    axs[0,2].axis('off')
    
    cv2.imwrite('data/ct_slice.png', ct_slice)
    cv2.imwrite('data/mask_resized.png', mask_resized)

    # second row (repeat or replace with other images)
    axs[1,0].imshow(calc_candidates, cmap='gray')
    axs[1,0].set_title("Calcifications")
    axs[1,0].axis('off')
    
    # mask_segs = mask_resized * calc_candidates
    RED1  = (0,   5)      # 0°-10°
    RED2  = (170, 179)    # 340°-360°
    GREEN = (50,  80)     # 100°-160°  (LAD in your screenshots)
    BLUE  = (100,130)     # 200°-260°  (CX)
    PINK  = (145,175)     # 290°-350°  (calcification contour)

    mask_segs = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2HSV)
    
    mask_segs = hue_mask(mask_segs, GREEN)

    # optional clean-up: remove isolated edge noise
    # kernel = np.ones((3,3), np.uint8)
    # mask_segs = cv2.morphologyEx(mask_segs, cv2.MORPH_OPEN, kernel)

    print(mask_segs.shape, mask_segs.min(), mask_segs.max())
    
    mask_segs2 = mask_segs * calc_candidates[:, :, 0]
    
    axs[1,1].imshow(mask_segs2, cmap='gray')
    axs[1,1].set_title("Mask")
    axs[1,1].axis('off')
    
    axs[1,2].imshow(ct_slice, cmap='gray')
    axs[1,2].imshow(mask_segs2, cmap='jet', alpha=0.2)
    axs[1,2].set_title("Mask + CT image")
    axs[1,2].axis('off')

    plt.tight_layout()
    fig.savefig('data/overlay1.png', dpi=300)
    plt.show()


