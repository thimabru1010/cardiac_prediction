import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import zoom, affine_transform
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift  # or use cv2.warpAffine for integer shift
import os
# import argparse
# from masks_auto_generation.extract_text_from_image import extract_text_from_image
# from masks_auto_generation.remove_text_from_image import remove_text_from_image

try:
    from masks_auto_generation.extract_text_from_image import extract_text_from_image
    print("import via package: masks_auto_generation.extract_text_from_image")
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "masks_auto_generation"))
    from .extract_text_from_image import extract_text_from_image
    print("import fallback: extract_text_from_image")
    
# from extract_text_from_image import extract_text_from_image

try:
    from masks_auto_generation.remove_text_from_image import remove_text_from_image
    print("import via package: masks_auto_generation.remove_text_from_image")
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "masks_auto_generation"))
    from .remove_text_from_image import remove_text_from_image
    print("import fallback: remove_text_from_image")

import google.generativeai as genai # type: ignore
from PIL import Image
from openai import OpenAI
import re
try:
    from masks_auto_generation.utils import plot_masks_and_exams_overlay, hue_mask, plot_compare_alignment, plot_lesion_classes, convert_rgb_to_binary_mask
except ImportError:
    from utils import plot_masks_and_exams_overlay, hue_mask, plot_compare_alignment, plot_lesion_classes, convert_rgb_to_binary_mask
# from utils import plot_masks_and_exams_overlay, hue_mask, plot_compare_alignment, plot_lesion_classes, convert_rgb_to_binary_mask

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

    # mask_f = cv2.normalize(mask_gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # ct_f = cv2.normalize(ct_gray, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    mask_f = mask_gray.astype(np.float32) / 255.0
    ct_f   = ct_gray.astype(np.float32)   / 255.0
    
    # Verifique se há NaN/Inf nas imagens
    print("Mask stats:", mask_f.min(), mask_f.max(), np.isnan(mask_f).any(), np.isinf(mask_f).any())
    print("CT stats:", ct_f.min(), ct_f.max(), np.isnan(ct_f).any(), np.isinf(ct_f).any())

    print("Mask shape:", mask_f.shape)
    print("CT shape:", ct_f.shape)
    
    # Verifique se há variação suficiente
    print("Mask variance:", np.var(mask_f))
    print("CT variance:", np.var(ct_f))
    
    # Verificação de variância
    if np.var(ct_f) < 1e-6:
        print("WARNING: CT variance too low, skipping alignment")

    # matriz inicial (identidade)
    warp_matrix = (
        np.eye(3, 3, dtype=np.float32)
        if motion == cv2.MOTION_HOMOGRAPHY
        else np.eye(2, 3, dtype=np.float32)
    )

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)

    # Finds the best warp_matrix using ECC
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

    return aligned

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
    # Find non-zero pixels
    ys, xs = np.where(mask > thr)
    # Get the lowest coordinate (not value) of each axis
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return arr[y0:y1+1, x0:x1+1, ...]

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
    # print("CT image shape:", ct_img.GetSize())
    # print("Mask image shape:", mask_img.GetSize())
    return ct_img, mask_img


if __name__ == "__main__":
    # dicom_folder = 'data/EXAMES/Patients_AutomatedMask/311180'
    dicom_folder = 'data/ExamesArya/123484'
    ct_img, mask_img = load_dicoms(dicom_folder)

    mask_np = sitk.GetArrayFromImage(mask_img)[-1]  # (z, y, x) or (z, y, x, c)
    
    # zoom_factor, slice_position = extract_text_from_image(
    #     model_name='gemini-2.0-flash-thinking-exp-01-21',
    #     prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
    #     image_pil=Image.fromarray(mask_np)) # type: ignore
    
    client = OpenAI()
    zoom_factor, slice_position = extract_text_from_image(
        client=client,
        model_name='gpt-4.1',
        prompt="Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'.",
        image_npy=mask_np) # type: ignore
    
    # zoom_factor = 1.51
    # slice_position = 9
    print(f"Zoom factor: {zoom_factor} - Slice position: {slice_position}")
    cv2.imwrite('data/Debug/mask_text.png', mask_np)
    mask_np, boxes = remove_text_from_image(mask_np)  # Remove text from mask
    orig_mask_np = mask_np.copy()
    #Save the mask for debugging
    cv2.imwrite('data/Debug/mask_no_text.png', mask_np)
    
    # Draw boxes for debugging
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    morphEx_kernel = np.ones((3,3), np.uint8)
    morphEx_kernel_big = np.ones((5,5), np.uint8)
    mask_np_boxes = mask_np.copy()
    
    for (i, (x0, y0, x1, y1)) in enumerate(boxes):
        print(f"Box {i}: ({x0}, {y0}), ({x1}, {y1})", colors[i])
        cv2.rectangle(mask_np_boxes, (x0, y0), (x1, y1), colors[i], 2)
    cv2.imwrite('data/Debug/mask_no_text_boxes.png', mask_np_boxes)

    ct_np = sitk.GetArrayFromImage(ct_img)
    print("CT image shape:", ct_np.shape)
    # slice_coord = ct_np.shape[0] + 1 - slice_position + 1
    print(f"Slice position: {slice_position}")
    ct_np = ct_np[slice_position-1]  #! O indice 0 é o final da série
    ct_slice = window_level(ct_np)       # Use first slice, apply window/level
    cv2.imwrite('data/Debug/ct_slice.png', ct_slice)
    
    # Apply manual crop (zoom effect, no resize)
    print(mask_np.shape)
    cropped_mask_np = artifficial_zoom_crop(mask_np, zoom_factor)
    cv2.imwrite('data/Debug/artificial_zoom.png', cropped_mask_np)
    
    # Remove as letras que orientam o exame
    # cropped_mask_np = cv2.morphologyEx(cropped_mask_np, cv2.MORPH_OPEN, morphEx_kernel_big)
    
    cropped_mask_np = tight_crop(cropped_mask_np, thr=10)
    cv2.imwrite('data/Debug/artificial_zoom_tight_crop.png', cropped_mask_np)
    
    no_zoom_mask_np = tight_crop(orig_mask_np, thr=10)

    print(ct_slice.shape)
    print(ct_slice.min(), ct_slice.max())
    ct_slice = tight_crop(ct_slice, thr=0)
    
    # Resize cropped mask to 512×512 for plotting
    target_size = (512, 512)
    
    no_zoom_mask_np = cv2.resize(no_zoom_mask_np, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    mask_resized = cv2.resize(cropped_mask_np, target_size, interpolation=cv2.INTER_NEAREST_EXACT)
    ct_slice = cv2.resize(ct_slice, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    cv2.imwrite('data/Debug/mask_resized.png', mask_resized)
    # Remove as letras que orientam o exame
    mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_OPEN, morphEx_kernel)
    mask_resized = tight_crop(mask_resized, thr=0)
    
    cv2.imwrite('data/Debug/mask_resized_filtered.png', mask_resized)
    
    aligned_mask = align_mask_to_ct(mask_resized, ct_slice, return_warp=True)
    # aligned_mask = cv2.morphologyEx(aligned_mask, cv2.MORPH_OPEN, morphEx_kernel)

    cv2.imwrite('data/Debug/aligned_mask.png', aligned_mask)
    print(aligned_mask.shape, aligned_mask.min(), aligned_mask.max())
    print(ct_slice.shape, ct_slice.min(), ct_slice.max())

    # first row
    calc_candidates = ct_slice.copy()
    calc_candidates[ct_slice < 130] = 0
    calc_candidates[ct_slice > 130] = 1
    ct_slice_norm = ct_slice.copy()
    ct_slice_norm = (ct_slice_norm - ct_slice_norm.min()) / (ct_slice_norm.max() - ct_slice_norm.min())

    calc_candidates2 = np.stack([
    calc_candidates,
    calc_candidates,
    calc_candidates
    ], axis=-1)
    aligned_mask_classes = convert_rgb_to_binary_mask(aligned_mask, calc_candidates2)
    print(np.unique(aligned_mask_classes))
    
    plot_masks_and_exams_overlay(ct_slice_norm, aligned_mask, calc_candidates)
    
    plot_compare_alignment(ct_slice_norm, no_zoom_mask_np, aligned_mask, calc_candidates)
    
    plot_lesion_classes(ct_slice_norm, aligned_mask, calc_candidates)


