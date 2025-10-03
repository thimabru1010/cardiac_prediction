import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def zero_out_regions(img: np.ndarray, bboxes):
    """
    Recebe uma imagem BGR (OpenCV) e uma lista de bounding-boxes
    no formato (x_min, y_min, x_max, y_max).
    Retorna uma cópia da imagem com cada retângulo preenchido por zeros (preto).
    """
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        # clip para garantir que não ultrapasse bordas
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(out.shape[1] - 1, x2); y2 = min(out.shape[0] - 1, y2)
        out[y1:y2 + 1, x1:x2 + 1] = 0
    return out

def find_text(mask):
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

def get_corner_rectangles(image_shape):
    """
    Retorna as coordenadas dos 4 retângulos nos cantos da imagem.
    Formato: (x_min, y_min, x_max, y_max)
    """
    h, w = image_shape[:2]  # height, width
    m = 0.30  # 35% de cada dimensão
    mh = 0.30
    mw = 0.25
    
    rectangles = {
        'top_left': (
            0,              # x_min
            0,              # y_min  
            int(w * m) + 100,     # x_max (35% da largura)
            int(h * m) + 50    # y_max (35% da altura)
        ),
        
        'top_right': (
            int(w * (1-m)) - 200, # x_min (65% da largura)
            0,              # y_min
            w,              # x_max (100% da largura)
            int(h * m) + 50     # y_max (35% da altura)
        ),
        
        'bottom_left': (
            0,              # x_min
            int(h * (1-m)), # y_min (65% da altura)
            int(w * m),     # x_max (35% da largura)
            h               # y_max (100% da altura)
        ),
        
        'bottom_right': (
            int(w * (1-mw)), # x_min (65% da largura) 
            int(h * (1-mh)), # y_min (65% da altura)
            w,              # x_max (100% da largura)
            h               # y_max (100% da altura)
        )
    }
    
    return rectangles

def detect_text(image):
    """
    Detects text in an image and returns bounding boxes.
    This is a placeholder function. Replace with actual text detection logic.
    """
    h, w, _ = image.shape
    print(f"Image shape: {image.shape}")

    # Detect text for bboxes again
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # m = 0.25
    # (x_min, y_min, x_max, y_max)
    zones = get_corner_rectangles(image.shape)
    # zones = [zones['bottom_right']]  # testar só bottom_right
    bboxes = []
    #! No dataset baixado só tem texto no bottom_right
    for name, (xmin_ori, ymin_ori, xmax_ori, ymax_ori) in zones.items():
        if name not in ['bottom_right']:
            continue
        bbox = find_text(mask[ymin_ori:ymax_ori, xmin_ori:xmax_ori] == 255) # (x_min, y_min, x_max, y_max) dentro da região
        # print((ymin_ori, ymax_ori), bbox)
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            x_min += xmin_ori; x_max += xmin_ori
            y_min += ymin_ori; y_max += ymin_ori
            margin = 5
            x_min = max(0, x_min-margin); y_min = max(0, y_min-margin)
            x_max = min(w-1, x_max+margin); y_max = min(h-1, y_max+margin)
            bboxes.append((x_min, y_min, x_max, y_max))
    return bboxes

def remove_text_from_image(image):
    """
    Removes text from an image with annotated labels by zeroing out the regions defined by bounding boxes.
    """
    bboxes = detect_text(image)
    return zero_out_regions(image, bboxes), bboxes

if __name__ == "__main__":
    # Load images
    root_folder = 'data/mask_generation_test/312323'
    orig_path = f'{root_folder}/exam_mask.png'
    # proc_path = f'{root_folder}/311122/exam_mask_no_text_v3.png'  # previously saved original zeroed

    orig = cv2.imread(orig_path)

    img_no_text, bboxes = remove_text_from_image(orig)

    # Create original with rectangles
    orig_boxes = orig.copy()
    for (x1,y1,x2,y2) in bboxes:
        cv2.rectangle(orig_boxes, (x1,y1), (x2,y2), (0,0,255), 4)

    # Plot side by side
    fig, axes = plt.subplots(1,2, figsize=(12,6))
    axes[0].imshow(cv2.cvtColor(orig_boxes, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original + Bounding Boxes')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_no_text, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Text Removed (No Boxes)')
    axes[1].axis('off')

    img_no_text_path = os.path.join(root_folder, 'exam_mask_no_text.png')
    cv2.imwrite(img_no_text_path, img_no_text)

    plt.tight_layout()
    plt.show()


