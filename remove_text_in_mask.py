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

# Load images
root_folder = 'data/mask_generation_test/311180'
orig_path = f'{root_folder}/exam_mask.png'
# proc_path = f'{root_folder}/311122/exam_mask_no_text_v3.png'  # previously saved original zeroed

orig = cv2.imread(orig_path)
# proc = cv2.imread(proc_path)
proc = orig.copy()  # Use the original image for processing

h, w, _ = orig.shape

# Detect text for bboxes again
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

def bbox_coords(mask):
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max

m = 0.35
zones = {
    'top_left':   (slice(0, int(h*m)), slice(0, int(w*m))),
    'top_right':  (slice(0, int(h*m)), slice(int(w*(1-m)), w)),
    'bottom_left':(slice(int(h*(1-m)), h), slice(0, int(w*m))),
    'bottom_right':(slice(int(h*(1-m)), h), slice(int(w*(1-m)), w))
}

bboxes = []
for name, (ys, xs) in zones.items():
    bbox = bbox_coords(mask[ys, xs] == 255)
    if bbox:
        x_min, y_min, x_max, y_max = bbox
        x_min += xs.start; x_max += xs.start
        y_min += ys.start; y_max += ys.start
        margin = 5
        x_min = max(0, x_min-margin); y_min = max(0, y_min-margin)
        x_max = min(w-1, x_max+margin); y_max = min(h-1, y_max+margin)
        bboxes.append((x_min, y_min, x_max, y_max))

img_no_text = zero_out_regions(orig, bboxes)

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


