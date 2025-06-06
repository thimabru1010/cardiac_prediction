import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Se estiver no Windows, pode ser necessário configurar o caminho do Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Exemplo

# Carregar a imagem original
image_path = 'data/mask_generation_test/original_image.png' # Coloque o nome do seu arquivo de imagem aqui
try:
    img_original = cv2.imread(image_path)
    if img_original is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem em: {image_path}")
    img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB) # Matplotlib usa RGB
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
except Exception as e:
    print(f"Erro ao carregar a imagem: {e}")
    exit()

H, W = img_gray.shape

# Definir as Regiões de Interesse (ROIs) para os cantos
# Ajuste essas proporções conforme necessário para cobrir bem as áreas de texto
roi_height_ratio = 0.25 # Considerar 25% da altura para as ROIs superior e inferior
roi_width_ratio = 0.40  # Considerar 40% da largura para as ROIs esquerda e direita

rois_coords = {
    "top_left": (0, 0, int(W * roi_width_ratio), int(H * roi_height_ratio)),
    "top_right": (int(W * (1 - roi_width_ratio)), 0, int(W * roi_width_ratio), int(H * roi_height_ratio)),
    "bottom_left": (0, int(H * (1 - roi_height_ratio)), int(W * roi_width_ratio), int(H * roi_height_ratio)),
    "bottom_right": (int(W * (1 - roi_width_ratio)), int(H * (1 - roi_height_ratio)), int(W * roi_width_ratio), int(H * roi_height_ratio))
}

detected_text_boxes_global = []

print("Detectando texto nos cantos...")
for corner_name, (rx, ry, rw, rh) in rois_coords.items():
    roi = img_gray[ry:ry+rh, rx:rx+rw]

    # Usar pytesseract para obter dados detalhados, incluindo caixas delimitadoras
    # --psm 6: Assume a single uniform block of text.
    # --oem 3: Default OCR Engine Mode
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT, config=custom_config, lang='por+eng')

    # Coletar todas as caixas de texto detectadas dentro desta ROI
    corner_boxes = []
    n_boxes = len(details['level'])
    for i in range(n_boxes):
        if int(details['conf'][i]) > 30: # Filtrar por confiança (ajuste conforme necessário)
            # As coordenadas são relativas à ROI
            (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
            # Converter para coordenadas globais da imagem
            global_x = rx + x
            global_y = ry + y
            corner_boxes.append((global_x, global_y, w, h))

    # Se houver caixas detectadas no canto, criar uma única caixa delimitadora que englobe todas elas
    if corner_boxes:
        min_x = min(b[0] for b in corner_boxes)
        min_y = min(b[1] for b in corner_boxes)
        max_x_plus_w = max(b[0] + b[2] for b in corner_boxes)
        max_y_plus_h = max(b[1] + b[3] for b in corner_boxes)

        overall_box_width = max_x_plus_w - min_x
        overall_box_height = max_y_plus_h - min_y
        detected_text_boxes_global.append((min_x, min_y, overall_box_width, overall_box_height))
        print(f"  {corner_name}: Caixa agregada ({min_x}, {min_y}, {overall_box_width}, {overall_box_height})")
    else:
        print(f"  {corner_name}: Nenhum texto com confiança suficiente detectado.")


# Criar uma cópia da imagem para desenhar as caixas e outra para filtrar
img_with_boxes = img_rgb.copy()
img_filtered = img_original.copy() # Usar a original BGR para salvar com cv2

print(f"\nTotal de {len(detected_text_boxes_global)} blocos de texto nos cantos detectados.")

# Desenhar as caixas delimitadoras e preencher com zero na imagem filtrada
for (x, y, w, h) in detected_text_boxes_global:
    # Desenhar na imagem com caixas (para visualização)
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2) # Vermelho

    # Preencher com zero (preto) na imagem filtrada
    # Garantir que as coordenadas não saiam dos limites da imagem
    y_end = min(y + h, H)
    x_end = min(x + w, W)
    img_filtered[y:y_end, x:x_end] = 0


# Salvar a imagem filtrada
filtered_image_path = "imagem_filtrada.png"
cv2.imwrite(filtered_image_path, img_filtered)
print(f"\nImagem filtrada salva em: {filtered_image_path}")

# Plotar a comparação
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(img_rgb) # Original RGB
axes[0].set_title("Imagem Original")
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)) # Filtrada convertida para RGB para plot
axes[1].set_title("Imagem Filtrada (Textos Zerados)")
axes[1].axis('off')

# Adicionar também a imagem com as caixas delimitadoras detectadas para depuração
plt.figure(figsize=(10,10))
plt.imshow(img_with_boxes)
plt.title("Textos Detectados nos Cantos (Caixas Agregadas)")
plt.axis('off')
plt.show()

print("\nBounding boxes dos textos agregados em cada canto (x, y, largura, altura):")
for i, (x, y, w, h) in enumerate(detected_text_boxes_global):
    print(f"  Bloco {i+1}: ({x}, {y}, {w}, {h})")