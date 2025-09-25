import base64
from openai import OpenAI
import cv2
import numpy as np
import pydicom

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_npy_image_png(npy_array):
    _, buffer = cv2.imencode('.png', npy_array)  # Sem perda de qualidade
    return base64.b64encode(buffer).decode('utf-8')

# Path to your image
dicom_path = "data/ExamesArya/105655/IA4.dcm"

dicom_npy = pydicom.dcmread(dicom_path).pixel_array
print(dicom_npy.shape)
print(dicom_npy.min(), dicom_npy.max())
# Getting the Base64 string
base64_npy_image = encode_npy_image_png(dicom_npy)

# response = client.responses.create(  # type: ignore
#     model="gpt-4.1",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 { "type": "Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'" },
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{base64_npy_image}",
#                 },
#             ],
#         }
#     ],
# )

prompt_text = (
    "Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'"
)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": prompt_text },
                { "type": "input_image", "image_url": f"data:image/png;base64,{base64_npy_image}" },
            ],
        }
    ],
)

print(response.output_text)