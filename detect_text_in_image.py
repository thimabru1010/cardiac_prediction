import os
from PIL import Image
import google.generativeai as genai # type: ignore
import json
import re

# --- Configuração da API Key ---
# RECOMENDADO: Use uma variável de ambiente chamada GOOGLE_API_KEY
try:
    api_key = "AIzaSyBrPHg64ZYQaiQmTw5oCvy9luZ8bvQTDHE"
    genai.configure(api_key=api_key)
    print("API Key configurada via variável de ambiente GOOGLE_API_KEY.")
except KeyError:
    print("ERRO: A variável de ambiente GOOGLE_API_KEY não está definida.")
    print("Por favor, defina GOOGLE_API_KEY com sua chave API.")
    exit() # Saia se a chave não estiver configurada

# --- Função para carregar a imagem ---
def load_image_from_path(image_path):
    """Carrega uma imagem de um caminho de arquivo."""
    try:
        img = Image.open(image_path)
        return img
    except FileNotFoundError:
        print(f"Erro: Arquivo de imagem não encontrado em '{image_path}'")
        return None
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        return None

# --- Função para listar modelos disponíveis (Opcional, mas útil) ---
def list_available_models():
    """Lista os modelos disponíveis e suas capacidades."""
    print("\nModelos disponíveis:")
    print("--------------------")
    for m in genai.list_models():
        # Filtra modelos que suportam geração de texto (geralmente todos generativos)
        # E mostra se suportam entrada de imagem ('generateContent')
        if 'generateContent' in getattr(m, 'supported_generation_methods', []):
            print(f"{m.name} - Suporta entrada multimodal (texto+imagem)")
        else:
            print(f"{m.name} - Apenas texto")
    print("--------------------")

def supports_multimodal(model_name):
    for m in genai.list_models():
        if m.name == model_name and 'generateContent' in getattr(m, 'supported_generation_methods', []):
            return True
    return False

# --- Função principal para interagir com o modelo ---

def interact_with_gemini(model_name: str, prompt_text: str, image_pil: Image.Image = None): # type: ignore
    """
    Interage com um modelo Gemini, enviando texto e opcionalmente uma imagem.

    Args:
        model_name: O nome do modelo a ser usado (ex: 'gemini-1.5-pro-latest').
        prompt_text: O texto do prompt para o modelo.
        image_pil: A imagem PIL a ser enviada (opcional, padrão: None).
    """
    try:
        # Inicializa o modelo especificado pelo nome
        print(f"\nTentando usar o modelo: {model_name}")
        model = genai.GenerativeModel(model_name)

        content = [prompt_text]
        if image_pil is not None:
            content.append(image_pil) # type: ignore
            
        response = model.generate_content(content)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        print(f"Ocorreu um erro ao interagir com a API: {e}")
        return None

def post_process_response(response_text):
    """
    Procura primeiro bloco de texto que começa e termina com chaves
    """
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
        else:
            print("Nenhum JSON encontrado na resposta.")
            return None
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        return None

def extract_zoom_and_number(response_dict):
    """
    Extrai os campos 'zoom' e 'numero' de um dicionário de resposta.
    """
    # Remove o simbolo '%' do campo 'zoom' e converte para float
    zoom = int(response_dict['zoom'][:-1]) / 100
    numero = response_dict['numero']
    return zoom, numero

def extract_text_from_image(model_name: str, prompt: str, image_pil: Image.Image):
    """
    Detecta texto em uma imagem usando o modelo Gemini.

    Args:
        model_name: O nome do modelo a ser usado.
        prompt: O prompt de texto para guiar a detecção.
        image_pil: A imagem PIL onde o texto será detectado.

    Returns:
        Um dicionário com os campos 'zoom' e 'numero' extraídos da resposta.
    """
    response = interact_with_gemini(model_name, prompt, image_pil)
    if response:
        response_dict = post_process_response(response)
        if response_dict:
            return extract_zoom_and_number(response_dict)
    return None

# --- Exemplo de como usar o script ---
if __name__ == "__main__":
    # Opcional: Liste os modelos disponíveis para ajudar na escolha
    list_available_models()

    # --- Defina os parâmetros para sua chamada API ---
    # Escolha o nome do modelo. Use 'gemini-1.5-pro-latest' ou 'gemini-1.5-flash-latest'
    # para modelos que suportam texto e imagem.
    modelo_escolhido = 'gemini-2.0-flash-thinking-exp-01-21' # Ou 'gemini-1.5-flash-latest'

    # Seu prompt de texto
    prompt_para_modelo = "Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'/ Me retorne um json contendo os seguintes campos: 'zoom' e 'numero'. O campo 'zoom' deve conter o texto após a palavra 'Zoom:' e o campo 'numero' deve conter o número após o caracter '#'."

    # Caminho para sua imagem. Mude para o caminho real de um arquivo de imagem local.
    # Deixe como None se você quiser enviar apenas texto.
    caminho_da_imagem = 'data/mask_generation_test/311180/exam_mask.png' # <--- MUDAR PARA O CAMINHO REAL DA IMAGEM

    img = load_image_from_path(caminho_da_imagem)

    zoom, numero = extract_text_from_image(modelo_escolhido, prompt_para_modelo, img) # type: ignore
    
    print(f"\nResposta do modelo: zoom={zoom}, Slice number={numero}")