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
def interact_with_gemini(model_name: str, prompt_text: str, image_path: str = ''):
    """
    Interage com um modelo Gemini, enviando texto e opcionalmente uma imagem.

    Args:
        model_name: O nome do modelo a ser usado (ex: 'gemini-1.5-pro-latest').
        prompt_text: O texto do prompt para o modelo.
        image_path: O caminho para o arquivo de imagem (opcional).
    """
    try:
        # Checa se o modelo suporta multimodalidade, se necessário
        # if image_path and not supports_multimodal(model_name):
        #     print(f"Erro: O modelo '{model_name}' não suporta entrada multimodal.")
        #     return None

        # Inicializa o modelo especificado pelo nome
        print(f"\nTentando usar o modelo: {model_name}")
        model = genai.GenerativeModel(model_name)

        # Prepara o conteúdo para enviar
        content = [prompt_text]
        if image_path:
            img = load_image_from_path(image_path)
            if img is not None:
                content.append(img)
            else:
                print("Imagem não carregada. Enviando apenas texto.")
        # Gera a resposta
        response = model.generate_content(content)
        return response.text if hasattr(response, 'text') else str(response)
    except Exception as e:
        print(f"Ocorreu um erro ao interagir com a API: {e}")
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
    caminho_da_imagem = 'data/mask_generation_test/312323/exam_mask.png' # <--- MUDAR PARA O CAMINHO REAL DA IMAGEM

    # --- Execute a interação com o modelo ---
    # Certifique-se de que a imagem existe no caminho especificado se image_path não for None.
    if caminho_da_imagem and not os.path.exists(caminho_da_imagem):
        print(f"Erro: O arquivo de imagem '{caminho_da_imagem}' não existe.")
        caminho_da_imagem = None

    print(f"\nChamando o modelo '{modelo_escolhido}' com o prompt e imagem (se fornecida)...")
    resposta_do_modelo = interact_with_gemini(modelo_escolhido, prompt_para_modelo, image_path=caminho_da_imagem)
    # resp_dict = json.loads(resposta_do_modelo)
    # Extrai apenas o bloco JSON
    # Procura primeiro bloco de texto que começa e termina com chaves
    match = re.search(r'\{.*?\}', resposta_do_modelo, re.DOTALL)
    if match:
        json_str = match.group(0)
        resp_dict = json.loads(json_str)
    else:
        print("Nenhum JSON encontrado.")
    
    print(resp_dict)
    # --- Exiba a resposta ---
    print("\n--- Resposta do Modelo ---")
    if resposta_do_modelo:
        print(resp_dict)
        # print(resposta_do_modelo)
    else:
        print("Não foi possível obter uma resposta do modelo.")