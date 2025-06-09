import google.generativeai as genai
import os
from PIL import Image
import io

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
        # E mostra se suportam entrada de imagem ('GenerateContent')
        if 'generateContent' in m.supported_generation_methods:
            print(f"Nome: {m.name}")
            # Você pode adicionar mais detalhes se quiser:
            # print(f"  Descrição: {m.description}")
            # print(f"  Entrada Suportada: {m.supported_generation_methods}")
            # print(f"  Janela de Contexto Máxima: {m.input_token_limit}")
            # print(f"  Saída Máxima de Tokens: {m.output_token_limit}")
            # print("-" * 20)
    print("--------------------")


# --- Função principal para interagir com o modelo ---
def interact_with_gemini(model_name: str, prompt_text: str, image_path: str = None):
    """
    Interage com um modelo Gemini, enviando texto e opcionalmente uma imagem.

    Args:
        model_name: O nome do modelo a ser usado (ex: 'gemini-1.5-pro-latest').
        prompt_text: O texto do prompt para o modelo.
        image_path: O caminho para o arquivo de imagem (opcional).
    """
    try:
        # Inicializa o modelo especificado pelo nome
        print(f"\nTentando usar o modelo: {model_name}")
        model = genai.GenerativeModel(model_name)

        # Verifica se o modelo suporta a entrada multimodal se uma imagem for fornecida
        if image_path:
            if 'GenerateContent' not in model.supported_generation_methods:
                 print(f"Erro: O modelo '{model_name}' não suporta a entrada multimodal (texto+imagem).")
                 print("Por favor, escolha um modelo como 'gemini-1.5-pro-latest' ou 'gemini-1.5-flash-latest'.")
                 return None # Retorna None ou raise uma exceção


        # Prepara o conteúdo para enviar
        content = [prompt_text]
        if image_path:
            image = load_image_from_path(image_path)
            if image:
                 # A biblioteca google-generativeai lida com a preparação da imagem
                 # quando você passa um objeto PIL Image diretamente
                 content.append(image)
            else:
                print("Não foi possível carregar a imagem. Prosseguindo apenas com o texto.")
                image_path = None # Define como None para não tentar enviar a imagem

        # --- Opção de usar o modo Thinking (se suportado pelo modelo e API) ---
        # A funcionalidade "thinking process" ou "grounding" geralmente é ativada
        # em modelos específicos ou via parâmetros adicionais na chamada generate_content.
        # No SDK atual e para modelos mais recentes, essa funcionalidade pode
        # ser acessada via `response.text` para a resposta principal e,
        # se o modelo suporta, talvez via outros campos no objeto response,
        # como `response.candidates[0].content.parts` que podem conter dados estruturados.
        # Para modelos como o "Gemini 2.0 Flash Thinking" (experimental), você
        # pode precisar usar o nome do modelo específico e verificar a documentação
        # para como acessar a saída do processo de pensamento.
        # Neste script básico, focamos na resposta de texto principal,
        # que já incorpora o resultado do "pensamento" do modelo.
        # Se você usar um modelo que explicitamente suporta e expõe o processo de pensamento,
        # você precisaria consultar a documentação para os detalhes de como acessá-lo no objeto `response`.
        # Por exemplo, para alguns modelos, pode haver um campo `grounding_metadata` ou similar.
        # Sem a confirmação de um modelo específico e um método no SDK para acessar
        # o "thinking process" de forma genérica, focaremos na resposta principal.
        # Para a maioria dos usos com Gemini 1.5, a resposta em `response.text` já é o que você precisa.

        # Chama a API para gerar conteúdo
        print("Enviando prompt para o modelo...")
        response = model.generate_content(content)
        print("Resposta recebida.")

        # Retorna o texto gerado
        # A resposta pode conter texto, e potencialmente outros componentes se for multimodal
        # Para simplificar, retornamos apenas o texto.
        # Se você precisa de uma análise mais profunda da resposta multimodal,
        # inspecione o objeto `response.candidates`.
        if response and response.text:
            return response.text
        else:
            # Se não houver texto na resposta (por exemplo, se o prompt foi inadequado)
            # A resposta pode conter "parts" não textuais ou estar vazia.
            # Você pode inspecionar `response.prompt_feedback` ou `response.candidates`
            # para mais detalhes sobre por que a geração pode ter falhado ou retornado vazio.
            print("A resposta do modelo não contém texto. Detalhes da resposta:")
            print(response) # Imprime o objeto de resposta completo para depuração
            return "Não foi possível gerar uma resposta em texto para este prompt/imagem."

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
    modelo_escolhido = 'gemini-1.5-pro-latest' # Ou 'gemini-1.5-flash-latest'

    # Seu prompt de texto
    prompt_para_modelo = "Me retorne o que está escrito no canto inferior direito da imagem após a palavra 'Zoom:' e o número após o caracter '#'"

    # Caminho para sua imagem. Mude para o caminho real de um arquivo de imagem local.
    # Deixe como None se você quiser enviar apenas texto.
    caminho_da_imagem = 'data/mask_generation_test/312323/exam_mask.png' # <--- MUDAR PARA O CAMINHO REAL DA IMAGEM

    # --- Execute a interação com o modelo ---
    # Certifique-se de que a imagem existe no caminho especificado se image_path não for None.
    if caminho_da_imagem and not os.path.exists(caminho_da_imagem):
        print(f"\nAVISO: O arquivo de imagem especificado '{caminho_da_imagem}' não existe.")
        print("Tentando interagir com o modelo APENAS com o prompt de texto.")
        caminho_da_imagem = None # Define como None para a chamada da função

    print(f"\nChamando o modelo '{modelo_escolhido}' com o prompt e imagem (se fornecida)...")
    resposta_do_modelo = interact_with_gemini(modelo_escolhido, prompt_para_modelo, image_path=caminho_da_imagem)

    # --- Exiba a resposta ---
    print("\n--- Resposta do Modelo ---")
    if resposta_do_modelo:
        print(resposta_do_modelo)
    else:
        print("Não foi possível obter uma resposta do modelo.")