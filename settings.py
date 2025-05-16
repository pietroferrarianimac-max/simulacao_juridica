# settings.py

import os
from dotenv import load_dotenv

# --- 0. Carregamento de Variáveis de Ambiente ---
# Certifique-se de ter um arquivo .env na raiz do projeto com suas chaves
# Exemplo de .env:
# GOOGLE_API_KEY="SUA_GOOGLE_API_KEY_AQUI"
# LANGSMITH_API_KEY="SUA_LANGSMITH_API_KEY_AQUI"
# GOOGLE_API_KEY_SEARCH="SUA_GOOGLE_API_KEY_PARA_SEARCH_AQUI"
# GOOGLE_CSE_ID="SEU_GOOGLE_CSE_ID_AQUI"

load_dotenv()

# Chaves de API e Configurações do Google
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY_SEARCH = os.getenv("GOOGLE_API_KEY_SEARCH")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Configurações do LangSmith (para tracing e debugging)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true") # Default to "true" if not set
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "SimulacaoJuridicaDebug") # Default project name

# Define as variáveis de ambiente para LangChain se LANGCHAIN_TRACING_V2 for 'true'
if LANGCHAIN_TRACING_V2 == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    if LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    if LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Verificação inicial da chave principal do Google (pode ser feita no main_app.py também)
# if not GOOGLE_API_KEY:
#     print("Erro Crítico: A variável de ambiente GOOGLE_API_KEY não foi definida.")
#     # Considerar levantar uma exceção ou ter uma flag para a aplicação principal checar
#     # exit() # Evitar exit() em módulos importáveis

# --- 1. Constantes e Configurações Globais ---

# Caminhos de Dados
DATA_PATH = "simulacao_juridica_data"
PATH_PROCESSO_EM_SI = os.path.join(DATA_PATH, "processo_em_si")
PATH_MODELOS_PETICOES = os.path.join(DATA_PATH, "modelos_peticoes")
PATH_MODELOS_JUIZ = os.path.join(DATA_PATH, "modelos_juiz")
FAISS_INDEX_PATH = "faiss_index_juridico" # Pasta para salvar o índice FAISS

# Nomes dos Nós do Grafo (Atores)
ADVOGADO_AUTOR = "advogado_autor"
JUIZ = "juiz"
ADVOGADO_REU = "advogado_reu"

# Etapas Processuais (conforme o rito ordinário do CPC)
ETAPA_PETICAO_INICIAL = "PETICAO_INICIAL"
ETAPA_DESPACHO_RECEBENDO_INICIAL = "DESPACHO_RECEBENDO_INICIAL"
ETAPA_CONTESTACAO = "CONTESTACAO"
ETAPA_DECISAO_SANEAMENTO = "DECISAO_SANEAMENTO"
ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR = "MANIFESTACAO_SEM_PROVAS_AUTOR"
ETAPA_MANIFESTACAO_SEM_PROVAS_REU = "MANIFESTACAO_SEM_PROVAS_REU"
ETAPA_SENTENCA = "SENTENCA"
ETAPA_FIM_PROCESSO = "_FIM_PROCESSO_" # Etapa final especial

# Modelos LLM
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"
EMBEDDING_MODEL_NAME = "models/embedding-001"

# Configurações de RAG
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
RETRIEVER_SEARCH_K = 5
RETRIEVER_FETCH_K = 10

# Configurações de UI (podem ser movidas para um ui_settings.py se crescerem muito)
FORM_STEPS = [
    "autor",
    "reu",
    "fatos",
    "direito",
    "pedidos",
    "natureza_acao",
    "documentos_autor",
    "revisar_e_simular"
]

TIPOS_DOCUMENTOS_COMUNS = [
    "Nenhum (Apenas Descrição Factual)",
    "Contrato (Partes, Objeto, Data)",
    "Termo de Declaração (Declarante, Data, Fatos Declarados)",
    "Laudo Pericial/Técnico (Perito, Objeto, Conclusões, Data)",
    "Procuração Ad Judicia",
    "Comprovante de Residência (Recente)",
    "Documento de Identidade (RG/CPF/CNH do Autor)",
    "Documento de Identidade (RG/CPF/CNH do Réu - se souber)",
    "CNPJ e Contrato Social (Pessoa Jurídica)",
    "Prints de Conversa (WhatsApp, E-mail, etc. - Datas, Interlocutores, Conteúdo Relevante)",
    "Ata Notarial (Fatos Constatados, Data)",
    "Fotografias/Vídeos (Descrição do Conteúdo, Data)",
    "Notas Fiscais/Recibos (Emitente, Destinatário, Valor, Data, Produto/Serviço)",
    "Extratos Bancários (Período, Transações Relevantes)",
    "Prontuário Médico/Atestado (Paciente, Médico, Data, Informações Relevantes)",
    "Boletim de Ocorrência (Fatos Narrados, Data, Envolvidos)",
    "Certidão (Casamento, Nascimento, Óbito, Imóvel, etc.)",
    "Notificação Extrajudicial (Enviada, Recebida, Conteúdo, Data)",
    "Outro (Especificar)"
]

# Cores para Análise de Sentimento (Pode ser útil se a lógica de exibição for modularizada também)
SENTIMENTO_CORES = {
    "Assertivo": "lightblue", "Confiante": "lightgreen", "Persuasivo": "lightyellow",
    "Combativo": "lightcoral", "Agressivo": "salmon", "Indignado": "lightpink",
    "Neutro": "lightgray", "Formal": "whitesmoke",
    "Conciliatório": "palegreen", "Colaborativo": "mediumaquamarine",
    "Defensivo": "thistle", "Emocional": "lavenderblush",
    "Não analisado": "silver", "Erro na análise": "orangered"
}
DEFAULT_SENTIMENTO_COR = "gainsboro"

if __name__ == '__main__':
    # Pequeno teste para verificar se as variáveis estão sendo carregadas
    print("--- Testando Configurações Carregadas ---")
    print(f"GOOGLE_API_KEY: {'********' if GOOGLE_API_KEY else 'NÃO DEFINIDA'}")
    print(f"LANGSMITH_API_KEY: {'********' if LANGSMITH_API_KEY else 'NÃO DEFINIDA'}")
    print(f"LANGCHAIN_PROJECT: {LANGCHAIN_PROJECT}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"ADVOGADO_AUTOR: {ADVOGADO_AUTOR}")
    print(f"ETAPA_PETICAO_INICIAL: {ETAPA_PETICAO_INICIAL}")
    print(f"Existem {len(FORM_STEPS)} etapas no formulário.")
    print(f"Primeiro tipo de documento comum: {TIPOS_DOCUMENTOS_COMUNS[0]}")