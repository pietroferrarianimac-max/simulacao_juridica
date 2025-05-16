import os
import traceback # For detailed error logging if search tool setup fails

# LangChain & Google imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper # Correct import
from langchain_google_community.search import GoogleSearchRun # Correct import

# Import necessary settings
from settings import (
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    GOOGLE_API_KEY_SEARCH,
    GOOGLE_CSE_ID
)

# --- LLM Initialization (Gemini) ---
llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.6, # As per original script
            convert_system_message_to_human=True,
            google_api_key=GOOGLE_API_KEY
        )
        print(f"[LLM] ChatGoogleGenerativeAI model '{GEMINI_MODEL_NAME}' initialized successfully.")
    except Exception as e:
        print(f"[LLM_ERROR] Failed to initialize ChatGoogleGenerativeAI: {e}")
        llm = None # Ensure llm is None if initialization fails
else:
    print("[LLM_WARNING] GOOGLE_API_KEY not found. LLM (ChatGoogleGenerativeAI) will not be available.")

# --- Google Search Tool Initialization ---
search_tool = None
if GOOGLE_API_KEY_SEARCH and GOOGLE_CSE_ID:
    try:
        search_api_wrapper_instance = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY_SEARCH,
            google_cse_id=GOOGLE_CSE_ID
        )
        search_tool = GoogleSearchRun( # This is the GoogleSearchRun tool instance
            api_wrapper=search_api_wrapper_instance
            # description="Uma ferramenta para buscar informações atuais na web usando o Google Search. Útil para encontrar jurisprudência recente ou notícias."
        )
        print("[LLM_TOOL] Google Search tool (GoogleSearchRun) configured successfully.")
    except Exception as e_config_tool:
        print(f"[LLM_TOOL_WARNING] Falha ao configurar search_tool (GoogleSearchRun): {e_config_tool}")
        print(traceback.format_exc())
        search_tool = None
else:
    print("[LLM_TOOL_WARNING] GOOGLE_API_KEY_SEARCH ou GOOGLE_CSE_ID não definidos. Ferramenta Google Search (search_tool) desabilitada.")

if __name__ == '__main__':
    print("\n--- Testando Configurações de LLM e Ferramentas ---")
    if llm:
        print(f"Modelo LLM ({GEMINI_MODEL_NAME}) está carregado.")

    else:
        print("Modelo LLM não está carregado (verifique GOOGLE_API_KEY e logs).")

    if search_tool:
        print("Ferramenta Google Search (search_tool) está configurada.")

    else:
        print("Ferramenta Google Search (search_tool) não está configurada (verifique chaves de API e logs).")
    print("--- Fim dos Testes ---")