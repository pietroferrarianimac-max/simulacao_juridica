import os
import shutil # Para limpar a pasta FAISS se necessário
from typing import List, Union, Any

# LangChain imports
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Importar constantes do settings.py
from settings import (
    FAISS_INDEX_PATH,
    PATH_PROCESSO_EM_SI,
    PATH_MODELOS_PETICOES,
    PATH_MODELOS_JUIZ,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_SEARCH_K,
    RETRIEVER_FETCH_K,
    GOOGLE_API_KEY # Necessário para GoogleGenerativeAIEmbeddings
)


def carregar_documentos_docx(
    caminho_pasta_ou_arquivo: str,
    tipo_fonte: str,
    id_processo_especifico: Union[str, None] = None
) -> List[Document]:
    """
    Carrega documentos .docx de uma pasta ou um arquivo específico.

    Args:
        caminho_pasta_ou_arquivo: Caminho para a pasta ou arquivo .docx.
        tipo_fonte: String indicando a origem dos documentos (ex: "processo_atual_arquivo", "modelo_peticao").
        id_processo_especifico: ID do processo, usado para metadados se 'processo_atual_arquivo'.

    Returns:
        Uma lista de objetos Document.
    """
    documentos = []
    if not os.path.exists(caminho_pasta_ou_arquivo):
        print(f"AVISO RAG: Caminho não encontrado: {caminho_pasta_ou_arquivo}")
        return documentos

    # Carregar um arquivo específico de processo
    if tipo_fonte == "processo_atual_arquivo" and id_processo_especifico and os.path.isfile(caminho_pasta_ou_arquivo):
        if caminho_pasta_ou_arquivo.endswith(".docx"):
            try:
                loader = Docx2txtLoader(caminho_pasta_ou_arquivo)
                docs_carregados = loader.load()
                for doc in docs_carregados:
                    doc.metadata = {
                        "source_type": tipo_fonte,
                        "file_name": os.path.basename(caminho_pasta_ou_arquivo),
                        "process_id": id_processo_especifico
                    }
                documentos.extend(docs_carregados)
                print(f"[RAG] Carregado processo de ARQUIVO '{os.path.basename(caminho_pasta_ou_arquivo)}' para ID '{id_processo_especifico}'.")
            except Exception as e:
                print(f"Erro ao carregar {caminho_pasta_ou_arquivo}: {e}")
        else:
            print(f"AVISO RAG: Arquivo de processo esperado .docx, encontrado: {caminho_pasta_ou_arquivo}")

    # Carregar todos os .docx de uma pasta (modelos)
    elif tipo_fonte in ["modelo_peticao", "modelo_juiz"] and os.path.isdir(caminho_pasta_ou_arquivo):
        try:
            loader = DirectoryLoader(
                caminho_pasta_ou_arquivo,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=False, # Geralmente não é ideal em logs de backend
                use_multithreading=True,
                silent_errors=True # Erros em arquivos individuais não param tudo
            )
            docs_carregados = loader.load()
            for doc in docs_carregados:
                # Garante que file_name e source_type estejam nos metadados
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "unknown.docx"))
                doc.metadata["source_type"] = tipo_fonte
            documentos.extend(docs_carregados)
            print(f"[RAG] Carregados {len(docs_carregados)} documentos da pasta de modelos '{os.path.basename(caminho_pasta_ou_arquivo)}'.")
        except Exception as e:
            print(f"Erro ao carregar modelos de {caminho_pasta_ou_arquivo}: {e}")
            
    return documentos

def criar_ou_carregar_retriever(
    id_processo: str,
    documento_caso_atual: Union[str, Document, None] = None,
    recriar_indice: bool = False
) -> Union[Any, None]: # Any é para o tipo 'VectorStoreRetriever'
    """
    Cria um novo índice FAISS ou carrega um existente.
    O índice incluirá os modelos (comuns) e o documento específico do processo atual (se fornecido).

    Args:
        id_processo: Identificador do processo.
        documento_caso_atual: Pode ser um objeto Document (gerado por formulários)
                                ou uma string com o nome do arquivo .docx (para fallback).
        recriar_indice: Força a recriação do índice.

    Returns:
        Uma instância de FAISS retriever ou None em caso de falha crítica.
    """
    if not GOOGLE_API_KEY:
        print("ERRO RAG: GOOGLE_API_KEY não configurada. Não é possível criar embeddings.")
        return None
        
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        task_type="retrieval_document",
        google_api_key=GOOGLE_API_KEY
    )

    if recriar_indice and os.path.exists(FAISS_INDEX_PATH):
        print(f"[RAG] Removendo índice FAISS antigo de '{FAISS_INDEX_PATH}' devido à flag recriar_indice.")
        try:
            shutil.rmtree(FAISS_INDEX_PATH)
        except OSError as e:
            print(f"Erro ao remover diretório FAISS antigo: {e}. Continuando com a recriação...")


    if os.path.exists(FAISS_INDEX_PATH) and not recriar_indice:
        try:
            print(f"[RAG] Tentando carregar índice FAISS existente de '{FAISS_INDEX_PATH}'.")
            vector_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings_model,
                allow_dangerous_deserialization=True # Necessário para FAISS com pickle
            )
            print("[RAG] Índice FAISS carregado com sucesso.")

            if documento_caso_atual:
                print("[RAG] Documento do caso atual fornecido e índice existente. Recriando para garantir inclusão.")
                # Chamada recursiva para recriar
                return criar_ou_carregar_retriever(id_processo, documento_caso_atual, recriar_indice=True)
            return vector_store.as_retriever(
                search_kwargs={'k': RETRIEVER_SEARCH_K, 'fetch_k': RETRIEVER_FETCH_K}
            )
        except Exception as e:
            print(f"[RAG] Erro ao carregar índice FAISS: {e}. Recriando...")
            if os.path.exists(FAISS_INDEX_PATH):
                try:
                    shutil.rmtree(FAISS_INDEX_PATH)
                except OSError as e_rem:
                    print(f"Erro ao remover diretório FAISS durante tentativa de recriação: {e_rem}. Falha pode ocorrer.")


    print("[RAG] Criando novo índice FAISS...")
    todos_documentos: List[Document] = []

    if isinstance(documento_caso_atual, Document):
        # Garante que os metadados essenciais estejam presentes
        doc_metadata = documento_caso_atual.metadata or {}
        doc_metadata.update({"source_type": "processo_atual_formulario", "process_id": id_processo})
        documento_caso_atual.metadata = doc_metadata
        todos_documentos.append(documento_caso_atual)
        print(f"[RAG] Adicionado documento do caso atual (gerado por formulário) para ID '{id_processo}'.")
    elif isinstance(documento_caso_atual, str): # É um nome de arquivo .docx
        # Assume que o nome do arquivo é apenas o basename e precisa ser juntado com PATH_PROCESSO_EM_SI
        caminho_completo_processo = os.path.join(PATH_PROCESSO_EM_SI, documento_caso_atual)
        todos_documentos.extend(
            carregar_documentos_docx(caminho_completo_processo, "processo_atual_arquivo", id_processo_especifico=id_processo)
        )
    
    # Carrega modelos de petições e de juiz
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_PETICOES, "modelo_peticao"))
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_JUIZ, "modelo_juiz"))

    if not todos_documentos:
        msg = "ERRO RAG: Nenhum documento foi carregado para o índice. Verifique os modelos e o caso atual."
        print(msg)
        # raise ValueError(msg) # Ou retornar None e deixar o chamador decidir
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_divididos = text_splitter.split_documents(todos_documentos)
    
    if not docs_divididos:
        msg = "ERRO RAG: Nenhum chunk gerado após a divisão dos documentos."
        print(msg)
        # raise ValueError(msg)
        return None

    print(f"[RAG] Documentos divididos em {len(docs_divididos)} chunks.")
    
    try:
        print(f"[RAG] Criando e salvando vector store FAISS em '{FAISS_INDEX_PATH}'.")
        vector_store = FAISS.from_documents(docs_divididos, embeddings_model)
        vector_store.save_local(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Erro fatal ao criar ou salvar FAISS: {e}")
        # raise e # Ou retornar None
        return None

    print("[RAG] Retriever criado e índice salvo com sucesso!")
    return vector_store.as_retriever(
        search_kwargs={'k': RETRIEVER_SEARCH_K, 'fetch_k': RETRIEVER_FETCH_K}
    )

if __name__ == '__main__':
    # Testes básicos (requerem que a estrutura de pastas e arquivos de modelo exista)
    print("--- Testando RAG Utils ---")


    if not os.path.exists(PATH_MODELOS_PETICOES):
        os.makedirs(PATH_MODELOS_PETICOES)
        print(f"Pasta {PATH_MODELOS_PETICOES} criada para teste. Adicione arquivos .docx nela.")
    if not os.path.exists(PATH_MODELOS_JUIZ):
        os.makedirs(PATH_MODELOS_JUIZ)
        print(f"Pasta {PATH_MODELOS_JUIZ} criada para teste. Adicione arquivos .docx nela.")

    docs_peticoes = carregar_documentos_docx(PATH_MODELOS_PETICOES, "modelo_peticao")
    print(f"Modelos de petições carregados: {len(docs_peticoes)}")
    docs_juiz = carregar_documentos_docx(PATH_MODELOS_JUIZ, "modelo_juiz")
    print(f"Modelos de juiz carregados: {len(docs_juiz)}")

    # Teste 2: Criar um retriever (pode demorar um pouco na primeira vez devido ao download do modelo de embedding)
    # Crie um documento de exemplo
    documento_teste_formulario = Document(
        page_content="Este é o conteúdo de um processo de teste vindo de um formulário.",
        metadata={"source_type": "processo_formulario_streamlit", "file_name": "teste_form.txt"}
    )
    
    print("\nTentando criar retriever (pode precisar da GOOGLE_API_KEY no .env)...")
    # Certifique-se que GOOGLE_API_KEY está no seu .env para este teste funcionar
    if GOOGLE_API_KEY:
        retriever = criar_ou_carregar_retriever(
            id_processo="teste_rag_utils_001",
            documento_caso_atual=documento_teste_formulario,
            recriar_indice=True # Força a recriação para o teste
        )
        if retriever:
            print("Retriever criado com sucesso!")
            # Teste de busca (opcional)
            try:
                relevant_docs = retriever.get_relevant_documents("qual o procedimento para petição inicial?")
                print(f"Busca por 'petição inicial' retornou {len(relevant_docs)} documentos.")
                # for i, doc_ret in enumerate(relevant_docs):
                # print(f"  Doc {i+1} (Fonte: {doc_ret.metadata.get('source_type', 'N/A')}, Arquivo: {doc_ret.metadata.get('file_name', 'N/A')})")
            except Exception as e_search:
                print(f"Erro ao testar busca no retriever: {e_search}")
        else:
            print("Falha ao criar o retriever.")
    else:
        print("Pulando teste de criação de retriever pois GOOGLE_API_KEY não foi encontrada.")

    print("\n--- Fim dos Testes RAG Utils ---")