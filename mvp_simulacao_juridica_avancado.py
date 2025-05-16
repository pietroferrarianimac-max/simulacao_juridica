# mvp_simulacao_juridica_avancado.py

import os
import shutil # Para limpar a pasta FAISS se necessário
import time # Para ID único de processo
from dotenv import load_dotenv
from typing import TypedDict, List, Union, Dict, Tuple, Any

# LangChain & LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document # Adicionado


import streamlit as st

# --- 0. Carregamento de Variáveis de Ambiente ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
    # No Streamlit, é melhor mostrar isso na UI
    # exit() # Evitar exit em apps Streamlit

os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "SimulacaoJuridicaDebug"

# --- 1. Constantes e Configurações Globais ---
DATA_PATH = "simulacao_juridica_data"
PATH_PROCESSO_EM_SI = os.path.join(DATA_PATH, "processo_em_si") # Pode ser menos usado agora
PATH_MODELOS_PETICOES = os.path.join(DATA_PATH, "modelos_peticoes")
PATH_MODELOS_JUIZ = os.path.join(DATA_PATH, "modelos_juiz")
FAISS_INDEX_PATH = "faiss_index_juridico" # Pasta para salvar o índice FAISS

# Nomes dos nós do grafo (atores)
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

# --- 2. Utilitários RAG (MODIFICADO) ---

def carregar_documentos_docx(caminho_pasta_ou_arquivo: str, tipo_fonte: str, id_processo_especifico: Union[str, None] = None) -> List[Document]:
    documentos = []
    if not os.path.exists(caminho_pasta_ou_arquivo):
        print(f"AVISO RAG: Caminho não encontrado: {caminho_pasta_ou_arquivo}")
        return documentos

    # Carregar um arquivo específico de processo (caso atual via upload de arquivo - fallback)
    if tipo_fonte == "processo_atual_arquivo" and id_processo_especifico and os.path.isfile(caminho_pasta_ou_arquivo):
        if caminho_pasta_ou_arquivo.endswith(".docx"):
            try:
                loader = Docx2txtLoader(caminho_pasta_ou_arquivo)
                docs_carregados = loader.load()
                for doc in docs_carregados:
                    doc.metadata = {"source_type": tipo_fonte, "file_name": os.path.basename(caminho_pasta_ou_arquivo), "process_id": id_processo_especifico}
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
                show_progress=False,
                use_multithreading=True,
                silent_errors=True
            )
            docs_carregados = loader.load()
            for doc in docs_carregados:
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "unknown.docx"))
                doc.metadata["source_type"] = tipo_fonte
            documentos.extend(docs_carregados)
            print(f"[RAG] Carregados {len(docs_carregados)} documentos da pasta de modelos '{os.path.basename(caminho_pasta_ou_arquivo)}'.")
        except Exception as e:
            print(f"Erro ao carregar modelos de {caminho_pasta_ou_arquivo}: {e}")
            
    return documentos

def criar_ou_carregar_retriever(id_processo: str, documento_caso_atual: Union[str, Document, None] = None, recriar_indice: bool = False):
    """
    Cria um novo índice FAISS ou carrega um existente.
    O índice incluirá os modelos (comuns) e o documento específico do processo atual (se fornecido).
    Args:
        id_processo: Identificador do processo.
        documento_caso_atual: Pode ser um objeto Document (gerado pelos formulários)
                                 ou uma string com o nome do arquivo .docx (para fallback).
        recriar_indice: Força a recriação do índice.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

    if recriar_indice and os.path.exists(FAISS_INDEX_PATH):
        print(f"[RAG] Removendo índice FAISS antigo de '{FAISS_INDEX_PATH}' devido à flag recriar_indice.")
        shutil.rmtree(FAISS_INDEX_PATH)

    if os.path.exists(FAISS_INDEX_PATH) and not recriar_indice:
        try:
            print(f"[RAG] Tentando carregar índice FAISS existente de '{FAISS_INDEX_PATH}'.")
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
            print("[RAG] Índice FAISS carregado com sucesso.")
            # Se um novo documento do caso atual foi fornecido e o índice já existe,
            # idealmente, adicionaríamos apenas esse novo documento.
            # Por simplicidade, se um novo doc for fornecido e o índice existir,
            # recriaremos para garantir que ele esteja incluído.
            if documento_caso_atual:
                print("[RAG] Documento do caso atual fornecido. Recriando índice para garantir sua inclusão.")
                # shutil.rmtree(FAISS_INDEX_PATH) # Comentado para permitir carregamento se não quiser recriar sempre
                # return criar_ou_carregar_retriever(id_processo, documento_caso_atual, recriar_indice=True)
            return vector_store.as_retriever(search_kwargs={'k': 5, 'fetch_k': 10})
        except Exception as e:
            print(f"[RAG] Erro ao carregar índice FAISS: {e}. Recriando...")
            if os.path.exists(FAISS_INDEX_PATH): shutil.rmtree(FAISS_INDEX_PATH)

    print("[RAG] Criando novo índice FAISS...")
    todos_documentos = []

    if isinstance(documento_caso_atual, Document):
        documento_caso_atual.metadata.update({"source_type": "processo_atual_formulario", "process_id": id_processo})
        todos_documentos.append(documento_caso_atual)
        print(f"[RAG] Adicionado documento do caso atual (gerado por formulário) para ID '{id_processo}'.")
    elif isinstance(documento_caso_atual, str): # É um nome de arquivo .docx
        caminho_completo_processo = os.path.join(PATH_PROCESSO_EM_SI, documento_caso_atual)
        todos_documentos.extend(carregar_documentos_docx(caminho_completo_processo, "processo_atual_arquivo", id_processo_especifico=id_processo))
    
    # Carrega modelos
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_PETICOES, "modelo_peticao"))
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_JUIZ, "modelo_juiz"))

    if not todos_documentos:
        msg = "ERRO RAG: Nenhum documento foi carregado para o índice. Verifique os modelos e o caso atual."
        print(msg)
        # Em Streamlit, é melhor mostrar isso na UI
        # raise ValueError(msg)
        st.error(msg)
        return None # Indica falha

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs_divididos = text_splitter.split_documents(todos_documentos)
    
    if not docs_divididos:
        msg = "ERRO RAG: Nenhum chunk gerado após a divisão dos documentos."
        print(msg)
        # raise ValueError(msg)
        st.error(msg)
        return None

    print(f"[RAG] Documentos divididos em {len(docs_divididos)} chunks.")
    
    try:
        print(f"[RAG] Criando e salvando vector store FAISS em '{FAISS_INDEX_PATH}'.")
        vector_store = FAISS.from_documents(docs_divididos, embeddings_model)
        vector_store.save_local(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Erro fatal ao criar ou salvar FAISS: {e}")
        st.error(f"Erro fatal ao criar ou salvar FAISS: {e}")
        # raise e
        return None

    print("[RAG] Retriever criado e índice salvo com sucesso!")
    return vector_store.as_retriever(search_kwargs={'k': 5, 'fetch_k': 10})

# --- 3. Inicialização do LLM (Modelo Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.6, convert_system_message_to_human=True)

# --- 4. Definição do Estado Processual (LangGraph) ---
class EstadoProcessual(TypedDict):
    id_processo: str
    retriever: Any # FAISS retriever instance
    
    nome_do_ultimo_no_executado: Union[str, None]
    etapa_concluida_pelo_ultimo_no: Union[str, None]
    proximo_ator_sugerido_pelo_ultimo_no: Union[str, None]
    
    etapa_a_ser_executada_neste_turno: str

    documento_gerado_na_etapa_recente: Union[str, None]
    historico_completo: List[Dict[str, str]]
    
    pontos_controvertidos_saneamento: Union[str, None]
    manifestacao_autor_sem_provas: bool
    manifestacao_reu_sem_provas: bool
    
    # NOVO: Para carregar dados do formulário Streamlit
    dados_formulario_entrada: Union[Dict[str, Any], None]


# --- 5. Mapa de Fluxo Processual (Rito Ordinário) ---
mapa_tarefa_no_atual: Dict[Tuple[Union[str, None], Union[str, None], str], str] = {
    (None, None, ADVOGADO_AUTOR): ETAPA_PETICAO_INICIAL,
    (ADVOGADO_AUTOR, ETAPA_PETICAO_INICIAL, JUIZ): ETAPA_DESPACHO_RECEBENDO_INICIAL,
    (JUIZ, ETAPA_DESPACHO_RECEBENDO_INICIAL, ADVOGADO_REU): ETAPA_CONTESTACAO,
    (ADVOGADO_REU, ETAPA_CONTESTACAO, JUIZ): ETAPA_DECISAO_SANEAMENTO,
    (JUIZ, ETAPA_DECISAO_SANEAMENTO, ADVOGADO_AUTOR): ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    (ADVOGADO_AUTOR, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR, ADVOGADO_REU): ETAPA_MANIFESTACAO_SEM_PROVAS_REU,
    (ADVOGADO_REU, ETAPA_MANIFESTACAO_SEM_PROVAS_REU, JUIZ): ETAPA_SENTENCA,
}

# --- 6. Função de Roteamento Condicional (Router) ---
def decidir_proximo_no_do_grafo(estado: EstadoProcessual):
    proximo_ator_sugerido = estado.get("proximo_ator_sugerido_pelo_ultimo_no")
    etapa_concluida = estado.get("etapa_concluida_pelo_ultimo_no")

    print(f"[Router] Estado recebido: { {k: v for k, v in estado.items() if k not in ['retriever', 'dados_formulario_entrada']} }") # Evita logar objetos grandes
    print(f"[Router] Decidindo próximo nó. Última etapa concluída: '{etapa_concluida}'. Próximo ator sugerido: '{proximo_ator_sugerido}'.")

    if proximo_ator_sugerido == ADVOGADO_AUTOR: return ADVOGADO_AUTOR
    if proximo_ator_sugerido == JUIZ: return JUIZ
    if proximo_ator_sugerido == ADVOGADO_REU: return ADVOGADO_REU
    if proximo_ator_sugerido == ETAPA_FIM_PROCESSO or etapa_concluida == ETAPA_SENTENCA:
        print("[Router] Fluxo direcionado para o FIM.")
        return END
    print(f"[Router] ERRO: Próximo ator '{proximo_ator_sugerido}' desconhecido ou fluxo não previsto. Encerrando.")
    return END

# --- 7. Helpers e Agentes (Nós do Grafo) ---

def criar_prompt_e_chain(template_string: str):
    prompt = ChatPromptTemplate.from_template(template_string)
    return prompt | llm | StrOutputParser()

def helper_logica_inicial_no(estado: EstadoProcessual, nome_do_no_atual: str) -> str:
    nome_ultimo_no = estado.get("nome_do_ultimo_no_executado")
    etapa_ultimo_no = estado.get("etapa_concluida_pelo_ultimo_no")
    chave_mapa = (nome_ultimo_no, etapa_ultimo_no, nome_do_no_atual)
    
    if nome_ultimo_no is None and etapa_ultimo_no is None and nome_do_no_atual == ADVOGADO_AUTOR:
        print(f"[{nome_do_no_atual}] Ponto de entrada, definindo etapa como PETICAO_INICIAL.")
        return ETAPA_PETICAO_INICIAL
    
    etapa_designada = mapa_tarefa_no_atual.get(chave_mapa)
    
    if not etapa_designada:
        print(f"ERRO [{nome_do_no_atual}]: Não foi possível determinar a etapa atual no mapa de tarefas com a chave: {chave_mapa}")
        print(f"Estado atual recebido pelo nó: nome_do_ultimo_no_executado='{nome_ultimo_no}', etapa_concluida_pelo_ultimo_no='{etapa_ultimo_no}'")
        return "ERRO_ETAPA_NAO_ENCONTRADA"
        
    print(f"[{nome_do_no_atual}] Iniciando. Etapa designada: {etapa_designada}.")
    return etapa_designada

def agente_advogado_autor(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_AUTOR)
    print(f"\n--- TURNO: {ADVOGADO_AUTOR} (Executando Etapa: {etapa_atual_do_no}) ---")

    # (Lógica de erro e inicialização de variáveis como antes)
    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        # ... (mesma lógica de erro)
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_AUTOR, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro crítico de fluxo.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_AUTOR, "documento": "Erro de fluxo."}],
            "id_processo": estado.get("id_processo"), "retriever": estado.get("retriever"),
            "dados_formulario_entrada": estado.get("dados_formulario_entrada")
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    dados_formulario = estado.get("dados_formulario_entrada", {})
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Este é o primeiro ato do processo."

    # --- Lógica de RAG para contexto (pode ser menos crucial para PI se dados do formulário são completos) ---
    # query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo}"
    # docs_fatos_caso = retriever.get_relevant_documents(query=query_fatos_caso) if retriever else []
    # contexto_fatos_caso_rag = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual do RAG."
    # print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso_rag[:200]}...")


    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        docs_modelo_pi = retriever.get_relevant_documents(query="modelo de petição inicial completa", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_pi[0].page_content if docs_modelo_pi else "ERRO RAG: Modelo de petição inicial não encontrado."
        
        # Usar dados do formulário como base para a Petição Inicial
        qualificacao_autor_form = dados_formulario.get("qualificacao_autor", "Qualificação do Autor não fornecida.")
        qualificacao_reu_form = dados_formulario.get("qualificacao_reu", "Qualificação do Réu não fornecida.")
        natureza_acao_form = dados_formulario.get("natureza_acao", "Natureza da ação não fornecida.")
        fatos_form = dados_formulario.get("fatos", "Fatos não fornecidos.")
        direito_form = dados_formulario.get("fundamentacao_juridica", "Fundamentação jurídica não fornecida.")
        pedidos_form = dados_formulario.get("pedidos", "Pedidos não fornecidos.")

        template_prompt_pi = f"""
        Você é um Advogado do Autor e está elaborando uma Petição Inicial completa e formal.
        **Processo ID:** {{id_processo}}

        **Dados Base Fornecidos para a Petição:**
        Qualificação do Autor:
        {qualificacao_autor_form}

        Qualificação do Réu:
        {qualificacao_reu_form}

        Natureza da Ação: {natureza_acao_form}

        Dos Fatos:
        {fatos_form}

        Do Direito (Fundamentação Jurídica):
        {direito_form}

        Dos Pedidos:
        {pedidos_form}

        **Modelo/Guia Estrutural de Petição Inicial (RAG - use para formatação e completude, mas priorize os dados fornecidos acima):**
        {{modelo_texto_guia}}
        
        **Histórico Processual (se houver, para outros contextos):**
        {{historico_formatado}}
        ---
        Com base em TODAS as informações acima, especialmente os DADOS BASE FORNECIDOS, redija a Petição Inicial completa e bem formatada.
        Certifique-se de que todos os elementos dos dados base (fatos, direito, pedidos, qualificações) estejam integralmente e corretamente incorporados.
        Petição Inicial:
        """
        chain_pi = criar_prompt_e_chain(template_prompt_pi)
        documento_gerado = chain_pi.invoke({
            "id_processo": id_processo,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
            # Os dados do formulário já estão no template_prompt_pi
        })
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR:
        # (Lógica como antes, usando RAG para modelo de manifestação)
        decisao_saneamento_recebida = estado.get("documento_gerado_na_etapa_recente", "ERRO: Decisão de Saneamento não encontrada no estado.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não detalhados.")
        docs_modelo_manifest = retriever.get_relevant_documents(query="modelo de petição ou manifestação declarando não ter mais provas a produzir", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_manifest[0].page_content if docs_modelo_manifest else "ERRO RAG: Modelo de manifestação sem provas não encontrado."

        template_prompt_manifest_autor = """
        Você é o Advogado do Autor. O juiz proferiu Decisão de Saneamento e intimou as partes a especificarem provas.
        **Processo ID:** {id_processo}
        **Tarefa:** Elaborar uma petição informando que o Autor não possui mais provas a produzir.
        **Decisão de Saneamento proferida pelo Juiz:**
        {decisao_saneamento_recebida}
        (Pontos Controvertidos principais: {pontos_controvertidos})
        **Modelo/Guia de Manifestação (use como referência):**
        {modelo_texto_guia}
        **Histórico Processual:**
        {historico_formatado}
        ---
        Redija a Petição de Manifestação do Autor.
        Manifestação:
        """
        chain_manifest_autor = criar_prompt_e_chain(template_prompt_manifest_autor)
        documento_gerado = chain_manifest_autor.invoke({
            "id_processo": id_processo,
            "decisao_saneamento_recebida": decisao_saneamento_recebida,
            "pontos_controvertidos": pontos_controvertidos,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_REU

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_AUTOR} não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no}."
        proximo_ator_logico = JUIZ

    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_AUTOR, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    # Preservar o estado
    estado_retorno = {k: v for k, v in estado.items()}
    estado_retorno.update({
        "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR),
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    })
    return estado_retorno

def agente_juiz(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, JUIZ)
    print(f"\n--- TURNO: {JUIZ} (Executando Etapa: {etapa_atual_do_no}) ---")
    # (Lógica de erro e inicialização de variáveis como antes)
    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        return {
            "nome_do_ultimo_no_executado": JUIZ, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro Juiz.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": JUIZ, "documento": "Erro Juiz."}],
             "id_processo": estado.get("id_processo"), "retriever": estado.get("retriever"),
             "dados_formulario_entrada": estado.get("dados_formulario_entrada")
        }

    documento_gerado = f"Decisão padrão para {JUIZ} na etapa {etapa_atual_do_no} não gerada."
    proximo_ator_logico = ETAPA_FIM_PROCESSO
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    # dados_formulario = estado.get("dados_formulario_entrada", {}) # Juiz geralmente não usa dados primários do form
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."
    documento_da_parte_para_analise = estado.get("documento_gerado_na_etapa_recente", "Nenhuma petição recente.")
    
    # Contexto RAG (se necessário, ex: para fundamentar com base nos fatos do caso já indexados)
    # query_fatos_caso_juiz = f"Resumo dos fatos e principais pontos do processo {id_processo} para decisão judicial"
    # docs_fatos_caso_juiz = retriever.get_relevant_documents(query=query_fatos_caso_juiz) if retriever else []
    # contexto_fatos_caso_rag_juiz = "\n".join([doc.page_content for doc in docs_fatos_caso_juiz]) if docs_fatos_caso_juiz else "Nenhum detalhe factual do RAG para o juiz."
    contexto_fatos_caso_rag_juiz = "Contexto RAG do caso para o juiz (simplificado para este exemplo)." # Placeholder

    pontos_controvertidos_definidos = estado.get("pontos_controvertidos_saneamento")

    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        docs_modelo_despacho = retriever.get_relevant_documents(query="modelo de despacho judicial recebendo petição inicial", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_despacho[0].page_content if docs_modelo_despacho else "ERRO RAG: Modelo de despacho inicial não encontrado."
        template_prompt = """
        Você é um Juiz de Direito. Analise a Petição Inicial e profira um despacho inicial.
        **Processo ID:** {id_processo}
        **Petição Inicial apresentada pelo Autor:**
        {peticao_inicial}
        **Modelo/Guia de Despacho (use como referência):**
        {modelo_texto_guia}
        **Histórico Processual:**
        {historico_formatado}
        ---
        Redija o Despacho Inicial.
        Despacho Inicial:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "peticao_inicial": documento_da_parte_para_analise,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_REU

    elif etapa_atual_do_no == ETAPA_DECISAO_SANEAMENTO:
        docs_modelo_saneamento = retriever.get_relevant_documents(query="modelo de decisão de saneamento", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_saneamento[0].page_content if docs_modelo_saneamento else "ERRO RAG: Modelo de saneamento não encontrado."
        template_prompt = """
        Você é um Juiz de Direito. Processe está na fase de saneamento.
        **Processo ID:** {id_processo}
        **Última manifestação das partes (Contestação):**
        {ultima_peticao_partes}
        **Contexto RAG do Caso (se necessário):**
        {contexto_fatos_caso_rag}
        **Modelo/Guia de Decisão de Saneamento:**
        {modelo_texto_guia}
        **Histórico Processual:**
        {historico_formatado}
        ---
        Redija a Decisão de Saneamento, definindo PONTOS CONTROVERTIDOS.
        Decisão de Saneamento:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "ultima_peticao_partes": documento_da_parte_para_analise,
            "contexto_fatos_caso_rag": contexto_fatos_caso_rag_juiz,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_AUTOR
        try: # Extração de pontos controvertidos
            inicio_pc = documento_gerado.upper().find("PONTOS CONTROVERTIDOS:")
            if inicio_pc != -1:
                fim_pc = documento_gerado.find("\n\n", inicio_pc) 
                if fim_pc == -1: fim_pc = len(documento_gerado)
                pontos_controvertidos_definidos = documento_gerado[inicio_pc + len("PONTOS CONTROVERTIDOS:"):fim_pc].strip()
            else: pontos_controvertidos_definidos = "Não extraído."
        except Exception: pontos_controvertidos_definidos = "Erro extração."
        print(f"[{JUIZ}-{etapa_atual_do_no}] Pontos Controvertidos: {pontos_controvertidos_definidos}")

    elif etapa_atual_do_no == ETAPA_SENTENCA:
        docs_modelo_sentenca = retriever.get_relevant_documents(query="modelo de sentença cível", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_sentenca[0].page_content if docs_modelo_sentenca else "ERRO RAG: Modelo de sentença não encontrado."
        template_prompt = """
        Você é um Juiz de Direito. O processo está concluso para sentença.
        **Processo ID:** {id_processo}
        **Última manifestação (Ex: Réu sem provas):**
        {ultima_peticao_partes}
        **Pontos Controvertidos definidos:**
        {pontos_controvertidos_saneamento}
        **Contexto RAG do Caso:**
        {contexto_fatos_caso_rag}
        **Modelo/Guia de Sentença:**
        {modelo_texto_guia}
        **Histórico Processual Completo:**
        {historico_formatado}
        ---
        Redija a Sentença.
        Sentença:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "ultima_peticao_partes": documento_da_parte_para_analise,
            "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento", "Não definidos."),
            "contexto_fatos_caso_rag": contexto_fatos_caso_rag_juiz,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ETAPA_FIM_PROCESSO
    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {JUIZ} não implementada.")
        documento_gerado = f"Conteúdo para {JUIZ} na etapa {etapa_atual_do_no}."
        proximo_ator_logico = ETAPA_FIM_PROCESSO

    print(f"[{JUIZ}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]
    
    estado_retorno = {k: v for k, v in estado.items()}
    estado_retorno.update({
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": pontos_controvertidos_definidos, # Atualiza
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    })
    return estado_retorno

def agente_advogado_reu(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_REU)
    print(f"\n--- TURNO: {ADVOGADO_REU} (Executando Etapa: {etapa_atual_do_no}) ---")
    # (Lógica de erro e inicialização de variáveis como antes)
    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_REU, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro Réu.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_REU, "documento": "Erro Réu."}],
            "id_processo": estado.get("id_processo"), "retriever": estado.get("retriever"),
            "dados_formulario_entrada": estado.get("dados_formulario_entrada")
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    # dados_formulario = estado.get("dados_formulario_entrada", {})
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."
    documento_relevante_anterior = estado.get("documento_gerado_na_etapa_recente", "Nenhum documento recente.")
    
    # Contexto RAG (fatos do caso sob perspectiva da defesa)
    # query_fatos_caso_reu = f"Resumo dos fatos e principais pontos do processo {id_processo} sob a perspectiva da defesa"
    # docs_fatos_caso_reu = retriever.get_relevant_documents(query=query_fatos_caso_reu) if retriever else []
    # contexto_fatos_caso_rag_reu = "\n".join([doc.page_content for doc in docs_fatos_caso_reu]) if docs_fatos_caso_reu else "Nenhum detalhe factual do RAG para a defesa."
    contexto_fatos_caso_rag_reu = "Contexto RAG do caso para o réu (simplificado)." # Placeholder

    if etapa_atual_do_no == ETAPA_CONTESTACAO:
        peticao_inicial_autor = "Petição Inicial não encontrada no histórico para contestação." 
        for item in reversed(estado.get("historico_completo", [])): # Busca a PI no histórico
            if item["etapa"] == ETAPA_PETICAO_INICIAL and item["ator"] == ADVOGADO_AUTOR:
                peticao_inicial_autor = item["documento"]
                break
        
        docs_modelo_contestacao = retriever.get_relevant_documents(query="modelo de contestação cível completa", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_contestacao[0].page_content if docs_modelo_contestacao else "ERRO RAG: Modelo de contestação não encontrado."
        template_prompt = """
        Você é um Advogado do Réu. Elabore uma Contestação.
        **Processo ID:** {id_processo}
        **Despacho Judicial Recebido (citação):**
        {despacho_judicial}
        **Petição Inicial do Autor (a ser contestada):**
        {peticao_inicial_autor}
        **Contexto RAG dos Fatos (perspectiva da defesa, se houver):**
        {contexto_fatos_caso_rag}
        **Modelo/Guia de Contestação:**
        {modelo_texto_guia}
        **Histórico Processual:**
        {historico_formatado}
        ---
        Redija a Contestação completa.
        Contestação:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "despacho_judicial": documento_relevante_anterior,
            "peticao_inicial_autor": peticao_inicial_autor,
            "contexto_fatos_caso_rag": contexto_fatos_caso_rag_reu,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        manifestacao_autor_sem_provas_doc = documento_relevante_anterior
        decisao_saneamento_texto = "Decisão de Saneamento não encontrada."
        if estado.get("pontos_controvertidos_saneamento"):
             decisao_saneamento_texto = f"Decisão de Saneamento anterior definiu: {estado['pontos_controvertidos_saneamento']}"
        
        docs_modelo_manifest_reu = retriever.get_relevant_documents(query="modelo de petição réu sem mais provas", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_manifest_reu[0].page_content if docs_modelo_manifest_reu else "ERRO RAG: Modelo manifestação réu sem provas não encontrado."
        template_prompt = """
        Você é o Advogado do Réu. O Autor já se manifestou sobre não ter mais provas.
        **Processo ID:** {id_processo}
        **Decisão de Saneamento (resumo):**
        {decisao_saneamento_texto}
        **Manifestação do Autor sobre não ter mais provas:**
        {manifestacao_autor_sem_provas_doc}
        **Modelo/Guia de Manifestação do Réu:**
        {modelo_texto_guia}
        **Histórico Processual:**
        {historico_formatado}
        ---
        Redija a Petição de Manifestação do Réu (informando não ter mais provas).
        Manifestação do Réu:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "decisao_saneamento_texto": decisao_saneamento_texto,
            "manifestacao_autor_sem_provas_doc": manifestacao_autor_sem_provas_doc,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ
    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_REU} não implementada.")
        documento_gerado = f"Conteúdo para {ADVOGADO_REU} na etapa {etapa_atual_do_no}."
        proximo_ator_logico = JUIZ

    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_REU, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    estado_retorno = {k: v for k, v in estado.items()}
    estado_retorno.update({
        "nome_do_ultimo_no_executado": ADVOGADO_REU,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU),
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    })
    return estado_retorno

# --- 8. Construção do Grafo LangGraph ---
workflow = StateGraph(EstadoProcessual)
workflow.add_node(ADVOGADO_AUTOR, agente_advogado_autor)
workflow.add_node(JUIZ, agente_juiz)
workflow.add_node(ADVOGADO_REU, agente_advogado_reu)
workflow.set_entry_point(ADVOGADO_AUTOR)
roteamento_mapa_edges = { ADVOGADO_AUTOR: ADVOGADO_AUTOR, JUIZ: JUIZ, ADVOGADO_REU: ADVOGADO_REU, END: END }
workflow.add_conditional_edges(ADVOGADO_AUTOR, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(JUIZ, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(ADVOGADO_REU, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
app = workflow.compile()

# --- 9. Interface Streamlit e Lógica de Execução ---

# --- Constantes e Funções da UI Streamlit ---
FORM_STEPS = ["autor", "reu", "natureza_acao", "fatos", "direito", "pedidos", "revisar_e_simular"]

def inicializar_estado_formulario():
    if 'current_form_step_index' not in st.session_state:
        st.session_state.current_form_step_index = 0
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            "id_processo": f"caso_sim_{int(time.time())}",
            "qualificacao_autor": "", "qualificacao_reu": "", "natureza_acao": "",
            "fatos": "", "fundamentacao_juridica": "", "pedidos": ""
        }
    if 'ia_generated_content_flags' not in st.session_state:
        # Inicializa flags para todos os campos em form_data, não apenas os de texto principais
        st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_results' not in st.session_state: # Garantir que simulation_results sempre exista
        st.session_state.simulation_results = {}
    if 'doc_visualizado' not in st.session_state: # Para a exibição de documentos
        st.session_state.doc_visualizado = None
    if 'doc_visualizado_titulo' not in st.session_state:
        st.session_state.doc_visualizado_titulo = ""

def gerar_conteudo_com_ia(prompt_template_str: str, campos_prompt: dict, campo_formulario_display: str, chave_estado: str):
    if not GOOGLE_API_KEY:
        st.error("A chave API do Google não foi configurada. Não é possível usar a IA.")
        return
    try:
        with st.spinner(f"Gerando conteúdo para '{campo_formulario_display}' com IA..."):
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | llm | StrOutputParser()
            conteudo_gerado = chain.invoke(campos_prompt)
            st.session_state.form_data[chave_estado] = conteudo_gerado
            st.session_state.ia_generated_content_flags[chave_estado] = True
        st.rerun() # Usar st.rerun() em vez de st.experimental_rerun()
    except Exception as e:
        st.error(f"Erro ao gerar conteúdo com IA: {e}")

def exibir_formulario_qualificacao_autor():
    st.subheader("1. Qualificação do Autor")
    with st.form("form_autor"):
        st.session_state.form_data["qualificacao_autor"] = st.text_area(
            "Qualificação Completa do Autor", value=st.session_state.form_data.get("qualificacao_autor", ""),
            height=150, key="autor_q_text_area",
            help="Ex: Nome completo, nacionalidade, estado civil, profissão, RG, CPF, endereço com CEP, e-mail."
        )
        col1, col2 = st.columns([1,5])
        with col1: submetido = st.form_submit_button("Próximo ➡")
        with col2:
            if st.form_submit_button("Autopreencher com IA (Dados Fictícios)"):
                prompt_str = "Gere uma qualificação completa fictícia para um autor de uma ação judicial (nome completo, nacionalidade, estado civil, profissão, RG, CPF, endereço completo com CEP e e-mail)."
                gerar_conteudo_com_ia(prompt_str, {}, "Qualificação do Autor", "qualificacao_autor")
        
        if st.session_state.ia_generated_content_flags.get("qualificacao_autor"):
            st.caption("📝 Conteúdo preenchido por IA. Revise e ajuste.")
        
        if submetido:
            if st.session_state.form_data.get("qualificacao_autor","").strip(): # Verificar se não está vazio ou só espaços
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Preencha a qualificação do autor.")

def exibir_formulario_qualificacao_reu():
    st.subheader("2. Qualificação do Réu")
    with st.form("form_reu"):
        st.session_state.form_data["qualificacao_reu"] = st.text_area(
            "Qualificação Completa do Réu", value=st.session_state.form_data.get("qualificacao_reu", ""),
            height=150, key="reu_q_text_area",
            help="Ex: Nome/Razão Social, CPF/CNPJ, endereço com CEP, e-mail (se pessoa física ou jurídica)."
        )
        col1, col2, col3 = st.columns([1,1,4]) # Ajustar proporção se necessário
        with col1:
            if st.form_submit_button("⬅ Voltar"):
                st.session_state.current_form_step_index -=1
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo ➡")
        with col3:
            if st.form_submit_button("Autopreencher com IA (Dados Fictícios)"):
                prompt_str = "Gere uma qualificação completa fictícia para um réu (pessoa física OU jurídica) em uma ação judicial (nome/razão social, CPF/CNPJ, endereço com CEP, e-mail)."
                gerar_conteudo_com_ia(prompt_str, {}, "Qualificação do Réu", "qualificacao_reu")

        if st.session_state.ia_generated_content_flags.get("qualificacao_reu"):
            st.caption("📝 Conteúdo preenchido por IA. Revise e ajuste.")
        
        if submetido:
            if st.session_state.form_data.get("qualificacao_reu","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Preencha a qualificação do réu.")

def exibir_formulario_natureza_acao():
    st.subheader("3. Natureza da Ação")
    with st.form("form_natureza"):
        st.session_state.form_data["natureza_acao"] = st.text_input(
            "Natureza da Ação",
            value=st.session_state.form_data.get("natureza_acao", ""), key="natureza_acao_text_input",
            help="Ex: Ação de Cobrança, Ação de Indenização por Danos Morais, Ação Revisional de Contrato."
        )
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            if st.form_submit_button("⬅ Voltar"):
                st.session_state.current_form_step_index -=1
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo ➡")
        with col3:
            if st.form_submit_button("Sugerir com IA (com base nos fatos, se preenchidos)"):
                fatos_previos = st.session_state.form_data.get("fatos","")[:300] # Pega um trecho dos fatos
                prompt_str = ("Com base nos seguintes fatos (se informados): '{fatos_previos}', sugira um nome conciso e técnico para a natureza de uma ação judicial. "
                              "Se os fatos não forem claros, sugira um exemplo comum como 'Ação de Indenização'. Exemplos de nomes: Ação de Cobrança, Ação Declaratória de Inexistência de Débito, Ação de Indenização por Danos Morais e Materiais.\nNatureza Sugerida:")
                gerar_conteudo_com_ia(prompt_str, {"fatos_previos": fatos_previos or "Não informado"}, "Natureza da Ação", "natureza_acao")
        
        if st.session_state.ia_generated_content_flags.get("natureza_acao"):
            st.caption("📝 Conteúdo sugerido por IA. Revise.")

        if submetido:
            if st.session_state.form_data.get("natureza_acao","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Defina a natureza da ação.")

def exibir_formulario_fatos():
    st.subheader("4. Breve Descrição dos Fatos")
    with st.form("form_fatos"):
        st.session_state.form_data["fatos"] = st.text_area(
            "Descreva os Fatos de forma clara e cronológica", value=st.session_state.form_data.get("fatos", ""),
            height=300, key="fatos_text_area",
            help="Relate os acontecimentos que deram origem à disputa, incluindo datas (mesmo que aproximadas), locais e pessoas envolvidas."
        )
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            if st.form_submit_button("⬅ Voltar"):
                st.session_state.current_form_step_index -=1
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo ➡")
        with col3:
            if st.form_submit_button("Gerar Fatos com IA (baseado na natureza da ação)"):
                natureza_acao_informada = st.session_state.form_data.get("natureza_acao","Ação não especificada")
                prompt_str = ("Para uma Ação de '{natureza_acao_informada}', elabore uma narrativa de fatos (2-4 parágrafos) que levaram à disputa. "
                              "Inclua elementos essenciais, datas aproximadas fictícias (ex: 'em meados de janeiro de 2023'), e o problema central. Use 'o Autor' e 'o Réu' para se referir às partes.\nDescrição dos Fatos:")
                gerar_conteudo_com_ia(prompt_str, {"natureza_acao_informada": natureza_acao_informada}, "Descrição dos Fatos", "fatos")

        if st.session_state.ia_generated_content_flags.get("fatos"):
            st.caption("📝 Conteúdo gerado por IA. Revise e detalhe conforme o caso real.")

        if submetido:
            if st.session_state.form_data.get("fatos","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Descreva os fatos.")

def exibir_formulario_direito():
    st.subheader("5. Fundamentação Jurídica (Do Direito)")
    with st.form("form_direito"):
        st.session_state.form_data["fundamentacao_juridica"] = st.text_area(
            "Insira a fundamentação jurídica aplicável ao caso", value=st.session_state.form_data.get("fundamentacao_juridica", ""),
            height=300, key="direito_text_area",
            help="Cite os artigos de lei, súmulas, jurisprudência e princípios jurídicos que amparam a sua pretensão, explicando a conexão com os fatos."
        )
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            if st.form_submit_button("⬅ Voltar"):
                st.session_state.current_form_step_index -=1
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo ➡")
        with col3:
            if st.form_submit_button("Sugerir Fundamentação com IA (baseado nos fatos e natureza)"):
                fatos_informados = st.session_state.form_data.get("fatos","Fatos não informados.")
                natureza_acao_informada = st.session_state.form_data.get("natureza_acao","Natureza da ação não informada.")
                prompt_str = ("Analise os Fatos: \n{fatos_informados}\n\nE a Natureza da Ação: '{natureza_acao_informada}'.\n"
                              "Com base nisso, elabore uma seção 'DO DIREITO' para uma petição inicial. "
                              "Cite artigos de lei relevantes (ex: Código Civil, CDC, Constituição Federal), e explique brevemente como se aplicam aos fatos para justificar os pedidos que seriam feitos. "
                              "Estruture em parágrafos.\nFundamentação Jurídica Sugerida:")
                gerar_conteudo_com_ia(prompt_str, {"fatos_informados": fatos_informados, "natureza_acao_informada": natureza_acao_informada}, "Fundamentação Jurídica", "fundamentacao_juridica")
        
        if st.session_state.ia_generated_content_flags.get("fundamentacao_juridica"):
            st.caption("📝 Conteúdo sugerido por IA. Revise, valide e complemente com referências específicas.")

        if submetido:
            if st.session_state.form_data.get("fundamentacao_juridica","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Insira a fundamentação jurídica.")

def exibir_formulario_pedidos():
    st.subheader("6. Pedidos")
    with st.form("form_pedidos"):
        st.session_state.form_data["pedidos"] = st.text_area(
            "Insira os pedidos da ação de forma clara e objetiva", value=st.session_state.form_data.get("pedidos", ""),
            height=300, key="pedidos_text_area",
            help="Liste os requerimentos finais ao juiz. Ex: citação do réu, procedência da ação para condenar o réu a..., condenação em custas e honorários. Use alíneas (a, b, c...)."
        )
        col1, col2, col3 = st.columns([1,1,4])
        with col1:
            if st.form_submit_button("⬅ Voltar"):
                st.session_state.current_form_step_index -=1
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo para Revisão ➡")
        with col3:
            if st.form_submit_button("Sugerir Pedidos com IA (baseado no caso)"):
                natureza_acao_informada = st.session_state.form_data.get("natureza_acao","")
                fatos_informados_trecho = st.session_state.form_data.get("fatos","")[:300] # Trecho para contexto
                direito_informado_trecho = st.session_state.form_data.get("fundamentacao_juridica","")[:300] # Trecho para contexto
                prompt_str = ("Com base na Natureza da Ação ('{natureza_acao_informada}'), um resumo dos Fatos ('{fatos_informados_trecho}...') e um resumo do Direito ('{direito_informado_trecho}...'), "
                              "elabore uma lista de pedidos típicos para uma petição inicial. Inclua pedidos como: citação do réu, procedência do pedido principal (seja específico se possível, ex: 'condenar o réu ao pagamento de X'), "
                              "condenação em custas processuais e honorários advocatícios. Formate os pedidos usando alíneas (a), (b), (c), etc.\nPedidos Sugeridos:")
                gerar_conteudo_com_ia(prompt_str, {
                    "natureza_acao_informada": natureza_acao_informada,
                    "fatos_informados_trecho": fatos_informados_trecho,
                    "direito_informado_trecho": direito_informado_trecho
                }, "Pedidos", "pedidos")

        if st.session_state.ia_generated_content_flags.get("pedidos"):
            st.caption("📝 Conteúdo sugerido por IA. Revise e ajuste conforme a especificidade do caso.")

        if submetido:
            if st.session_state.form_data.get("pedidos","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Insira os pedidos.")

def exibir_revisao_e_iniciar_simulacao():
    st.subheader("7. Revisar Dados e Iniciar Simulação")
    form_data_local = st.session_state.form_data
    st.info(f"**ID do Processo (Gerado):** `{form_data_local.get('id_processo', 'N/A')}`")

    # Usar st.text_area com disabled=True para exibir os dados de forma consistente
    # Adicionar uma altura padrão menor para os expanders fechados e maior para os abertos se desejado
    
    with st.expander("Qualificação do Autor", expanded=False):
        st.text_area("Revisão - Autor", value=form_data_local.get("qualificacao_autor", "Não preenchido"), height=100, disabled=True, key="rev_autor_area")
    with st.expander("Qualificação do Réu", expanded=False):
        st.text_area("Revisão - Réu", value=form_data_local.get("qualificacao_reu", "Não preenchido"), height=100, disabled=True, key="rev_reu_area")
    with st.expander("Natureza da Ação", expanded=False):
        st.text_input("Revisão - Natureza", value=form_data_local.get("natureza_acao", "Não preenchido"), disabled=True, key="rev_nat_input") # text_input aqui é ok
    with st.expander("Fatos", expanded=True): # Fatos podem ser mais longos, expandir por padrão
        st.text_area("Revisão - Fatos", value=form_data_local.get("fatos", "Não preenchido"), height=200, disabled=True, key="rev_fatos_area")
    with st.expander("Fundamentação Jurídica", expanded=False):
        st.text_area("Revisão - Direito", value=form_data_local.get("fundamentacao_juridica", "Não preenchido"), height=200, disabled=True, key="rev_dir_area")
    with st.expander("Pedidos", expanded=False):
        st.text_area("Revisão - Pedidos", value=form_data_local.get("pedidos", "Não preenchido"), height=200, disabled=True, key="rev_ped_area")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Voltar para Editar Pedidos", use_container_width=True):
            # Navega para o penúltimo passo (índice de "pedidos")
            st.session_state.current_form_step_index = FORM_STEPS.index("pedidos")
            st.rerun()
    with col2:
        # Verifica se todos os campos essenciais foram preenchidos antes de habilitar o botão de simulação
        campos_obrigatorios = ["qualificacao_autor", "qualificacao_reu", "natureza_acao", "fatos", "fundamentacao_juridica", "pedidos"]
        todos_preenchidos = all(form_data_local.get(campo, "").strip() for campo in campos_obrigatorios)
        
        if st.button("🚀 Iniciar Simulação com estes Dados", type="primary", disabled=not todos_preenchidos, use_container_width=True):
            st.session_state.simulation_running = True
            # Limpar resultados de simulações anteriores para o ID de processo atual, se for um novo.
            # O ID do processo já é atualizado em inicializar_estado_formulario se for uma "Nova Simulação".
            # Se o usuário apenas navegou e voltou, o ID é o mesmo.
            current_pid = form_data_local.get('id_processo')
            if current_pid in st.session_state.simulation_results:
                 del st.session_state.simulation_results[current_pid] # Garante que rodará novamente para este ID se clicado

            st.rerun()
        elif not todos_preenchidos:
            st.warning("Alguns campos obrigatórios (Autor, Réu, Natureza, Fatos, Direito, Pedidos) parecem não estar preenchidos. Por favor, revise e complete-os antes de iniciar a simulação.")


def rodar_simulacao_principal(dados_coletados: dict):
    st.markdown(f"--- INICIANDO SIMULAÇÃO PARA O CASO: **{dados_coletados.get('id_processo','N/A')}** ---")
    
    # Validação defensiva
    if not dados_coletados or not dados_coletados.get('id_processo'):
        st.error("Erro: Dados do caso incompletos para iniciar a simulação.")
        st.session_state.simulation_running = False
        if st.button("Retornar ao formulário"):
            st.rerun()
        return

    conteudo_processo_texto = f"""
ID do Processo: {dados_coletados.get('id_processo')}
Qualificação do Autor:
{dados_coletados.get('qualificacao_autor')}

Qualificação do Réu:
{dados_coletados.get('qualificacao_reu')}

Natureza da Ação: {dados_coletados.get('natureza_acao')}

Fatos:
{dados_coletados.get('fatos')}

Fundamentação Jurídica:
{dados_coletados.get('fundamentacao_juridica')}

Pedidos:
{dados_coletados.get('pedidos')}
    """
    documento_do_caso_atual = Document(
        page_content=conteudo_processo_texto,
        metadata={
            "source_type": "processo_formulario_streamlit", 
            "file_name": f"{dados_coletados.get('id_processo')}_formulario.txt", 
            "process_id": dados_coletados.get('id_processo')
        }
    )
    
    retriever_do_caso = None
    placeholder_rag = st.empty() # Usar placeholder para mensagens de status do RAG
    with placeholder_rag.status("⚙️ Inicializando sistema RAG com dados do formulário...", expanded=True):
        st.write("Carregando modelos e criando índice vetorial...")
        try:
            retriever_do_caso = criar_ou_carregar_retriever(
                dados_coletados.get('id_processo'), 
                documento_caso_atual=documento_do_caso_atual, 
                recriar_indice=True # Sempre recriar para garantir que os dados do formulário atual sejam usados
            )
            if retriever_do_caso:
                st.write("✅ Retriever RAG pronto!")
            else:
                st.write("⚠️ Falha ao inicializar o retriever RAG.")
        except Exception as e_rag:
            st.error(f"Erro crítico na inicialização do RAG: {e_rag}")
            retriever_do_caso = None # Garante que está None em caso de falha

    if not retriever_do_caso:
        placeholder_rag.empty() # Limpa o status
        st.error("Falha crítica ao criar o retriever com os dados do formulário. A simulação não pode continuar.")
        st.session_state.simulation_running = False
        if st.button("Tentar Novamente (Recarregar Formulário)"):
             st.session_state.current_form_step_index = FORM_STEPS.index("revisar_e_simular") # Volta para revisão
             st.rerun()
        return

    placeholder_rag.success("🚀 Sistema RAG inicializado e pronto!")
    time.sleep(1.5) # Pequena pausa para o usuário ver a mensagem de sucesso
    placeholder_rag.empty()


    estado_inicial = EstadoProcessual(
        id_processo=dados_coletados.get('id_processo'),
        retriever=retriever_do_caso,
        nome_do_ultimo_no_executado=None, etapa_concluida_pelo_ultimo_no=None,
        proximo_ator_sugerido_pelo_ultimo_no=ADVOGADO_AUTOR, # O grafo define o ponto de entrada, mas aqui é uma sugestão inicial
        documento_gerado_na_etapa_recente=None, historico_completo=[],
        pontos_controvertidos_saneamento=None, manifestacao_autor_sem_provas=False,
        manifestacao_reu_sem_provas=False, etapa_a_ser_executada_neste_turno="", # Será definido pelo nó
        dados_formulario_entrada=dados_coletados 
    )

    st.subheader("⏳ Acompanhamento da Simulação:")
    if 'expand_all_steps' not in st.session_state: st.session_state.expand_all_steps = True
    
    # Usar colunas para o botão não ocupar a largura toda se não desejado
    # col_toggle, _ = st.columns([1,3])
    # with col_toggle:
    if st.checkbox("Expandir todos os passos da simulação", value=st.session_state.expand_all_steps, key="cb_expand_all_sim_steps", on_change=lambda: setattr(st.session_state, 'expand_all_steps', st.session_state.cb_expand_all_sim_steps)):
        pass # Ação já feita pelo on_change
        
    progress_bar_placeholder = st.empty()
    steps_container = st.container()
    max_passos_simulacao = 10 
    passo_atual_simulacao = 0
    estado_final_simulacao = None

    try:
        for s_idx, s_event in enumerate(app.stream(input=estado_inicial, config={"recursion_limit": max_passos_simulacao})):
            passo_atual_simulacao += 1
            if not s_event or not isinstance(s_event, dict) or not list(s_event.keys()):
                # st.write(f"Debug: Evento {s_idx} vazio ou formato inesperado: {s_event}")
                continue

            nome_do_no_executado = list(s_event.keys())[0]
            # st.write(f"Debug: Evento {s_idx} - Nó: {nome_do_no_executado}, Conteúdo: {s_event[nome_do_no_executado]}")


            if nome_do_no_executado == "__end__":
                estado_final_simulacao = list(s_event.values())[0] # O estado final está no valor
                nome_do_no_executado = END # Alinha com a lógica de exibição
            else:
                # Assume que o valor associado à chave do nó é o estado atualizado
                estado_parcial_apos_no = s_event[nome_do_no_executado]
                if not isinstance(estado_parcial_apos_no, dict): # Checagem extra
                    # st.write(f"Debug: Conteúdo inesperado para nó {nome_do_no_executado}: {estado_parcial_apos_no}")
                    # Se o estado não for um dicionário, pode ser um erro ou formato diferente.
                    # Tentar pegar o último estado conhecido se possível, ou registrar erro.
                    if estado_final_simulacao: # Usa o último estado válido se este for problemático
                         pass # Mantém o estado_final_simulacao anterior
                    else: # Se for o primeiro e já deu problema
                         st.error(f"Formato de estado inesperado no nó {nome_do_no_executado}. A simulação pode estar inconsistente.")
                         break
                else:
                    estado_final_simulacao = estado_parcial_apos_no


            etapa_concluida_log = estado_final_simulacao.get('etapa_concluida_pelo_ultimo_no', 'N/A')
            doc_gerado_completo = str(estado_final_simulacao.get('documento_gerado_na_etapa_recente', ''))
            prox_ator_sug_log = estado_final_simulacao.get('proximo_ator_sugerido_pelo_ultimo_no', 'N/A')

            expander_title = f"Passo {passo_atual_simulacao}: Nó '{nome_do_no_executado}' concluiu etapa '{etapa_concluida_log}'"
            if nome_do_no_executado == END: expander_title = f"🏁 Passo {passo_atual_simulacao}: Fim da Simulação"
            
            with steps_container.expander(expander_title, expanded=st.session_state.get('expand_all_steps', True)):
                st.markdown(f"**Nó Executado:** `{nome_do_no_executado}`")
                st.markdown(f"**Etapa Concluída:** `{etapa_concluida_log}`")
                if etapa_concluida_log not in ["ERRO_FLUXO_IRRECUPERAVEL", "ERRO_ETAPA_NAO_ENCONTRADA"] and doc_gerado_completo:
                    st.text_area("Documento Gerado:", value=doc_gerado_completo, height=200, key=f"doc_step_sim_{passo_atual_simulacao}", disabled=True)
                elif doc_gerado_completo: # Se for erro, mas tiver documento, mostrar como erro
                    st.error(f"Detalhe do Erro/Documento: {doc_gerado_completo}")
                st.markdown(f"**Próximo Ator Sugerido (pelo nó):** `{prox_ator_sug_log}`")
            
            progress_val = min(1.0, passo_atual_simulacao / (len(mapa_tarefa_no_atual) + 1) ) # Estimativa baseada no número de transições possíveis
            progress_bar_placeholder.progress(progress_val, text=f"Simulando... {int(progress_val*100)}%")

            if nome_do_no_executado == END or prox_ator_sug_log == ETAPA_FIM_PROCESSO:
                steps_container.success("🎉 Fluxo da simulação concluído!")
                break 
            if etapa_concluida_log == "ERRO_FLUXO_IRRECUPERAVEL" or etapa_concluida_log == "ERRO_ETAPA_NAO_ENCONTRADA":
                steps_container.error(f"❌ Erro crítico no fluxo em '{nome_do_no_executado}'. Simulação interrompida.")
                break
            if passo_atual_simulacao >= max_passos_simulacao:
                steps_container.warning(f"Simulação atingiu o limite máximo de {max_passos_simulacao} passos e foi interrompida.")
                break
        
        progress_bar_placeholder.progress(1.0, text="Simulação Concluída!")
        if estado_final_simulacao:
            st.session_state.simulation_results[dados_coletados.get('id_processo')] = estado_final_simulacao
            exibir_resultados_simulacao(estado_final_simulacao)
        else:
            st.warning("A simulação terminou, mas não foi possível obter o estado final completo.")

    except Exception as e_sim:
        st.error(f"ERRO INESPERADO DURANTE A EXECUÇÃO DA SIMULAÇÃO: {e_sim}")
        import traceback
        st.text_area("Stack Trace do Erro:", traceback.format_exc(), height=300)
    finally:
        progress_bar_placeholder.empty()
        # Considerar se simulation_running deve ser resetado aqui ou por um botão de "Nova Simulação"
        # st.session_state.simulation_running = False # Descomentar se quiser permitir nova simulação automaticamente após esta terminar/falhar

def exibir_resultados_simulacao(estado_final_simulacao: dict):
    st.markdown("---")
    st.subheader("📊 Resultados da Simulação")
    
    # O placeholder para visualização de documento individual já é gerenciado no escopo da função
    doc_completo_placeholder_sim = st.empty()

    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        st.markdown("#### Linha do Tempo Interativa do Processo")
        historico = estado_final_simulacao["historico_completo"]
        icon_map = { 
            ADVOGADO_AUTOR: "🙋‍♂️", JUIZ: "⚖️", ADVOGADO_REU: "🙋‍♀️", 
            ETAPA_PETICAO_INICIAL: "📄", ETAPA_DESPACHO_RECEBENDO_INICIAL: "➡️", 
            ETAPA_CONTESTACAO: "🛡️", ETAPA_DECISAO_SANEAMENTO: "🛠️", 
            ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR: "🗣️", ETAPA_MANIFESTACAO_SEM_PROVAS_REU: "🗣️", 
            ETAPA_SENTENCA: "🏁", "DEFAULT_ACTOR": "👤", "DEFAULT_ETAPA": "📑",
            "ERRO_FLUXO_IRRECUPERAVEL": "❌", "ERRO_ETAPA_NAO_ENCONTRADA": "❓"
        }
        num_etapas = len(historico)
        if num_etapas > 0 :
            cols = st.columns(min(num_etapas, 7)) # Limitar colunas para melhor visualização
            for i, item_hist in enumerate(historico):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                
                ator_icon = icon_map.get(ator_hist, icon_map["DEFAULT_ACTOR"])
                etapa_icon = icon_map.get(etapa_hist, icon_map["DEFAULT_ETAPA"])
                
                # Adicionar cor de fundo baseada em erro
                cor_fundo = "rgba(255, 0, 0, 0.1)" if "ERRO" in etapa_hist else "rgba(0, 0, 0, 0.03)"

                with cols[i % len(cols)]:
                    container_style = f"""
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        padding: 10px; 
                        text-align: center; 
                        background-color: {cor_fundo};
                        height: 120px; 
                        display: flex; 
                        flex-direction: column; 
                        justify-content: space-around;
                        margin-bottom: 5px; /* Espaço entre os cards da linha do tempo */
                    """
                    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 28px;'>{ator_icon}{etapa_icon}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px;'><b>{ator_hist}</b><br>{etapa_hist[:30]}{'...' if len(etapa_hist)>30 else ''}</div>", unsafe_allow_html=True)
                    if st.button(f"Ver Doc {i+1}", key=f"btn_timeline_sim_{i}", help=f"Visualizar: {etapa_hist}", use_container_width=True):
                        st.session_state.doc_visualizado = doc_completo_hist
                        st.session_state.doc_visualizado_titulo = f"Documento da Linha do Tempo (Passo {i+1}): {ator_hist} - {etapa_hist}"
                        st.rerun() 
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True) # Separador visual
    else: st.warning("Nenhum histórico completo para exibir na linha do tempo.")

    if st.session_state.get('doc_visualizado') is not None: # Usar get para segurança
        with doc_completo_placeholder_sim.container():
            st.subheader(st.session_state.get('doc_visualizado_titulo', "Visualização de Documento"))
            st.text_area("Conteúdo do Documento:", st.session_state.doc_visualizado, height=350, key="doc_view_sim_area_main_results", disabled=True)
            if st.button("Fechar Visualização do Documento", key="close_doc_view_sim_btn_main_results", type="primary"):
                st.session_state.doc_visualizado = None
                st.session_state.doc_visualizado_titulo = ""
                st.rerun()
    
    st.markdown("#### Histórico Detalhado (Conteúdo Completo das Etapas)")
    if 'expand_all_history' not in st.session_state: st.session_state.expand_all_history = False
    
    if st.checkbox("Expandir todo o histórico detalhado", value=st.session_state.expand_all_history, key="cb_expand_all_hist_detail_sim", on_change=lambda: setattr(st.session_state, 'expand_all_history', st.session_state.cb_expand_all_hist_detail_sim)):
        pass

    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
            ator_hist = item_hist.get('ator', 'N/A')
            etapa_hist = item_hist.get('etapa', 'N/A')
            doc_completo_hist = str(item_hist.get('documento', 'N/A'))
            with st.expander(f"Detalhe {i+1}: Ator '{ator_hist}' | Etapa '{etapa_hist}'", expanded=st.session_state.expand_all_history):
                st.text_area(f"Documento Completo (Passo {i+1}):", value=doc_completo_hist, height=200, key=f"doc_hist_detail_sim_{i}", disabled=True)
    st.markdown("--- FIM DA EXIBIÇÃO DOS RESULTADOS ---")

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="IA-Mestra: Simulação Jurídica Avançada", page_icon="⚖️")
    st.title("IA-Mestra: Simulação Jurídica Avançada ⚖️")
    st.caption("Uma ferramenta para simular o fluxo processual com assistência de IA, utilizando LangGraph e RAG.")

    if not GOOGLE_API_KEY:
        st.error("🔴 ERRO CRÍTICO: A variável de ambiente GOOGLE_API_KEY não foi definida. A aplicação não pode funcionar sem ela.")
        st.stop() # Impede a execução do restante do app se a chave estiver ausente
    
    # Inicializa o estado da sessão (deve ser chamado no início)
    inicializar_estado_formulario() 
    
    st.sidebar.title("Painel de Controle 🕹️")
    if st.sidebar.button("🔄 Nova Simulação (Limpar Formulário)", key="nova_sim_btn_sidebar", type="primary", use_container_width=True):
        # Reseta o índice do formulário para o início
        st.session_state.current_form_step_index = 0
        # Cria um novo ID de processo para a nova simulação
        novo_id_processo = f"caso_sim_{int(time.time())}"
        # Limpa os dados do formulário, mas já define o novo ID
        st.session_state.form_data = {
            "id_processo": novo_id_processo, 
            "qualificacao_autor": "", "qualificacao_reu": "", "natureza_acao": "",
            "fatos": "", "fundamentacao_juridica": "", "pedidos": ""
        }
        # Reseta as flags de conteúdo gerado por IA
        st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
        # Não limpa 'simulation_results' aqui, pois pode ser útil ver o histórico de execuções anteriores.
        # Em vez disso, a lógica em 'rodar_simulacao_principal' ou 'exibir_revisao_e_iniciar_simulacao'
        # deve garantir que uma nova simulação para um ID existente seja forçada ou os resultados sejam limpos seletivamente.
        # Se o ID é sempre novo, não há conflito.
        
        st.session_state.simulation_running = False # Garante que voltará para a tela de formulário
        if 'doc_visualizado' in st.session_state: st.session_state.doc_visualizado = None 
        if 'doc_visualizado_titulo' in st.session_state: st.session_state.doc_visualizado_titulo = ""
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ Preencha os formulários sequenciais para definir os parâmetros do caso. "
        "A IA pode auxiliar no preenchimento com dados fictícios ou sugestões jurídicas contextuais."
    )
    st.sidebar.markdown("---")
    # Link para o LangSmith (se configurado)
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and os.getenv("LANGCHAIN_PROJECT"):
        project_name = os.getenv("LANGCHAIN_PROJECT")
        st.sidebar.markdown(f"🔍 [Monitorar no LangSmith](https://smith.langchain.com/o/{os.getenv('LANGCHAIN_TENANT_ID', 'default')}/projects/p/{project_name})", unsafe_allow_html=True)


    # --- Lógica Principal de Exibição da UI ---
    if st.session_state.get('simulation_running', False):
        id_processo_atual = st.session_state.form_data.get('id_processo')
        
        # Se a simulação está marcada como rodando, mas não temos resultados para o ID atual, rodar.
        # Ou, se o usuário explicitamente iniciou (botão em exibir_revisao_e_iniciar_simulacao seta simulation_running = True)
        if id_processo_atual and id_processo_atual not in st.session_state.get('simulation_results', {}):
            rodar_simulacao_principal(st.session_state.form_data)
        # Se temos resultados para o ID atual, exibir.
        elif id_processo_atual and st.session_state.get('simulation_results', {}).get(id_processo_atual):
            st.info(f"📖 Exibindo resultados da simulação para o ID: {id_processo_atual}")
            exibir_resultados_simulacao(st.session_state.simulation_results[id_processo_atual])
            if st.button("Iniciar uma Nova Simulação (Limpar Formulário)"): # Botão para facilitar o reinício após ver resultados
                # Código similar ao botão da sidebar para resetar
                st.session_state.current_form_step_index = 0
                novo_id_processo = f"caso_sim_{int(time.time())}"
                st.session_state.form_data = { "id_processo": novo_id_processo, "qualificacao_autor": "", "qualificacao_reu": "", "natureza_acao": "", "fatos": "", "fundamentacao_juridica": "", "pedidos": "" }
                st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
                st.session_state.simulation_running = False
                if 'doc_visualizado' in st.session_state: st.session_state.doc_visualizado = None
                st.rerun()
        else: 
            st.warning("⚠️ A simulação anterior não produziu resultados ou houve um problema de estado. Por favor, inicie uma nova simulação.")
            st.session_state.simulation_running = False 
            if st.button("Ir para o início do formulário"):
                 st.session_state.current_form_step_index = 0
                 st.rerun()

    else: # Exibir formulários
        passo_atual_idx = st.session_state.current_form_step_index
        
        # --- Indicador de Progresso do Formulário ---
        if 0 <= passo_atual_idx < len(FORM_STEPS): # Checagem de segurança para o índice
            nome_passo_atual = FORM_STEPS[passo_atual_idx]
            if nome_passo_atual != "revisar_e_simular":
                progresso_percentual = (passo_atual_idx + 1) / (len(FORM_STEPS) -1) # -1 porque revisar não é um passo de preenchimento ativo
                st.progress(progresso_percentual)
                titulo_passo_formatado = nome_passo_atual.replace('_', ' ').title()
                st.markdown(f"#### Etapa de Preenchimento: **{titulo_passo_formatado}** (Passo {passo_atual_idx + 1} de {len(FORM_STEPS)-1})")
            else: # Etapa de Revisão
                 st.markdown(f"#### Etapa Final: **Revisar Dados e Iniciar Simulação** (Passo {len(FORM_STEPS)} de {len(FORM_STEPS)})")
            st.markdown("---")
        # --- Fim do Indicador de Progresso ---

            current_step_key = FORM_STEPS[passo_atual_idx] # Reconfirmar, já que usamos nome_passo_atual
            if current_step_key == "autor": exibir_formulario_qualificacao_autor()
            elif current_step_key == "reu": exibir_formulario_qualificacao_reu()
            elif current_step_key == "natureza_acao": exibir_formulario_natureza_acao()
            elif current_step_key == "fatos": exibir_formulario_fatos()
            elif current_step_key == "direito": exibir_formulario_direito()
            elif current_step_key == "pedidos": exibir_formulario_pedidos()
            elif current_step_key == "revisar_e_simular": exibir_revisao_e_iniciar_simulacao()
            else: # --- Else para Robustez ---
                st.error(f"🔴 ERRO INTERNO: Etapa do formulário desconhecida ou não tratada: '{current_step_key}'.")
                st.warning("Por favor, tente reiniciar a simulação a partir do menu lateral.")
        else: # Índice fora do esperado
            st.error("🔴 ERRO INTERNO: Índice da etapa do formulário inválido. Tentando reiniciar...")
            st.session_state.current_form_step_index = 0 # Reseta para o início
            st.rerun()