# mvp_simulacao_juridica_avancado.py

import os
import shutil # Para limpar a pasta FAISS se necessário
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


import streamlit as st

# --- 0. Carregamento de Variáveis de Ambiente ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
    exit()

# --- 1. Constantes e Configurações Globais ---
DATA_PATH = "simulacao_juridica_data"
PATH_PROCESSO_EM_SI = os.path.join(DATA_PATH, "processo_em_si")
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

# --- 2. Utilitários RAG ---

def carregar_documentos_docx(caminho_pasta_ou_arquivo: str, tipo_fonte: str, id_processo_especifico: Union[str, None] = None) -> List[Any]:
    documentos = []
    if not os.path.exists(caminho_pasta_ou_arquivo):
        print(f"AVISO RAG: Caminho não encontrado: {caminho_pasta_ou_arquivo}")
        return documentos

    # Carregar um arquivo específico de processo (caso atual)
    if tipo_fonte == "processo_atual" and id_processo_especifico and os.path.isfile(caminho_pasta_ou_arquivo):
        if caminho_pasta_ou_arquivo.endswith(".docx"):
            try:
                loader = Docx2txtLoader(caminho_pasta_ou_arquivo)
                docs_carregados = loader.load()
                for doc in docs_carregados:
                    doc.metadata = {"source_type": tipo_fonte, "file_name": os.path.basename(caminho_pasta_ou_arquivo), "process_id": id_processo_especifico}
                documentos.extend(docs_carregados)
                print(f"[RAG] Carregado processo '{os.path.basename(caminho_pasta_ou_arquivo)}' para ID '{id_processo_especifico}'.")
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
                show_progress=False, # Reduzir verbosidade
                use_multithreading=True,
                silent_errors=True
            )
            docs_carregados = loader.load()
            for doc in docs_carregados:
                # Garante que 'source' (nome do arquivo) está em 'file_name' para consistência
                doc.metadata["file_name"] = os.path.basename(doc.metadata.get("source", "unknown.docx"))
                doc.metadata["source_type"] = tipo_fonte
            documentos.extend(docs_carregados)
            print(f"[RAG] Carregados {len(docs_carregados)} documentos da pasta de modelos '{os.path.basename(caminho_pasta_ou_arquivo)}'.")
        except Exception as e:
            print(f"Erro ao carregar modelos de {caminho_pasta_ou_arquivo}: {e}")
            
    return documentos

def criar_ou_carregar_retriever(id_processo: str, arquivo_processo_especifico: str):
    """
    Cria um novo índice FAISS ou carrega um existente.
    O índice incluirá os modelos (comuns a todos os processos) e o arquivo específico do processo atual.
    """
    # Limpar índice antigo para garantir que estamos sempre com os dados mais recentes (opcional)
    # if os.path.exists(FAISS_INDEX_PATH):
    #     print(f"[RAG] Removendo índice FAISS antigo de '{FAISS_INDEX_PATH}'.")
    #     shutil.rmtree(FAISS_INDEX_PATH)

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            print(f"[RAG] Tentando carregar índice FAISS existente de '{FAISS_INDEX_PATH}'.")
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings_model, allow_dangerous_deserialization=True)
            print("[RAG] Índice FAISS carregado com sucesso.")
            # TODO: Adicionar lógica para verificar se o arquivo_processo_especifico já está no índice
            # ou se precisamos adicionar/atualizar apenas esse documento.
            # Por simplicidade neste MVP, se o índice existe, usamos como está ou recriamos.
            # Para uma solução robusta, seria necessário um versionamento ou uma forma de atualizar o índice.
            # Vamos recriar para garantir que o processo específico está lá.
            print("[RAG] Recriando índice para garantir inclusão do processo específico e modelos atualizados.")
            if os.path.exists(FAISS_INDEX_PATH): shutil.rmtree(FAISS_INDEX_PATH) # Força recriação
            return criar_ou_carregar_retriever(id_processo, arquivo_processo_especifico) # Chama recursivamente para recriar

        except Exception as e:
            print(f"[RAG] Erro ao carregar índice FAISS: {e}. Recriando...")
            if os.path.exists(FAISS_INDEX_PATH): shutil.rmtree(FAISS_INDEX_PATH) # Limpa se deu erro ao carregar


    print("[RAG] Criando novo índice FAISS...")
    todos_documentos = []
    # Carrega o arquivo de processo específico
    caminho_completo_processo = os.path.join(PATH_PROCESSO_EM_SI, arquivo_processo_especifico)
    todos_documentos.extend(carregar_documentos_docx(caminho_completo_processo, "processo_atual", id_processo_especifico=id_processo))
    
    # Carrega modelos
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_PETICOES, "modelo_peticao"))
    todos_documentos.extend(carregar_documentos_docx(PATH_MODELOS_JUIZ, "modelo_juiz"))

    if not todos_documentos:
        print("ERRO RAG: Nenhum documento foi carregado para o índice. Verifique as pastas e arquivos .docx.")
        raise ValueError("RAG: Falha ao carregar documentos para o índice.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300) # Chunks maiores para .docx
    docs_divididos = text_splitter.split_documents(todos_documentos)
    
    if not docs_divididos:
        print("ERRO RAG: Nenhum chunk gerado após a divisão dos documentos.")
        raise ValueError("RAG: Falha ao dividir documentos.")
        
    print(f"[RAG] Documentos divididos em {len(docs_divididos)} chunks.")
    
    try:
        print(f"[RAG] Criando e salvando vector store FAISS em '{FAISS_INDEX_PATH}'.")
        vector_store = FAISS.from_documents(docs_divididos, embeddings_model)
        vector_store.save_local(FAISS_INDEX_PATH)
    except Exception as e:
        print(f"Erro fatal ao criar ou salvar FAISS: {e}")
        raise e

    print("[RAG] Retriever real criado e índice salvo com sucesso!")
    return vector_store.as_retriever(search_kwargs={'k': 5, 'fetch_k': 10}) # Pega os 5 mais relevantes de 10 buscados

# --- 3. Inicialização do LLM (Modelo Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.6, convert_system_message_to_human=True)

# --- 4. Definição do Estado Processual (LangGraph) ---
class EstadoProcessual(TypedDict):
    id_processo: str
    retriever: Any # FAISS retriever instance
    
    # Rastreamento do fluxo
    nome_do_ultimo_no_executado: Union[str, None]
    etapa_concluida_pelo_ultimo_no: Union[str, None]
    proximo_ator_sugerido_pelo_ultimo_no: Union[str, None]
    
    # Informação para o nó atual
    etapa_a_ser_executada_neste_turno: str # Definida pelo router para o nó atual

    # Dados gerados e de contexto
    documento_gerado_na_etapa_recente: Union[str, None] # Peça/decisão gerada na etapa mais recente
    historico_completo: List[Dict[str, str]] # Lista de {"etapa": str, "ator": str, "documento": str}
    
    # Campos específicos para certas etapas
    pontos_controvertidos_saneamento: Union[str, None] # Da decisão de saneamento
    manifestacao_autor_sem_provas: bool # Flag
    manifestacao_reu_sem_provas: bool # Flag

# --- 5. Mapa de Fluxo Processual (Rito Ordinário) ---
mapa_tarefa_no_atual: Dict[Tuple[Union[str, None], Union[str, None], str], str] = {
    (None, None, ADVOGADO_AUTOR): ETAPA_PETICAO_INICIAL,
    (ADVOGADO_AUTOR, ETAPA_PETICAO_INICIAL, JUIZ): ETAPA_DESPACHO_RECEBENDO_INICIAL,
    (JUIZ, ETAPA_DESPACHO_RECEBENDO_INICIAL, ADVOGADO_REU): ETAPA_CONTESTACAO,
    (ADVOGADO_REU, ETAPA_CONTESTACAO, JUIZ): ETAPA_DECISAO_SANEAMENTO, # Originalmente ia para Réplica, simplificado para Saneamento
    (JUIZ, ETAPA_DECISAO_SANEAMENTO, ADVOGADO_AUTOR): ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    (ADVOGADO_AUTOR, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR, ADVOGADO_REU): ETAPA_MANIFESTACAO_SEM_PROVAS_REU,
    (ADVOGADO_REU, ETAPA_MANIFESTACAO_SEM_PROVAS_REU, JUIZ): ETAPA_SENTENCA,
}

# --- 6. Função de Roteamento Condicional (Router) ---
def decidir_proximo_no_do_grafo(estado: EstadoProcessual):
    proximo_ator_sugerido = estado.get("proximo_ator_sugerido_pelo_ultimo_no")
    etapa_concluida = estado.get("etapa_concluida_pelo_ultimo_no")

    print(f"[Router] Estado recebido: { {k: v for k, v in estado.items() if k != 'retriever'} }")
    print(f"[Router] Decidindo próximo nó. Última etapa concluída: '{etapa_concluida}'. Próximo ator sugerido: '{proximo_ator_sugerido}'.")

    if proximo_ator_sugerido == ADVOGADO_AUTOR:
        return ADVOGADO_AUTOR
    elif proximo_ator_sugerido == JUIZ:
        return JUIZ
    elif proximo_ator_sugerido == ADVOGADO_REU:
        return ADVOGADO_REU
    elif proximo_ator_sugerido == ETAPA_FIM_PROCESSO or etapa_concluida == ETAPA_SENTENCA:
        print("[Router] Fluxo direcionado para o FIM.")
        return END
    else:
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

# (Definições dos agentes ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU - Versões Completas)

def agente_advogado_autor(estado: EstadoProcessual) -> Dict[str, Any]:
    # print(f"[DEBUG ADVOGADO_AUTOR] Etapa atual do nó: {estado.get('etapa_a_ser_executada_neste_turno')}") # Redundante com helper
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_AUTOR)
    # estado["etapa_a_ser_executada_neste_turno"] = etapa_atual_do_no # O helper já loga isso.

    print(f"\n--- TURNO: {ADVOGADO_AUTOR} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        print(f"ERRO FATAL: {ADVOGADO_AUTOR} não conseguiu determinar sua etapa. Encerrando fluxo.")
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, # Sinaliza fim para o router
            "documento_gerado_na_etapa_recente": "Erro crítico de fluxo.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_AUTOR, "documento": "Erro de fluxo irrecuperável."}],
            "id_processo": estado.get("id_processo", "ID_DESCONHECIDO"), # Preservar estado
            "retriever": estado.get("retriever"),
            "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"),
            "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False),
            "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
            "etapa_a_ser_executada_neste_turno": "ERRO_FLUXO_IRRECUPERAVEL"
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Default, será sobrescrito pela lógica da etapa
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado:
        historico_formatado = "Este é o primeiro ato do processo."

    # --- Lógica de RAG para contexto do caso (comum a várias etapas) ---
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo}"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo} # Filtra pelo ID do processo!
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si."
    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")

    # --- Tratamento específico para cada etapa do Advogado Autor ---
    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        docs_modelo_pi = retriever.get_relevant_documents(query="modelo de petição inicial completa", filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_pi[0].page_content if docs_modelo_pi else "ERRO RAG: Modelo de petição inicial não encontrado."
        
        template_prompt = """
        Você é um Advogado do Autor experiente e está elaborando uma Petição Inicial.
        **Processo ID:** {id_processo}
        **Tarefa:** Elaborar a Petição Inicial completa e fundamentada para o caso.

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}

        **Modelo/Guia de Petição Inicial (use como referência estrutural e de linguagem jurídica, mas adapte ao caso concreto):**
        {modelo_texto_guia}
        
        **Histórico Processual até o momento:**
        {historico_formatado}
        ---
        Com base em todas as informações acima, redija a Petição Inicial. Seja completo, claro e objetivo.
        Atenção aos fatos específicos do caso para argumentação.
        Não inclua saudações genéricas no início ou fim se já estiverem no modelo.
        Petição Inicial:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR:
        decisao_saneamento_recebida = estado.get("documento_gerado_na_etapa_recente", "ERRO: Decisão de Saneamento não encontrada no estado.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não detalhados.")

        docs_modelo_manifest = retriever.get_relevant_documents(query="modelo de petição ou manifestação declarando não ter mais provas a produzir", filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_manifest[0].page_content if docs_modelo_manifest else "ERRO RAG: Modelo de manifestação sem provas não encontrado."

        template_prompt = """
        Você é o Advogado do Autor. O juiz proferiu Decisão de Saneamento e intimou as partes a especificarem provas.
        **Processo ID:** {id_processo}
        **Tarefa:** Elaborar uma petição informando que o Autor não possui mais provas a produzir, requerendo o julgamento do feito no estado em que se encontra, ou protestando por memoriais se for o caso.

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}

        **Decisão de Saneamento proferida pelo Juiz:**
        {decisao_saneamento_recebida}
        (Pontos Controvertidos principais: {pontos_controvertidos})
        
        **Modelo/Guia de Manifestação (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual até o momento:**
        {historico_formatado}
        ---
        Redija a Petição de Manifestação do Autor.
        Manifestação:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "decisao_saneamento_recebida": decisao_saneamento_recebida,
            "pontos_controvertidos": pontos_controvertidos,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_REU # Próximo é o réu se manifestar sobre provas

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_AUTOR} não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não foi gerado pela lógica específica."
        proximo_ator_logico = JUIZ # Ou ETAPA_FIM_PROCESSO em caso de erro irrecuperável

    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_AUTOR, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    return {
        "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"), # Preserva se já existia
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR),
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

def agente_juiz(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, JUIZ)
    # estado["etapa_a_ser_executada_neste_turno"] = etapa_atual_do_no # Helper já loga

    print(f"\n--- TURNO: {JUIZ} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        print(f"ERRO FATAL: {JUIZ} não conseguiu determinar sua etapa. Encerrando fluxo.")
        return {
            "nome_do_ultimo_no_executado": JUIZ,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": "Erro crítico de fluxo para o Juiz.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": JUIZ, "documento": "Erro de fluxo irrecuperável do Juiz."}],
            "id_processo": estado.get("id_processo", "ID_DESCONHECIDO"), # Preservar estado
            "retriever": estado.get("retriever"),
            "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"),
            "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False),
            "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
            "etapa_a_ser_executada_neste_turno": "ERRO_FLUXO_IRRECUPERAVEL"
        }

    documento_gerado = f"Decisão padrão para {JUIZ} na etapa {etapa_atual_do_no} não gerada."
    proximo_ator_logico = ETAPA_FIM_PROCESSO # Default, será sobrescrito
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: 
        historico_formatado = "Histórico processual não disponível ou PI ainda não processada."

    documento_da_parte_para_analise = estado.get("documento_gerado_na_etapa_recente", "Nenhuma petição recente das partes para análise.")
    
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo} para decisão judicial"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo}
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si."
    print(f"[{JUIZ}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")
    
    pontos_controvertidos_definidos = estado.get("pontos_controvertidos_saneamento") 

    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        docs_modelo_despacho = retriever.get_relevant_documents(query="modelo de despacho judicial recebendo petição inicial e citando o réu", filter={"source_type": "modelo_juiz"})
        modelo_texto_guia = docs_modelo_despacho[0].page_content if docs_modelo_despacho else "ERRO RAG: Modelo de despacho inicial não encontrado."

        template_prompt = """
        Você é um Juiz de Direito e acaba de receber uma Petição Inicial.
        **Processo ID:** {id_processo}
        **Tarefa:** Analisar a Petição Inicial e proferir um despacho inicial (ex: recebendo a inicial, determinando citação do réu, ou outras providências preliminares).

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}
        
        **Petição Inicial apresentada pelo Autor:**
        {peticao_inicial}

        **Modelo/Guia de Despacho Inicial (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual até o momento:**
        {historico_formatado}
        ---
        Com base nisso, redija o Despacho Inicial. Seja claro e objetivo, seguindo a praxe jurídica.
        Despacho Inicial:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "peticao_inicial": documento_da_parte_para_analise,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_REU

    elif etapa_atual_do_no == ETAPA_DECISAO_SANEAMENTO:
        docs_modelo_saneamento = retriever.get_relevant_documents(query="modelo de decisão de saneamento e organização do processo", filter={"source_type": "modelo_juiz"})
        modelo_texto_guia = docs_modelo_saneamento[0].page_content if docs_modelo_saneamento else "ERRO RAG: Modelo de decisão de saneamento não encontrado."

        template_prompt = """
        Você é um Juiz de Direito e o processo está na fase de saneamento, após petição inicial e contestação (e eventual réplica, que deve ser inferida do histórico se aplicável).
        **Processo ID:** {id_processo}
        **Tarefa:** Proferir uma Decisão de Saneamento e Organização do Processo. Defina as questões processuais pendentes, as questões de fato sobre as quais recairá a atividade probatória (pontos controvertidos), distribua o ônus da prova e, se for o caso, designe audiência ou determine as provas a serem produzidas.

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}
        
        **Última manifestação relevante das partes (ex: Contestação do Réu ou Réplica do Autor):**
        {ultima_peticao_partes}

        **Modelo/Guia de Decisão de Saneamento (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual Completo (incluindo PI, Contestação, etc.):**
        {historico_formatado}
        ---
        Com base nisso, redija a Decisão de Saneamento.
        Inclua uma seção clara definindo os PONTOS CONTROVERTIDOS. Ex: "PONTOS CONTROVERTIDOS: 1. A existência do contrato; 2. O alegado inadimplemento."
        Decisão de Saneamento:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "ultima_peticao_partes": documento_da_parte_para_analise, # Esta é a contestação do réu
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_AUTOR 
        
        try:
            inicio_pc = documento_gerado.upper().find("PONTOS CONTROVERTIDOS:")
            if inicio_pc != -1:
                fim_pc = documento_gerado.find("\n\n", inicio_pc) 
                if fim_pc == -1: fim_pc = len(documento_gerado)
                pontos_controvertidos_definidos = documento_gerado[inicio_pc + len("PONTOS CONTROVERTIDOS:"):fim_pc].strip()
            else:
                pontos_controvertidos_definidos = "Não foi possível extrair os pontos controvertidos automaticamente. Verificar decisão."
        except Exception:
            pontos_controvertidos_definidos = "Erro ao tentar extrair pontos controvertidos."
        print(f"[{JUIZ}-{etapa_atual_do_no}] Pontos Controvertidos extraídos/definidos: {pontos_controvertidos_definidos}")

    elif etapa_atual_do_no == ETAPA_SENTENCA:
        docs_modelo_sentenca = retriever.get_relevant_documents(query="modelo de sentença judicial cível", filter={"source_type": "modelo_juiz"})
        modelo_texto_guia = docs_modelo_sentenca[0].page_content if docs_modelo_sentenca else "ERRO RAG: Modelo de sentença não encontrado."

        template_prompt = """
        Você é um Juiz de Direito e o processo está concluso para sentença, após as partes indicarem não ter mais provas a produzir.
        **Processo ID:** {id_processo}
        **Tarefa:** Analisar todo o processo (fatos, argumentos das partes, provas constantes nos autos - simuladas pelo histórico) e proferir a Sentença.

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}
        
        **Última manifestação relevante das partes (ex: Réu indicando não ter provas):**
        {ultima_peticao_partes}

        **Pontos Controvertidos definidos no Saneamento:**
        {pontos_controvertidos_saneamento}

        **Modelo/Guia de Sentença (use como referência para estrutura: relatório, fundamentação, dispositivo):**
        {modelo_texto_guia}
        
        **Histórico Processual Completo (incluindo PI, Contestação, Réplica, Saneamento, Manifestações sobre provas):**
        {historico_formatado}
        ---
        Com base em todas as informações e no seu conhecimento jurídico, redija a Sentença completa.
        Sentença:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "ultima_peticao_partes": documento_da_parte_para_analise,
            "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento", "Não definidos anteriormente."),
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ETAPA_FIM_PROCESSO

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {JUIZ} não implementada completamente.")
        documento_gerado = f"Conteúdo para {JUIZ} na etapa {etapa_atual_do_no} não foi gerado pela lógica específica."
        proximo_ator_logico = ETAPA_FIM_PROCESSO 

    print(f"[{JUIZ}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")

    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]
    
    return {
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": pontos_controvertidos_definidos,
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False), 
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),   
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

def agente_advogado_reu(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_REU)
    # estado["etapa_a_ser_executada_neste_turno"] = etapa_atual_do_no # Helper já loga

    print(f"\n--- TURNO: {ADVOGADO_REU} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        print(f"ERRO FATAL: {ADVOGADO_REU} não conseguiu determinar sua etapa. Encerrando fluxo.")
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_REU,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": "Erro crítico de fluxo para o Advogado do Réu.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_REU, "documento": "Erro de fluxo irrecuperável do Advogado Réu."}],
            "id_processo": estado.get("id_processo", "ID_DESCONHECIDO"), # Preservar estado
            "retriever": estado.get("retriever"),
            "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"),
            "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False),
            "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
            "etapa_a_ser_executada_neste_turno": "ERRO_FLUXO_IRRECUPERAVEL"
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Default, será sobrescrito pela lógica da etapa
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado:
        historico_formatado = "Histórico processual não disponível."

    documento_relevante_anterior = estado.get("documento_gerado_na_etapa_recente", "Nenhum documento recente para análise imediata.")
    
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo} sob a perspectiva da defesa"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo}
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si para a defesa."
    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")

    if etapa_atual_do_no == ETAPA_CONTESTACAO:
        peticao_inicial_autor = "Petição Inicial não encontrada no histórico para contestação." 
        for item in reversed(estado.get("historico_completo", [])):
            if item["etapa"] == ETAPA_PETICAO_INICIAL and item["ator"] == ADVOGADO_AUTOR:
                peticao_inicial_autor = item["documento"]
                break
        
        docs_modelo_contestacao = retriever.get_relevant_documents(query="modelo de contestação cível completa", filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_contestacao[0].page_content if docs_modelo_contestacao else "ERRO RAG: Modelo de contestação não encontrado."

        template_prompt = """
        Você é um Advogado do Réu experiente e está elaborando uma Contestação.
        **Processo ID:** {id_processo}
        **Tarefa:** Analisar a Petição Inicial do Autor e o despacho judicial, e elaborar uma Contestação completa, rebatendo os argumentos do autor, apresentando preliminares (se houver) e o mérito da defesa.

        **Contexto dos Fatos do Caso (do RAG, com foco na perspectiva da defesa):**
        {contexto_fatos_caso}

        **Despacho Judicial Recebido (ex: citação):**
        {despacho_judicial}
        
        **Petição Inicial do Autor (documento a ser contestado):**
        {peticao_inicial_autor}

        **Modelo/Guia de Contestação (use como referência estrutural e de linguagem):**
        {modelo_texto_guia}
        
        **Histórico Processual até o momento:**
        {historico_formatado}
        ---
        Com base em todas as informações acima, redija a Contestação. Seja completo, claro e objetivo.
        Atenção aos fatos específicos do caso para a argumentação da defesa.
        Contestação:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "despacho_judicial": documento_relevante_anterior, # Este é o despacho do juiz recebendo a inicial
            "peticao_inicial_autor": peticao_inicial_autor,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ # Após contestação, juiz decide sobre saneamento (no fluxo simplificado). Poderia ser réplica do autor.

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        manifestacao_autor_sem_provas_doc = documento_relevante_anterior # Manifestação do autor
        decisao_saneamento_texto = "Decisão de Saneamento não encontrada no histórico recente." 
        if estado.get("pontos_controvertidos_saneamento"):
             decisao_saneamento_texto = f"Decisão de Saneamento anterior definiu: {estado['pontos_controvertidos_saneamento']}"
        else: 
            for item in reversed(estado.get("historico_completo", [])):
                if item["etapa"] == ETAPA_DECISAO_SANEAMENTO and item["ator"] == JUIZ:
                    decisao_saneamento_texto = item["documento"]
                    break

        docs_modelo_manifest = retriever.get_relevant_documents(query="modelo de petição ou manifestação declarando não ter mais provas a produzir pela defesa", filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_manifest[0].page_content if docs_modelo_manifest else "ERRO RAG: Modelo de manifestação sem provas (Réu) não encontrado."

        template_prompt = """
        Você é o Advogado do Réu. O juiz proferiu Decisão de Saneamento e o Autor já se manifestou sobre não ter mais provas.
        **Processo ID:** {id_processo}
        **Tarefa:** Elaborar uma petição informando que o Réu também não possui mais provas a produzir, ou especificando as últimas provas, se houver. Assumindo que não há mais provas, requerer o julgamento do feito.

        **Contexto dos Fatos do Caso (do RAG, com foco na perspectiva da defesa):**
        {contexto_fatos_caso}

        **Decisão de Saneamento proferida pelo Juiz:**
        {decisao_saneamento_texto}
        
        **Manifestação do Autor sobre não ter mais provas:**
        {manifestacao_autor_sem_provas_doc}

        **Modelo/Guia de Manifestação (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual até o momento:**
        {historico_formatado}
        ---
        Redija a Petição de Manifestação do Réu.
        Manifestação do Réu:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "contexto_fatos_caso": contexto_fatos_caso,
            "decisao_saneamento_texto": decisao_saneamento_texto,
            "manifestacao_autor_sem_provas_doc": manifestacao_autor_sem_provas_doc,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ 

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_REU} não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não foi gerado pela lógica específica."
        proximo_ator_logico = JUIZ 

    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_REU, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    return {
        "nome_do_ultimo_no_executado": ADVOGADO_REU,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"), 
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False), 
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU),
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

# --- 8. Construção do Grafo LangGraph ---
workflow = StateGraph(EstadoProcessual)

workflow.add_node(ADVOGADO_AUTOR, agente_advogado_autor)
workflow.add_node(JUIZ, agente_juiz)
workflow.add_node(ADVOGADO_REU, agente_advogado_reu)

workflow.set_entry_point(ADVOGADO_AUTOR)

roteamento_mapa_edges = {
    ADVOGADO_AUTOR: ADVOGADO_AUTOR,
    JUIZ: JUIZ,
    ADVOGADO_REU: ADVOGADO_REU,
    END: END
}

workflow.add_conditional_edges(ADVOGADO_AUTOR, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(JUIZ, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(ADVOGADO_REU, decidir_proximo_no_do_grafo, roteamento_mapa_edges)

app = workflow.compile()

# --- 9. Execução da Simulação com Streamlit ---
if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Simulação Jurídica Avançada")
    st.subheader("Rito Ordinário do CPC")

    # Inicializar session_state para visualização de documentos
    if 'doc_visualizado' not in st.session_state:
        st.session_state.doc_visualizado = None
        st.session_state.doc_visualizado_titulo = ""
    if 'expand_all_steps' not in st.session_state:
        st.session_state.expand_all_steps = True
    if 'expand_all_history' not in st.session_state: # Para os expanders detalhados
        st.session_state.expand_all_history = False

    id_processo_simulado = st.text_input("ID do Processo:", "caso_001")
    arquivo_processo_upload = st.file_uploader("Carregar Arquivo do Processo (.docx):", type=["docx"])

    # Placeholder para o documento completo (será usado pela timeline e pelos expanders)
    doc_completo_placeholder = st.empty()

    if arquivo_processo_upload is not None:
        # ... (lógica de upload e inicialização do retriever como antes) ...
        nome_arquivo_temporario = f"temp_{id_processo_simulado}_{arquivo_processo_upload.name}"
        caminho_temp_salvo = os.path.join(PATH_PROCESSO_EM_SI, nome_arquivo_temporario)
        
        os.makedirs(PATH_PROCESSO_EM_SI, exist_ok=True)
        
        with open(caminho_temp_salvo, "wb") as f:
            f.write(arquivo_processo_upload.read())

        st.markdown(f"--- INICIANDO SIMULAÇÃO PARA O PROCESSO: **{id_processo_simulado}** ---")

        retriever_do_caso = None
        try:
            with st.spinner("Inicializando sistema RAG e carregando documentos..."):
                retriever_do_caso = criar_ou_carregar_retriever(id_processo_simulado, nome_arquivo_temporario)
            st.success("Sistema RAG inicializado com sucesso.")
        except Exception as e:
            st.error(f"ERRO FATAL: Falha crítica ao inicializar o sistema RAG. Detalhe: {e}")
            if os.path.exists(caminho_temp_salvo):
                os.remove(caminho_temp_salvo) 
            st.stop()

        estado_inicial = EstadoProcessual(
            id_processo=id_processo_simulado,
            retriever=retriever_do_caso,
            nome_do_ultimo_no_executado=None,
            etapa_concluida_pelo_ultimo_no=None,
            proximo_ator_sugerido_pelo_ultimo_no=ADVOGADO_AUTOR,
            documento_gerado_na_etapa_recente=None,
            historico_completo=[],
            pontos_controvertidos_saneamento=None,
            manifestacao_autor_sem_provas=False,
            manifestacao_reu_sem_provas=False,
            etapa_a_ser_executada_neste_turno=""
        )
        # ... (lógica da execução da simulação com app.stream e st.expander para os passos como antes) ...
        st.subheader("Acompanhamento da Simulação:")
        if st.button("Alternar Expansão dos Passos da Execução"):
            st.session_state.expand_all_steps = not st.session_state.expand_all_steps
        progress_bar = st.progress(0)
        steps_container = st.container()
        max_passos_simulacao = 15 
        passo_atual_simulacao = 0
        estado_final_simulacao = None
        # ... (Loop app.stream como na versão anterior) ...
        try:
            for s in app.stream(input=estado_inicial, config={"recursion_limit": max_passos_simulacao}):
                passo_atual_simulacao += 1
                num_total_etapas_estimadas = len(mapa_tarefa_no_atual) + 1
                progress_value = min(100, int((passo_atual_simulacao / num_total_etapas_estimadas) * 100))
                progress_bar.progress(progress_value)

                if not isinstance(s, dict) or not s:
                    msg = f"Erro: Stream retornou valor inesperado: {s}. Encerrando."
                    steps_container.error(msg)
                    break

                nome_do_no_executado = list(s.keys())[0]
                estado_parcial_apos_no = s[nome_do_no_executado]
                estado_final_simulacao = estado_parcial_apos_no

                etapa_concluida_log = estado_parcial_apos_no.get('etapa_concluida_pelo_ultimo_no', 'N/A')
                doc_gerado_completo = str(estado_parcial_apos_no.get('documento_gerado_na_etapa_recente', ''))
                prox_ator_sug_log = estado_parcial_apos_no.get('proximo_ator_sugerido_pelo_ultimo_no', 'N/A')
                
                expander_title = f"Passo {passo_atual_simulacao}: {nome_do_no_executado} concluiu '{etapa_concluida_log}'"
                if nome_do_no_executado == END:
                     expander_title = f"Passo {passo_atual_simulacao}: Fim da Simulação"
                
                with steps_container.expander(expander_title, expanded=st.session_state.expand_all_steps):
                    st.markdown(f"**Nó Executado:** `{nome_do_no_executado}`")
                    st.markdown(f"**Etapa Concluída:** `{etapa_concluida_log}`")
                    if etapa_concluida_log != "ERRO_FLUXO_IRRECUPERAVEL" and doc_gerado_completo:
                        st.text_area("Documento Gerado:", value=doc_gerado_completo, height=200, key=f"doc_step_{passo_atual_simulacao}", disabled=True)
                    elif doc_gerado_completo:
                        st.error(f"Detalhe do Erro/Documento: {doc_gerado_completo}")
                    st.markdown(f"**Próximo Ator Sugerido:** `{prox_ator_sug_log}`")

                if nome_do_no_executado == END or prox_ator_sug_log == ETAPA_FIM_PROCESSO:
                    steps_container.success("Fluxo da simulação atingiu o nó FINAL ou etapa de fim.")
                    progress_bar.progress(100)
                    break
                if passo_atual_simulacao >= max_passos_simulacao:
                    steps_container.warning(f"Simulação atingiu o limite máximo de {max_passos_simulacao} passos.")
                    break
        except Exception as e:
            st.error(f"ERRO INESPERADO DURANTE A EXECUÇÃO DA SIMULAÇÃO: {e}")
            import traceback
            st.text_area("Stack Trace do Erro:", traceback.format_exc(), height=300)
        finally:
            if os.path.exists(caminho_temp_salvo):
                os.remove(caminho_temp_salvo)
                print(f"[Main] Arquivo temporário '{caminho_temp_salvo}' removido.")
            progress_bar.empty()

        st.markdown("---")
        st.subheader("Linha do Tempo Interativa do Processo")

        if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
            historico = estado_final_simulacao["historico_completo"]
            
            # Mapeamento de ícones (Unicode emojis)
            icon_map = {
                ADVOGADO_AUTOR: "🙋‍♂️", # Pessoa levantando a mão (autor)
                JUIZ: "⚖️",          # Balança da justiça
                ADVOGADO_REU: "🙋‍♀️", # Pessoa levantando a mão (réu - outra figura)
                ETAPA_PETICAO_INICIAL: "📄",
                ETAPA_DESPACHO_RECEBENDO_INICIAL: "➡️",
                ETAPA_CONTESTACAO: " دفاع ", # Defesa em árabe (exemplo visual) ou "🛡️",
                ETAPA_DECISAO_SANEAMENTO: "🛠️",
                ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR: "🗣️",
                ETAPA_MANIFESTACAO_SEM_PROVAS_REU: "🗣️",
                ETAPA_SENTENCA: "🏁",
                "DEFAULT_ACTOR": "👤",
                "DEFAULT_ETAPA": "📑"
            }

            # Criar colunas para a linha do tempo. O número de colunas é o número de etapas.
            # Se for muito grande, o Streamlit vai empilhar.
            # Para um visual mais controlado, poderíamos limitar o número de itens por linha
            # ou usar um container com overflow CSS (mais complexo).
            # Por simplicidade, vamos usar colunas diretas.
            
            num_etapas = len(historico)
            cols = st.columns(num_etapas)

            for i, item_hist in enumerate(historico):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                
                ator_icon = icon_map.get(ator_hist, icon_map["DEFAULT_ACTOR"])
                etapa_icon = icon_map.get(etapa_hist, icon_map["DEFAULT_ETAPA"])

                with cols[i]:
                    # Exibir o ícone do ator e da etapa de forma proeminente
                    st.markdown(f"<div style='text-align: center; font-size: 24px;'>{ator_icon}{etapa_icon}</div>", unsafe_allow_html=True)
                    # Nome do ator e etapa abaixo dos ícones
                    st.markdown(f"<p style='text-align: center; font-size: 12px; margin-bottom: 5px;'><b>{ator_hist}</b><br>{etapa_hist}</p>", unsafe_allow_html=True)
                    
                    # Botão para ver o documento
                    if st.button(f"Doc {i+1}", key=f"btn_timeline_{i}", help=f"Ver documento: {ator_hist} - {etapa_hist}"):
                        st.session_state.doc_visualizado = doc_completo_hist
                        st.session_state.doc_visualizado_titulo = f"Documento (Etapa {i+1} da Linha do Tempo): {ator_hist} - {etapa_hist}"
                        # st.rerun() # Geralmente não necessário se o placeholder estiver fora do loop de colunas

                # Adicionar uma "seta" ou conector entre as colunas, exceto para a última
                # Isso é difícil de fazer de forma robusta com st.columns diretamente.
                # Poderíamos tentar com st.markdown entre as colunas, mas o alinhamento é complexo.
                # Para esta versão, omitiremos as setas conectoras complexas entre colunas.
                # Uma linha simples abaixo da "faixa" da timeline pode ser uma opção.
            
            if num_etapas > 0:
                 st.markdown("---") # Linha separadora após a timeline

        else:
            st.warning("Nenhum histórico completo disponível para exibir na linha do tempo.")

        # Exibição do documento completo selecionado (fora do loop do histórico)
        # Este placeholder é crucial e deve estar FORA de qualquer loop de colunas ou expanders.
        if st.session_state.doc_visualizado:
            with doc_completo_placeholder.container():
                st.subheader(st.session_state.doc_visualizado_titulo)
                st.text_area("Conteúdo do Documento:", st.session_state.doc_visualizado, height=400, key="doc_view_timeline_area")
                if st.button("Fechar Visualização do Documento", key="close_doc_view_timeline_btn"):
                    st.session_state.doc_visualizado = None
                    st.session_state.doc_visualizado_titulo = ""
                    st.rerun()
        
        st.markdown("---")
        # Manter a seção de Histórico Detalhado com expanders se ainda for útil
        st.subheader("Histórico Detalhado (Conteúdo das Etapas):")
        # (O código dos expanders do histórico detalhado pode permanecer aqui, como antes)
        if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
            if st.button("Alternar Expansão dos Itens do Histórico Detalhado"):
                st.session_state.expand_all_history = not st.session_state.expand_all_history
            for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                with st.expander(f"{i+1}. Ator: {ator_hist} | Etapa: {etapa_hist}", expanded=st.session_state.expand_all_history):
                    # ... (conteúdo do expander como antes, mas o botão de ver documento aqui pode ser removido se a timeline for suficiente)
                    st.text_area(f"Documento Gerado (Detalhe {i+1}):", value=doc_completo_hist, height=150, key=f"doc_hist_detail_{i}", disabled=True)

        st.markdown("--- FIM DA SIMULAÇÃO ---")
        print("\n[Main] Execução da simulação Streamlit concluída.")

    else:
        st.info("Aguardando upload do arquivo do processo para iniciar a simulação.")