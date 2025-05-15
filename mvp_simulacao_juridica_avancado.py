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
from langchain_core.runnables import RunnablePassthrough

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
        print("[RAG] Criando e salvando vector store FAISS em '{FAISS_INDEX_PATH}'.")
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

    # (Continuação da Parte 1: Imports, Constantes, Utilitários RAG, LLM, EstadoProcessual já definidos)

# --- 5. Mapa de Fluxo Processual (Rito Ordinário) ---
# Este mapa define QUAL ETAPA um NÓ ATUAL deve executar,
# baseado em QUEM foi o NÓ ANTERIOR e QUAL ETAPA o nó anterior CONCLUIU.
# Chave: (nome_do_no_anterior, etapa_concluida_pelo_no_anterior, nome_do_no_atual_que_ira_rodar)
# Valor: etapa_que_o_no_atual_deve_executar
mapa_tarefa_no_atual: Dict[Tuple[Union[str, None], Union[str, None], str], str] = {
    # 1. Início: Advogado do Autor faz a Petição Inicial
    (None, None, ADVOGADO_AUTOR): ETAPA_PETICAO_INICIAL, # Ponto de entrada

    # 2. Juiz recebe a Petição Inicial do Advogado Autor
    (ADVOGADO_AUTOR, ETAPA_PETICAO_INICIAL, JUIZ): ETAPA_DESPACHO_RECEBENDO_INICIAL,

    # 3. Advogado do Réu recebe o Despacho do Juiz e contesta
    (JUIZ, ETAPA_DESPACHO_RECEBENDO_INICIAL, ADVOGADO_REU): ETAPA_CONTESTACAO,

    # 4. Juiz recebe a Contestação do Réu e profere Decisão de Saneamento
    (ADVOGADO_REU, ETAPA_CONTESTACAO, JUIZ): ETAPA_DECISAO_SANEAMENTO,

    # 5. Advogado do Autor recebe a Decisão de Saneamento e se manifesta sobre provas
    (JUIZ, ETAPA_DECISAO_SANEAMENTO, ADVOGADO_AUTOR): ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,

    # 6. Advogado do Réu recebe a manifestação do Autor (via Juiz, implicitamente) e também se manifesta
    (ADVOGADO_AUTOR, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR, ADVOGADO_REU): ETAPA_MANIFESTACAO_SEM_PROVAS_REU, # Adicione esta linha

    # 7. Juiz recebe as manifestações das partes e profere Sentença
    (ADVOGADO_REU, ETAPA_MANIFESTACAO_SEM_PROVAS_REU, JUIZ): ETAPA_SENTENCA,

    # 8. Fim do processo após sentença (Juiz sugere FIM)
    # (JUIZ, ETAPA_SENTENCA, ETAPA_FIM_PROCESSO): ETAPA_FIM_PROCESSO # Não é uma tarefa, mas um estado final
}

# --- 6. Função de Roteamento Condicional (Router) ---
def decidir_proximo_no_do_grafo(estado: EstadoProcessual):
    """
    Decide qual o próximo nó a ser executado com base na sugestão do nó anterior.
    """
    proximo_ator_sugerido = estado.get("proximo_ator_sugerido_pelo_ultimo_no")
    etapa_concluida = estado.get("etapa_concluida_pelo_ultimo_no")

    print(f"[Router] Estado recebido: {estado}") # Adicione esta linha
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

# --- 7. Esqueleto dos Agentes (Nós do Grafo) ---
# Serão detalhados na Parte 3. Por enquanto, apenas a estrutura.

def criar_prompt_e_chain(template_string: str):
    prompt = ChatPromptTemplate.from_template(template_string)
    return prompt | llm | StrOutputParser()

def helper_logica_inicial_no(estado: EstadoProcessual, nome_do_no_atual: str) -> str:
    """
    Ajuda cada nó a determinar qual etapa ele deve executar.
    Retorna a string da etapa ou uma string de erro/vazia.
    """
    nome_ultimo_no = estado.get("nome_do_ultimo_no_executado")
    etapa_ultimo_no = estado.get("etapa_concluida_pelo_ultimo_no")

    chave_mapa = (nome_ultimo_no, etapa_ultimo_no, nome_do_no_atual)
    
    if nome_ultimo_no is None and etapa_ultimo_no is None and nome_do_no_atual == ADVOGADO_AUTOR:
        # Ponto de entrada especial para o primeiro nó do grafo
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
    estado["etapa_a_ser_executada_neste_turno"] = etapa_atual_do_no
    print(f"\n--- TURNO: {ADVOGADO_AUTOR} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": "Erro de fluxo.",
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado:
        historico_formatado = "Este é o primeiro ato do processo."

    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo}"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo}
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si."
    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")

    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        query_modelo_pi = f"modelo de petição inicial completa para o caso {id_processo}"
        docs_modelo_pi = retriever.get_relevant_documents(query=query_modelo_pi, filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_pi[0].page_content if docs_modelo_pi else "ERRO RAG: Modelo de petição inicial não encontrado."

        template_prompt = """
        Você é um Advogado do Autor experiente e está elaborando uma Petição Inicial para o **Processo ID:** {id_processo}.
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
        # A decisão de saneamento deve estar no 'documento_gerado_na_etapa_recente'
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
        # Definir um próximo ator padrão ou de erro se a etapa não for tratada
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
        "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"),
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR),
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

def agente_juiz(estado: EstadoProcessual) -> Dict[str, Any]:
    # A lógica detalhada virá na Parte 3
    etapa_atual_do_no = helper_logica_inicial_no(estado, JUIZ)
    estado["etapa_a_ser_executada_neste_turno"] = etapa_atual_do_no
    print(f"\n--- TURNO: {JUIZ} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
         return {
            "nome_do_ultimo_no_executado": JUIZ,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": "Erro de fluxo.",
        }

    # Lógica específica para cada etapa do JUIZ (DESPACHO_RECEBENDO_INICIAL, etc.)
    # ... a ser preenchida ...
    documento_gerado = f"Documento simulado para {JUIZ} na etapa {etapa_atual_do_no}"
    proximo_ator_logico = ADVOGADO_REU # Placeholder
    pontos_controvertidos = estado.get("pontos_controvertidos_saneamento")

    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        proximo_ator_logico = ADVOGADO_REU
    elif etapa_atual_do_no == ETAPA_DECISAO_SANEAMENTO:
        proximo_ator_logico = ADVOGADO_AUTOR # Autor se manifesta sobre provas primeiro
        pontos_controvertidos = "Ponto X e Ponto Y definidos no saneamento (simulado)" # Juiz define aqui
    elif etapa_atual_do_no == ETAPA_SENTENCA:
        proximo_ator_logico = ETAPA_FIM_PROCESSO # Fim
    # ... outras condições ...

    print(f"{JUIZ} ({etapa_atual_do_no}) gerou: {documento_gerado[:100]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}

    return {
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": estado["historico_completo"] + [novo_historico_item],
        "pontos_controvertidos_saneamento": pontos_controvertidos,
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False),
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),
    }

def agente_advogado_reu(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_REU)
    print(f"\n--- TURNO: {ADVOGADO_REU} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        print(f"ERRO FATAL: {ADVOGADO_REU} não conseguiu determinar sua etapa. Encerrando fluxo.")
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_REU,
            "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": "Erro crítico de fluxo para o Advogado do Réu.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_REU, "documento": "Erro de fluxo irrecuperável do Advogado Réu."}],
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ
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

        query_modelo_contestacao = f"modelo de contestação cível completa para o caso {id_processo}"
        docs_modelo_contestacao = retriever.get_relevant_documents(query=query_modelo_contestacao, filter={"source_type": "modelo_peticao"})
        modelo_texto_guia = docs_modelo_contestacao[0].page_content if docs_modelo_contestacao else "ERRO RAG: Modelo de contestação não encontrado."

        template_prompt = """
        Você é um Advogado do Réu experiente e está elaborando uma Contestação para o **Processo ID:** {id_processo}.
        **Tarefa:** Analisar a Petição Inicial do Autor e elaborar uma Contestação completa, rebatendo os argumentos do autor, apresentando preliminares (se houver) e o mérito da defesa.

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
            "despacho_judicial": documento_relevante_anterior,
            "peticao_inicial_autor": peticao_inicial_autor,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = JUIZ
    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        # Lógica para a manifestação do réu sobre não ter provas virá em um próximo turno
        documento_gerado = f"Documento simulado para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
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

# Mapeamento para as conditional_edges:
# O valor retornado pela função router (ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU, ou END)
# deve corresponder às chaves neste dicionário.
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

def agente_advogado_autor(estado: EstadoProcessual) -> Dict[str, Any]:
    print(f"[DEBUG ADVOGADO_AUTOR] Etapa atual do nó: {estado.get('etapa_a_ser_executada_neste_turno')}")
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
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Default, será sobrescrito pela lógica da etapa
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado:
        historico_formatado = "Este é o primeiro ato do processo."

    # --- Lógica de RAG para contexto do caso (comum a várias etapas) ---
    # Tenta buscar fatos do processo. Pode ser uma query genérica ou mais específica.
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo}"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo} # Filtra pelo ID do processo!
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si."
    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")

    # --- Tratamento específico para cada etapa do Advogado Autor ---
    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        # RAG para modelo de Petição Inicial
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
        # A decisão de saneamento deve estar no 'documento_gerado_na_etapa_recente'
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
        # Definir um próximo ator padrão ou de erro se a etapa não for tratada
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
        # Preservar outros campos do estado que não foram modificados diretamente
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no, # A etapa que este nó acabou de executar
    }

# (Continuação das Partes 1, 2 e 3.1)
# As definições de ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU,
# ETAPA_DESPACHO_RECEBENDO_INICIAL, ETAPA_DECISAO_SANEAMENTO, ETAPA_SENTENCA, ETAPA_FIM_PROCESSO etc.
# llm, helper_logica_inicial_no, criar_prompt_e_chain
# já devem existir no seu script.

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
        }

    documento_gerado = f"Decisão padrão para {JUIZ} na etapa {etapa_atual_do_no} não gerada."
    proximo_ator_logico = ETAPA_FIM_PROCESSO # Default, será sobrescrito
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: # Deveria sempre ter histórico se o juiz está atuando após a PI
        historico_formatado = "Histórico processual não disponível ou PI ainda não processada."

    documento_da_parte_para_analise = estado.get("documento_gerado_na_etapa_recente", "Nenhuma petição recente das partes para análise.")
    
    # --- Lógica de RAG para contexto do caso (comum a várias etapas) ---
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo} para decisão judicial"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo}
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si."
    print(f"[{JUIZ}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")
    
    pontos_controvertidos_definidos = estado.get("pontos_controvertidos_saneamento") # Mantém se já existia

    # --- Tratamento específico para cada etapa do Juiz ---
    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        # Petição Inicial está em 'documento_da_parte_para_analise'
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
        proximo_ator_logico = ADVOGADO_REU # Após despacho inicial, réu é citado para contestar

    elif etapa_atual_do_no == ETAPA_DECISAO_SANEAMENTO:
        # Réplica do autor (ou ausência dela) está em 'documento_da_parte_para_analise'
        # Também pode considerar a contestação do réu, que está no histórico.
        # Vamos focar no estado atual e no que o LLM pode inferir do histórico geral.
        
        docs_modelo_saneamento = retriever.get_relevant_documents(query="modelo de decisão de saneamento e organização do processo", filter={"source_type": "modelo_juiz"})
        modelo_texto_guia = docs_modelo_saneamento[0].page_content if docs_modelo_saneamento else "ERRO RAG: Modelo de decisão de saneamento não encontrado."

        template_prompt = """
        Você é um Juiz de Direito e o processo está na fase de saneamento, após petição inicial, contestação e réplica.
        **Processo ID:** {id_processo}
        **Tarefa:** Proferir uma Decisão de Saneamento e Organização do Processo. Defina as questões processuais pendentes, as questões de fato sobre as quais recairá a atividade probatória (pontos controvertidos), distribua o ônus da prova e, se for o caso, designe audiência ou determine as provas a serem produzidas.

        **Contexto dos Fatos do Caso (extraído do arquivo do processo):**
        {contexto_fatos_caso}
        
        **Última manifestação relevante das partes (ex: Réplica do Autor):**
        {ultima_peticao_partes}

        **Modelo/Guia de Decisão de Saneamento (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual Completo (incluindo PI, Contestação, Réplica):**
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
            "ultima_peticao_partes": documento_da_parte_para_analise,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_AUTOR # Partes são intimadas a especificar provas, começando pelo autor.
        
        # Extrair pontos controvertidos da decisão gerada (simplificado - idealmente usaríamos técnicas de NLP)
        # Por agora, vamos assumir que o LLM os formata de uma maneira que podemos extrair ou pedir ao LLM para listar.
        # Para o MVP, pode ser um placeholder ou uma tentativa simples de extração.
        try:
            inicio_pc = documento_gerado.upper().find("PONTOS CONTROVERTIDOS:")
            if inicio_pc != -1:
                fim_pc = documento_gerado.find("\n\n", inicio_pc) # Tenta encontrar o fim da seção
                if fim_pc == -1: fim_pc = len(documento_gerado)
                pontos_controvertidos_definidos = documento_gerado[inicio_pc + len("PONTOS CONTROVERTIDOS:"):fim_pc].strip()
            else:
                pontos_controvertidos_definidos = "Não foi possível extrair os pontos controvertidos automaticamente. Verificar decisão."
        except Exception:
            pontos_controvertidos_definidos = "Erro ao tentar extrair pontos controvertidos."
        print(f"[{JUIZ}-{etapa_atual_do_no}] Pontos Controvertidos extraídos/definidos: {pontos_controvertidos_definidos}")


    elif etapa_atual_do_no == ETAPA_SENTENCA:
        # Manifestações das partes sobre não ter mais provas estão no histórico.
        # O 'documento_da_parte_para_analise' seria a última manifestação (do réu).
        
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
        proximo_ator_logico = ETAPA_FIM_PROCESSO # Após a sentença, o fluxo principal do rito ordinário se encerra aqui.

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {JUIZ} não implementada completamente.")
        documento_gerado = f"Conteúdo para {JUIZ} na etapa {etapa_atual_do_no} não foi gerado pela lógica específica."
        proximo_ator_logico = ETAPA_FIM_PROCESSO # Erro ou etapa não tratada leva ao fim

    print(f"[{JUIZ}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")

    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]
    
    return {
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": pontos_controvertidos_definidos, # Atualiza se foi definido no saneamento
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False), # Preserva
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False),   # Preserva
        # Preservar outros campos do estado
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

# (Continuação das Partes 1, 2, 3.1 e 3.2)
# As definições de ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU,
# ETAPA_CONTESTACAO, ETAPA_MANIFESTACAO_SEM_PROVAS_REU, etc.
# llm, helper_logica_inicial_no, criar_prompt_e_chain
# já devem existir no seu script.

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
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Default, será sobrescrito pela lógica da etapa
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado:
        historico_formatado = "Histórico processual não disponível." # Improvável nesta fase para o réu

    # Documento do juiz ou da parte contrária para análise
    documento_relevante_anterior = estado.get("documento_gerado_na_etapa_recente", "Nenhum documento recente para análise imediata.")
    
    # --- Lógica de RAG para contexto do caso (comum a várias etapas) ---
    query_fatos_caso = f"Resumo dos fatos e principais pontos do processo {id_processo} sob a perspectiva da defesa"
    docs_fatos_caso = retriever.get_relevant_documents(
        query=query_fatos_caso,
        filter={"source_type": "processo_atual", "process_id": id_processo}
    )
    contexto_fatos_caso = "\n".join([doc.page_content for doc in docs_fatos_caso]) if docs_fatos_caso else "Nenhum detalhe factual específico do caso encontrado no RAG do processo_em_si para a defesa."
    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Contexto do caso (RAG): {contexto_fatos_caso[:200]}...")

    # --- Tratamento específico para cada etapa do Advogado Réu ---
    if etapa_atual_do_no == ETAPA_CONTESTACAO:
        # O despacho do juiz (citando o réu) está em 'documento_relevante_anterior'.
        # A Petição Inicial original do autor está no histórico.
        peticao_inicial_autor = "Petição Inicial não encontrada no histórico para contestação." # Default
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
            "despacho_judicial": documento_relevante_anterior,
            "peticao_inicial_autor": peticao_inicial_autor,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_AUTOR # Após contestação, autor tem direito à réplica

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        # A manifestação do autor sobre não ter provas está em 'documento_relevante_anterior'.
        # A decisão de saneamento também é importante e está no histórico ou em 'pontos_controvertidos_saneamento'.
        manifestacao_autor_sem_provas_doc = documento_relevante_anterior
        decisao_saneamento_texto = "Decisão de Saneamento não encontrada no histórico recente." # Default
        if estado.get("pontos_controvertidos_saneamento"):
             decisao_saneamento_texto = f"Decisão de Saneamento anterior definiu: {estado['pontos_controvertidos_saneamento']}"
        else: # Tenta buscar no histórico
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
        proximo_ator_logico = JUIZ # Após ambas as partes dizerem não ter provas, processo vai para sentença

    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_REU} não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não foi gerado pela lógica específica."
        proximo_ator_logico = JUIZ # Default para erro ou etapa não tratada

    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_REU, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    return {
        "nome_do_ultimo_no_executado": ADVOGADO_REU,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": estado.get("pontos_controvertidos_saneamento"), # Preserva
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False), # Preserva
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU),
        # Preservar outros campos do estado
        "id_processo": estado["id_processo"],
        "retriever": estado["retriever"],
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }

# (Continuação das Partes 1, 2, 3.1, 3.2 e 3.3)
# Todas as definições anteriores (imports, constantes, RAG, LLM, Estado, Mapa, Agentes, Grafo)
# já devem estar presentes no script.

# --- 9. Execução da Simulação com Streamlit ---
if __name__ == "__main__":
    st.title("Simulação Jurídica Avançada")
    st.subheader("Rito Ordinário do CPC")

    id_processo_simulado = st.text_input("ID do Processo:", "caso_001")
    arquivo_processo_upload = st.file_uploader("Carregar Arquivo do Processo (.docx):", type=["docx"])

    if arquivo_processo_upload is not None:
        # Salvar o arquivo temporariamente para que a função carregar_documentos_docx possa acessá-lo
        nome_arquivo_temporario = f"temp_{id_processo_simulado}_processo.docx"
        with open(nome_arquivo_temporario, "wb") as f:
            f.write(arquivo_processo_upload.read())

        print(f"--- INICIANDO SIMULAÇÃO JURÍDICA AVANÇADA PARA O PROCESSO: {id_processo_simulado} ---")
        print(f"--- Rito Ordinário do CPC ---")

        retriever_do_caso = None
        try:
            print("\n[Main] Tentando inicializar o sistema RAG (Retriever)...")
            retriever_do_caso = criar_ou_carregar_retriever(id_processo_simulado, nome_arquivo_temporario)
            print("[Main] Sistema RAG inicializado com sucesso.")
        except Exception as e:
            st.error(f"ERRO FATAL: Falha crítica ao inicializar o sistema RAG. Encerrando. Detalhe do erro: {e}")
            print(f"[Main] ERRO FATAL: Falha crítica ao inicializar o sistema RAG. Encerrando.")
            print(f"Detalhe do erro: {e}")
            print("Verifique se as pastas de dados existem, se o arquivo .docx está correto e se a API de embeddings está acessível.")
            os.remove(nome_arquivo_temporario) # Limpar arquivo temporário em caso de erro
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

        st.subheader("Execução da Simulação:")
        progress_bar = st.progress(0)
        log_area = st.empty()
        historico_area = st.empty()

        print("\n[Main] Estado inicial definido. Compilando e iniciando a execução do grafo LangGraph...")

        max_passos_simulacao = 15
        passo_atual_simulacao = 0
        estado_final_simulacao = None

        try:
            for s in app.stream(input=estado_inicial, config={"recursion_limit": max_passos_simulacao}):
                passo_atual_simulacao += 1
                progress_value = int((passo_atual_simulacao / max_passos_simulacao) * 100)
                progress_bar.progress(progress_value)

                if not isinstance(s, dict) or not s:
                    log_area.error(f"[Main - Passo {passo_atual_simulacao}] Stream retornou valor inesperado: {s}. Encerrando.")
                    print(f"[Main - Passo {passo_atual_simulacao}] Stream retornou valor inesperado: {s}. Encerrando.")
                    break

                nome_do_no_executado = list(s.keys())[0]
                estado_parcial_apos_no = s[nome_do_no_executado]
                estado_final_simulacao = estado_parcial_apos_no

                log_area.info(f"[Main - Passo {passo_atual_simulacao}] Nó executado: '{nome_do_no_executado}'")
                print(f"\n[Main - Passo {passo_atual_simulacao}] Nó executado: '{nome_do_no_executado}'")
                if estado_parcial_apos_no:
                    etapa_concluida_log = estado_parcial_apos_no.get('etapa_concluida_pelo_ultimo_no', 'N/A')
                    doc_gerado_log = str(estado_parcial_apos_no.get('documento_gerado_na_etapa_recente', ''))[:200]
                    prox_ator_sug_log = estado_parcial_apos_no.get('proximo_ator_sugerido_pelo_ultimo_no', 'N/A')

                    log_area.info(f"  Etapa Concluída por '{nome_do_no_executado}': {etapa_concluida_log}")
                    log_area.info(f"  Documento Gerado (trecho): {doc_gerado_log}...")
                    log_area.info(f"  Próximo Ator Sugerido: {prox_ator_sug_log}")
                    print(f"  Etapa Concluída por '{nome_do_no_executado}': {etapa_concluida_log}")
                    print(f"  Documento Gerado (trecho): {doc_gerado_log}...")
                    print(f"  Próximo Ator Sugerido: {prox_ator_sug_log}")

                    historico_str = "### Histórico da Simulação:\n"
                    for i, item_hist in enumerate(estado_parcial_apos_no.get("historico_completo", [])):
                        ator_hist = item_hist.get('ator', 'N/A')
                        etapa_hist = item_hist.get('etapa', 'N/A')
                        doc_hist = str(item_hist.get('documento', 'N/A'))[:100] + "..."
                        historico_str += f"{i+1}. **Ator:** {ator_hist}, **Etapa:** {etapa_hist}\n   **Documento:** {doc_hist}\n"
                    historico_area.markdown(historico_str)

                else:
                    log_area.warning(" AVISO: Estado parcial após nó está vazio ou None.")
                    print("  AVISO: Estado parcial após nó está vazio ou None.")

                if nome_do_no_executado == END:
                    log_area.success("\n[Main] Fluxo da simulação atingiu o nó FINAL (END).")
                    print("\n[Main] Fluxo da simulação atingiu o nó FINAL (END).")
                    break
                if passo_atual_simulacao >= max_passos_simulacao:
                    log_area.warning(f"\n[Main] Simulação atingiu o limite máximo de {max_passos_simulacao} passos.")
                    print(f"\n[Main] Simulação atingiu o limite máximo de {max_passos_simulacao} passos.")
                    break

        except Exception as e:
            log_area.error(f"\n[Main] ERRO INESPERADO DURANTE A EXECUÇÃO DA SIMULAÇÃO: {e}")
            print(f"\n[Main] ERRO INESPERADO DURANTE A EXECUÇÃO DA SIMULAÇÃO:")
            import traceback
            traceback.print_exc()
            print(f"Erro: {e}")
            if estado_final_simulacao:
                log_area.info("\n[Main] Último estado conhecido antes do erro:")
                print("\n[Main] Último estado conhecido antes do erro:")
                for key, value in estado_final_simulacao.items():
                    if key == "retriever":
                        log_area.info(f"  {key}: <Instância do Retriever>")
                        print(f"  {key}: <Instância do Retriever>")
                    elif key == "historico_completo":
                        log_area.info(f"  {key}: ({len(value)} itens no histórico)")
                        print(f"  {key}: ({len(value)} itens no histórico)")
                    else:
                        log_area.info(f"  {key}: {str(value)[:300]}")
                        print(f"  {key}: {str(value)[:300]}")

        finally:
            if 'nome_arquivo_temporario' in locals() and os.path.exists(nome_arquivo_temporario):
                os.remove(nome_arquivo_temporario)
                print(f"[Main] Arquivo temporário '{nome_arquivo_temporario}' removido.")
            elif 'nome_arquivo_temporario' not in locals():
                print("[Main] Nenhum arquivo temporário para remover.")
            elif not os.path.exists(nome_arquivo_temporario):
                print(f"[Main] Arquivo temporário '{nome_arquivo_temporario}' já foi removido ou não existe.")

        st.subheader("Fim da Simulação Jurídica")
        if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
            historico_str_final = "### Histórico Completo de Ações e Documentos Gerados:\n"
            for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_hist = str(item_hist.get('documento', 'N/A'))[:200] + "..."
                historico_str_final += f"{i+1}. **Ator:** {ator_hist}, **Etapa:** {etapa_hist}\n   **Documento:** {doc_hist}\n"
            st.markdown(historico_str_final)
            print("\n\n--- FIM DA SIMULAÇÃO JURÍDICA ---")
            print("\nHistórico Completo de Ações e Documentos Gerados:")
            for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_hist = str(item_hist.get('documento', 'N/A'))[:200] + "..."
                print(f"  {i+1}. Ator: {ator_hist}, Etapa: {etapa_hist}\n    Documento (trecho): {doc_hist}...")
        else:
            st.warning("Nenhum histórico completo disponível ou simulação não produziu estado final válido.")
            print("Nenhum histórico completo disponível ou simulação não produziu estado final válido.")

        print("\n[Main] Execução concluída.")