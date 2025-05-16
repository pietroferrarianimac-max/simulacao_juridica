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
from langchain_core.documents import Document
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_community.search import GoogleSearchRun
from langchain_core.tools import Tool



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

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GOOGLE_API_KEY_SEARCH = os.getenv("GOOGLE_API_KEY_SEARCH")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

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
    
    # Para carregar dados do formulário Streamlit (incluirá documentos do autor)
    dados_formulario_entrada: Union[Dict[str, Any], None]

    # Para armazenar documentos juntados pelo Réu após a contestação
    documentos_juntados_pelo_reu: Union[List[Dict[str, str]], None]

    # CAMPOS PARA ANÁLISE DE SENTIMENTO
    sentimento_peticao_inicial: Union[str, None]
    sentimento_contestacao: Union[str, None]

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

def formatar_lista_documentos_para_prompt(documentos: List[Dict[str, str]], parte_nome: str) -> str:
    if not documentos:
        return f"Nenhum documento específico foi listado por {parte_nome} para esta etapa."
    texto_docs = f"\n\n**Documentos que acompanham esta petição ({parte_nome}):**\n"
    for i, doc in enumerate(documentos):
        texto_docs += f"{i+1}. **Tipo:** {doc.get('tipo', 'N/A')}\n   **Descrição/Propósito:** {doc.get('descricao', 'N/A')}\n"
    return texto_docs

def agente_advogado_autor(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_AUTOR)
    print(f"\n--- TURNO: {ADVOGADO_AUTOR} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_AUTOR, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro crítico de fluxo.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_AUTOR, "documento": "Erro de fluxo."}],
            **{k: v for k, v in estado.items() if k not in ["nome_do_ultimo_no_executado", "etapa_concluida_pelo_ultimo_no", "proximo_ator_sugerido_pelo_ultimo_no", "documento_gerado_na_etapa_recente", "historico_completo"]}
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Padrão
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    dados_formulario = estado.get("dados_formulario_entrada", {})
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Este é o primeiro ato do processo."

    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        docs_modelo_pi = retriever.get_relevant_documents(query="modelo de petição inicial cível completa e bem estruturada", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_pi[0].page_content if docs_modelo_pi else "ERRO RAG: Modelo de petição inicial não encontrado."
        
        qualificacao_autor_form = dados_formulario.get("qualificacao_autor", "Qualificação do Autor não fornecida.")
        qualificacao_reu_form = dados_formulario.get("qualificacao_reu", "Qualificação do Réu não fornecida.")
        # natureza_acao_form é preenchida pelo usuário ao final do formulário, com sugestão da IA
        natureza_acao_form = dados_formulario.get("natureza_acao", "Natureza da ação não fornecida ou a ser definida.")
        fatos_form = dados_formulario.get("fatos", "Fatos não fornecidos.")
        direito_form = dados_formulario.get("fundamentacao_juridica", "Fundamentação jurídica não fornecida.")
        pedidos_form = dados_formulario.get("pedidos", "Pedidos não fornecidos.")
        
        # MODIFICAÇÃO: Recuperar e formatar documentos do autor
        documentos_autor_lista = dados_formulario.get("documentos_autor", [])
        documentos_autor_texto_formatado = formatar_lista_documentos_para_prompt(documentos_autor_lista, "Autor")

        template_prompt_pi = f"""
        Você é um Advogado do Autor experiente e está elaborando uma Petição Inicial completa, formal e persuasiva.
        **Processo ID:** {{id_processo}}

        **Dados Base Fornecidos para a Petição:**
        Qualificação do Autor:
        {qualificacao_autor_form}

        Qualificação do Réu:
        {qualificacao_reu_form}

        Natureza da Ação (definida pelo usuário/IA): {natureza_acao_form}

        Dos Fatos:
        {fatos_form}

        Do Direito (Fundamentação Jurídica):
        {direito_form}

        Dos Pedidos:
        {pedidos_form}
        {documentos_autor_texto_formatado} 
        **Modelo/Guia Estrutural de Petição Inicial (RAG - use para formatação, completude e referências legais, mas priorize os dados fornecidos acima para o conteúdo do caso):**
        {{modelo_texto_guia}}
        
        **Instruções Adicionais:**
        1. Redija a Petição Inicial completa e bem formatada, seguindo a praxe forense.
        2. Certifique-se de que todos os elementos dos DADOS BASE (fatos, direito, pedidos, qualificações, natureza da ação) estejam integralmente e corretamente incorporados.
        3. No corpo da petição (especialmente na narração dos fatos ou antes dos pedidos), faça menção aos principais documentos listados em "Documentos que acompanham esta petição (Autor)", indicando sua relevância para comprovar as alegações.
        4. Conclua com os requerimentos de praxe (data, assinatura do advogado).

        Petição Inicial:
        """
        chain_pi = criar_prompt_e_chain(template_prompt_pi)
        documento_gerado = chain_pi.invoke({
            "id_processo": id_processo,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado # Mantido para outros contextos, menos relevante para PI
        })
        # ---> INÍCIO: Análise de Sentimento da Petição Inicial <---
        sentimento_pi_texto = "Não analisado"
        try:
            prompt_sentimento_pi = f"""
            Analise o tom e o sentimento predominante do seguinte texto jurídico (Petição Inicial).
            Responda com uma única palavra ou expressão curta que melhor descreva o sentimento (ex: Assertivo, Conciliatório, Agressivo, Neutro, Persuasivo, Formal, Emocional, Confiante, Defensivo, Indignado, Colaborativo).
            Seja conciso.

            Texto da Petição Inicial:
            {documento_gerado[:3000]} # Limita para evitar estouro de token, se necessário

            Sentimento Predominante:"""
            chain_sentimento_pi = criar_prompt_e_chain(prompt_sentimento_pi)
            sentimento_pi_texto = chain_sentimento_pi.invoke({}) # documento_gerado está no prompt
            print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Sentimento da PI: {sentimento_pi_texto}")
        except Exception as e_sent:
            print(f"Erro ao analisar sentimento da PI: {e_sent}")
            sentimento_pi_texto = "Erro na análise"
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR:
        decisao_saneamento_recebida = estado.get("documento_gerado_na_etapa_recente", "ERRO: Decisão de Saneamento não encontrada no estado.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não definidos na decisão de saneamento.")
        historico_completo_formatado_para_prompt = "\n".join([f"### Documento da Etapa: {item['etapa']} (Ator: {item['ator']})\n{item['documento']}\n---" for item in estado.get("historico_completo", [])])


        template_prompt_manifestacao_autor = f"""
        Você é o Advogado do Autor. O Juiz proferiu a Decisão de Saneamento e intimou as partes para especificarem as provas que pretendem produzir, ou manifestarem desinteresse na produção de mais provas.
        Seu cliente (Autor) informou que não possui mais provas a produzir e deseja o julgamento antecipado da lide, se o Réu também não tiver provas.

        **Processo ID:** {{id_processo}}

        **Decisão de Saneamento Recebida do Juiz:**
        {decisao_saneamento_recebida}

        **Pontos Controvertidos Fixados na Decisão de Saneamento:**
        {pontos_controvertidos}

        **Histórico Processual Anterior (para contexto):**
        {historico_completo_formatado_para_prompt}

        **Instruções:**
        1. Redija uma petição de "Manifestação Sobre Provas (Autor)".
        2. Na petição, declare que o Autor não tem outras provas a produzir, além daquelas já constantes nos autos (documentais).
        3. Requeira o julgamento do processo no estado em que se encontra (julgamento antecipado do mérito), caso o Réu também não especifique provas a produzir ou se as provas especificadas por ele forem apenas documentais já apresentadas ou impertinentes.
        4. Mantenha a formalidade e praxe forense. Conclua com data e assinatura do advogado.

        Manifestação Sobre Provas (Autor):
        """
        chain_manifestacao_autor = criar_prompt_e_chain(template_prompt_manifestacao_autor)
        documento_gerado = chain_manifestacao_autor.invoke({
            "id_processo": id_processo,
            # "historico_formatado": historico_formatado # Já incluído no prompt acima de forma mais completa
        })
        proximo_ator_logico = ADVOGADO_REU


    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_AUTOR} não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no}."
        # proximo_ator_logico já é JUIZ por padrão, pode precisar de ajuste dependendo da etapa não implementada

    print(f"[{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_AUTOR, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    estado_retorno_parcial = {
        "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR),
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
        "sentimento_peticao_inicial": sentimento_pi_texto if etapa_atual_do_no == ETAPA_PETICAO_INICIAL else estado.get("sentimento_peticao_inicial"),
    }
    # Preserva outros campos do estado que não foram explicitamente modificados
    estado_final_retorno = {**estado, **estado_retorno_parcial}
    return estado_final_retorno


def agente_juiz(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, JUIZ)
    print(f"\n--- TURNO: {JUIZ} (Executando Etapa: {etapa_atual_do_no}) ---")
    
    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
        return {
            "nome_do_ultimo_no_executado": JUIZ, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro Juiz.",
             "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": JUIZ, "documento": "Erro Juiz."}],
            **{k: v for k, v in estado.items() if k not in ["nome_do_ultimo_no_executado", "etapa_concluida_pelo_ultimo_no", "proximo_ator_sugerido_pelo_ultimo_no", "documento_gerado_na_etapa_recente", "historico_completo"]}
        }

    documento_gerado = f"Decisão padrão para {JUIZ} na etapa {etapa_atual_do_no} não gerada."
    proximo_ator_logico = ETAPA_FIM_PROCESSO # Padrão
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."
    documento_da_parte_para_analise = estado.get("documento_gerado_na_etapa_recente", "Nenhuma petição ou manifestação recente para análise.")
    
    contexto_fatos_caso_rag_juiz = "Contexto RAG do caso para o juiz (simplificado para este exemplo)." 
    pontos_controvertidos_definidos = estado.get("pontos_controvertidos_saneamento") # Para uso posterior

    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        docs_modelo_despacho = retriever.get_relevant_documents(query="modelo de despacho judicial cível recebendo petição inicial e determinando citação", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_despacho[0].page_content if docs_modelo_despacho else "ERRO RAG: Modelo de despacho inicial não encontrado."
        
        # A petição inicial (documento_da_parte_para_analise) já contém a menção aos docs do autor.
        template_prompt = """
        Você é um Juiz de Direito. Analise a Petição Inicial apresentada e, se estiver em ordem, profira um despacho inicial determinando a citação do réu.
        Considere os documentos que acompanham a inicial, conforme nela mencionados.
        **Processo ID:** {id_processo}
        **Petição Inicial apresentada pelo Autor (pode incluir menção a documentos anexos):**
        {peticao_inicial}
        **Modelo/Guia de Despacho (use como referência para estrutura e formalidades):**
        {modelo_texto_guia}
        **Histórico Processual (se houver):**
        {historico_formatado}
        ---
        Redija o Despacho Inicial. Se a petição estiver apta, defira a inicial e ordene a citação do réu para apresentar contestação no prazo legal.
        Mencione brevemente o recebimento da inicial e dos documentos que a instruem, se relevante.
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
        docs_modelo_saneamento = retriever.get_relevant_documents(query="modelo de decisão de saneamento e organização do processo cível", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_saneamento[0].page_content if docs_modelo_saneamento else "ERRO RAG: Modelo de saneamento não encontrado."

        # Recuperar documentos do autor (do formulário/estado inicial) e do réu (do estado)
        documentos_autor_lista = estado.get("dados_formulario_entrada", {}).get("documentos_autor", [])
        documentos_autor_texto = formatar_lista_documentos_para_prompt(documentos_autor_lista, "Autor")
        
        documentos_reu_lista = estado.get("documentos_juntados_pelo_reu", [])
        documentos_reu_texto = formatar_lista_documentos_para_prompt(documentos_reu_lista, "Réu")

        template_prompt = """
        Você é um Juiz de Direito. O processo está na fase de saneamento após a apresentação da contestação.
        Analise a Petição Inicial, a Contestação e os documentos juntados por ambas as partes.
        **Processo ID:** {id_processo}
        
        **Petição Inicial e Documentos do Autor (resumo/menção):**
        (O conteúdo completo da Petição Inicial, que menciona os documentos do Autor, foi apresentado anteriormente no histórico)
        {documentos_autor_info}

        **Contestação do Réu e Documentos do Réu (para análise):**
        {contestacao_e_docs_reu} 
        (O conteúdo da Contestação está em 'contestacao_e_docs_reu'. Esta variável também pode conter a lista de documentos do réu ao final)
        {documentos_reu_info}

        **Contexto RAG Adicional do Caso (se necessário):**
        {contexto_fatos_caso_rag}
        
        **Modelo/Guia de Decisão de Saneamento (use como referência):**
        {modelo_texto_guia}
        
        **Histórico Processual Anterior:**
        {historico_formatado}
        ---
        Tarefa: Redija a Decisão de Saneamento e Organização do Processo.
        1. Verifique preliminares e condições da ação.
        2. Delimite as questões de fato sobre as quais recairá a atividade probatória (PONTOS CONTROVERTIDOS).
        3. Especifique os meios de prova admitidos.
        4. Defina as questões de direito relevantes para a decisão do mérito.
        5. Nunca designe audiência de instrução e julgamento, sempre determine que as parte
        especifiquem as provas que pretendem produzir, advertindo que audiência não está 
        prevista neste MVP
        Certifique-se de que a decisão seja clara e objetiva.
        Decisão de Saneamento:
        """
        chain = criar_prompt_e_chain(template_prompt)
        # 'documento_da_parte_para_analise' aqui é a contestação (que pode ou não já incluir a lista de docs do réu)
        documento_gerado = chain.invoke({
            "id_processo": id_processo,
            "documentos_autor_info": documentos_autor_texto, # Passa a lista formatada
            "contestacao_e_docs_reu": documento_da_parte_para_analise, # Contestação do agente_advogado_reu
            "documentos_reu_info": documentos_reu_texto, # Passa a lista formatada
            "contexto_fatos_caso_rag": contexto_fatos_caso_rag_juiz,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })
        proximo_ator_logico = ADVOGADO_AUTOR # Para manifestação sobre provas ou saneamento
        try:
            inicio_pc = documento_gerado.upper().find("PONTOS CONTROVERTIDOS:")
            if inicio_pc != -1:
                fim_pc = documento_gerado.find("\n\n", inicio_pc) 
                if fim_pc == -1: fim_pc = len(documento_gerado)
                pontos_controvertidos_definidos = documento_gerado[inicio_pc + len("PONTOS CONTROVERTIDOS:"):fim_pc].strip()
            else: pontos_controvertidos_definidos = "Não extraído explicitamente da decisão de saneamento."
        except Exception: pontos_controvertidos_definidos = "Erro na extração dos pontos controvertidos."
        print(f"[{JUIZ}-{etapa_atual_do_no}] Pontos Controvertidos Definidos/Extraídos: {pontos_controvertidos_definidos}")


    elif etapa_atual_do_no == ETAPA_SENTENCA:
        peticao_inicial_completa = "Petição Inicial não encontrada."
        contestacao_completa = "Contestação não encontrada."
        decisao_saneamento_completa = "Decisão de Saneamento não encontrada."
        manifestacao_autor_sem_provas_texto = "Manifestação do Autor sobre provas não encontrada ou não ocorreu."
        manifestacao_reu_sem_provas_texto = "Manifestação do Réu sobre provas não encontrada ou não ocorreu."
        
        historico_completo_para_sentenca = []
        for item in estado.get("historico_completo", []):
            historico_completo_para_sentenca.append(f"### Documento da Etapa: {item['etapa']} (Ator: {item['ator']})\n{item['documento']}\n---")
            if item['etapa'] == ETAPA_PETICAO_INICIAL:
                peticao_inicial_completa = item['documento']
            elif item['etapa'] == ETAPA_CONTESTACAO:
                contestacao_completa = item['documento']
            elif item['etapa'] == ETAPA_DECISAO_SANEAMENTO:
                decisao_saneamento_completa = item['documento']
            elif item['etapa'] == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR:
                 manifestacao_autor_sem_provas_texto = item['documento']
            elif item['etapa'] == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
                 manifestacao_reu_sem_provas_texto = item['documento']

        historico_formatado_completo_str = "\n".join(historico_completo_para_sentenca)

        docs_modelo_sentenca = retriever.get_relevant_documents(query="modelo de sentença cível completa de mérito", filter={"source_type": "modelo_juiz"}) if retriever else []
        modelo_texto_guia = docs_modelo_sentenca[0].page_content if docs_modelo_sentenca else "ERRO RAG: Modelo de sentença não encontrado."
        
        # Informações sobre documentos juntados (do estado)
        documentos_autor_lista_estado = estado.get("dados_formulario_entrada", {}).get("documentos_autor", [])
        documentos_autor_texto_formatado_estado = formatar_lista_documentos_para_prompt(documentos_autor_lista_estado, "Autor")
        
        documentos_reu_lista_estado = estado.get("documentos_juntados_pelo_reu", [])
        documentos_reu_texto_formatado_estado = formatar_lista_documentos_para_prompt(documentos_reu_lista_estado, "Réu")

        template_prompt_sentenca = f"""
        Você é um Juiz de Direito e deve proferir a Sentença neste processo.
        As partes (Autor e Réu) manifestaram desinteresse na produção de outras provas, requerendo o julgamento antecipado da lide.

        **Processo ID:** {{id_processo}}

        **Peças Processuais Principais e Histórico Completo (para sua análise):**
        Petição Inicial:
        {peticao_inicial_completa}
        ---
        Documentos do Autor (listados na inicial ou formulário):
        {documentos_autor_texto_formatado_estado}
        ---
        Contestação:
        {contestacao_completa}
        ---
        Documentos do Réu (listados na contestação ou gerados):
        {documentos_reu_texto_formatado_estado}
        ---
        Decisão de Saneamento (contém os pontos controvertidos):
        {decisao_saneamento_completa}
        ---
        Manifestação do Autor sobre Provas:
        {manifestacao_autor_sem_provas_texto}
        ---
        Manifestação do Réu sobre Provas:
        {manifestacao_reu_sem_provas_texto}
        ---
        Histórico Processual Completo Adicional (se necessário):
        {historico_formatado_completo_str} 
        ---
        **Modelo/Guia de Sentença (use como referência para estrutura e formalidades):**
        {modelo_texto_guia}
        ---
        **Instruções para a Sentença:**
        1.  Elabore um relatório conciso, mencionando as principais ocorrências do processo.
        2.  Apresente a fundamentação, analisando as questões de fato e de direito, examinando as provas produzidas (documentais, no caso) em relação aos pontos controvertidos.
        3.  Profira o dispositivo, julgando procedente, parcialmente procedente ou improcedente o pedido do Autor, de forma clara e precisa.
        4.  Condene a parte vencida ao pagamento das custas processuais e honorários advocatícios (considere um percentual sobre o valor da causa ou condenação, ex: 10%).
        5.  Conclua com local, data e assinatura.

        Sentença:
        """
        chain_sentenca = criar_prompt_e_chain(template_prompt_sentenca)
        documento_gerado = chain_sentenca.invoke({
            "id_processo": id_processo,
            # Os demais campos já estão no template string
        })
        proximo_ator_logico = ETAPA_FIM_PROCESSO
        # 'pontos_controvertidos_saneamento' não precisa ser modificado aqui, mas é usado no prompt.

    print(f"[{JUIZ}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]
    
    estado_retorno_parcial = {
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": historico_atualizado,
        "pontos_controvertidos_saneamento": pontos_controvertidos_definidos, # Atualiza
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
    }
    estado_final_retorno = {**estado, **estado_retorno_parcial}
    return estado_final_retorno


def agente_advogado_reu(estado: EstadoProcessual) -> Dict[str, Any]:
    etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_REU)
    print(f"\n--- TURNO: {ADVOGADO_REU} (Executando Etapa: {etapa_atual_do_no}) ---")

    if etapa_atual_do_no == "ERRO_ETAPA_NAO_ENCONTRADA":
         return {
            "nome_do_ultimo_no_executado": ADVOGADO_REU, "etapa_concluida_pelo_ultimo_no": "ERRO_FLUXO_IRRECUPERAVEL",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO, "documento_gerado_na_etapa_recente": "Erro Réu.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO", "ator": ADVOGADO_REU, "documento": "Erro Réu."}],
            **{k: v for k, v in estado.items() if k not in ["nome_do_ultimo_no_executado", "etapa_concluida_pelo_ultimo_no", "proximo_ator_sugerido_pelo_ultimo_no", "documento_gerado_na_etapa_recente", "historico_completo"]}
        }

    documento_gerado_principal = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} não gerado."
    proximo_ator_logico = JUIZ # Padrão
    retriever = estado["retriever"]
    id_processo = estado["id_processo"]
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Documento: {item['documento'][:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."
    # O 'documento_relevante_anterior' é o despacho do juiz que ordenou a citação ou a manifestação do autor
    documento_relevante_anterior = estado.get("documento_gerado_na_etapa_recente", "Nenhum documento anterior relevante informado.")
    
    contexto_fatos_caso_rag_reu = "Contexto RAG do caso para o réu (simplificado)." 
    lista_documentos_juntados_pelo_reu_final = [] # Inicializa

    if etapa_atual_do_no == ETAPA_CONTESTACAO:
        peticao_inicial_autor_texto_completo = "Petição Inicial do Autor não encontrada no histórico para contestação." 
        # Tenta buscar a Petição Inicial completa do histórico
        for item_hist in reversed(estado.get("historico_completo", [])):
            if item_hist["etapa"] == ETAPA_PETICAO_INICIAL and item_hist["ator"] == ADVOGADO_AUTOR:
                peticao_inicial_autor_texto_completo = item_hist["documento"]
                break
        
        docs_modelo_contestacao = retriever.get_relevant_documents(query="modelo de contestação cível completa e bem fundamentada", filter={"source_type": "modelo_peticao"}) if retriever else []
        modelo_texto_guia = docs_modelo_contestacao[0].page_content if docs_modelo_contestacao else "ERRO RAG: Modelo de contestação não encontrado."
        
        template_prompt_contestacao = """
        Você é um Advogado do Réu experiente. Sua tarefa é elaborar uma Contestação completa e robusta.
        **Processo ID:** {id_processo}
        
        **Despacho Judicial Recebido (determinando a citação/contestação):**
        {despacho_judicial}
        
        **Petição Inicial do Autor (que originou esta contestação e pode mencionar documentos juntados pelo Autor):**
        {peticao_inicial_autor}
        
        **Contexto Adicional dos Fatos (perspectiva da defesa, se houver, via RAG):**
        {contexto_fatos_caso_rag}
        
        **Modelo/Guia de Contestação (RAG - use para estrutura, formalidades e teses defensivas comuns):**
        {modelo_texto_guia}
        
        **Histórico Processual Anterior:**
        {historico_formatado}
        ---
        Instruções para a Contestação:
        1. Analise cuidadosamente a Petição Inicial do Autor.
        2. Apresente as defesas processuais (preliminares), se houver (ex: incompetência, inépcia da inicial).
        3. No mérito, impugne especificamente os fatos narrados pelo Autor e os fundamentos jurídicos apresentados.
        4. Apresente a versão dos fatos sob a ótica do Réu e a fundamentação jurídica que ampara sua defesa.
        5. Formule os pedidos da contestação (ex: acolhimento das preliminares, improcedência dos pedidos do autor, condenação em custas e honorários).
        6. A contestação deve ser bem estruturada (endereçamento, qualificação das partes se necessário reiterar, fatos, preliminares, mérito, pedidos, fecho).

        Contestação:
        """
        chain_contestacao = criar_prompt_e_chain(template_prompt_contestacao)
        documento_gerado_principal = chain_contestacao.invoke({
            "id_processo": id_processo,
            "despacho_judicial": documento_relevante_anterior,
            "peticao_inicial_autor": peticao_inicial_autor_texto_completo,
            "contexto_fatos_caso_rag": contexto_fatos_caso_rag_reu,
            "modelo_texto_guia": modelo_texto_guia,
            "historico_formatado": historico_formatado
        })

        # --- SEGUNDA CHAMADA LLM: Gerar lista de documentos do Réu ---
        print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Contestação gerada. Solicitando lista de documentos do Réu...")
        fatos_gerais_caso = estado.get("dados_formulario_entrada", {}).get("fatos", "Fatos do caso não disponíveis.")
        
        prompt_docs_reu_template = """
        Com base na Petição Inicial do Autor, na Contestação do Réu recém-elaborada, e nos fatos gerais do caso, você deve listar de 2 a 4 documentos principais que o Réu provavelmente juntaria para dar suporte à sua defesa.
        Para cada documento, forneça o tipo e uma descrição MUITO SUCINTA (1 frase, máximo 20 palavras).

        Petição Inicial do Autor (Resumo/Contexto - o Réu teve acesso completo a ela):
        {peticao_inicial_autor_resumo} 

        Contestação do Réu (Completa - elaborada para este caso):
        {contestacao_texto_completa}

        Fatos Gerais do Caso (fornecidos no início da simulação):
        {fatos_do_caso}

        Sua resposta DEVE SER uma lista de strings, onde cada string representa um documento no formato: "Tipo do Documento: Descrição sucinta."
        Exemplo de formato de saída:
        Documento de Identidade do Réu: RG e CPF para qualificação nos autos.
        Contrato de Locação: Cópia do contrato que estabelece as obrigações das partes.
        Comprovantes de Pagamento: Recibos dos aluguéis dos meses X, Y e Z.

        Liste os documentos do Réu:
        """
        # Usaremos StrOutputParser e depois parsearemos a string.
        # Para JSON, precisaríamos de JsonOutputParser e um prompt que explicitamente peça JSON.
        
        chain_docs_reu = criar_prompt_e_chain(prompt_docs_reu_template)
        # Para o resumo da PI, podemos pegar os primeiros N caracteres.
        pi_resumo_para_prompt_docs = peticao_inicial_autor_texto_completo[:1000] + "..." if len(peticao_inicial_autor_texto_completo) > 1000 else peticao_inicial_autor_texto_completo
        
        resposta_docs_reu_str = chain_docs_reu.invoke({
            "peticao_inicial_autor_resumo": pi_resumo_para_prompt_docs,
            "contestacao_texto_completa": documento_gerado_principal,
            "fatos_do_caso": fatos_gerais_caso
        })
        
        # Parsear a string da resposta para uma lista de dicts
        lista_documentos_juntados_pelo_reu_final = []
        if resposta_docs_reu_str and resposta_docs_reu_str.strip():
            linhas_docs = resposta_docs_reu_str.strip().split('\n')
            for linha in linhas_docs:
                if ':' in linha:
                    partes = linha.split(':', 1)
                    tipo_doc = partes[0].strip()
                    desc_doc = partes[1].strip()
                    if tipo_doc and desc_doc: # Só adiciona se ambos existirem
                         lista_documentos_juntados_pelo_reu_final.append({"tipo": tipo_doc, "descricao": desc_doc})
        
        if not lista_documentos_juntados_pelo_reu_final:
            print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Não foi possível gerar ou parsear a lista de documentos do Réu.")
            lista_documentos_juntados_pelo_reu_final = [{"tipo": "Informação", "descricao": "A IA não especificou documentos para o réu nesta etapa ou houve falha no parsing."}]
        else:
            print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Documentos do Réu gerados: {lista_documentos_juntados_pelo_reu_final}")

        # Anexar a lista de documentos ao final da contestação (opcional, para o histórico)
        documentos_reu_texto_para_anexar = formatar_lista_documentos_para_prompt(lista_documentos_juntados_pelo_reu_final, "Réu")
        documento_gerado_principal += f"\n\n---\n{documentos_reu_texto_para_anexar}"
        
        # ---> INÍCIO: Análise de Sentimento da Contestação <---
        sentimento_contestacao_texto = "Não analisado"
        try:
            prompt_sentimento_contestacao = f"""
            Analise o tom e o sentimento predominante do seguinte texto jurídico (Contestação).
            Responda com uma única palavra ou expressão curta que melhor descreva o sentimento (ex: Assertivo, Conciliatório, Agressivo, Neutro, Persuasivo, Formal, Emocional, Confiante, Defensivo, Indignado, Colaborativo).
            Seja conciso.

            Texto da Contestação:
            {documento_gerado_principal[:3000]} # Limita para evitar estouro de token

            Sentimento Predominante:"""
            chain_sentimento_contestacao = criar_prompt_e_chain(prompt_sentimento_contestacao)
            sentimento_contestacao_texto = chain_sentimento_contestacao.invoke({}) # documento_gerado_principal está no prompt
            print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Sentimento da Contestação: {sentimento_contestacao_texto}")
        except Exception as e_sent:
            print(f"Erro ao analisar sentimento da Contestação: {e_sent}")
            sentimento_contestacao_texto = "Erro na análise"
        proximo_ator_logico = JUIZ


    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        decisao_saneamento_juiz = "Decisão de Saneamento não encontrada no histórico."
        manifestacao_autor_recente = estado.get("documento_gerado_na_etapa_recente", "Manifestação do Autor não encontrada no estado.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não definidos.")
        historico_completo_formatado_para_prompt = "\n".join([f"### Documento da Etapa: {item['etapa']} (Ator: {item['ator']})\n{item['documento']}\n---" for item in estado.get("historico_completo", [])])


        # Busca a decisão de saneamento no histórico para melhor contexto
        for item_hist in reversed(estado.get("historico_completo", [])):
            if item_hist["etapa"] == ETAPA_DECISAO_SANEAMENTO and item_hist["ator"] == JUIZ:
                decisao_saneamento_juiz = item_hist["documento"]
                break
        
        template_prompt_manifestacao_reu = f"""
        Você é o Advogado do Réu. O Juiz proferiu a Decisão de Saneamento e o Autor já se manifestou informando não ter mais provas a produzir.
        Seu cliente (Réu) também informou que não possui mais provas a produzir e deseja o julgamento antecipado da lide.

        **Processo ID:** {{id_processo}}

        **Decisão de Saneamento do Juiz (para referência):**
        {decisao_saneamento_juiz}

        **Manifestação do Autor Recebida:**
        {manifestacao_autor_recente}

        **Pontos Controvertidos Fixados na Decisão de Saneamento:**
        {pontos_controvertidos}

        **Histórico Processual Anterior (para contexto):**
        {historico_completo_formatado_para_prompt}

        **Instruções:**
        1. Redija uma petição de "Manifestação Sobre Provas (Réu)".
        2. Na petição, declare que o Réu também não tem outras provas a produzir, além daquelas já constantes nos autos (documentais, incluindo as da contestação).
        3. Requeira o julgamento do processo no estado em que se encontra (julgamento antecipado do mérito), reiterando o pedido do Autor nesse sentido.
        4. Mantenha a formalidade e praxe forense. Conclua com data e assinatura do advogado.

        Manifestação Sobre Provas (Réu):
        """
        chain_manifestacao_reu = criar_prompt_e_chain(template_prompt_manifestacao_reu)
        documento_gerado_principal = chain_manifestacao_reu.invoke({
            "id_processo": id_processo,
            # "historico_formatado": historico_formatado # Já incluído
        })
        proximo_ator_logico = JUIZ
        # A flag "manifestacao_reu_sem_provas" já é atualizada no bloco de retorno do estado.
        # Não há "lista_documentos_juntados_pelo_reu_final" nesta etapa, pois é só manifestação.
        # Se precisar garantir que não sobrescreva, pode-se fazer:
        lista_documentos_juntados_pelo_reu_final = estado.get("documentos_juntados_pelo_reu", [])
        
    else:
        print(f"AVISO: Lógica para etapa '{etapa_atual_do_no}' do {ADVOGADO_REU} não implementada.")
        documento_gerado_principal = f"Conteúdo para {ADVOGADO_REU} na etapa {etapa_atual_do_no}."
        proximo_ator_logico = JUIZ

    print(f"[{ADVOGADO_REU}-{etapa_atual_do_no}] Documento Principal Gerado (trecho): {documento_gerado_principal[:250]}...")
    
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_REU, "documento": documento_gerado_principal}
    historico_atualizado = estado.get("historico_completo", []) + [novo_historico_item]

    estado_retorno_parcial = {
        "nome_do_ultimo_no_executado": ADVOGADO_REU,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado_principal, # Contestação + lista de docs
        "historico_completo": historico_atualizado,
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU),
        "etapa_a_ser_executada_neste_turno": etapa_atual_do_no,
        "documentos_juntados_pelo_reu": lista_documentos_juntados_pelo_reu_final, # Salva a lista estruturada no estado
        "sentimento_contestacao": sentimento_contestacao_texto if etapa_atual_do_no == ETAPA_CONTESTACAO else estado.get("sentimento_contestacao"),
    }
    estado_final_retorno = {**estado, **estado_retorno_parcial}
    return estado_final_retorno

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


# --- 8-A. Funcionalidades Adicionais (Ementa, Sentimentos, Verificador) ---

def gerar_ementa_cnj_padrao(texto_sentenca: str, id_processo: str, llm_usado: ChatGoogleGenerativeAI) -> str:
    """
    Gera uma ementa para a sentença fornecida, seguindo o padrão da Recomendação CNJ 154/2024.
    """
    prompt_template_ementa = """
    Você é um especialista em direito e Diretor de Secretaria experiente, encarregado de gerar uma ementa para a seguinte sentença, seguindo RIGOROSAMENTE o padrão da Recomendação CNJ 154/2024 (EMENTA-PADRÃO).

    **SENTENÇA COMPLETA:**
    {texto_sentenca}

    **PADRÃO DA EMENTA (Recomendação CNJ 154/2024) A SER SEGUIDO:**
    Ementa: [Ramo do Direito]. [Classe processual]. [Frase ou palavras que indiquem o assunto principal]. [Conclusão].

    I. Caso em exame
    1. Apresentação do caso, com a indicação dos fatos relevantes, do pedido principal da ação ou do recurso e, se for o caso, da decisão recorrida.

    II. Questão em discussão
    2. A questão em discussão consiste em (...). / Há duas questões em discussão: (i) saber se (...); e (ii) saber se (...). (incluir todas as questões, com os seus respectivos fatos e fundamentos, utilizando-se de numeração em romano, letras minúsculas e entre parênteses).

    III. Razões de decidir
    3. Exposição do fundamento de maneira resumida (cada fundamento deve integrar um item).
    4. Exposição de outro fundamento de maneira resumida. (Adicionar mais itens conforme necessário, seguindo a numeração)

    IV. Dispositivo e tese
    5. Ex: Pedido procedente/improcedente. Recurso provido/desprovido.
    Tese de julgamento: frases objetivas das conclusões da decisão, ordenadas por numerais cardinais entre aspas e sem itálico. “1. [texto da tese]. 2. [texto da tese]” (quando houver tese).
    _________
    Dispositivos relevantes citados: ex.: CF/1988, art. 1º, III e IV; CC, arts. 1.641, II, e 1.639, § 2º.
    Jurisprudência relevante citada: ex.: STF, ADPF nº 130, Rel. Min. Ayres Britto, Plenário, j. 30.04.2009.

    **INSTRUÇÕES CRÍTICAS:**
    - Extraia as informações DIRETAMENTE da sentença fornecida. NÃO INVENTE informações.
    - Preencha TODAS as seções do padrão da ementa conforme especificado.
    - Seja fiel ao conteúdo e à terminologia da sentença.
    - Para "Ramo do Direito" e "Classe Processual", infira da sentença ou, se não explícito, deduza com base no conteúdo (ex: Direito Civil, Ação de Indenização por Danos Morais).
    - Mantenha a formatação EXATA, incluindo numeração, marcadores (I, II, III, IV), letras minúsculas entre parênteses para sub-itens de questões, e a linha "_________" antes dos dispositivos/jurisprudência citados.
    - Se uma seção não tiver conteúdo direto na sentença (ex: ausência de tese explícita), indique "Não consta expressamente na sentença." ou similar, mas tente ao máximo extrair ou inferir.

    Responda APENAS com a ementa formatada.

    **EMENTA GERADA (no padrão CNJ):**
    """
    chain_ementa = criar_prompt_e_chain(prompt_template_ementa) # Reutiliza sua função existente
    try:
        ementa_gerada = chain_ementa.invoke({
            "texto_sentenca": texto_sentenca,
            # "id_processo": id_processo # Pode ser útil para o LLM ter o ID, mas não é estritamente parte do template CNJ
        })
        return ementa_gerada
    except Exception as e:
        print(f"Erro ao gerar ementa CNJ: {e}")
        return f"Erro ao gerar ementa: {e}"

search_tool = None
if GOOGLE_API_KEY_SEARCH and GOOGLE_CSE_ID:
    try:
        # 1. Crie a instância do API Wrapper
        search_api_wrapper_instance = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY_SEARCH,
            google_cse_id=GOOGLE_CSE_ID
        )

        # 2. Crie a instância da ferramenta GoogleSearchRun,
        #    passando o wrapper, e atribua à sua variável
        search_tool = GoogleSearchRun( # Esta é a CLASSE GoogleSearchRun
            api_wrapper=search_api_wrapper_instance
            # Você pode adicionar outros parâmetros opcionais aqui, como 'description'
        )
        print("[INFO] Ferramenta Google Search_tool (instância de GoogleSearchRun) configurada com sucesso.")

    except Exception as e_config_tool:
        import traceback
        print(f"[AVISO] Falha ao configurar search_tool: {e_config_tool}")
        print(traceback.format_exc())
        search_tool = None
else:
    print("[AVISO] Chaves de API para busca não definidas. Ferramenta search_tool desabilitada.")


def verificar_sentenca_com_jurisprudencia(texto_sentenca: str, llm_usado: ChatGoogleGenerativeAI) -> str:
    """
    Verifica a sentença comparando-a com jurisprudência encontrada via Google Search.
    """
    if not search_tool:
        return "Ferramenta de busca Google não está configurada ou disponível. Verifique as chaves GOOGLE_API_KEY_SEARCH e GOOGLE_CSE_ID."

    # 1. Extrair teses/palavras-chave da sentença para busca
    prompt_extracao_teses = f"""
    Dada a seguinte sentença, extraia 2-3 teses jurídicas centrais ou os principais pontos de direito decididos.
    Formate cada tese como uma frase curta e objetiva, ideal para uma busca de jurisprudência.
    Se a sentença for complexa, foque nos pontos que seriam mais controversos ou relevantes para pesquisa jurisprudencial.
    Responda com cada tese em uma nova linha.

    Sentença (trecho inicial para identificação, o conteúdo completo foi analisado internamente):
    {texto_sentenca[:1500]}

    Teses/Palavras-chave para Busca (uma por linha):
    """
    chain_extracao = criar_prompt_e_chain(prompt_extracao_teses)
    try:
        teses_str = chain_extracao.invoke({"texto_sentenca": texto_sentenca}) # Passa o texto completo aqui
        teses_para_busca = [t.strip() for t in teses_str.split('\n') if t.strip()]
        if not teses_para_busca:
            return "Não foi possível extrair teses da sentença para a busca."
        print(f"Teses extraídas para busca: {teses_para_busca}")
    except Exception as e:
        return f"Erro ao extrair teses da sentença: {e}"

    # 2. Buscar jurisprudência para cada tese
    todos_resultados_busca = []
    with st.spinner(f"Buscando jurisprudência para {len(teses_para_busca)} tese(s)..."):
        for i, tese in enumerate(teses_para_busca):
            if not tese: continue
            st.write(f"Buscando por: '{tese}'...")
            try:
                # Adicionar termos como "jurisprudência", "acórdão" pode refinar a busca
                query_busca = f'jurisprudência {tese}' #  OR ementa {tese}
                resultados_tese_str = search_tool.invoke(query_busca)
                todos_resultados_busca.append(f"Resultados da busca para '{tese}':\n{resultados_tese_str}\n---\n")
                st.write(f"Resultados parciais para '{tese}' obtidos.")
            except Exception as e_busca:
                todos_resultados_busca.append(f"Erro ao buscar por '{tese}': {e_busca}\n---\n")
                st.warning(f"Erro na busca por '{tese}': {e_busca}")
        st.success("Busca de jurisprudência concluída.")

    snippets_jurisprudencia_formatados = "\n".join(todos_resultados_busca)
    if not snippets_jurisprudencia_formatados.strip():
        snippets_jurisprudencia_formatados = "Nenhum resultado encontrado nas buscas."

    # 3. Análise comparativa pelo LLM
    prompt_analise_sentenca = f"""
    Você é um jurista sênior analisando uma sentença judicial à luz da jurisprudência encontrada.

    **SENTENÇA ORIGINAL (Trechos mais relevantes ou resumo):**
    (Considere os seguintes pontos principais da sentença que foi proferida no caso simulado)
    {teses_str}
    (Fim dos trechos da sentença)

    **JURISPRUDÊNCIA ENCONTRADA (Snippets e Resumos de Buscas):**
    {snippets_jurisprudencia_formatados}

    **Tarefa:**
    Com base EXCLUSIVAMENTE na jurisprudência fornecida acima, avalie se as teses principais da sentença original parecem estar, em termos gerais, alinhadas ou desalinhadas com essa jurisprudência.
    Seja cauteloso e objetivo. Se a jurisprudência não for clara, suficiente ou diretamente aplicável, afirme isso.

    **Formato da Resposta:**
    1.  **Avaliação Geral:** (Ex: "Alinhada com a jurisprudência apresentada.", "Aparentemente desalinhada em relação a X.", "Parcialmente alinhada.", "Jurisprudência insuficiente para uma conclusão definitiva.")
    2.  **Justificativa Sucinta:** (Explique brevemente, apontando pontos de convergência ou divergência com base nos trechos da jurisprudência, ou a dificuldade de comparação.)
    3.  **Observação:** Lembre-se que esta é uma análise preliminar baseada em snippets de busca.

    **Análise da Sentença vs. Jurisprudência:**
    """
    chain_analise_final = criar_prompt_e_chain(prompt_analise_sentenca)
    try:
        analise_final = chain_analise_final.invoke({}) # Contexto já está no prompt
        return analise_final
    except Exception as e:
        return f"Erro ao realizar análise comparativa da sentença: {e}"


# --- 9. Interface Streamlit e Lógica de Execução ---

# --- Constantes e Funções da UI Streamlit ---
# MODIFICADO: Nova ordem e nova etapa
FORM_STEPS = [
    "autor", 
    "reu", 
    "fatos", 
    "direito", 
    "pedidos", 
    "natureza_acao",      # Movido para após "pedidos"
    "documentos_autor",   # Nova etapa adicionada
    "revisar_e_simular"
]

# NOVO: Lista de tipos de documentos
TIPOS_DOCUMENTOS_COMUNS = [
    "Nenhum (Apenas Descrição Factual)", # Opção para quando não há um "documento" formal
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


def inicializar_estado_formulario():
    if 'current_form_step_index' not in st.session_state:
        st.session_state.current_form_step_index = 0
    
    # Campos que devem ser inicializados/resetados para um novo formulário
    default_form_data = {
        "id_processo": f"caso_sim_{int(time.time())}",
        "qualificacao_autor": "", "qualificacao_reu": "", 
        "fatos": "", "fundamentacao_juridica": "", "pedidos": "",
        "natureza_acao": "", 
        "documentos_autor": [] # Lista para armazenar os documentos do autor
    }
    default_ia_flags = {key: False for key in default_form_data.keys()}
    default_ia_flags["documentos_autor_descricoes"] = {} # Flags para descrições individuais

    if 'form_data' not in st.session_state:
        st.session_state.form_data = default_form_data.copy()
    else: # Garante que todos os campos existem, útil se novos campos são adicionados posteriormente
        for key, value in default_form_data.items():
            if key not in st.session_state.form_data:
                st.session_state.form_data[key] = value
        if "documentos_autor" not in st.session_state.form_data: # Caso específico para a lista
             st.session_state.form_data["documentos_autor"] = []


    if 'ia_generated_content_flags' not in st.session_state:
        st.session_state.ia_generated_content_flags = default_ia_flags.copy()
    else:
        for key, value in default_ia_flags.items():
            if key not in st.session_state.ia_generated_content_flags:
                st.session_state.ia_generated_content_flags[key] = value


    if 'num_documentos_autor' not in st.session_state: # Para a UI de documentos_autor
        st.session_state.num_documentos_autor = 0 
        # Será ajustado para 1 na primeira vez que exibir_formulario_documentos_autor for chamado, se a lista estiver vazia.

    # Outros estados da UI
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    if 'doc_visualizado' not in st.session_state:
        st.session_state.doc_visualizado = None
    if 'doc_visualizado_titulo' not in st.session_state:
        st.session_state.doc_visualizado_titulo = ""

    # NOVOS ESTADOS DA UI PARA FUNCIONALIDADES ADICIONAIS
    if 'ementa_cnj_gerada' not in st.session_state:
        st.session_state.ementa_cnj_gerada = None
    if 'verificacao_sentenca_resultado' not in st.session_state:
        st.session_state.verificacao_sentenca_resultado = None
    if 'show_ementa_popup' not in st.session_state: # Para controlar popup/modal
        st.session_state.show_ementa_popup = False
    if 'show_verificacao_popup' not in st.session_state:
        st.session_state.show_verificacao_popup = False

def gerar_conteudo_com_ia(prompt_template_str: str, campos_prompt: dict, campo_formulario_display: str, chave_estado: str, sub_chave_lista: Union[str, int, None] = None, indice_lista: Union[int, None] = None):
    """
    Gera conteúdo com IA.
    Se sub_chave_lista e indice_lista são fornecidos, atualiza um item dentro de uma lista de dicionários em form_data.
    Ex: form_data['documentos_autor'][indice_lista][sub_chave_lista] = conteudo_gerado
    Caso contrário, atualiza form_data[chave_estado] diretamente.
    """
    if not GOOGLE_API_KEY:
        st.error("A chave API do Google não foi configurada. Não é possível usar a IA.")
        return
    try:
        with st.spinner(f"Gerando conteúdo para '{campo_formulario_display}' com IA..."):
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | llm | StrOutputParser()
            conteudo_gerado = chain.invoke(campos_prompt)
            
            if sub_chave_lista is not None and indice_lista is not None and chave_estado == "documentos_autor":
                # Garante que a lista e o dicionário no índice existem
                while len(st.session_state.form_data["documentos_autor"]) <= indice_lista:
                    st.session_state.form_data["documentos_autor"].append({})
                st.session_state.form_data["documentos_autor"][indice_lista][sub_chave_lista] = conteudo_gerado
                st.session_state.ia_generated_content_flags.setdefault("documentos_autor_descricoes", {})[f"doc_{indice_lista}"] = True
            else:
                st.session_state.form_data[chave_estado] = conteudo_gerado
                st.session_state.ia_generated_content_flags[chave_estado] = True
        st.rerun()
    except Exception as e:
        st.error(f"Erro ao gerar conteúdo com IA para '{campo_formulario_display}': {e}")

# --- Funções de Exibição dos Formulários (Atualizadas e Novas) ---

def exibir_formulario_qualificacao_autor():
    idx_etapa = FORM_STEPS.index("autor")
    st.subheader(f"{idx_etapa + 1}. Qualificação do Autor")
    # ... (restante da função como na última versão fornecida, apenas ajustando o botão "Próximo" se necessário)
    # O "Próximo" de "autor" vai para "reu", que é o próximo em FORM_STEPS, então a lógica de +=1 no índice funciona.
    with st.form("form_autor"):
        st.session_state.form_data["qualificacao_autor"] = st.text_area(
            "Qualificação Completa do Autor", value=st.session_state.form_data.get("qualificacao_autor", ""),
            height=150, key="autor_q_text_area_final",
            help="Ex: Nome completo, nacionalidade, estado civil, profissão, RG, CPF, endereço com CEP, e-mail."
        )
        col1, col2 = st.columns([1,5])
        with col1: submetido = st.form_submit_button("Próximo (Réu) ➡")
        with col2:
            if st.form_submit_button("Autopreencher com IA (Dados Fictícios)"):
                prompt_str = "Gere uma qualificação completa fictícia para um autor de uma ação judicial (nome completo, nacionalidade, estado civil, profissão, RG, CPF, endereço completo com CEP e e-mail)."
                gerar_conteudo_com_ia(prompt_str, {}, "Qualificação do Autor", "qualificacao_autor")
        
        if st.session_state.ia_generated_content_flags.get("qualificacao_autor"):
            st.caption("📝 Conteúdo preenchido por IA. Revise e ajuste.")
        
        if submetido:
            if st.session_state.form_data.get("qualificacao_autor","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Preencha a qualificação do autor.")

def exibir_formulario_qualificacao_reu():
    idx_etapa = FORM_STEPS.index("reu")
    st.subheader(f"{idx_etapa + 1}. Qualificação do Réu")
    # O "Voltar" de "reu" vai para "autor". O "Próximo" vai para "fatos".
    with st.form("form_reu"):
        st.session_state.form_data["qualificacao_reu"] = st.text_area(
            "Qualificação Completa do Réu", value=st.session_state.form_data.get("qualificacao_reu", ""),
            height=150, key="reu_q_text_area_final",
            help="Ex: Nome/Razão Social, CPF/CNPJ, endereço com CEP, e-mail (se pessoa física ou jurídica)."
        )
        col1, col2, col3 = st.columns([1,1,4]) 
        with col1:
            if st.form_submit_button("⬅ Voltar (Autor)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("autor")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Fatos) ➡")
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

def exibir_formulario_fatos():
    idx_etapa = FORM_STEPS.index("fatos")
    st.subheader(f"{idx_etapa + 1}. Descrição dos Fatos") # Título atualizado
    # O "Voltar" de "fatos" vai para "reu". O "Próximo" vai para "direito".
    with st.form("form_fatos"):
        st.session_state.form_data["fatos"] = st.text_area(
            "Descreva os Fatos de forma clara e cronológica", value=st.session_state.form_data.get("fatos", ""),
            height=300, key="fatos_text_area_final",
            help="Relate os acontecimentos que deram origem à disputa, incluindo datas (mesmo que aproximadas), locais e pessoas envolvidas."
        )
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Réu)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("reu")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Direito) ➡")
        with col3:
            if st.form_submit_button("Gerar Fatos com IA (para um caso fictício)"):
                # Natureza da ação ainda não foi definida, então o prompt é mais genérico ou pode pedir um tipo de caso.
                # Para simplificar, vamos manter um gerador de fatos mais genérico.
                prompt_str = ("Elabore uma narrativa de fatos (2-4 parágrafos) para um caso judicial cível fictício comum (ex: cobrança, dano moral simples, acidente de trânsito leve). "
                              "Inclua elementos essenciais, datas aproximadas fictícias (ex: 'em meados de janeiro de 2023'), e o problema central. Use 'o Autor' e 'o Réu' para se referir às partes.\nDescrição dos Fatos:")
                gerar_conteudo_com_ia(prompt_str, {}, "Descrição dos Fatos", "fatos")

        if st.session_state.ia_generated_content_flags.get("fatos"):
            st.caption("📝 Conteúdo gerado por IA. Revise e detalhe conforme o caso real.")

        if submetido:
            if st.session_state.form_data.get("fatos","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Descreva os fatos.")

def exibir_formulario_direito():
    idx_etapa = FORM_STEPS.index("direito")
    st.subheader(f"{idx_etapa + 1}. Fundamentação Jurídica (Do Direito)")
    # O "Voltar" de "direito" vai para "fatos". O "Próximo" vai para "pedidos".
    with st.form("form_direito"):
        st.session_state.form_data["fundamentacao_juridica"] = st.text_area(
            "Insira a fundamentação jurídica aplicável ao caso", value=st.session_state.form_data.get("fundamentacao_juridica", ""),
            height=300, key="direito_text_area_final",
            help="Cite os artigos de lei, súmulas, jurisprudência e princípios jurídicos que amparam a sua pretensão, explicando a conexão com os fatos."
        )
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Fatos)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("fatos")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Pedidos) ➡")
        with col3:
            if st.form_submit_button("Sugerir Fundamentação com IA (baseado nos fatos)"):
                fatos_informados = st.session_state.form_data.get("fatos","Fatos não informados para contextualizar a fundamentação do direito.")
                # Natureza da ação ainda não foi definida aqui.
                prompt_str = ("Analise os Fatos: \n{fatos_informados}\n\n"
                              "Com base nisso, elabore uma seção 'DO DIREITO' para uma petição inicial. "
                              "Sugira institutos jurídicos aplicáveis, cite artigos de lei relevantes (ex: Código Civil, CDC, Constituição Federal), e explique brevemente como se aplicam aos fatos para justificar os pedidos que seriam feitos. "
                              "Estruture em parágrafos.\nFundamentação Jurídica Sugerida:")
                gerar_conteudo_com_ia(prompt_str, {"fatos_informados": fatos_informados}, "Fundamentação Jurídica", "fundamentacao_juridica")
        
        if st.session_state.ia_generated_content_flags.get("fundamentacao_juridica"):
            st.caption("📝 Conteúdo sugerido por IA. Revise, valide e complemente com referências específicas.")

        if submetido:
            if st.session_state.form_data.get("fundamentacao_juridica","").strip():
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Insira a fundamentação jurídica.")

def exibir_formulario_pedidos():
    idx_etapa = FORM_STEPS.index("pedidos")
    st.subheader(f"{idx_etapa + 1}. Pedidos")
    # O "Voltar" de "pedidos" vai para "direito". O "Próximo" vai para "natureza_acao".
    with st.form("form_pedidos"):
        st.session_state.form_data["pedidos"] = st.text_area(
            "Insira os pedidos da ação de forma clara e objetiva", value=st.session_state.form_data.get("pedidos", ""),
            height=300, key="pedidos_text_area_final",
            help="Liste os requerimentos finais ao juiz. Ex: citação do réu, procedência da ação para condenar o réu a..., condenação em custas e honorários. Use alíneas (a, b, c...)."
        )
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Direito)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("direito")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Natureza da Ação) ➡") # MODIFICADO: Próximo é Natureza da Ação
        with col3:
            if st.form_submit_button("Sugerir Pedidos com IA (baseado nos fatos e direito)"):
                # Natureza da ação ainda não foi definida aqui.
                fatos_informados_trecho = st.session_state.form_data.get("fatos","")[:300] 
                direito_informado_trecho = st.session_state.form_data.get("fundamentacao_juridica","")[:300]
                prompt_str = ("Com base um resumo dos Fatos ('{fatos_informados_trecho}...') e um resumo do Direito ('{direito_informado_trecho}...'), "
                              "elabore uma lista de pedidos típicos para uma petição inicial. Inclua pedidos como: citação do réu, procedência do pedido principal (seja específico se possível, ex: 'condenar o réu ao pagamento de X'), "
                              "condenação em custas processuais e honorários advocatícios. Formate os pedidos usando alíneas (a), (b), (c), etc.\nPedidos Sugeridos:")
                gerar_conteudo_com_ia(prompt_str, {
                    "fatos_informados_trecho": fatos_informados_trecho,
                    "direito_informado_trecho": direito_informado_trecho
                }, "Pedidos", "pedidos")

        if st.session_state.ia_generated_content_flags.get("pedidos"):
            st.caption("📝 Conteúdo sugerido por IA. Revise e ajuste conforme a especificidade do caso.")

        if submetido:
            if st.session_state.form_data.get("pedidos","").strip():
                st.session_state.current_form_step_index += 1 # Vai para natureza_acao
                st.rerun()
            else: st.warning("Insira os pedidos.")

# Função para Natureza da Ação (agora após Pedidos)
def exibir_formulario_natureza_acao():
    idx_etapa = FORM_STEPS.index('natureza_acao')
    st.subheader(f"{idx_etapa + 1}. Definição da Natureza da Ação")
    with st.form("form_natureza_final"): # Chave do formulário atualizada
        fatos_contexto = st.session_state.form_data.get("fatos", "Fatos não fornecidos.")
        direito_contexto = st.session_state.form_data.get("fundamentacao_juridica", "Fundamentação não fornecida.")
        pedidos_contexto = st.session_state.form_data.get("pedidos", "Pedidos não fornecidos.")

        st.info("Com base nos fatos, direito e pedidos que você informou, a IA pode sugerir a natureza técnica da ação.")
        
        with st.expander("Revisar Contexto para IA (Fatos, Direito, Pedidos)", expanded=False):
            st.text_area("Fatos (Resumo)", value=fatos_contexto[:500] + ("..." if len(fatos_contexto)>500 else ""), height=100, disabled=True, key="natureza_fatos_ctx_final")
            st.text_area("Direito (Resumo)", value=direito_contexto[:500] + ("..." if len(direito_contexto)>500 else ""), height=100, disabled=True, key="natureza_direito_ctx_final")
            st.text_area("Pedidos (Resumo)", value=pedidos_contexto[:500] + ("..." if len(pedidos_contexto)>500 else ""), height=100, disabled=True, key="natureza_pedidos_ctx_final")

        st.session_state.form_data["natureza_acao"] = st.text_input(
            "Natureza da Ação (Ex: Ação de Indenização por Danos Morais c/c Danos Materiais)",
            value=st.session_state.form_data.get("natureza_acao", ""), 
            key="natureza_acao_text_input_final_val",
            help="A IA pode sugerir. Refine ou altere conforme necessário."
        )
        
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Pedidos)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("pedidos")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Documentos) ➡") # MODIFICADO: Próximo é Documentos do Autor
        with col3:
            if st.form_submit_button("✨ Sugerir Natureza da Ação com IA"):
                prompt_str = (
                    "Você é um jurista experiente. Com base nos seguintes elementos de um caso:\n"
                    "FATOS:\n{fatos_completos}\n\n"
                    "FUNDAMENTAÇÃO JURÍDICA:\n{direito_completo}\n\n"
                    "PEDIDOS:\n{pedidos_completos}\n\n"
                    "Sugira o 'nomen iuris' (natureza da ação) mais adequado e técnico para este caso. "
                    "Seja específico e, se aplicável, mencione cumulações (c/c). Exemplos: 'Ação de Cobrança pelo Rito Comum', 'Ação de Indenização por Danos Morais e Materiais', "
                    "'Ação Declaratória de Inexistência de Débito c/c Repetição de Indébito e Indenização por Danos Morais'."
                    "\nNatureza da Ação Sugerida:"
                )
                gerar_conteudo_com_ia(
                    prompt_str, 
                    {
                        "fatos_completos": fatos_contexto,
                        "direito_completo": direito_contexto,
                        "pedidos_completos": pedidos_contexto
                    }, 
                    "Natureza da Ação", 
                    "natureza_acao"
                )
        
        if st.session_state.ia_generated_content_flags.get("natureza_acao"):
            st.caption("📝 Conteúdo sugerido por IA. Revise e ajuste para precisão técnica.")

        if submetido:
            if st.session_state.form_data.get("natureza_acao","").strip():
                st.session_state.current_form_step_index += 1 # Avança para 'documentos_autor'
                st.rerun()
            else: st.warning("Defina a natureza da ação ou peça uma sugestão à IA.")

# Para Documentos do Autor
def exibir_formulario_documentos_autor():
    idx_etapa = FORM_STEPS.index('documentos_autor')
    st.subheader(f"{idx_etapa + 1}. Documentos Juntados pelo Autor com a Petição Inicial")
    st.markdown("Liste os principais documentos que o Autor juntaria. A IA pode ajudar a gerar descrições sucintas (1-2 frases).")

    if "documentos_autor" not in st.session_state.form_data: # Inicialização defensiva
        st.session_state.form_data["documentos_autor"] = []
    if "num_documentos_autor" not in st.session_state or st.session_state.num_documentos_autor < 0:
         st.session_state.num_documentos_autor = 0


    if st.session_state.num_documentos_autor == 0:
        st.info("Nenhum documento adicionado. Clique em 'Adicionar Documento' para começar ou prossiga se não houver documentos a listar.")
    
    # Garante que a lista 'documentos_autor' em form_data tenha o mesmo número de elementos que 'num_documentos_autor'
    # Adiciona dicts vazios se num_documentos_autor for maior
    while len(st.session_state.form_data["documentos_autor"]) < st.session_state.num_documentos_autor:
        st.session_state.form_data["documentos_autor"].append({"tipo": TIPOS_DOCUMENTOS_COMUNS[0], "descricao": ""})
    # Remove excesso se num_documentos_autor diminuiu (exceto pelo botão de remover)
    if len(st.session_state.form_data["documentos_autor"]) > st.session_state.num_documentos_autor:
         st.session_state.form_data["documentos_autor"] = st.session_state.form_data["documentos_autor"][:st.session_state.num_documentos_autor]


    for i in range(st.session_state.num_documentos_autor):
        with st.expander(f"Documento {i+1}: {st.session_state.form_data['documentos_autor'][i].get('tipo', 'Novo Documento')}", expanded=True):
            doc_atual_ref = st.session_state.form_data["documentos_autor"][i] # Referência para facilitar

            cols_doc = st.columns([3, 6]) 
            doc_atual_ref["tipo"] = cols_doc[0].selectbox(
                f"Tipo do Documento {i+1}", options=TIPOS_DOCUMENTOS_COMUNS, 
                index=TIPOS_DOCUMENTOS_COMUNS.index(doc_atual_ref.get("tipo", TIPOS_DOCUMENTOS_COMUNS[0])),
                key=f"doc_autor_tipo_{i}_final"
            )
            
            doc_atual_ref["descricao"] = cols_doc[1].text_area(
                f"Descrição/Conteúdo Sucinto do Documento {i+1}", 
                value=doc_atual_ref.get("descricao", ""), 
                key=f"doc_autor_desc_{i}_final", height=100,
                help="Ex: 'Contrato de aluguel datado de 01/01/2022, assinado por X e Y, estabelecendo aluguel de R$ Z.' OU 'RG do autor, Sr. A, CPF nº B, comprovando seus dados de qualificação.'"
            )
            
            if st.button(f"✨ Gerar Descrição IA para Doc. {i+1}", key=f"doc_autor_ia_btn_{i}_final"):
                tipo_selecionado = doc_atual_ref["tipo"]
                fatos_contexto = st.session_state.form_data.get("fatos", "Contexto factual não disponível.")
                prompt_desc_doc = (
                    f"Você é um assistente jurídico. Para um documento do tipo '{tipo_selecionado}' que será juntado por um Autor em uma ação judicial, "
                    f"gere uma descrição MUITO SUCINTA (1 a 2 frases curtas, máximo 30 palavras) sobre seu conteúdo e propósito. "
                    f"Contexto dos fatos do caso (resumido): '{fatos_contexto[:300]}...'. "
                    "Exemplos de descrição:\n"
                    "- Tipo: Contrato - Descrição: 'Contrato de prestação de serviços de consultoria, assinado em 10/03/2023 entre Autor e Réu, detalhando o escopo e valor dos serviços.'\n"
                    "- Tipo: Documento de Identidade - Descrição: 'RG e CPF do Autor, Fulano de Tal, para fins de sua correta qualificação nos autos.'\n"
                    "- Tipo: Prints de WhatsApp - Descrição: 'Capturas de tela de conversas via WhatsApp entre Autor e Réu, datadas de 05/04/2023 a 10/04/2023, evidenciando a negociação do acordo X.'\n"
                    "\nDescrição Sucinta:"
                )
                gerar_conteudo_com_ia(
                    prompt_desc_doc, 
                    {}, # Campos do prompt já estão no template_string
                    f"Descrição do Documento {i+1} ({tipo_selecionado})", 
                    "documentos_autor", # Chave principal da lista
                    sub_chave_lista="descricao", # Subchave a ser atualizada no dict da lista
                    indice_lista=i # Índice na lista documentos_autor
                )
            
            if st.session_state.ia_generated_content_flags.get("documentos_autor_descricoes", {}).get(f"doc_{i}"):
                st.caption("📝 Descrição gerada/sugerida por IA. Revise.")
    
    st.markdown("---")
    col_botoes_add_rem_1, col_botoes_add_rem_2 = st.columns(2)
    if col_botoes_add_rem_1.button("➕ Adicionar Documento", key="add_doc_autor_btn_final", help="Adiciona um novo campo para listar um documento."):
        st.session_state.num_documentos_autor += 1
        # O loop acima já garante que form_data["documentos_autor"] será estendido se necessário no rerun
        st.rerun()

    if st.session_state.num_documentos_autor > 0:
        if col_botoes_add_rem_2.button("➖ Remover Último Documento", key="rem_doc_autor_btn_final", help="Remove o último campo de documento da lista."):
            st.session_state.num_documentos_autor -= 1
            if st.session_state.form_data["documentos_autor"]: # Garante que não tentará pop de lista vazia
                st.session_state.form_data["documentos_autor"].pop()
            # Limpa flag de IA se existir para o índice removido
            if f"doc_{st.session_state.num_documentos_autor}" in st.session_state.ia_generated_content_flags.get("documentos_autor_descricoes", {}):
                del st.session_state.ia_generated_content_flags["documentos_autor_descricoes"][f"doc_{st.session_state.num_documentos_autor}"]
            st.rerun()
    st.markdown("---")

    with st.form("form_documentos_autor_nav_final"):
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.form_submit_button("⬅ Voltar (Natureza da Ação)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("natureza_acao")
                st.rerun()
        with col_nav2:
            if st.form_submit_button("Próximo (Revisar e Simular) ➡"):
                documentos_validos = True
                # Remove documentos vazios (sem tipo definido como não sendo "Nenhum..." e sem descrição) antes de validar
                docs_filtrados = []
                for doc_item in st.session_state.form_data.get("documentos_autor", []):
                    tipo_valido = doc_item.get("tipo") != TIPOS_DOCUMENTOS_COMUNS[0] # Não é "Nenhum..."
                    descricao_presente = doc_item.get("descricao","").strip()
                    if tipo_valido and descricao_presente: # Se tem tipo (que não "Nenhum") E descrição
                        docs_filtrados.append(doc_item)
                    elif not tipo_valido and descricao_presente: # Se é "Nenhum..." mas tem descrição (descrição factual)
                         docs_filtrados.append({"tipo": TIPOS_DOCUMENTOS_COMUNS[0], "descricao": descricao_presente})
                    # Se for "Nenhum" e sem descrição, ou tipo válido mas sem descrição, será ignorado/removido
                
                st.session_state.form_data["documentos_autor"] = docs_filtrados
                st.session_state.num_documentos_autor = len(docs_filtrados) # Atualiza o contador

                # Validação após limpeza: se ainda há documentos, devem ter tipo e descrição.
                # Esta validação já foi mais ou menos feita pela filtragem.
                # Apenas para garantir, mas a lógica de filtragem acima deve ser suficiente.
                # for idx, doc in enumerate(st.session_state.form_data["documentos_autor"]):
                #     if not doc.get("tipo") or (doc.get("tipo") != TIPOS_DOCUMENTOS_COMUNS[0] and not doc.get("descricao","").strip()):
                #         st.warning(f"Documento {idx+1} ('{doc.get('tipo')}') parece incompleto. Verifique tipo e descrição ou remova.")
                #         documentos_validos = False
                #         break
                
                if documentos_validos: # Se passou na validação (ou se a filtragem é suficiente)
                    st.session_state.current_form_step_index += 1 
                    st.rerun()


def exibir_revisao_e_iniciar_simulacao():
    idx_etapa = FORM_STEPS.index('revisar_e_simular')
    st.subheader(f"{idx_etapa + 1}. Revisar Dados e Iniciar Simulação")
    form_data_local = st.session_state.form_data
    st.info(f"**ID do Processo (Gerado):** `{form_data_local.get('id_processo', 'N/A')}`")

    with st.expander("Qualificação do Autor", expanded=False): 
        st.text_area("Revisão - Autor", value=form_data_local.get("qualificacao_autor", "Não preenchido"), height=100, disabled=True, key="rev_autor_area_final_2")
    with st.expander("Qualificação do Réu", expanded=False): 
        st.text_area("Revisão - Réu", value=form_data_local.get("qualificacao_reu", "Não preenchido"), height=100, disabled=True, key="rev_reu_area_final_2")
    with st.expander("Fatos", expanded=True): 
        st.text_area("Revisão - Fatos", value=form_data_local.get("fatos", "Não preenchido"), height=200, disabled=True, key="rev_fatos_area_final_2")
    with st.expander("Fundamentação Jurídica", expanded=False): 
        st.text_area("Revisão - Direito", value=form_data_local.get("fundamentacao_juridica", "Não preenchido"), height=200, disabled=True, key="rev_dir_area_final_2")
    with st.expander("Pedidos", expanded=False): 
        st.text_area("Revisão - Pedidos", value=form_data_local.get("pedidos", "Não preenchido"), height=200, disabled=True, key="rev_ped_area_final_2")
    with st.expander("Natureza da Ação", expanded=False): 
        st.text_input("Revisão - Natureza da Ação", value=form_data_local.get("natureza_acao", "Não preenchido"), disabled=True, key="rev_nat_input_final_2")
    
    # MODIFICADO: Seção para revisar documentos do autor
    with st.expander("Documentos Juntados pelo Autor", expanded=True):
        documentos_autor_revisao = form_data_local.get("documentos_autor", [])
        if documentos_autor_revisao:
            for i, doc in enumerate(documentos_autor_revisao):
                st.markdown(f"**Documento {i+1}:** {doc.get('tipo', 'N/A')}")
                st.text_area(f"Descrição Doc. {i+1}", value=doc.get('descricao', 'Sem descrição'), height=75, disabled=True, key=f"rev_doc_autor_{i}_final")
        else:
            st.write("Nenhum documento foi listado pelo autor.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Voltar (Documentos do Autor)", use_container_width=True): # MODIFICADO: Botão Voltar
            st.session_state.current_form_step_index = FORM_STEPS.index("documentos_autor")
            st.rerun()
    with col2:
        campos_obrigatorios = ["qualificacao_autor", "qualificacao_reu", "natureza_acao", "fatos", "fundamentacao_juridica", "pedidos"]
        todos_preenchidos = all(form_data_local.get(campo, "").strip() for campo in campos_obrigatorios)
        
        if st.button("🚀 Iniciar Simulação com estes Dados", type="primary", disabled=not todos_preenchidos, use_container_width=True):
            st.session_state.simulation_running = True
            current_pid = form_data_local.get('id_processo')
            # Limpa resultados para este ID se já existirem, para forçar nova simulação
            if current_pid in st.session_state.get('simulation_results', {}):
                 del st.session_state.simulation_results[current_pid] 
            st.rerun()
        elif not todos_preenchidos:
            st.warning("Campos essenciais (Autor, Réu, Fatos, Direito, Pedidos, Natureza da Ação) devem ser preenchidos.")

def rodar_simulacao_principal(dados_coletados: dict):
    st.markdown(f"--- INICIANDO SIMULAÇÃO PARA O CASO: **{dados_coletados.get('id_processo','N/A')}** ---")
    
    if not dados_coletados or not dados_coletados.get('id_processo'):
        st.error("Erro: Dados do caso incompletos para iniciar a simulação.")
        st.session_state.simulation_running = False
        if st.button("Retornar ao formulário"): st.rerun()
        return

    # MODIFICADO: Incluir documentos do autor no texto do processo
    documentos_autor_formatado = "\n\n--- Documentos Juntados pelo Autor (conforme informado no formulário) ---\n"
    docs_autor_lista = dados_coletados.get("documentos_autor", [])
    if docs_autor_lista:
        for i, doc in enumerate(docs_autor_lista):
            documentos_autor_formatado += f"{i+1}. Tipo: {doc.get('tipo', 'N/A')}\n   Descrição/Propósito: {doc.get('descricao', 'N/A')}\n"
    else:
        documentos_autor_formatado += "Nenhum documento específico foi listado pelo autor no formulário inicial para esta simulação.\n"

    conteudo_processo_texto = f"""
ID do Processo: {dados_coletados.get('id_processo')}
Qualificação do Autor:
{dados_coletados.get('qualificacao_autor')}

Qualificação do Réu:
{dados_coletados.get('qualificacao_reu')}

Natureza da Ação: {dados_coletados.get('natureza_acao')}

Dos Fatos:
{dados_coletados.get('fatos')}

Da Fundamentação Jurídica:
{dados_coletados.get('fundamentacao_juridica')}

Dos Pedidos:
{dados_coletados.get('pedidos')}
{documentos_autor_formatado}
    """
    documento_do_caso_atual = Document(
        page_content=conteudo_processo_texto,
        metadata={
            "source_type": "processo_formulario_streamlit", 
            "file_name": f"{dados_coletados.get('id_processo')}_formulario.txt", 
            "process_id": dados_coletados.get('id_processo')
        }
    )
    
    # ... (Restante da função rodar_simulacao_principal como na versão anterior, pois as mudanças nela já foram boas)
    # A inicialização do RAG, o estado_inicial, e o loop de app.stream permanecem os mesmos.
    # A exibição dos resultados também.

    retriever_do_caso = None
    placeholder_rag = st.empty() 
    with placeholder_rag.status("⚙️ Inicializando sistema RAG com dados do formulário...", expanded=True):
        st.write("Carregando modelos e criando índice vetorial...")
        try:
            retriever_do_caso = criar_ou_carregar_retriever(
                dados_coletados.get('id_processo'), 
                documento_caso_atual=documento_do_caso_atual, 
                recriar_indice=True 
            )
            if retriever_do_caso:
                st.write("✅ Retriever RAG pronto!")
            else:
                st.write("⚠️ Falha ao inicializar o retriever RAG.")
        except Exception as e_rag:
            st.error(f"Erro crítico na inicialização do RAG: {e_rag}")
            retriever_do_caso = None 

    if not retriever_do_caso:
        placeholder_rag.empty() 
        st.error("Falha crítica ao criar o retriever com os dados do formulário. A simulação não pode continuar.")
        st.session_state.simulation_running = False
        if st.button("Tentar Novamente (Recarregar Formulário)"):
             st.session_state.current_form_step_index = FORM_STEPS.index("revisar_e_simular") 
             st.rerun()
        return

    placeholder_rag.success("🚀 Sistema RAG inicializado e pronto!")
    time.sleep(1.5) 
    placeholder_rag.empty()

    # Adicionando o campo 'documentos_juntados_pelo_reu' ao estado inicial, mesmo que vazio.
    estado_inicial = EstadoProcessual(
        id_processo=dados_coletados.get('id_processo'),
        retriever=retriever_do_caso,
        nome_do_ultimo_no_executado=None, etapa_concluida_pelo_ultimo_no=None,
        proximo_ator_sugerido_pelo_ultimo_no=ADVOGADO_AUTOR, 
        documento_gerado_na_etapa_recente=None, historico_completo=[],
        pontos_controvertidos_saneamento=None, manifestacao_autor_sem_provas=False,
        manifestacao_reu_sem_provas=False, etapa_a_ser_executada_neste_turno="",
        dados_formulario_entrada=dados_coletados,
        documentos_juntados_pelo_reu=None # Inicializa como None
    )

    st.subheader("⏳ Acompanhamento da Simulação:")
    if 'expand_all_steps' not in st.session_state: st.session_state.expand_all_steps = True
    
    if st.checkbox("Expandir todos os passos da simulação", value=st.session_state.expand_all_steps, key="cb_expand_all_sim_steps_final", on_change=lambda: setattr(st.session_state, 'expand_all_steps', st.session_state.cb_expand_all_sim_steps_final)):
        pass 
        
    progress_bar_placeholder = st.empty()
    steps_container = st.container()
    max_passos_simulacao = 12 # Aumentado um pouco devido à complexidade
    passo_atual_simulacao = 0
    estado_final_simulacao = None

    try:
        for s_idx, s_event in enumerate(app.stream(input=estado_inicial, config={"recursion_limit": max_passos_simulacao})):
            passo_atual_simulacao += 1
            if not s_event or not isinstance(s_event, dict) or not list(s_event.keys()):
                continue

            nome_do_no_executado = list(s_event.keys())[0]
            
            if nome_do_no_executado == "__end__":
                estado_final_simulacao = list(s_event.values())[0] 
                nome_do_no_executado = END 
            else:
                estado_parcial_apos_no = s_event[nome_do_no_executado]
                if not isinstance(estado_parcial_apos_no, dict): 
                    if estado_final_simulacao: 
                         pass 
                    else: 
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
                    st.text_area("Documento Gerado:", value=doc_gerado_completo, height=200, key=f"doc_step_sim_{passo_atual_simulacao}_final", disabled=True)
                elif doc_gerado_completo: 
                    st.error(f"Detalhe do Erro/Documento: {doc_gerado_completo}")
                st.markdown(f"**Próximo Ator Sugerido (pelo nó):** `{prox_ator_sug_log}`")
            
            # Estimativa de progresso pode ser mais complexa agora, mas vamos simplificar
            num_total_etapas_estimadas = len(mapa_tarefa_no_atual) + 2 # Adiciona 2 para PI e possível etapa final
            progress_val = min(1.0, passo_atual_simulacao / num_total_etapas_estimadas ) 
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


# --- Função exibir_resultados_simulacao (sem grandes mudanças estruturais aqui, mas pode exibir docs do réu) ---
def exibir_resultados_simulacao(estado_final_simulacao: dict):
    # Definição do mapa de cores para sentimentos (pode ser global ou dentro da função de UI)
    SENTIMENTO_CORES = {
        "Assertivo": "lightblue", "Confiante": "lightgreen", "Persuasivo": "lightyellow",
        "Combativo": "lightcoral", "Agressivo": "salmon", "Indignado": "lightpink",
        "Neutro": "lightgray", "Formal": "whitesmoke",
        "Conciliatório": "palegreen", "Colaborativo": "mediumaquamarine",
        "Defensivo": "thistle", "Emocional": "lavenderblush",
        "Não analisado": "silver", "Erro na análise": "orangered"
    }
    DEFAULT_SENTIMENTO_COR = "gainsboro"

    doc_completo_placeholder_sim = st.empty() # Para visualização de docs da timeline

    if estado_final_simulacao:
        sentimento_pi = estado_final_simulacao.get("sentimento_peticao_inicial")
        sentimento_cont = estado_final_simulacao.get("sentimento_contestacao")
        
        if sentimento_pi or sentimento_cont:
            st.markdown("#### Análise de Sentimentos (IA)")
            cols_sent = st.columns(2)
            if sentimento_pi:
                cor_pi = SENTIMENTO_CORES.get(sentimento_pi, DEFAULT_SENTIMENTO_COR)
                cols_sent[0].markdown(f"**Petição Inicial:** <span style='background-color:{cor_pi}; color:black; padding: 3px 6px; border-radius: 5px;'>{sentimento_pi}</span>", unsafe_allow_html=True)
            else:
                cols_sent[0].markdown("**Petição Inicial:** Sentimento não analisado.")
            if sentimento_cont:
                cor_cont = SENTIMENTO_CORES.get(sentimento_cont, DEFAULT_SENTIMENTO_COR)
                cols_sent[1].markdown(f"**Contestação:** <span style='background-color:{cor_cont}; color:black; padding: 3px 6px; border-radius: 5px;'>{sentimento_cont}</span>", unsafe_allow_html=True)
            else:
                cols_sent[1].markdown("**Contestação:** Sentimento não analisado.")
            st.markdown("---") # Separador após a análise de sentimentos

    # Linha do Tempo Interativa do Processo
    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        st.markdown("#### Linha do Tempo Interativa do Processo")
        historico = estado_final_simulacao["historico_completo"]
        icon_map = { # Seu icon_map ...
            ADVOGADO_AUTOR: "🙋‍♂️", JUIZ: "⚖️", ADVOGADO_REU: "🙋‍♀️",
            ETAPA_PETICAO_INICIAL: "📄", ETAPA_DESPACHO_RECEBENDO_INICIAL: "➡️",
            ETAPA_CONTESTACAO: "🛡️", ETAPA_DECISAO_SANEAMENTO: "🛠️",
            ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR: "🗣️", ETAPA_MANIFESTACAO_SEM_PROVAS_REU: "🗣️",
            ETAPA_SENTENCA: "🏁", "DEFAULT_ACTOR": "👤", "DEFAULT_ETAPA": "📑",
            "ERRO_FLUXO_IRRECUPERAVEL": "❌", "ERRO_ETAPA_NAO_ENCONTRADA": "❓"
        }
        num_etapas = len(historico)
        if num_etapas > 0 :
            cols = st.columns(min(num_etapas, 8))
            for i, item_hist in enumerate(historico):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                
                ator_icon = icon_map.get(ator_hist, icon_map["DEFAULT_ACTOR"])
                etapa_icon = icon_map.get(etapa_hist, icon_map["DEFAULT_ETAPA"])
                cor_fundo = "rgba(255, 0, 0, 0.1)" if "ERRO" in etapa_hist else "rgba(0, 0, 0, 0.03)"

                with cols[i % len(cols)]:
                    container_style = f"border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center; background-color: {cor_fundo}; height: 130px; display: flex; flex-direction: column; justify-content: space-around; margin-bottom: 5px;"
                    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 28px;'>{ator_icon}{etapa_icon}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px;'><b>{ator_hist}</b><br>{etapa_hist[:30]}{'...' if len(etapa_hist)>30 else ''}</div>", unsafe_allow_html=True)
                    if st.button(f"Ver Doc {i+1}", key=f"btn_timeline_sim_{i}_final", help=f"Visualizar: {etapa_hist}", use_container_width=True):
                        st.session_state.doc_visualizado = doc_completo_hist
                        st.session_state.doc_visualizado_titulo = f"Documento da Linha do Tempo (Passo {i+1}): {ator_hist} - {etapa_hist}"
                        # doc_completo_placeholder_sim.empty() # Limpa antes de re-exibir o placeholder
                        st.rerun() 
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True) 
    else:
        st.warning("Nenhum histórico completo para exibir na linha do tempo.")

    # Visualização do Documento da Timeline (quando st.session_state.doc_visualizado é setado)
    if st.session_state.get('doc_visualizado') is not None: 
        with doc_completo_placeholder_sim.container(): # Usar o placeholder definido no início da função
            st.subheader(st.session_state.get('doc_visualizado_titulo', "Visualização de Documento"))
            st.text_area("Conteúdo do Documento:", st.session_state.doc_visualizado, height=350, key="doc_view_sim_area_main_results_final", disabled=True)
            if st.button("Fechar Visualização do Documento", key="close_doc_view_sim_btn_main_results_final", type="primary"):
                st.session_state.doc_visualizado = None
                st.session_state.doc_visualizado_titulo = ""
                doc_completo_placeholder_sim.empty() # Limpa o conteúdo do placeholder
                st.rerun()

    # Funcionalidades Adicionais da Sentença (Ementa e Verificação)
    sentenca_texto_completo = None
    houve_sentenca = False
    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        for item_hist in reversed(estado_final_simulacao["historico_completo"]):
            if item_hist.get("etapa") == ETAPA_SENTENCA:
                sentenca_texto_completo = str(item_hist.get("documento", ""))
                houve_sentenca = True
                break
    
    if houve_sentenca and sentenca_texto_completo:
        st.markdown("---")
        st.markdown("#### Funcionalidades Adicionais da Sentença")
        id_proc = estado_final_simulacao.get("id_processo", "desconhecido")

        col_ementa, col_verificador = st.columns(2)

        with col_ementa:
            if st.button("📄 Gerar Ementa (Padrão CNJ)", key="btn_gerar_ementa_final", use_container_width=True):
                # ... (lógica do botão como você tem)
                if sentenca_texto_completo:
                    with st.spinner("Gerando ementa no padrão CNJ... Isso pode levar um momento."):
                        st.session_state.ementa_cnj_gerada = gerar_ementa_cnj_padrao(sentenca_texto_completo, id_proc, llm)
                        st.session_state.show_ementa_popup = True
                        st.rerun()
                else:
                    st.warning("Texto da sentença não encontrado para gerar ementa.")
        
        with col_verificador:
            if search_tool: # CORREÇÃO: usa a variável correta 'search_tool'
                if st.button("🔍 Verificar Sentença com Jurisprudência (Google)", key="btn_verificar_sentenca_final", use_container_width=True):
                    # ... (lógica do botão como você tem)
                    if sentenca_texto_completo:
                        with st.spinner("Verificando sentença com jurisprudência... (Pode demorar alguns instantes)"):
                            st.session_state.verificacao_sentenca_resultado = verificar_sentenca_com_jurisprudencia(sentenca_texto_completo, llm)
                            st.session_state.show_verificacao_popup = True
                            st.rerun()
                    else:
                        st.warning("Texto da sentença não encontrado para verificação.")
            else:
                col_verificador.info("Verificação com Google desabilitada (API não configurada ou falha na inicialização).") # Mensagem atualizada

    # Pop-ups para Ementa e Verificação
    if st.session_state.get('show_ementa_popup', False) and st.session_state.get('ementa_cnj_gerada'):
        # REMOVER o if hasattr(st, 'dialog') e usar diretamente o fallback
        # ---- INÍCIO DO BLOCO DE FALLBACK PARA EMENTA ----
        with st.container(): # Usar container para simular um modal sobreposto
            st.markdown("---")
            st.subheader("📄 Ementa Gerada (Padrão CNJ)")
            st.markdown(st.session_state.ementa_cnj_gerada)
            if st.button("Fechar Ementa", key="close_ementa_fallback_v3_corrected"): # Nova chave única
                st.session_state.show_ementa_popup = False
                st.session_state.ementa_cnj_gerada = None
                st.rerun() # Pode ser necessário para fechar/limpar visualmente
            st.markdown("---")
        # ---- FIM DO BLOCO DE FALLBACK PARA EMENTA ----

    if st.session_state.get('show_verificacao_popup', False) and st.session_state.get('verificacao_sentenca_resultado'):
        # REMOVER o if hasattr(st, 'dialog') e usar diretamente o fallback
        # ---- INÍCIO DO BLOCO DE FALLBACK PARA VERIFICAÇÃO ----
        with st.container():
            st.markdown("---")
            st.subheader("🔍 Verificação da Sentença com Jurisprudência")
            st.markdown(st.session_state.verificacao_sentenca_resultado)
            if st.button("Fechar Verificação", key="close_verif_fallback_v3_corrected"): # Nova chave única
                st.session_state.show_verificacao_popup = False
                st.session_state.verificacao_sentenca_resultado = None
                st.rerun() # Pode ser necessário
            st.markdown("---")
        # ---- FIM DO BLOCO DE FALLBACK PARA VERIFICAÇÃO ---

    st.subheader("📊 Resultados da Simulação")
    doc_completo_placeholder_sim = st.empty()

    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        st.markdown("#### Linha do Tempo Interativa do Processo")
        # ... (código da linha do tempo como antes) ...
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
            cols = st.columns(min(num_etapas, 8)) # Aumentado para 8 se houver mais etapas com docs
            for i, item_hist in enumerate(historico):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                
                ator_icon = icon_map.get(ator_hist, icon_map["DEFAULT_ACTOR"])
                etapa_icon = icon_map.get(etapa_hist, icon_map["DEFAULT_ETAPA"])
                cor_fundo = "rgba(255, 0, 0, 0.1)" if "ERRO" in etapa_hist else "rgba(0, 0, 0, 0.03)"

                with cols[i % len(cols)]:
                    container_style = f"border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center; background-color: {cor_fundo}; height: 130px; display: flex; flex-direction: column; justify-content: space-around; margin-bottom: 5px;"
                    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 28px;'>{ator_icon}{etapa_icon}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px;'><b>{ator_hist}</b><br>{etapa_hist[:30]}{'...' if len(etapa_hist)>30 else ''}</div>", unsafe_allow_html=True)
                    if st.button(f"Ver Doc {i+1}", key=f"btn_timeline_sim_{i}_{estado_final_simulacao.get('id_processo', 'default_pid')}", help=f"Visualizar: {etapa_hist}", use_container_width=True):
                        st.session_state.doc_visualizado = doc_completo_hist
                        st.session_state.doc_visualizado_titulo = f"Documento da Linha do Tempo (Passo {i+1}): {ator_hist} - {etapa_hist}"
                        st.rerun() 
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True) 
    else: st.warning("Nenhum histórico completo para exibir na linha do tempo.")

    if st.session_state.get('doc_visualizado') is not None: 
        with doc_completo_placeholder_sim.container():
            st.subheader(st.session_state.get('doc_visualizado_titulo', "Visualização de Documento"))
            st.text_area("Conteúdo do Documento:", st.session_state.doc_visualizado, height=350, key="doc_view_sim_area_main_results_final", disabled=True)
            if st.button("Fechar Visualização do Documento", key="close_doc_view_sim_btn_main_results_final", type="primary"):
                st.session_state.doc_visualizado = None
                st.session_state.doc_visualizado_titulo = ""
                doc_completo_placeholder_sim.empty()
                
    
    st.markdown("#### Histórico Detalhado (Conteúdo Completo das Etapas)")
    if 'expand_all_history' not in st.session_state: st.session_state.expand_all_history = False
    
    if st.checkbox("Expandir todo o histórico detalhado", value=st.session_state.expand_all_history, key="cb_expand_all_hist_detail_sim_final", on_change=lambda: setattr(st.session_state, 'expand_all_history', st.session_state.cb_expand_all_hist_detail_sim_final)):
        pass

    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
            ator_hist = item_hist.get('ator', 'N/A'); etapa_hist = item_hist.get('etapa', 'N/A')
            doc_completo_hist = str(item_hist.get('documento', 'N/A'))
            with st.expander(f"Detalhe {i+1}: Ator '{ator_hist}' | Etapa '{etapa_hist}'", expanded=st.session_state.expand_all_history):
                st.text_area(f"Documento Completo (Passo {i+1}):", value=doc_completo_hist, height=200, key=f"doc_hist_detail_sim_{i}_final", disabled=True)
    
    # NOVO: Exibir documentos juntados pelo Réu (se existirem)
    if estado_final_simulacao and estado_final_simulacao.get("documentos_juntados_pelo_reu"):
        st.markdown("#### Documentos Juntados pelo Réu (Gerados pela IA)")
        with st.expander("Ver Documentos do Réu", expanded=False):
            for i, doc_reu in enumerate(estado_final_simulacao.get("documentos_juntados_pelo_reu", [])):
                st.markdown(f"**Documento {i+1} (Réu):** {doc_reu.get('tipo', 'N/A')}")
                st.text_area(f"Descrição Doc. Réu {i+1}", value=doc_reu.get('descricao', 'Sem descrição'), height=75, disabled=True, key=f"res_doc_reu_{i}")

    st.markdown("--- FIM DA EXIBIÇÃO DOS RESULTADOS ---")


# --- Bloco Principal de Execução do Streamlit ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="IA-Mestra: Simulação Jurídica Avançada", page_icon="⚖️")
    st.title("IA-Mestra: Simulação Jurídica Avançada ⚖️")
    st.caption("Uma ferramenta para simular o fluxo processual com assistência de IA, utilizando LangGraph e RAG.")

    if not GOOGLE_API_KEY:
        st.error("🔴 ERRO CRÍTICO: A variável de ambiente GOOGLE_API_KEY não foi definida. A aplicação não pode funcionar sem ela.")
        st.stop() 
    
    inicializar_estado_formulario() 
    
    st.sidebar.title("Painel de Controle 🕹️")
    if st.sidebar.button("🔄 Nova Simulação (Limpar Formulário)", key="nova_sim_btn_sidebar_final", type="primary", use_container_width=True):
        st.session_state.current_form_step_index = 0
        novo_id_processo = f"caso_sim_{int(time.time())}"
        # Reinicializa completamente form_data e flags relacionadas
        st.session_state.form_data = {
            "id_processo": novo_id_processo, "qualificacao_autor": "", "qualificacao_reu": "", 
            "fatos": "", "fundamentacao_juridica": "", "pedidos": "",
            "natureza_acao": "", "documentos_autor": []
        }
        st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
        st.session_state.ia_generated_content_flags["documentos_autor_descricoes"] = {}
        st.session_state.num_documentos_autor = 0 # Reset do contador de documentos

        st.session_state.simulation_running = False 
        if 'doc_visualizado' in st.session_state: st.session_state.doc_visualizado = None 
        if 'doc_visualizado_titulo' in st.session_state: st.session_state.doc_visualizado_titulo = ""
        # Considerar limpar st.session_state.simulation_results se desejar apagar histórico entre simulações
        # st.session_state.simulation_results = {} 
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ Preencha os formulários sequenciais para definir os parâmetros do caso. "
        "A IA pode auxiliar no preenchimento com dados fictícios ou sugestões jurídicas contextuais."
    )
    st.sidebar.markdown("---")
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" and os.getenv("LANGCHAIN_PROJECT"):
        project_name = os.getenv("LANGCHAIN_PROJECT")
        # LANGCHAIN_TENANT_ID pode não ser o termo correto para o URL do LangSmith; pode ser o nome da organização ou UUID.
        # O URL típico é smith.langchain.com/o/{organization_id}/projects/p/{project_name}
        # Se LANGCHAIN_ENDPOINT existir e for algo como https://api.smith.langchain.com, o URL base é smith.langchain.com.
        # Por simplicidade, manter o link genérico se o tenant ID não for facilmente obtido.
        langsmith_url_base = "https://smith.langchain.com/"
        org_id_ou_tenant = os.getenv('LANGCHAIN_TENANT_ID', None) # Ou outro env var que possa conter o ID da organização
        if org_id_ou_tenant:
            langsmith_url = f"{langsmith_url_base}o/{org_id_ou_tenant}/projects/{project_name}"
        else: # Fallback para um link mais genérico se o ID da org não estiver disponível
            langsmith_url = f"{langsmith_url_base}projects/{project_name}" 
            # Ou até mesmo apenas "https://smith.langchain.com/" se o nome do projeto não garante acesso direto.
            # O link mais seguro sem ID da organização é para a página de projetos geral, e o usuário navega.
            # langsmith_url = f"https://smith.langchain.com/projects" (mais seguro)

        st.sidebar.markdown(f"🔍 [Monitorar no LangSmith]({langsmith_url})", unsafe_allow_html=True)


    # --- Lógica Principal de Exibição da UI ---
    if st.session_state.get('simulation_running', False):
        # ... (lógica de rodar/exibir simulação como antes, já estava boa) ...
        id_processo_atual = st.session_state.form_data.get('id_processo')
        if id_processo_atual and id_processo_atual not in st.session_state.get('simulation_results', {}):
            rodar_simulacao_principal(st.session_state.form_data)
        elif id_processo_atual and st.session_state.get('simulation_results', {}).get(id_processo_atual):
            st.info(f"📖 Exibindo resultados da simulação para o ID: {id_processo_atual}")
            exibir_resultados_simulacao(st.session_state.simulation_results[id_processo_atual])
            if st.button("Iniciar uma Nova Simulação (Limpar Formulário)", key="nova_sim_btn_results_final"): 
                st.session_state.current_form_step_index = 0
                novo_id_processo = f"caso_sim_{int(time.time())}"
                st.session_state.form_data = { "id_processo": novo_id_processo, "qualificacao_autor": "", "qualificacao_reu": "", "fatos": "", "fundamentacao_juridica": "", "pedidos": "", "natureza_acao": "", "documentos_autor": [] }
                st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
                st.session_state.ia_generated_content_flags["documentos_autor_descricoes"] = {}
                st.session_state.num_documentos_autor = 0
                st.session_state.simulation_running = False
                if 'doc_visualizado' in st.session_state: st.session_state.doc_visualizado = None
                st.rerun()
        else: 
            st.warning("⚠️ A simulação anterior não produziu resultados ou houve um problema de estado. Por favor, inicie uma nova simulação.")
            st.session_state.simulation_running = False 
            if st.button("Ir para o início do formulário", key="goto_form_start_final"):
                 st.session_state.current_form_step_index = 0
                 st.rerun()
    else: # Exibir formulários
        passo_atual_idx = st.session_state.current_form_step_index
        
        # --- Indicador de Progresso do Formulário ---
        if 0 <= passo_atual_idx < len(FORM_STEPS):
            nome_passo_atual = FORM_STEPS[passo_atual_idx]
            # O total de passos de preenchimento é len(FORM_STEPS) - 1 (excluindo a revisão)
            total_passos_preenchimento = len(FORM_STEPS) -1
            
            if nome_passo_atual != "revisar_e_simular":
                # Progresso baseado no índice atual sobre o total de passos de preenchimento
                progresso_percentual = (passo_atual_idx) / total_passos_preenchimento if total_passos_preenchimento > 0 else 0
                st.progress(progresso_percentual)
                titulo_passo_formatado = nome_passo_atual.replace('_', ' ').title()
                st.markdown(f"#### Etapa de Preenchimento: **{titulo_passo_formatado}** (Passo {passo_atual_idx + 1} de {total_passos_preenchimento})")
            else: # Etapa de Revisão
                 st.markdown(f"#### Etapa Final: **Revisar Dados e Iniciar Simulação** (Passo {len(FORM_STEPS)} de {len(FORM_STEPS)})")
            st.markdown("---")
        # --- Fim do Indicador de Progresso ---

            current_step_key = FORM_STEPS[passo_atual_idx] 
            if current_step_key == "autor": exibir_formulario_qualificacao_autor()
            elif current_step_key == "reu": exibir_formulario_qualificacao_reu()
            elif current_step_key == "fatos": exibir_formulario_fatos()
            elif current_step_key == "direito": exibir_formulario_direito()
            elif current_step_key == "pedidos": exibir_formulario_pedidos()
            elif current_step_key == "natureza_acao": exibir_formulario_natureza_acao() # MODIFICADO: Chamada correta
            elif current_step_key == "documentos_autor": exibir_formulario_documentos_autor() # NOVO: Chamada correta
            elif current_step_key == "revisar_e_simular": exibir_revisao_e_iniciar_simulacao()
            else: 
                st.error(f"🔴 ERRO INTERNO: Etapa do formulário desconhecida ou não tratada: '{current_step_key}'.")
                st.warning("Por favor, tente reiniciar a simulação a partir do menu lateral.")
        else: 
            st.error("🔴 ERRO INTERNO: Índice da etapa do formulário inválido. Tentando reiniciar...")
            st.session_state.current_form_step_index = 0 
            st.rerun()