import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Union
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
# Para um RAG real, você usaria:
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader # ou similar

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Carregamento de Variáveis de Ambiente e Configuração Inicial ---
load_dotenv()

# Validação da API Key (opcional, mas bom para feedback rápido)
if not os.getenv("GOOGLE_API_KEY"):
    print("Erro: A variável de ambiente GOOGLE_API_KEY não foi definida.")
    exit()

# --- 2. Inicialização do LLM (Modelo Gemini) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)

# --- 3. Simulação do RAG (para MVP) ---
# Em um sistema completo, aqui você carregaria, dividiria, criaria embeddings
# e indexaria os documentos do processo.
class SimRAGRetriever:
    def __init__(self, id_processo: str):
        self.id_processo = id_processo
        self.documentos_simulados = {
            "caso_001": [
                "Petição Inicial: Reclamação por danos morais devido a inscrição indevida em cadastro de inadimplentes.",
                "Prova Documental: Comprovante de pagamento da dívida antes da inscrição.",
                "Prova Documental: Notificação de negativação.",
            ],
            "caso_002": [
                "Petição Inicial: Ação de despejo por falta de pagamento.",
                "Contrato de Locação: Cláusula X prevê multa por atraso.",
                "Notificação Extrajudicial: Cobrança dos aluguéis atrasados.",
            ]
        }
        print(f"[SimRAGRetriever] Inicializado para o processo: {id_processo}")

    def get_relevant_documents(self, query: str, top_k: int = 2) -> List[str]:
        print(f"[SimRAGRetriever] Buscando por: '{query}' no processo {self.id_processo}")
        docs_processo = self.documentos_simulados.get(self.id_processo, ["Processo não encontrado ou sem documentos."])
        # Simulação simples de relevância (poderia ser mais sofisticada)
        relevant_docs = [doc for doc in docs_processo if query.lower().split(" ")[0] in doc.lower()]
        if not relevant_docs:
            relevant_docs = docs_processo # Retorna todos se nenhuma palavra chave bater
        return relevant_docs[:top_k]

def criar_retriever_simulado(id_processo: str) -> SimRAGRetriever:
    return SimRAGRetriever(id_processo)

# --- 4. Definição do Estado do Processo (LangGraph) ---
class EstadoProcessual(TypedDict):
    id_processo: str
    retriever: SimRAGRetriever # No MVP, usamos nosso SimRAGRetriever
    historico_acoes: List[str]
    proximo_ator: str
    peticao_rascunho: Union[str, None] # Petição sendo trabalhada
    decisao_juiz_rascunho: Union[str, None] # Decisão sendo trabalhada
    entrada_usuario_atual: Union[str, None] # Para instruções específicas

# --- 5. Definição dos Agentes (Nós do Grafo) ---

def agente_advogado_autor(estado: EstadoProcessual):
    print("\n--- TURNO: ADVOGADO DO AUTOR ---")
    prompt_adv_autor_template = """
    Você é o Advogado do Autor. Seu objetivo é iniciar o processo ou responder a despachos.
    Processo ID: {id_processo}
    Documentos relevantes dos autos (simulado via RAG):
    {documentos_rag}

    Histórico de Ações:
    {historico}

    Instrução para esta atuação: {instrucao}
    Elabore a petição solicitada de forma concisa para uma simulação.
    """
    prompt_adv_autor = ChatPromptTemplate.from_template(prompt_adv_autor_template)
    chain_adv_autor = prompt_adv_autor | llm | StrOutputParser()

    documentos_relevantes = estado["retriever"].get_relevant_documents(estado["entrada_usuario_atual"] or "petição inicial")
    
    peticao = chain_adv_autor.invoke({
        "id_processo": estado["id_processo"],
        "documentos_rag": "\n".join([f"- {doc}" for doc in documentos_relevantes]),
        "historico": "\n".join(estado["historico_acoes"]) if estado["historico_acoes"] else "Nenhuma ação anterior.",
        "instrucao": estado["entrada_usuario_atual"] or "Elaborar petição inicial concisa sobre uma disputa contratual simples."
    })
    print(f"Adv. Autor elaborou: {peticao}")
    return {
        "peticao_rascunho": peticao,
        "historico_acoes": estado["historico_acoes"] + [f"Adv. Autor peticionou: {peticao[:100]}..."],
        "proximo_ator": "juiz",
        "entrada_usuario_atual": None # Limpa a instrução
    }

def agente_juiz(estado: EstadoProcessual):
    print("\n--- TURNO: JUIZ ---")
    prompt_juiz_template = """
    Você é o Juiz. Analise a petição apresentada e os documentos. Profira uma decisão/despacho.
    Processo ID: {id_processo}
    Petição Atual para análise:
    {peticao}

    Documentos relevantes dos autos (simulado via RAG):
    {documentos_rag}

    Histórico de Ações:
    {historico}

    Sua decisão/despacho (seja conciso para a simulação):
    """
    prompt_juiz = ChatPromptTemplate.from_template(prompt_juiz_template)
    chain_juiz = prompt_juiz | llm | StrOutputParser()

    documentos_relevantes = estado["retriever"].get_relevant_documents(estado["peticao_rascunho"] or "análise geral")

    decisao = chain_juiz.invoke({
        "id_processo": estado["id_processo"],
        "peticao": estado["peticao_rascunho"] or "Nenhuma petição para análise.",
        "documentos_rag": "\n".join([f"- {doc}" for doc in documentos_relevantes]),
        "historico": "\n".join(estado["historico_acoes"])
    })
    print(f"Juiz decidiu: {decisao}")

    # Lógica simples de fluxo para MVP
    proximo = "adv_reu"
    if "inicial" in (estado["peticao_rascunho"] or "").lower():
        proximo = "adv_reu" # Após a inicial, vai para o réu
    elif "contesta" in (estado["peticao_rascunho"] or "").lower():
        proximo = "adv_autor" # Após contestação, pode ir para réplica do autor (simplificado)
    elif len(estado["historico_acoes"]) > 4 : # Limita o ciclo para MVP
        proximo = "fim_processo"


    return {
        "decisao_juiz_rascunho": decisao,
        "historico_acoes": estado["historico_acoes"] + [f"Juiz decidiu: {decisao[:100]}..."],
        "peticao_rascunho": None, # Limpa a petição analisada
        "proximo_ator": proximo
    }

def agente_advogado_reu(estado: EstadoProcessual):
    print("\n--- TURNO: ADVOGADO DO RÉU ---")
    prompt_adv_reu_template = """
    Você é o Advogado do Réu. O Juiz proferiu uma decisão e/ou há uma petição do autor para você responder.
    Processo ID: {id_processo}
    Última decisão do Juiz:
    {decisao_juiz}

    Petição do Autor (se houver, para contestar):
    {peticao_autor_original_para_contestar} 
    
    Documentos relevantes dos autos (simulado via RAG):
    {documentos_rag}

    Histórico de Ações:
    {historico}

    Instrução para esta atuação: {instrucao}
    Elabore sua manifestação/contestação de forma concisa para a simulação.
    """
    prompt_adv_reu = ChatPromptTemplate.from_template(prompt_adv_reu_template)
    chain_adv_reu = prompt_adv_reu | llm | StrOutputParser()
    
    # Para contestar, o advogado do réu precisaria da petição inicial original.
    # No nosso fluxo simplificado, vamos assumir que a "petição_rascunho" do estado anterior (do autor)
    # foi implicitamente a que o juiz analisou para citar o réu.
    # Para simplificar, pegamos a última petição do histórico que foi do autor, se houver.
    peticao_autor_original = "N/A"
    for acao in reversed(estado["historico_acoes"]):
        if "Adv. Autor peticionou:" in acao:
            peticao_autor_original = acao.replace("Adv. Autor peticionou:", "").strip()
            break
            
    documentos_relevantes = estado["retriever"].get_relevant_documents(estado["entrada_usuario_atual"] or "contestação")

    contestacao = chain_adv_reu.invoke({
        "id_processo": estado["id_processo"],
        "decisao_juiz": estado["decisao_juiz_rascunho"] or "Nenhuma decisão recente do juiz.",
        "peticao_autor_original_para_contestar": peticao_autor_original,
        "documentos_rag": "\n".join([f"- {doc}" for doc in documentos_relevantes]),
        "historico": "\n".join(estado["historico_acoes"]),
        "instrucao": estado["entrada_usuario_atual"] or "Elaborar contestação concisa à petição inicial."
    })
    print(f"Adv. Réu elaborou: {contestacao}")
    return {
        "peticao_rascunho": contestacao,
        "historico_acoes": estado["historico_acoes"] + [f"Adv. Réu contestou: {contestacao[:100]}..."],
        "proximo_ator": "juiz",
        "entrada_usuario_atual": None # Limpa a instrução
    }

# --- 6. Construção do Grafo (LangGraph) ---
workflow = StateGraph(EstadoProcessual)

workflow.add_node("adv_autor", agente_advogado_autor)
workflow.add_node("juiz", agente_juiz)
workflow.add_node("adv_reu", agente_advogado_reu)

# Definir o ponto de entrada
workflow.set_entry_point("adv_autor")

# Lógica de transição condicional
def decidir_proximo_passo(estado: EstadoProcessual):
    print(f"[Router] Próximo ator definido: {estado['proximo_ator']}")
    if estado["proximo_ator"] == "juiz":
        return "juiz"
    elif estado["proximo_ator"] == "adv_reu":
        return "adv_reu"
    elif estado["proximo_ator"] == "adv_autor": # Para réplica, por exemplo
        return "adv_autor"
    elif estado["proximo_ator"] == "fim_processo":
        return END
    return END # Padrão para fim, se algo der errado

workflow.add_conditional_edges("adv_autor", decidir_proximo_passo, {"juiz": "juiz"})
workflow.add_conditional_edges(
    "juiz",
    decidir_proximo_passo,
    {
        "adv_reu": "adv_reu",
        "adv_autor": "adv_autor", # Ex: Para o autor replicar após contestação
        END: END
    }
)
workflow.add_conditional_edges("adv_reu", decidir_proximo_passo, {"juiz": "juiz"})

# Compilar o grafo
app = workflow.compile()

# --- 7. Execução da Simulação ---
if __name__ == "__main__":
    id_processo_simulado = "caso_001"
    retriever_simulado = criar_retriever_simulado(id_processo_simulado)

    estado_inicial = EstadoProcessual(
        id_processo=id_processo_simulado,
        retriever=retriever_simulado,
        historico_acoes=[],
        proximo_ator="adv_autor",
        peticao_rascunho=None,
        decisao_juiz_rascunho=None,
        entrada_usuario_atual="Elaborar petição inicial concisa referente a uma cobrança indevida."
    )

    print(f"\n--- INÍCIO DA SIMULAÇÃO DO PROCESSO {id_processo_simulado} ---")
    
    # Limitar o número de passos para o MVP
    max_passos = 6 
    passo_atual = 0
    for s in app.stream(estado_inicial, {"recursion_limit": 15}):
        passo_atual += 1
        nome_no, estado_parcial = list(s.items())[0]
        print(f"\n... ESTADO APÓS NÓ '{nome_no}' (Passo {passo_atual}) ...")
        # print(f"Estado parcial: {estado_parcial}") # Descomente para debug detalhado
        
        if nome_no == END or passo_atual >= max_passos :
            print("\n--- FIM DA SIMULAÇÃO (ATINGIU NÓ FINAL OU LIMITE DE PASSOS) ---")
            print("Histórico Final de Ações:")
            for i, acao in enumerate(estado_parcial.get("historico_acoes", [])):
                print(f"  {i+1}. {acao}")
            break
        # Para a próxima iteração, podemos (opcionalmente) fornecer uma nova instrução
        # Aqui, deixaremos o fluxo seguir ou o usuário poderia intervir.
        # Exemplo de como seria uma nova instrução para o próximo ator:
        # if estado_parcial.get("proximo_ator") == "adv_reu":
        #     estado_parcial["entrada_usuario_atual"] = "Contestar veementemente a petição inicial."


    # Visualizar o grafo (opcional, requer graphviz)
    try:
        # Salvar imagem do grafo
        # app.get_graph().draw_mermaid_png(output_file_path="mvp_simulacao_juridica_graph.png")
        # print("\nGrafo da simulação salvo como 'mvp_simulacao_juridica_graph.png'")
        pass # Mermaid PNG export can be tricky with dependencies, skip for basic MVP if issues.
    except Exception as e:
        print(f"\nNão foi possível gerar a imagem do grafo (requer graphviz e dependências): {e}")