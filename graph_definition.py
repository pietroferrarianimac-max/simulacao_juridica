# graph_definition.py

from typing import TypedDict, List, Union, Dict, Tuple, Any
from functools import partial # Para passar argumentos fixos aos nós do grafo

# LangGraph
from langgraph.graph import StateGraph, END

# Agentes (do nosso arquivo agents.py)
from agents import (
    agente_advogado_autor,
    agente_juiz,
    agente_advogado_reu
)

# Constantes (do nosso arquivo settings.py)
from settings import (
    ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU,
    ETAPA_PETICAO_INICIAL, ETAPA_DESPACHO_RECEBENDO_INICIAL, ETAPA_CONTESTACAO,
    ETAPA_DECISAO_SANEAMENTO, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    ETAPA_MANIFESTACAO_SEM_PROVAS_REU, ETAPA_SENTENCA, ETAPA_FIM_PROCESSO
)

# --- Definição do Estado Processual (LangGraph) ---
class EstadoProcessual(TypedDict):
    id_processo: str
    retriever: Any # Instância do retriever FAISS (ou similar)

    nome_do_ultimo_no_executado: Union[str, None]
    etapa_concluida_pelo_ultimo_no: Union[str, None]
    proximo_ator_sugerido_pelo_ultimo_no: Union[str, None]

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

# --- Mapa de Fluxo Processual (Rito Ordinário) ---
# Chave: (ultimo_ator, etapa_concluida_pelo_ultimo_ator, ator_atual_designado_pelo_router)
# Valor: etapa_a_ser_executada_pelo_ator_atual
mapa_tarefa_no_atual: Dict[Tuple[Union[str, None], Union[str, None], str], str] = {
    (None, None, ADVOGADO_AUTOR): ETAPA_PETICAO_INICIAL,
    (ADVOGADO_AUTOR, ETAPA_PETICAO_INICIAL, JUIZ): ETAPA_DESPACHO_RECEBENDO_INICIAL,
    (JUIZ, ETAPA_DESPACHO_RECEBENDO_INICIAL, ADVOGADO_REU): ETAPA_CONTESTACAO,
    (ADVOGADO_REU, ETAPA_CONTESTACAO, JUIZ): ETAPA_DECISAO_SANEAMENTO,
    (JUIZ, ETAPA_DECISAO_SANEAMENTO, ADVOGADO_AUTOR): ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    (ADVOGADO_AUTOR, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR, ADVOGADO_REU): ETAPA_MANIFESTACAO_SEM_PROVAS_REU,
    (ADVOGADO_REU, ETAPA_MANIFESTACAO_SEM_PROVAS_REU, JUIZ): ETAPA_SENTENCA,
}

# --- Função de Roteamento Condicional (Router) ---
def decidir_proximo_no_do_grafo(estado: EstadoProcessual) -> str:
    proximo_ator_sugerido = estado.get("proximo_ator_sugerido_pelo_ultimo_no")
    etapa_concluida = estado.get("etapa_concluida_pelo_ultimo_no")

    # Log verboso para depuração do roteamento
    print(f"[Router] Decidindo próximo nó com base em:")
    print(f"  Último nó executado: {estado.get('nome_do_ultimo_no_executado')}")
    print(f"  Etapa concluída pelo último nó: {etapa_concluida}")
    print(f"  Próximo ator sugerido pelo último nó: {proximo_ator_sugerido}")

    if proximo_ator_sugerido == ADVOGADO_AUTOR:
        print("[Router] Direcionando para ADVOGADO_AUTOR.")
        return ADVOGADO_AUTOR
    if proximo_ator_sugerido == JUIZ:
        print("[Router] Direcionando para JUIZ.")
        return JUIZ
    if proximo_ator_sugerido == ADVOGADO_REU:
        print("[Router] Direcionando para ADVOGADO_REU.")
        return ADVOGADO_REU

    if proximo_ator_sugerido == ETAPA_FIM_PROCESSO or etapa_concluida == ETAPA_SENTENCA:
        print("[Router] Fluxo direcionado para o FIM do processo.")
        return END

    print(f"[Router_ERRO] Próximo ator '{proximo_ator_sugerido}' desconhecido ou fluxo não previsto após etapa '{etapa_concluida}'. Encerrando.")
    return END

# --- Construção do Grafo LangGraph ---
workflow = StateGraph(EstadoProcessual)

# Como nossos agentes agora esperam 'mapa_tarefas' como argumento,
# usamos functools.partial para criar novas funções que já têm 'mapa_tarefas' embutido.
advogado_autor_node = partial(agente_advogado_autor, mapa_tarefas=mapa_tarefa_no_atual)
juiz_node = partial(agente_juiz, mapa_tarefas=mapa_tarefa_no_atual)
advogado_reu_node = partial(agente_advogado_reu, mapa_tarefas=mapa_tarefa_no_atual)

# Adicionar nós ao grafo
workflow.add_node(ADVOGADO_AUTOR, advogado_autor_node)
workflow.add_node(JUIZ, juiz_node)
workflow.add_node(ADVOGADO_REU, advogado_reu_node)

# Definir ponto de entrada
workflow.set_entry_point(ADVOGADO_AUTOR)

# Mapa para roteamento nas conditional_edges
roteamento_mapa_edges = {
    ADVOGADO_AUTOR: ADVOGADO_AUTOR,
    JUIZ: JUIZ,
    ADVOGADO_REU: ADVOGADO_REU,
    END: END # Palavra-chave especial do LangGraph para terminar o fluxo
}

# Adicionar arestas condicionais
# Após cada nó de agente, o router 'decidir_proximo_no_do_grafo' é chamado.
# O retorno do router (que é uma chave em 'roteamento_mapa_edges')
# determina para qual nó o fluxo seguirá.
workflow.add_conditional_edges(ADVOGADO_AUTOR, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(JUIZ, decidir_proximo_no_do_grafo, roteamento_mapa_edges)
workflow.add_conditional_edges(ADVOGADO_REU, decidir_proximo_no_do_grafo, roteamento_mapa_edges)

# Compilar o grafo para obter a aplicação executável
app = workflow.compile()

if __name__ == '__main__':
    print("--- Testando Definições do Grafo e Compilação ---")
    print(f"Tipo EstadoProcessual definido: {hasattr(EstadoProcessual, '__annotations__')}")
    print(f"Mapa de tarefas 'mapa_tarefa_no_atual' definido com {len(mapa_tarefa_no_atual)} entradas.")
    print(f"Função de roteamento 'decidir_proximo_no_do_grafo' definida: {callable(decidir_proximo_no_do_grafo)}")

    # Verifica se os nós foram criados corretamente com partial
    # (Isso é mais um teste da estrutura do que da funcionalidade em si sem execução)
    print(f"Nó para ADVOGADO_AUTOR criado: {callable(advogado_autor_node)}")
    print(f"Nó para JUIZ criado: {callable(juiz_node)}")
    print(f"Nó para ADVOGADO_REU criado: {callable(advogado_reu_node)}")

    if app:
        print("Grafo LangGraph ('app') compilado com sucesso.")
        print("Para testar a execução do grafo, você precisaria configurar um estado inicial completo,")
        print("incluindo um retriever e garantir que o LLM (via llm_models.py) esteja acessível.")
        print("Exemplo (não executado automaticamente):")
        print("""
        # estado_inicial_teste = EstadoProcessual(
        #     id_processo="teste_grafo_comp_001",
        #     retriever=meu_retriever_mock_ou_real, # Precisa ser inicializado
        #     nome_do_ultimo_no_executado=None,
        #     etapa_concluida_pelo_ultimo_no=None,
        #     proximo_ator_sugerido_pelo_ultimo_no=ADVOGADO_AUTOR,
        #     documento_gerado_na_etapa_recente=None,
        #     historico_completo=[],
        #     pontos_controvertidos_saneamento=None,
        #     manifestacao_autor_sem_provas=False,
        #     manifestacao_reu_sem_provas=False,
        #     dados_formulario_entrada={
        #         "qualificacao_autor": "Autor Teste...", "qualificacao_reu": "Réu Teste...",
        #         "natureza_acao": "Ação de Teste", "fatos": "Fatos.", "fundamentacao_juridica": "Direito.",
        #         "pedidos": "Pedidos.", "documentos_autor": []
        #     },
        #     documentos_juntados_pelo_reu=None,
        #     sentimento_peticao_inicial=None,
        #     sentimento_contestacao=None
        # )
        # for event in app.stream(input=estado_inicial_teste, config={"recursion_limit": 15}):
        #     for key, value in event.items():
        #         print(f"  Nó: {key}, Etapa Concluída: {value.get('etapa_concluida_pelo_ultimo_no', 'N/A')}")
        #     print("---")
        """)
    else:
        print("ERRO: Falha ao compilar o grafo LangGraph ('app').")

    print("\n--- Fim dos Testes de graph_definition.py ---")