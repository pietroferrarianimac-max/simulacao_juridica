from typing import List, Dict, Any

# LangChain Core (se os helpers interagirem diretamente com componentes LangChain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Importar LLM de llm_models.py
from llm_models import llm

# Importar EstadoProcessual e mapa_tarefa_no_atual de graph_definition.py (será criado depois)
# Para evitar dependência circular no momento da criação, vamos definir o tipo EstadoProcessual
# e mapa_tarefa_no_atual como 'Any' por enquanto, ou importar apenas o necessário.
# Idealmente, o 'mapa_tarefa_no_atual' e as constantes de etapa seriam de settings.py
# e 'EstadoProcessual' seria de graph_definition.py.

from settings import (
    ADVOGADO_AUTOR, # Necessário para helper_logica_inicial_no
    # (outras constantes de ator/etapa se o helper_logica_inicial_no precisar delas diretamente)
)
# Nota: mapa_tarefa_no_atual será passado como argumento para helper_logica_inicial_no
# para evitar importação direta de graph_definition aqui e potencial ciclo.

def criar_prompt_e_chain(template_string: str) -> Any: # Retorna uma LangChain Runnable
    """Cria uma cadeia simples de prompt, LLM e parser de string."""
    if not llm:
        # Esta é uma condição crítica. Se o LLM não estiver disponível,
        # a aplicação principal (Streamlit) deve ser notificada.
        # Levantar uma exceção aqui é uma boa prática.
        raise EnvironmentError(
            "LLM não inicializado. "
            "Verifique a configuração da GOOGLE_API_KEY em settings.py e llm_models.py."
        )
    prompt = ChatPromptTemplate.from_template(template_string)
    return prompt | llm | StrOutputParser()

def helper_logica_inicial_no(
    nome_ultimo_no: str | None,
    etapa_ultimo_no: str | None,
    nome_do_no_atual: str,
    mapa_tarefas: Dict[tuple[str | None, str | None, str], str]
) -> str:
    """
    Determina qual etapa processual o nó atual deve executar.

    Args:
        nome_ultimo_no: Nome do último nó executado.
        etapa_ultimo_no: Etapa concluída pelo último nó.
        nome_do_no_atual: Nome do nó que está iniciando sua execução.
        mapa_tarefas: O dicionário 'mapa_tarefa_no_atual' vindo da definição do grafo.

    Returns:
        A string da etapa a ser executada ou uma string de erro.
    """
    chave_mapa = (nome_ultimo_no, etapa_ultimo_no, nome_do_no_atual)

    etapa_designada: str | None = "" # Inicializa como None ou string vazia

    # Caso especial: Ponto de entrada do grafo
    if nome_ultimo_no is None and etapa_ultimo_no is None and nome_do_no_atual == ADVOGADO_AUTOR:
        etapa_designada = mapa_tarefas.get((None, None, ADVOGADO_AUTOR))
        if etapa_designada:
            print(f"INFO [{nome_do_no_atual}] Ponto de entrada. Etapa designada: {etapa_designada}.")
        else:
            # Este é um erro de configuração do mapa se a entrada não estiver definida
            print(f"ERRO [{nome_do_no_atual}] Ponto de entrada não encontrado no mapa_tarefas para ADVOGADO_AUTOR.")
            return "ERRO_CONFIG_ENTRADA_NAO_MAPEADA"
    else:
        etapa_designada = mapa_tarefas.get(chave_mapa)
        if etapa_designada:
            print(f"INFO [{nome_do_no_atual}] Etapa designada pelo mapa: {etapa_designada} (usando chave: {chave_mapa}).")
        else:
            # Log detalhado se a chave não for encontrada no mapa
            print(f"ERRO [{nome_do_no_atual}]: Chave {chave_mapa} não encontrada no mapa_tarefas fornecido.")
            # Para depuração, você pode querer logar as chaves disponíveis no mapa:
            # print(f"  Chaves disponíveis no mapa_tarefas: {list(mapa_tarefas.keys())}")
            return "ERRO_ETAPA_NAO_ENCONTRADA_NO_MAPA"

    if not etapa_designada: # Checagem final, embora os caminhos acima devam cobrir
        print(f"ALERTA [{nome_do_no_atual}]: Não foi possível determinar a etapa processual com a chave {chave_mapa}.")
        return "ERRO_ETAPA_INDETERMINADA_FINAL"

    return etapa_designada


def formatar_lista_documentos_para_prompt(documentos: List[Dict[str, str]], parte_nome: str) -> str:
    """
    Formata uma lista de dicionários de documentos para inclusão em prompts dos agentes.
    Cada dicionário deve ter 'tipo' e 'descricao'.
    """
    if not documentos:
        return f"Nenhum documento específico foi listado por {parte_nome} para esta etapa."

    texto_docs = f"\n\n**Documentos que acompanham esta peça processual ({parte_nome}):**\n"
    for i, doc_info in enumerate(documentos):
        tipo = doc_info.get('tipo', 'N/A (Tipo não especificado)')
        # Tenta ser flexível com a chave da descrição ('descricao' ou 'description')
        descricao = doc_info.get('descricao', doc_info.get('description', 'N/A (Descrição não fornecida)'))
        texto_docs += f"{i+1}. **Tipo do Documento:** {tipo}\n   **Descrição/Propósito:** {descricao}\n"
    return texto_docs

if __name__ == '__main__':
    print("--- Testando Agent Helpers ---")

    # Teste para criar_prompt_e_chain (requer LLM configurado e GOOGLE_API_KEY)
    print("\nTestando criar_prompt_e_chain:")
    try:
        chain_teste = criar_prompt_e_chain("Qual a capital do Brasil? Resposta: {capital}")
        # print(f"Chain criada: {chain_teste}")
        # Se o LLM estiver funcionando, você pode tentar invocar:
        # resposta = chain_teste.invoke({"capital": "Brasília"})
        # print(f"  Resposta da chain (exemplo): {resposta}")
        print("  criar_prompt_e_chain executado (verifique se o LLM está configurado para teste completo).")
    except EnvironmentError as e:
        print(f"  ERRO ao testar criar_prompt_e_chain: {e}")
    except Exception as e:
        print(f"  ERRO INESPERADO ao testar criar_prompt_e_chain: {e}")


    # Teste para helper_logica_inicial_no
    print("\nTestando helper_logica_inicial_no:")
    mapa_teste = {
        (None, None, "advogado_autor"): "PETICAO_INICIAL_TESTE",
        ("advogado_autor", "PETICAO_INICIAL_TESTE", "juiz"): "DESPACHO_INICIAL_TESTE",
        ("juiz", "DESPACHO_INICIAL_TESTE", "advogado_reu"): "CONTESTACAO_TESTE",
    }
    # Cenário 1: Ponto de entrada
    etapa1 = helper_logica_inicial_no(None, None, "advogado_autor", mapa_teste)
    print(f"  Cenário 1 (Entrada): Esperado PETICAO_INICIAL_TESTE -> Obtido: {etapa1}")
    assert etapa1 == "PETICAO_INICIAL_TESTE"

    # Cenário 2: Fluxo normal
    etapa2 = helper_logica_inicial_no("advogado_autor", "PETICAO_INICIAL_TESTE", "juiz", mapa_teste)
    print(f"  Cenário 2 (Fluxo): Esperado DESPACHO_INICIAL_TESTE -> Obtido: {etapa2}")
    assert etapa2 == "DESPACHO_INICIAL_TESTE"

    # Cenário 3: Chave não encontrada
    etapa3 = helper_logica_inicial_no("advogado_autor", "ETAPA_ERRADA", "juiz", mapa_teste)
    print(f"  Cenário 3 (Erro): Esperado ERRO_ETAPA_NAO_ENCONTRADA_NO_MAPA -> Obtido: {etapa3}")
    assert etapa3 == "ERRO_ETAPA_NAO_ENCONTRADA_NO_MAPA"

    # Teste para formatar_lista_documentos_para_prompt
    print("\nTestando formatar_lista_documentos_para_prompt:")
    docs_teste_vazio = []
    docs_teste_com_dados = [
        {"tipo": "RG", "descricao": "Documento de identidade do autor."},
        {"tipo": "Contrato", "description": "Contrato de prestação de serviços assinado."}, # Testando 'description'
        {"tipo": "Comprovante"} # Testando descrição faltando
    ]
    print(f"  Formatado (vazio):\n{formatar_lista_documentos_para_prompt(docs_teste_vazio, 'Autor Teste')}")
    print(f"  Formatado (com dados):\n{formatar_lista_documentos_para_prompt(docs_teste_com_dados, 'Autor Teste')}")

    print("\n--- Fim dos Testes Agent Helpers ---")