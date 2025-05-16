from typing import List # Necessário para List[str] em teses_para_busca

# Nossos módulos
from agent_helpers import criar_prompt_e_chain # Para interagir com o LLM
from llm_models import search_tool # Para a busca de jurisprudência


def gerar_ementa_cnj_padrao(
    texto_sentenca: str,
    id_processo: str, # Mantido como argumento, embora o prompt original o comente
    # llm_usado: ChatGoogleGenerativeAI # O llm é acessado via criar_prompt_e_chain
) -> str:
    """
    Gera uma ementa para a sentença fornecida, seguindo o padrão da Recomendação CNJ 154/2024.
    O LLM é acessado através da função criar_prompt_e_chain.
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
    - Extraia as informações DIRETAMENTE da sentença fornecida para o Processo ID: """ + str(id_processo) + """. NÃO INVENTE informações.
    - Preencha TODAS as seções do padrão da ementa conforme especificado.
    - Seja fiel ao conteúdo e à terminologia da sentença.
    - Para "Ramo do Direito" e "Classe Processual", infira da sentença ou, se não explícito, deduza com base no conteúdo (ex: Direito Civil, Ação de Indenização por Danos Morais).
    - Mantenha a formatação EXATA, incluindo numeração, marcadores (I, II, III, IV), letras minúsculas entre parênteses para sub-itens de questões, e a linha "_________" antes dos dispositivos/jurisprudência citados.
    - Se uma seção não tiver conteúdo direto na sentença (ex: ausência de tese explícita), indique "Não consta expressamente na sentença." ou similar, mas tente ao máximo extrair ou inferir.

    Responda APENAS com a ementa formatada.

    **EMENTA GERADA (no padrão CNJ):**
    """
    try:
        chain_ementa = criar_prompt_e_chain(prompt_template_ementa)
        ementa_gerada = chain_ementa.invoke({
            "texto_sentenca": texto_sentenca,
            # "id_processo": id_processo # Já está no template string
        })
        return ementa_gerada
    except EnvironmentError as e_env: # Erro se o LLM não estiver configurado em criar_prompt_e_chain
        print(f"ERRO DE AMBIENTE ao gerar ementa CNJ: {e_env}")
        return f"Erro de ambiente ao gerar ementa: {e_env}. Verifique a configuração da API do LLM."
    except Exception as e:
        print(f"ERRO ao gerar ementa CNJ para o processo {id_processo}: {e}")
        return f"Erro ao gerar ementa: {e}"

def verificar_sentenca_com_jurisprudencia(
    texto_sentenca: str,
    # llm_usado: ChatGoogleGenerativeAI # LLM é acessado via criar_prompt_e_chain
) -> str:
    """
    Verifica a sentença comparando-a com jurisprudência encontrada via Google Search.
    Retorna uma string com a análise ou uma mensagem de erro/aviso.
    As chamadas de UI (st.spinner, etc.) foram removidas; o chamador é responsável por elas.
    """
    if not search_tool:
        msg = "Ferramenta de busca Google (search_tool) não está configurada ou disponível. Verifique llm_models.py e as chaves GOOGLE_API_KEY_SEARCH e GOOGLE_CSE_ID."
        print(f"AVISO [verificar_sentenca]: {msg}")
        return msg

    # 1. Extrair teses/palavras-chave da sentença para busca
    print("INFO [verificar_sentenca]: Extraindo teses da sentença...")
    prompt_extracao_teses = f"""
    Dada a seguinte sentença, extraia 2-3 teses jurídicas centrais ou os principais pontos de direito decididos.
    Formate cada tese como uma frase curta e objetiva, ideal para uma busca de jurisprudência.
    Se a sentença for complexa, foque nos pontos que seriam mais controversos ou relevantes para pesquisa jurisprudencial.
    Responda com cada tese em uma nova linha.

    Sentença (trecho inicial para identificação, o conteúdo completo foi analisado internamente):
    {texto_sentenca[:1500]}

    Teses/Palavras-chave para Busca (uma por linha):
    """
    try:
        chain_extracao = criar_prompt_e_chain(prompt_extracao_teses)
        teses_str = chain_extracao.invoke({}) # texto_sentenca já está no f-string
        teses_para_busca: List[str] = [t.strip() for t in teses_str.split('\n') if t.strip()]
        if not teses_para_busca:
            msg = "Não foi possível extrair teses da sentença para a busca."
            print(f"AVISO [verificar_sentenca]: {msg}")
            return msg
        print(f"INFO [verificar_sentenca]: Teses extraídas para busca: {teses_para_busca}")
    except EnvironmentError as e_env:
        print(f"ERRO DE AMBIENTE [verificar_sentenca] ao extrair teses: {e_env}")
        return f"Erro de ambiente ao extrair teses: {e_env}. Verifique a API do LLM."
    except Exception as e:
        print(f"ERRO [verificar_sentenca] ao extrair teses da sentença: {e}")
        return f"Erro ao extrair teses da sentença: {e}"

    # 2. Buscar jurisprudência para cada tese
    todos_resultados_busca_formatados: List[str] = []
    print(f"INFO [verificar_sentenca]: Buscando jurisprudência para {len(teses_para_busca)} tese(s)...")
    for i, tese in enumerate(teses_para_busca):
        if not tese:
            continue
        print(f"  Buscando por: '{tese}'...")
        try:
            query_busca = f'jurisprudência {tese}' # Adicionar "jurisprudência" refina a busca
            resultados_tese_str = search_tool.invoke(query_busca) # Passa a string diretamente
            todos_resultados_busca_formatados.append(f"Resultados da busca para '{tese}':\n{resultados_tese_str}\n---\n")
            print(f"  Resultados parciais para '{tese}' obtidos.")
        except Exception as e_busca:
            error_msg = f"Erro ao buscar jurisprudência por '{tese}': {e_busca}"
            print(f"  ERRO [verificar_sentenca]: {error_msg}")
            todos_resultados_busca_formatados.append(f"{error_msg}\n---\n")
    print("INFO [verificar_sentenca]: Busca de jurisprudência concluída.")

    snippets_jurisprudencia_str = "\n".join(todos_resultados_busca_formatados)
    if not snippets_jurisprudencia_str.strip() or all("Erro ao buscar" in res for res in todos_resultados_busca_formatados):
        snippets_jurisprudencia_str = "Nenhum resultado relevante encontrado nas buscas ou todas as buscas falharam."

    # 3. Análise comparativa pelo LLM
    print("INFO [verificar_sentenca]: Realizando análise comparativa da sentença com jurisprudência...")
    prompt_analise_sentenca = f"""
    Você é um jurista sênior analisando uma sentença judicial à luz da jurisprudência encontrada.

    **SENTENÇA ORIGINAL (Teses principais extraídas):**
    {teses_str} 
    (Fim das teses da sentença)

    **JURISPRUDÊNCIA ENCONTRADA (Snippets e Resumos de Buscas):**
    {snippets_jurisprudencia_str}

    **Tarefa:**
    Com base EXCLUSIVAMENTE na jurisprudência fornecida acima, avalie se as teses principais da sentença original parecem estar, em termos gerais, alinhadas ou desalinhadas com essa jurisprudência.
    Seja cauteloso e objetivo. Se a jurisprudência não for clara, suficiente ou diretamente aplicável, afirme isso.

    **Formato da Resposta:**
    1.  **Avaliação Geral:** (Ex: "Alinhada com a jurisprudência apresentada.", "Aparentemente desalinhada em relação a X.", "Parcialmente alinhada.", "Jurisprudência insuficiente para uma conclusão definitiva.")
    2.  **Justificativa Sucinta:** (Explique brevemente, apontando pontos de convergência ou divergência com base nos trechos da jurisprudência, ou a dificuldade de comparação.)
    3.  **Observação:** Lembre-se que esta é uma análise preliminar baseada em snippets de busca.

    **Análise da Sentença vs. Jurisprudência:**
    """
    try:
        chain_analise_final = criar_prompt_e_chain(prompt_analise_sentenca)
        analise_final = chain_analise_final.invoke({}) # Contexto já está no prompt
        print("INFO [verificar_sentenca]: Análise comparativa concluída.")
        return analise_final
    except EnvironmentError as e_env:
        print(f"ERRO DE AMBIENTE [verificar_sentenca] na análise final: {e_env}")
        return f"Erro de ambiente na análise comparativa: {e_env}. Verifique a API do LLM."
    except Exception as e:
        print(f"ERRO [verificar_sentenca] ao realizar análise comparativa da sentença: {e}")
        return f"Erro ao realizar análise comparativa da sentença: {e}"

if __name__ == '__main__':
    print("--- Testando Funcionalidades Judiciais ---")

    # Simular texto de sentença (exemplo muito breve)
    sentenca_exemplo = """
    JULGO PROCEDENTE o pedido para condenar a Ré ao pagamento de R$ 5.000,00 a título de danos morais.
    Fundamento na falha da prestação de serviço e no art. 14 do CDC. A Ré não comprovou excludente de responsabilidade.
    Ponto controvertido principal: existência do defeito no serviço.
    """
    id_processo_exemplo = "proc_judicial_001"

    print("\nTestando gerar_ementa_cnj_padrao:")
    # Este teste requer que o LLM (via criar_prompt_e_chain) esteja funcional.
    # Se GOOGLE_API_KEY não estiver no .env, criar_prompt_e_chain levantará EnvironmentError.
    try:
        ementa = gerar_ementa_cnj_padrao(sentenca_exemplo, id_processo_exemplo)
        print(f"  Ementa Gerada (ou mensagem de erro):\n{ementa}")
    except Exception as e_test_ementa:
        print(f"  ERRO INESPERADO no teste de gerar_ementa_cnj_padrao: {e_test_ementa}")


    print("\nTestando verificar_sentenca_com_jurisprudencia:")
    # Este teste requer que o LLM E o search_tool estejam funcionais.
    # Se as chaves não estiverem no .env, as funções internas lidarão com isso.
    try:
        verificacao = verificar_sentenca_com_jurisprudencia(sentenca_exemplo)
        print(f"  Resultado da Verificação (ou mensagem de erro):\n{verificacao}")
    except Exception as e_test_verif:
        print(f"  ERRO INESPERADO no teste de verificar_sentenca_com_jurisprudencia: {e_test_verif}")

    print("\n--- Fim dos Testes de Funcionalidades Judiciais ---")