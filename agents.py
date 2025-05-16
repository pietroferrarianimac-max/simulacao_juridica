# agents.py
from typing import TYPE_CHECKING, Dict, Any, Tuple # Adicionar TYPE_CHECKING

if TYPE_CHECKING:
    from graph_definition import EstadoProcessual # Apenas para type hinting



from agent_helpers import (
    criar_prompt_e_chain,
    helper_logica_inicial_no,
    formatar_lista_documentos_para_prompt
)
from settings import (
    ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU,
    ETAPA_PETICAO_INICIAL, ETAPA_DESPACHO_RECEBENDO_INICIAL, ETAPA_CONTESTACAO,
    ETAPA_DECISAO_SANEAMENTO, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    ETAPA_MANIFESTACAO_SEM_PROVAS_REU, ETAPA_SENTENCA, ETAPA_FIM_PROCESSO
)


EstadoProcessual = Dict[str, Any]


def agente_advogado_autor(estado: EstadoProcessual, mapa_tarefas: Dict[Tuple[str | None, str | None, str], str]) -> Dict[str, Any]:
    # etapa_atual_do_no = helper_logica_inicial_no(estado, ADVOGADO_AUTOR) # Original
    nome_ultimo_no = estado.get("nome_do_ultimo_no_executado")
    etapa_ultimo_no = estado.get("etapa_concluida_pelo_ultimo_no")
    etapa_atual_do_no = helper_logica_inicial_no(nome_ultimo_no, etapa_ultimo_no, ADVOGADO_AUTOR, mapa_tarefas)

    print(f"\n--- TURNO: {ADVOGADO_AUTOR} (Executando Etapa: {etapa_atual_do_no}) ---")

    if "ERRO" in etapa_atual_do_no:
        # Simplified error state, assuming all necessary keys are preserved or re-added by the graph
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
            "etapa_concluida_pelo_ultimo_no": f"ERRO_FLUXO_AUTOR_{etapa_atual_do_no}",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": f"Erro crítico de fluxo no {ADVOGADO_AUTOR}: {etapa_atual_do_no}.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO_FLUXO", "ator": ADVOGADO_AUTOR, "documento": f"Erro: {etapa_atual_do_no}"}],
        }

    documento_gerado = f"Documento padrão para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no} (lógica pendente)."
    proximo_ator_logico = JUIZ
    sentimento_pi_texto_gerado = estado.get("sentimento_peticao_inicial")

    retriever = estado.get("retriever") # Get retriever from state
    id_processo = estado.get("id_processo", "ID_DESCONHECIDO")
    dados_formulario = estado.get("dados_formulario_entrada", {})
    historico_formatado = "\n".join([
        f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Doc (trecho): {str(item.get('documento',''))[:150]}..."
        for item in estado.get("historico_completo", [])
    ])
    if not historico_formatado: historico_formatado = "Este é o primeiro ato do processo."

    if etapa_atual_do_no == ETAPA_PETICAO_INICIAL:
        modelo_texto_guia = "Modelo de Petição Inicial não carregado (RAG não disponível ou falhou)."
        if retriever:
            try:
                docs_modelo_pi = retriever.get_relevant_documents(
                    query="modelo de petição inicial cível completa e bem estruturada",
                )
                if docs_modelo_pi:
                    modelo_texto_guia = docs_modelo_pi[0].page_content
                else:
                    print(f"AVISO [{ADVOGADO_AUTOR}-{etapa_atual_do_no}]: Nenhum modelo de PI encontrado via RAG.")
            except Exception as e_rag:
                print(f"ERRO [{ADVOGADO_AUTOR}-{etapa_atual_do_no}]: Falha ao buscar modelo de PI via RAG: {e_rag}")
        else:
            print(f"ALERTA [{ADVOGADO_AUTOR}-{etapa_atual_do_no}]: Retriever não disponível no estado.")

        qualificacao_autor_form = dados_formulario.get("qualificacao_autor", "Qualificação do Autor não fornecida.")
        qualificacao_reu_form = dados_formulario.get("qualificacao_reu", "Qualificação do Réu não fornecida.")
        natureza_acao_form = dados_formulario.get("natureza_acao", "Natureza da ação não fornecida.")
        fatos_form = dados_formulario.get("fatos", "Fatos não fornecidos.")
        direito_form = dados_formulario.get("fundamentacao_juridica", "Fundamentação jurídica não fornecida.")
        pedidos_form = dados_formulario.get("pedidos", "Pedidos não fornecidos.")
        documentos_autor_lista = dados_formulario.get("documentos_autor", [])
        documentos_autor_texto_formatado = formatar_lista_documentos_para_prompt(documentos_autor_lista, "Autor")

        template_prompt_pi = f"""
        Você é um Advogado do Autor experiente e está elaborando uma Petição Inicial completa, formal e persuasiva.
        **Processo ID:** {id_processo}
        **Dados Base Fornecidos para a Petição:**
        Qualificação do Autor: {qualificacao_autor_form}
        Qualificação do Réu: {qualificacao_reu_form}
        Natureza da Ação: {natureza_acao_form}
        Dos Fatos: {fatos_form}
        Do Direito (Fundamentação Jurídica): {direito_form}
        Dos Pedidos: {pedidos_form}
        {documentos_autor_texto_formatado}
        **Modelo/Guia Estrutural de Petição Inicial (RAG - use para formatação, completude e referências legais, mas priorize os dados fornecidos acima para o conteúdo do caso):**
        {modelo_texto_guia}
        **Instruções Adicionais:**
        1. Redija a Petição Inicial completa e bem formatada, seguindo a praxe forense.
        2. Certifique-se de que todos os elementos dos DADOS BASE (fatos, direito, pedidos, qualificações, natureza da ação) estejam integralmente e corretamente incorporados.
        3. No corpo da petição (especialmente na narração dos fatos ou antes dos pedidos), faça menção aos principais documentos listados em "Documentos que acompanham esta petição (Autor)", indicando sua relevância para comprovar as alegações.
        4. Conclua com os requerimentos de praxe (data, assinatura do advogado).
        Petição Inicial:
        """
        chain_pi = criar_prompt_e_chain(template_prompt_pi)
        documento_gerado = chain_pi.invoke({})

        sentimento_pi_texto_gerado = "Não analisado" # Reset before analysis
        try:
            prompt_sentimento_pi = f"""
            Analise o tom e o sentimento predominante do seguinte texto jurídico (Petição Inicial).
            Responda com uma única palavra ou expressão curta que melhor descreva o sentimento (ex: Assertivo, Conciliatório, Agressivo, Neutro, Persuasivo, Formal, Emocional, Confiante, Defensivo, Indignado, Colaborativo).
            Seja conciso.
            Texto da Petição Inicial:
            {documento_gerado[:3000]}
            Sentimento Predominante:"""
            chain_sentimento_pi = criar_prompt_e_chain(prompt_sentimento_pi)
            sentimento_pi_texto_gerado = chain_sentimento_pi.invoke({})
            print(f"INFO [{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Sentimento da PI: {sentimento_pi_texto_gerado}")
        except Exception as e_sent:
            print(f"ERRO [{ADVOGADO_AUTOR}-{etapa_atual_do_no}] ao analisar sentimento da PI: {e_sent}")
            sentimento_pi_texto_gerado = "Erro na análise"
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR:
        decisao_saneamento_recebida = estado.get("documento_gerado_na_etapa_recente", "ERRO: Decisão de Saneamento não encontrada no estado.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não definidos na decisão de saneamento.")
        historico_completo_formatado_para_prompt = "\n".join([f"### Documento da Etapa: {item['etapa']} (Ator: {item['ator']})\n{item['documento']}\n---" for item in estado.get("historico_completo", [])])

        template_prompt_manifestacao_autor = f"""
        Você é o Advogado do Autor. O Juiz proferiu a Decisão de Saneamento e intimou as partes para especificarem as provas que pretendem produzir, ou manifestarem desinteresse na produção de mais provas.
        Seu cliente (Autor) informou que não possui mais provas a produzir e deseja o julgamento antecipado da lide, se o Réu também não tiver provas.
        **Processo ID:** {id_processo}
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
        documento_gerado = chain_manifestacao_autor.invoke({})
        proximo_ator_logico = ADVOGADO_REU
    else:
        print(f"AVISO [{ADVOGADO_AUTOR}]: Lógica para etapa '{etapa_atual_do_no}' não implementada completamente.")
        documento_gerado = f"Conteúdo para {ADVOGADO_AUTOR} na etapa {etapa_atual_do_no}."

    print(f"INFO [{ADVOGADO_AUTOR}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_AUTOR, "documento": documento_gerado}

    # Retorna apenas os campos que este agente modifica ou que são essenciais para o próximo passo.
    # O LangGraph se encarrega de mesclar isso com o estado existente.
    return {
        "nome_do_ultimo_no_executado": ADVOGADO_AUTOR,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": estado.get("historico_completo", []) + [novo_historico_item],
        "manifestacao_autor_sem_provas": estado.get("manifestacao_autor_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR),
        "sentimento_peticao_inicial": sentimento_pi_texto_gerado,
    }

def agente_juiz(estado: EstadoProcessual, mapa_tarefas: Dict[Tuple[str | None, str | None, str], str]) -> Dict[str, Any]:
    nome_ultimo_no = estado.get("nome_do_ultimo_no_executado")
    etapa_ultimo_no = estado.get("etapa_concluida_pelo_ultimo_no")
    etapa_atual_do_no = helper_logica_inicial_no(nome_ultimo_no, etapa_ultimo_no, JUIZ, mapa_tarefas)

    print(f"\n--- TURNO: {JUIZ} (Executando Etapa: {etapa_atual_do_no}) ---")

    if "ERRO" in etapa_atual_do_no:
        return {
            "nome_do_ultimo_no_executado": JUIZ,
            "etapa_concluida_pelo_ultimo_no": f"ERRO_FLUXO_JUIZ_{etapa_atual_do_no}",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": f"Erro crítico de fluxo no {JUIZ}: {etapa_atual_do_no}.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO_FLUXO", "ator": JUIZ, "documento": f"Erro: {etapa_atual_do_no}"}],
        }

    documento_gerado = f"Decisão padrão para {JUIZ} na etapa {etapa_atual_do_no} (lógica pendente)."
    proximo_ator_logico = ETAPA_FIM_PROCESSO
    pontos_controvertidos_definidos_nesta_etapa = estado.get("pontos_controvertidos_saneamento")

    retriever = estado.get("retriever")
    id_processo = estado.get("id_processo", "ID_DESCONHECIDO")
    documento_da_parte_para_analise = estado.get("documento_gerado_na_etapa_recente", "Nenhuma peça recente para análise.")
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Doc: {str(item.get('documento',''))[:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."

    if etapa_atual_do_no == ETAPA_DESPACHO_RECEBENDO_INICIAL:
        modelo_texto_guia = "Modelo de Despacho não carregado."
        if retriever:
            try:
                docs_modelo_despacho = retriever.get_relevant_documents(query="modelo de despacho judicial cível recebendo petição inicial e determinando citação")
                if docs_modelo_despacho: modelo_texto_guia = docs_modelo_despacho[0].page_content
            except Exception as e_rag: print(f"ERRO RAG [{JUIZ}-{etapa_atual_do_no}]: {e_rag}")
        else: print(f"ALERTA [{JUIZ}-{etapa_atual_do_no}]: Retriever não disponível.")

        template_prompt = f"""
        Você é um Juiz de Direito. Analise a Petição Inicial apresentada e, se estiver em ordem, profira um despacho inicial determinando a citação do réu.
        Considere os documentos que acompanham a inicial, conforme nela mencionados.
        **Processo ID:** {id_processo}
        **Petição Inicial apresentada pelo Autor (pode incluir menção a documentos anexos):**
        {documento_da_parte_para_analise}
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
        documento_gerado = chain.invoke({})
        proximo_ator_logico = ADVOGADO_REU

    elif etapa_atual_do_no == ETAPA_DECISAO_SANEAMENTO:
        modelo_texto_guia = "Modelo de Saneamento não carregado."
        if retriever:
            try:
                docs_modelo_saneamento = retriever.get_relevant_documents(query="modelo de decisão de saneamento e organização do processo cível")
                if docs_modelo_saneamento: modelo_texto_guia = docs_modelo_saneamento[0].page_content
            except Exception as e_rag: print(f"ERRO RAG [{JUIZ}-{etapa_atual_do_no}]: {e_rag}")
        else: print(f"ALERTA [{JUIZ}-{etapa_atual_do_no}]: Retriever não disponível.")

        documentos_autor_lista = estado.get("dados_formulario_entrada", {}).get("documentos_autor", [])
        documentos_autor_texto = formatar_lista_documentos_para_prompt(documentos_autor_lista, "Autor")
        documentos_reu_lista = estado.get("documentos_juntados_pelo_reu", [])
        documentos_reu_texto = formatar_lista_documentos_para_prompt(documentos_reu_lista, "Réu")
        # 'documento_da_parte_para_analise' aqui é a contestação.

        template_prompt = f"""
        Você é um Juiz de Direito. O processo está na fase de saneamento após a apresentação da contestação.
        Analise a Petição Inicial (no histórico), a Contestação e os documentos juntados por ambas as partes.
        **Processo ID:** {id_processo}
        **Petição Inicial e Documentos do Autor (resumo/menção - conteúdo completo no histórico):**
        {documentos_autor_texto}
        **Contestação do Réu e Documentos do Réu (para análise):**
        {documento_da_parte_para_analise} 
        {documentos_reu_texto} 
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
        5. Intime as partes para especificarem as provas que pretendem produzir, advertindo que audiência não está prevista neste MVP.
        Certifique-se de que a decisão seja clara e objetiva.
        Decisão de Saneamento:
        """
        chain = criar_prompt_e_chain(template_prompt)
        documento_gerado = chain.invoke({})
        proximo_ator_logico = ADVOGADO_AUTOR

        try:
            inicio_pc = documento_gerado.upper().find("PONTOS CONTROVERTIDOS:")
            if inicio_pc != -1:
                fim_pc = documento_gerado.find("\n\n", inicio_pc)
                if fim_pc == -1: fim_pc = len(documento_gerado)
                pontos_controvertidos_definidos_nesta_etapa = documento_gerado[inicio_pc + len("PONTOS CONTROVERTIDOS:"):fim_pc].strip()
            else: pontos_controvertidos_definidos_nesta_etapa = "Não extraído explicitamente da decisão de saneamento."
            print(f"INFO [{JUIZ}-{etapa_atual_do_no}] Pontos Controvertidos Definidos/Extraídos: {pontos_controvertidos_definidos_nesta_etapa}")
        except Exception as e_pc:
            print(f"ERRO [{JUIZ}-{etapa_atual_do_no}] ao extrair Pontos Controvertidos: {e_pc}")
            pontos_controvertidos_definidos_nesta_etapa = "Erro na extração dos pontos controvertidos."

    elif etapa_atual_do_no == ETAPA_SENTENCA:
        peticao_inicial_completa, contestacao_completa, decisao_saneamento_completa = "N/A", "N/A", "N/A"
        manifestacao_autor_sem_provas_texto, manifestacao_reu_sem_provas_texto = "N/A", "N/A"
        for item in estado.get("historico_completo", []):
            if item['etapa'] == ETAPA_PETICAO_INICIAL: peticao_inicial_completa = item['documento']
            elif item['etapa'] == ETAPA_CONTESTACAO: contestacao_completa = item['documento']
            elif item['etapa'] == ETAPA_DECISAO_SANEAMENTO: decisao_saneamento_completa = item['documento']
            elif item['etapa'] == ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR: manifestacao_autor_sem_provas_texto = item['documento']
            elif item['etapa'] == ETAPA_MANIFESTACAO_SEM_PROVAS_REU: manifestacao_reu_sem_provas_texto = item['documento']

        modelo_texto_guia = "Modelo de Sentença não carregado."
        if retriever:
            try:
                docs_modelo_sentenca = retriever.get_relevant_documents(query="modelo de sentença cível completa de mérito")
                if docs_modelo_sentenca: modelo_texto_guia = docs_modelo_sentenca[0].page_content
            except Exception as e_rag: print(f"ERRO RAG [{JUIZ}-{etapa_atual_do_no}]: {e_rag}")
        else: print(f"ALERTA [{JUIZ}-{etapa_atual_do_no}]: Retriever não disponível.")
        
        documentos_autor_lista_estado = estado.get("dados_formulario_entrada", {}).get("documentos_autor", [])
        documentos_autor_texto_formatado_estado = formatar_lista_documentos_para_prompt(documentos_autor_lista_estado, "Autor")
        documentos_reu_lista_estado = estado.get("documentos_juntados_pelo_reu", [])
        documentos_reu_texto_formatado_estado = formatar_lista_documentos_para_prompt(documentos_reu_lista_estado, "Réu")

        template_prompt_sentenca = f"""
        Você é um Juiz de Direito e deve proferir a Sentença neste processo.
        As partes (Autor e Réu) manifestaram desinteresse na produção de outras provas, requerendo o julgamento antecipado da lide.
        **Processo ID:** {id_processo}
        **Peças Processuais Principais e Histórico Completo (para sua análise):**
        Petição Inicial: {peticao_inicial_completa}
        --- Documentos do Autor (listados na inicial ou formulário): {documentos_autor_texto_formatado_estado}
        Contestação: {contestacao_completa}
        --- Documentos do Réu (listados na contestação ou gerados): {documentos_reu_texto_formatado_estado}
        Decisão de Saneamento (contém os pontos controvertidos): {decisao_saneamento_completa}
        Manifestação do Autor sobre Provas: {manifestacao_autor_sem_provas_texto}
        Manifestação do Réu sobre Provas: {manifestacao_reu_sem_provas_texto}
        **Modelo/Guia de Sentença (use como referência para estrutura e formalidades):**
        {modelo_texto_guia}
        **Histórico Processual Completo Adicional (se necessário):**
        {historico_formatado}
        ---
        **Instruções para a Sentença:**
        1. Elabore um relatório conciso.
        2. Apresente a fundamentação, analisando as questões de fato e de direito, examinando as provas (documentais) em relação aos pontos controvertidos.
        3. Profira o dispositivo (procedente, parcialmente procedente ou improcedente).
        4. Condene a parte vencida em custas e honorários (ex: 10%).
        Sentença:
        """
        chain_sentenca = criar_prompt_e_chain(template_prompt_sentenca)
        documento_gerado = chain_sentenca.invoke({})
        proximo_ator_logico = ETAPA_FIM_PROCESSO
    else:
        print(f"AVISO [{JUIZ}]: Lógica para etapa '{etapa_atual_do_no}' não implementada.")

    print(f"INFO [{JUIZ}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": JUIZ, "documento": documento_gerado}

    return {
        "nome_do_ultimo_no_executado": JUIZ,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado,
        "historico_completo": estado.get("historico_completo", []) + [novo_historico_item],
        "pontos_controvertidos_saneamento": pontos_controvertidos_definidos_nesta_etapa,
    }

def agente_advogado_reu(estado: EstadoProcessual, mapa_tarefas: Dict[Tuple[str | None, str | None, str], str]) -> Dict[str, Any]:
    nome_ultimo_no = estado.get("nome_do_ultimo_no_executado")
    etapa_ultimo_no = estado.get("etapa_concluida_pelo_ultimo_no")
    etapa_atual_do_no = helper_logica_inicial_no(nome_ultimo_no, etapa_ultimo_no, ADVOGADO_REU, mapa_tarefas)

    print(f"\n--- TURNO: {ADVOGADO_REU} (Executando Etapa: {etapa_atual_do_no}) ---")

    if "ERRO" in etapa_atual_do_no:
        return {
            "nome_do_ultimo_no_executado": ADVOGADO_REU,
            "etapa_concluida_pelo_ultimo_no": f"ERRO_FLUXO_REU_{etapa_atual_do_no}",
            "proximo_ator_sugerido_pelo_ultimo_no": ETAPA_FIM_PROCESSO,
            "documento_gerado_na_etapa_recente": f"Erro crítico de fluxo no {ADVOGADO_REU}: {etapa_atual_do_no}.",
            "historico_completo": estado.get("historico_completo", []) + [{"etapa": "ERRO_FLUXO", "ator": ADVOGADO_REU, "documento": f"Erro: {etapa_atual_do_no}"}],
        }

    documento_gerado_principal = f"Documento padrão para {ADVOGADO_REU} na etapa {etapa_atual_do_no} (lógica pendente)."
    proximo_ator_logico = JUIZ
    # Inicializa para preservar valores de execuções anteriores se não for a etapa de contestação
    lista_documentos_juntados_pelo_reu_final = estado.get("documentos_juntados_pelo_reu", [])
    sentimento_contestacao_texto_gerado = estado.get("sentimento_contestacao")


    retriever = estado.get("retriever")
    id_processo = estado.get("id_processo", "ID_DESCONHECIDO")
    documento_relevante_anterior = estado.get("documento_gerado_na_etapa_recente", "Nenhum doc anterior informado.")
    historico_formatado = "\n".join([f"- Etapa: {item['etapa']}, Ator: {item['ator']}:\n  Doc: {str(item.get('documento',''))[:150]}..." for item in estado.get("historico_completo", [])])
    if not historico_formatado: historico_formatado = "Histórico não disponível."


    if etapa_atual_do_no == ETAPA_CONTESTACAO:
        peticao_inicial_autor_texto_completo = "Petição Inicial do Autor não encontrada no histórico."
        for item_hist in reversed(estado.get("historico_completo", [])): # Procura de trás para frente
            if item_hist["etapa"] == ETAPA_PETICAO_INICIAL and item_hist["ator"] == ADVOGADO_AUTOR:
                peticao_inicial_autor_texto_completo = item_hist["documento"]
                break
        
        modelo_texto_guia = "Modelo de Contestação não carregado."
        if retriever:
            try:
                docs_modelo_contestacao = retriever.get_relevant_documents(query="modelo de contestação cível completa e bem fundamentada")
                if docs_modelo_contestacao: modelo_texto_guia = docs_modelo_contestacao[0].page_content
            except Exception as e_rag: print(f"ERRO RAG [{ADVOGADO_REU}-{etapa_atual_do_no}]: {e_rag}")
        else: print(f"ALERTA [{ADVOGADO_REU}-{etapa_atual_do_no}]: Retriever não disponível.")

        template_prompt_contestacao = f"""
        Você é um Advogado do Réu experiente. Sua tarefa é elaborar uma Contestação completa e robusta.
        **Processo ID:** {id_processo}
        **Despacho Judicial Recebido (determinando a citação/contestação):**
        {documento_relevante_anterior}
        **Petição Inicial do Autor (que originou esta contestação e pode mencionar documentos juntados pelo Autor):**
        {peticao_inicial_autor_texto_completo}
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
        6. A contestação deve ser bem estruturada.
        Contestação:
        """
        chain_contestacao = criar_prompt_e_chain(template_prompt_contestacao)
        documento_gerado_principal = chain_contestacao.invoke({})

        # Gerar lista de documentos do Réu
        fatos_gerais_caso = estado.get("dados_formulario_entrada", {}).get("fatos", "Fatos do caso não disponíveis.")
        pi_resumo_para_prompt_docs = peticao_inicial_autor_texto_completo[:1000] + ("..." if len(peticao_inicial_autor_texto_completo) > 1000 else peticao_inicial_autor_texto_completo)
        prompt_docs_reu_template = f"""
        Com base na Petição Inicial do Autor, na Contestação do Réu recém-elaborada, e nos fatos gerais do caso, você deve listar de 2 a 4 documentos principais que o Réu provavelmente juntaria para dar suporte à sua defesa.
        Para cada documento, forneça o tipo e uma descrição MUITO SUCINTA (1 frase, máximo 20 palavras).
        Petição Inicial do Autor (Resumo): {pi_resumo_para_prompt_docs}
        Contestação do Réu (Completa - elaborada para este caso): {documento_gerado_principal}
        Fatos Gerais do Caso (fornecidos no início da simulação): {fatos_gerais_caso}
        Sua resposta DEVE SER uma lista de strings, onde cada string representa um documento no formato: "Tipo do Documento: Descrição sucinta."
        Exemplo:
        Documento de Identidade do Réu: RG e CPF para qualificação.
        Contrato de Locação: Cópia do contrato que estabelece obrigações.
        Liste os documentos do Réu:
        """
        chain_docs_reu = criar_prompt_e_chain(prompt_docs_reu_template)
        resposta_docs_reu_str = chain_docs_reu.invoke({})
        
        parsed_docs_reu = []
        if resposta_docs_reu_str and resposta_docs_reu_str.strip():
            linhas_docs = resposta_docs_reu_str.strip().split('\n')
            for linha in linhas_docs:
                if ':' in linha:
                    partes = linha.split(':', 1)
                    tipo_doc, desc_doc = partes[0].strip(), partes[1].strip()
                    if tipo_doc and desc_doc: parsed_docs_reu.append({"tipo": tipo_doc, "descricao": desc_doc})
        
        if not parsed_docs_reu:
            print(f"AVISO [{ADVOGADO_REU}-{etapa_atual_do_no}]: Não foi possível gerar ou parsear lista de documentos do Réu.")
            lista_documentos_juntados_pelo_reu_final = [{"tipo": "Informação", "descricao": "A IA não especificou docs para o réu nesta etapa."}]
        else:
            lista_documentos_juntados_pelo_reu_final = parsed_docs_reu
            print(f"INFO [{ADVOGADO_REU}-{etapa_atual_do_no}] Documentos do Réu gerados: {lista_documentos_juntados_pelo_reu_final}")

        documentos_reu_texto_para_anexar = formatar_lista_documentos_para_prompt(lista_documentos_juntados_pelo_reu_final, "Réu")
        documento_gerado_principal += f"\n\n---\n{documentos_reu_texto_para_anexar}"
        
        sentimento_contestacao_texto_gerado = "Não analisado" # Reset
        try:
            prompt_sentimento_contestacao = f"""
            Analise o tom e o sentimento predominante do seguinte texto jurídico (Contestação).
            Responda com uma única palavra ou expressão curta (ex: Assertivo, Conciliatório, Agressivo, Neutro).
            Texto da Contestação:
            {documento_gerado_principal[:3000]}
            Sentimento Predominante:"""
            chain_sentimento_contestacao = criar_prompt_e_chain(prompt_sentimento_contestacao)
            sentimento_contestacao_texto_gerado = chain_sentimento_contestacao.invoke({})
            print(f"INFO [{ADVOGADO_REU}-{etapa_atual_do_no}] Sentimento da Contestação: {sentimento_contestacao_texto_gerado}")
        except Exception as e_sent_cont:
            print(f"ERRO [{ADVOGADO_REU}-{etapa_atual_do_no}] ao analisar sentimento da Contestação: {e_sent_cont}")
            sentimento_contestacao_texto_gerado = "Erro na análise"
        proximo_ator_logico = JUIZ

    elif etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU:
        decisao_saneamento_juiz = "Decisão de Saneamento não encontrada no histórico."
        for item_hist in reversed(estado.get("historico_completo", [])):
            if item_hist["etapa"] == ETAPA_DECISAO_SANEAMENTO and item_hist["ator"] == JUIZ:
                decisao_saneamento_juiz = item_hist["documento"]
                break
        manifestacao_autor_recente = estado.get("documento_gerado_na_etapa_recente", "Manifestação do Autor não encontrada.")
        pontos_controvertidos = estado.get("pontos_controvertidos_saneamento", "Pontos controvertidos não definidos.")
        historico_completo_formatado_para_prompt = "\n".join([f"### Documento da Etapa: {item['etapa']} (Ator: {item['ator']})\n{item['documento']}\n---" for item in estado.get("historico_completo", [])])
        
        template_prompt_manifestacao_reu = f"""
        Você é o Advogado do Réu. O Juiz proferiu a Decisão de Saneamento e o Autor já se manifestou informando não ter mais provas a produzir.
        Seu cliente (Réu) também informou que não possui mais provas a produzir e deseja o julgamento antecipado da lide.
        **Processo ID:** {id_processo}
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
        2. Na petição, declare que o Réu também não tem outras provas a produzir.
        3. Requeira o julgamento do processo no estado em que se encontra.
        Manifestação Sobre Provas (Réu):
        """
        chain_manifestacao_reu = criar_prompt_e_chain(template_prompt_manifestacao_reu)
        documento_gerado_principal = chain_manifestacao_reu.invoke({})
        proximo_ator_logico = JUIZ
        # Mantém os documentos do réu que já estavam no estado (da contestação)
        lista_documentos_juntados_pelo_reu_final = estado.get("documentos_juntados_pelo_reu", [])

    else:
        print(f"AVISO [{ADVOGADO_REU}]: Lógica para etapa '{etapa_atual_do_no}' não implementada.")

    print(f"INFO [{ADVOGADO_REU}-{etapa_atual_do_no}] Documento Gerado (trecho): {documento_gerado_principal[:250]}...")
    novo_historico_item = {"etapa": etapa_atual_do_no, "ator": ADVOGADO_REU, "documento": documento_gerado_principal}

    return {
        "nome_do_ultimo_no_executado": ADVOGADO_REU,
        "etapa_concluida_pelo_ultimo_no": etapa_atual_do_no,
        "proximo_ator_sugerido_pelo_ultimo_no": proximo_ator_logico,
        "documento_gerado_na_etapa_recente": documento_gerado_principal,
        "historico_completo": estado.get("historico_completo", []) + [novo_historico_item],
        "manifestacao_reu_sem_provas": estado.get("manifestacao_reu_sem_provas", False) or (etapa_atual_do_no == ETAPA_MANIFESTACAO_SEM_PROVAS_REU),
        "documentos_juntados_pelo_reu": lista_documentos_juntados_pelo_reu_final,
        "sentimento_contestacao": sentimento_contestacao_texto_gerado,
    }


if __name__ == '__main__':
    print("--- Testando Módulo de Agentes ---")
    # Para testar os agentes individualmente de forma completa, seria necessário:
    # 1. Mockar ou ter uma instância real do 'llm' (via llm_models.py e GOOGLE_API_KEY).
    # 2. Mockar ou ter uma instância real do 'retriever'.
    # 3. Criar um objeto 'mapa_tarefas' de teste.
    # 4. Criar um objeto 'estado' de teste com os campos esperados.

    # Exemplo de estrutura de teste (simplificado, sem RAG/LLM real):
    mapa_teste_agentes = {
        (None, None, ADVOGADO_AUTOR): ETAPA_PETICAO_INICIAL,
        (ADVOGADO_AUTOR, ETAPA_PETICAO_INICIAL, JUIZ): ETAPA_DESPACHO_RECEBENDO_INICIAL,
        # Adicionar mais mapeamentos para testar outros fluxos
    }

    estado_inicial_teste_autor = {
        "id_processo": "teste_agente_001",
        "retriever": None, # Mock retriever aqui se quiser testar RAG
        "nome_do_ultimo_no_executado": None,
        "etapa_concluida_pelo_ultimo_no": None,
        "proximo_ator_sugerido_pelo_ultimo_no": ADVOGADO_AUTOR,
        "documento_gerado_na_etapa_recente": None,
        "historico_completo": [],
        "pontos_controvertidos_saneamento": None,
        "manifestacao_autor_sem_provas": False,
        "manifestacao_reu_sem_provas": False,
        "dados_formulario_entrada": {
            "qualificacao_autor": "Autor Fictício", "qualificacao_reu": "Réu Fictício",
            "natureza_acao": "Ação de Cobrança Fictícia", "fatos": "Fatos da cobrança.",
            "fundamentacao_juridica": "Art. X do CC.", "pedidos": "Pede-se a condenação.",
            "documentos_autor": [{"tipo": "Contrato", "descricao": "Contrato de dívida."}]
        },
        "documentos_juntados_pelo_reu": None,
        "sentimento_peticao_inicial": None,
        "sentimento_contestacao": None
    }
    print("\nTestando agente_advogado_autor (Petição Inicial - sem LLM real):")
    try:
        # Para um teste real, o LLM precisa estar configurado
        # Aqui, a chamada a criar_prompt_e_chain pode falhar se o LLM não estiver ok
        resultado_autor = agente_advogado_autor(estado_inicial_teste_autor, mapa_teste_agentes)
        print(f"  Resultado do agente_advogado_autor (etapa concluída): {resultado_autor.get('etapa_concluida_pelo_ultimo_no')}")
        print(f"  Próximo ator sugerido: {resultado_autor.get('proximo_ator_sugerido_pelo_ultimo_no')}")
        # print(f"  Documento gerado (trecho): {str(resultado_autor.get('documento_gerado_na_etapa_recente',''))[:100]}...")
    except EnvironmentError as e:
        print(f"  ERRO no teste do agente_advogado_autor (provavelmente LLM não configurado): {e}")
    except Exception as e:
        print(f"  ERRO INESPERADO no teste do agente_advogado_autor: {e}")


    # Você pode adicionar mais testes para os outros agentes e etapas aqui.
    print("\n--- Fim dos Testes do Módulo de Agentes ---")