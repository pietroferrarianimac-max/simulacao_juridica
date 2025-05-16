# ui_components.py

import streamlit as st
import time
from typing import  Union

# LangChain Core (para gerar_conteudo_com_ia e rodar_simulacao_principal)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END

# Nossos Módulos
from settings import (
    GOOGLE_API_KEY, # Para gerar_conteudo_com_ia
    FORM_STEPS, TIPOS_DOCUMENTOS_COMUNS, SENTIMENTO_CORES, DEFAULT_SENTIMENTO_COR,
    ADVOGADO_AUTOR, JUIZ, ADVOGADO_REU, # Para icon_map e lógica de simulação
    ETAPA_PETICAO_INICIAL, ETAPA_DESPACHO_RECEBENDO_INICIAL, ETAPA_CONTESTACAO,
    ETAPA_DECISAO_SANEAMENTO, ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR,
    ETAPA_MANIFESTACAO_SEM_PROVAS_REU, ETAPA_SENTENCA, ETAPA_FIM_PROCESSO,
    # Adicione outras constantes de etapa se usadas diretamente aqui
)
from llm_models import llm # Para gerar_conteudo_com_ia e judicial_features
from rag_utils import criar_ou_carregar_retriever # Para rodar_simulacao_principal
from graph_definition import app, EstadoProcessual # Para rodar_simulacao_principal
from judicial_features import gerar_ementa_cnj_padrao, verificar_sentenca_com_jurisprudencia

# --- Funções da UI Streamlit ---

def inicializar_estado_formulario():
    """Inicializa ou reseta o estado do formulário no st.session_state."""
    if 'current_form_step_index' not in st.session_state:
        st.session_state.current_form_step_index = 0

    default_form_data = {
        "id_processo": f"caso_sim_{int(time.time())}",
        "qualificacao_autor": "", "qualificacao_reu": "",
        "fatos": "", "fundamentacao_juridica": "", "pedidos": "",
        "natureza_acao": "",
        "documentos_autor": [] # Lista para armazenar os documentos do autor
    }
    # Flags para conteúdo gerado por IA
    default_ia_flags = {key: False for key in default_form_data.keys()}
    default_ia_flags["documentos_autor_descricoes"] = {}

    if 'form_data' not in st.session_state:
        st.session_state.form_data = default_form_data.copy()
    else: # Garante que todos os campos existam, útil para atualizações
        for key, value in default_form_data.items():
            if key not in st.session_state.form_data:
                st.session_state.form_data[key] = value
        if "documentos_autor" not in st.session_state.form_data:
             st.session_state.form_data["documentos_autor"] = []


    if 'ia_generated_content_flags' not in st.session_state:
        st.session_state.ia_generated_content_flags = default_ia_flags.copy()
    else:
        for key, value in default_ia_flags.items():
            if key not in st.session_state.ia_generated_content_flags:
                 st.session_state.ia_generated_content_flags[key] = value

    if 'num_documentos_autor' not in st.session_state:
        st.session_state.num_documentos_autor = 0

    # Outros estados da UI
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'simulation_results' not in st.session_state: # Guarda resultados por ID de processo
        st.session_state.simulation_results = {}
    if 'doc_visualizado' not in st.session_state:
        st.session_state.doc_visualizado = None
    if 'doc_visualizado_titulo' not in st.session_state:
        st.session_state.doc_visualizado_titulo = ""

    # Estados para funcionalidades adicionais da sentença
    if 'ementa_cnj_gerada' not in st.session_state:
        st.session_state.ementa_cnj_gerada = None
    if 'verificacao_sentenca_resultado' not in st.session_state:
        st.session_state.verificacao_sentenca_resultado = None
    if 'show_ementa_popup' not in st.session_state:
        st.session_state.show_ementa_popup = False
    if 'show_verificacao_popup' not in st.session_state:
        st.session_state.show_verificacao_popup = False

def gerar_conteudo_com_ia(
    prompt_template_str: str,
    campos_prompt: dict,
    campo_formulario_display: str, # Nome amigável para o spinner
    chave_estado_form_data: str, # Chave em st.session_state.form_data
    sub_chave_lista: Union[str, None] = None, # Para listas de dicts, ex: 'descricao' em um doc
    indice_lista: Union[int, None] = None # Índice na lista, ex: para documentos_autor[i]
):
    """Gera conteúdo com IA e atualiza o st.session_state.form_data."""
    if not GOOGLE_API_KEY or not llm:
        st.error("A chave API do Google não foi configurada ou o LLM não foi inicializado. Não é possível usar a IA.")
        return
    try:
        with st.spinner(f"Gerando conteúdo para '{campo_formulario_display}' com IA..."):
            # Reutiliza a lógica de criar_prompt_e_chain, que já tem o LLM
            # Se criar_prompt_e_chain não estivesse em agent_helpers, seria definida aqui.
            prompt = ChatPromptTemplate.from_template(prompt_template_str)
            chain = prompt | llm | StrOutputParser() # llm importado de llm_models
            conteudo_gerado = chain.invoke(campos_prompt)

            if sub_chave_lista is not None and indice_lista is not None and chave_estado_form_data == "documentos_autor":
                # Garante que a lista e o dicionário no índice existem
                while len(st.session_state.form_data["documentos_autor"]) <= indice_lista:
                    st.session_state.form_data["documentos_autor"].append({}) # Adiciona dict vazio
                st.session_state.form_data["documentos_autor"][indice_lista][sub_chave_lista] = conteudo_gerado
                st.session_state.ia_generated_content_flags.setdefault("documentos_autor_descricoes", {})[f"doc_{indice_lista}"] = True
            else:
                st.session_state.form_data[chave_estado_form_data] = conteudo_gerado
                st.session_state.ia_generated_content_flags[chave_estado_form_data] = True
        st.rerun() # Re-renderiza a UI para mostrar o conteúdo gerado
    except Exception as e:
        st.error(f"Erro ao gerar conteúdo com IA para '{campo_formulario_display}': {e}")

# --- Funções de Exibição dos Formulários ---

def exibir_formulario_qualificacao_autor():
    idx_etapa = FORM_STEPS.index("autor")
    st.subheader(f"{idx_etapa + 1}. Qualificação do Autor")
    with st.form("form_autor_ui"): # Chave única para o form
        st.session_state.form_data["qualificacao_autor"] = st.text_area(
            "Qualificação Completa do Autor", value=st.session_state.form_data.get("qualificacao_autor", ""),
            height=150, key="ui_autor_q_text_area",
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
    with st.form("form_reu_ui"):
        st.session_state.form_data["qualificacao_reu"] = st.text_area(
            "Qualificação Completa do Réu", value=st.session_state.form_data.get("qualificacao_reu", ""),
            height=150, key="ui_reu_q_text_area",
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
    st.subheader(f"{idx_etapa + 1}. Descrição dos Fatos")
    with st.form("form_fatos_ui"):
        st.session_state.form_data["fatos"] = st.text_area(
            "Descreva os Fatos de forma clara e cronológica", value=st.session_state.form_data.get("fatos", ""),
            height=300, key="ui_fatos_text_area",
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
    with st.form("form_direito_ui"):
        st.session_state.form_data["fundamentacao_juridica"] = st.text_area(
            "Insira a fundamentação jurídica aplicável ao caso", value=st.session_state.form_data.get("fundamentacao_juridica", ""),
            height=300, key="ui_direito_text_area",
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
    with st.form("form_pedidos_ui"):
        st.session_state.form_data["pedidos"] = st.text_area(
            "Insira os pedidos da ação de forma clara e objetiva", value=st.session_state.form_data.get("pedidos", ""),
            height=300, key="ui_pedidos_text_area",
            help="Liste os requerimentos finais ao juiz. Ex: citação do réu, procedência da ação para condenar o réu a..., condenação em custas e honorários. Use alíneas (a, b, c...)."
        )
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Direito)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("direito")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Natureza da Ação) ➡")
        with col3:
            if st.form_submit_button("Sugerir Pedidos com IA (baseado nos fatos e direito)"):
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
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Insira os pedidos.")

def exibir_formulario_natureza_acao():
    idx_etapa = FORM_STEPS.index('natureza_acao')
    st.subheader(f"{idx_etapa + 1}. Definição da Natureza da Ação")
    with st.form("form_natureza_ui"):
        fatos_contexto = st.session_state.form_data.get("fatos", "Fatos não fornecidos.")
        direito_contexto = st.session_state.form_data.get("fundamentacao_juridica", "Fundamentação não fornecida.")
        pedidos_contexto = st.session_state.form_data.get("pedidos", "Pedidos não fornecidos.")

        st.info("Com base nos fatos, direito e pedidos que você informou, a IA pode sugerir a natureza técnica da ação.")
        with st.expander("Revisar Contexto para IA (Fatos, Direito, Pedidos)", expanded=False):
            st.text_area("Fatos (Resumo)", value=fatos_contexto[:500] + ("..." if len(fatos_contexto)>500 else ""), height=100, disabled=True, key="ui_natureza_fatos_ctx")
            st.text_area("Direito (Resumo)", value=direito_contexto[:500] + ("..." if len(direito_contexto)>500 else ""), height=100, disabled=True, key="ui_natureza_direito_ctx")
            st.text_area("Pedidos (Resumo)", value=pedidos_contexto[:500] + ("..." if len(pedidos_contexto)>500 else ""), height=100, disabled=True, key="ui_natureza_pedidos_ctx")

        st.session_state.form_data["natureza_acao"] = st.text_input(
            "Natureza da Ação (Ex: Ação de Indenização por Danos Morais c/c Danos Materiais)",
            value=st.session_state.form_data.get("natureza_acao", ""),
            key="ui_natureza_acao_text_input",
            help="A IA pode sugerir. Refine ou altere conforme necessário."
        )
        
        col1, col2, col3 = st.columns([1,1,3])
        with col1:
            if st.form_submit_button("⬅ Voltar (Pedidos)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("pedidos")
                st.rerun()
        with col2: submetido = st.form_submit_button("Próximo (Documentos) ➡")
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
                st.session_state.current_form_step_index += 1
                st.rerun()
            else: st.warning("Defina a natureza da ação ou peça uma sugestão à IA.")

def exibir_formulario_documentos_autor():
    idx_etapa = FORM_STEPS.index('documentos_autor')
    st.subheader(f"{idx_etapa + 1}. Documentos Juntados pelo Autor com a Petição Inicial")
    st.markdown("Liste os principais documentos que o Autor juntaria. A IA pode ajudar a gerar descrições sucintas (1-2 frases).")

    if "documentos_autor" not in st.session_state.form_data:
        st.session_state.form_data["documentos_autor"] = []
    if "num_documentos_autor" not in st.session_state or st.session_state.num_documentos_autor < 0:
       st.session_state.num_documentos_autor = 0

    if st.session_state.num_documentos_autor == 0 and not st.session_state.form_data["documentos_autor"]:
        st.info("Nenhum documento adicionado. Clique em 'Adicionar Documento' para começar ou prossiga se não houver documentos a listar.")
    
    # Sincroniza a lista 'documentos_autor' com 'num_documentos_autor'
    while len(st.session_state.form_data["documentos_autor"]) < st.session_state.num_documentos_autor:
        st.session_state.form_data["documentos_autor"].append({"tipo": TIPOS_DOCUMENTOS_COMUNS[0], "descricao": ""})
    if len(st.session_state.form_data["documentos_autor"]) > st.session_state.num_documentos_autor:
       st.session_state.form_data["documentos_autor"] = st.session_state.form_data["documentos_autor"][:st.session_state.num_documentos_autor]

    for i in range(st.session_state.num_documentos_autor):
        # Garante que o dict exista para o índice 'i'
        if i >= len(st.session_state.form_data["documentos_autor"]):
            st.session_state.form_data["documentos_autor"].append({"tipo": TIPOS_DOCUMENTOS_COMUNS[0], "descricao": ""})
        
        doc_atual_ref = st.session_state.form_data["documentos_autor"][i]

        with st.expander(f"Documento {i+1}: {doc_atual_ref.get('tipo', 'Novo Documento')}", expanded=True):
            cols_doc = st.columns([3, 6]) 
            doc_atual_ref["tipo"] = cols_doc[0].selectbox(
                f"Tipo do Documento {i+1}", options=TIPOS_DOCUMENTOS_COMUNS, 
                index=TIPOS_DOCUMENTOS_COMUNS.index(doc_atual_ref.get("tipo", TIPOS_DOCUMENTOS_COMUNS[0])),
                key=f"ui_doc_autor_tipo_{i}"
            )
            
            doc_atual_ref["descricao"] = cols_doc[1].text_area(
                f"Descrição/Conteúdo Sucinto do Documento {i+1}", 
                value=doc_atual_ref.get("descricao", ""), 
                key=f"ui_doc_autor_desc_{i}", height=100,
                help="Ex: 'Contrato de aluguel datado de 01/01/2022...' OU 'RG do autor...'"
            )
            
            if st.button(f"✨ Gerar Descrição IA para Doc. {i+1}", key=f"ui_doc_autor_ia_btn_{i}"):
                tipo_selecionado = doc_atual_ref["tipo"]
                fatos_contexto = st.session_state.form_data.get("fatos", "Contexto factual não disponível.")
                prompt_desc_doc = (
                    f"Você é um assistente jurídico. Para um documento do tipo '{tipo_selecionado}' que será juntado por um Autor, "
                    f"gere uma descrição SUCINTA (1-2 frases, máx 30 palavras) sobre seu conteúdo e propósito. "
                    f"Contexto dos fatos (resumido): '{fatos_contexto[:300]}...'. \nDescrição Sucinta:"
                )
                gerar_conteudo_com_ia(
                    prompt_desc_doc, 
                    {}, 
                    f"Descrição do Documento {i+1} ({tipo_selecionado})", 
                    "documentos_autor",
                    sub_chave_lista="descricao",
                    indice_lista=i
                )
            
            if st.session_state.ia_generated_content_flags.get("documentos_autor_descricoes", {}).get(f"doc_{i}"):
                st.caption("📝 Descrição gerada/sugerida por IA. Revise.")
    
    st.markdown("---")
    col_botoes_add_rem_1, col_botoes_add_rem_2 = st.columns(2)
    if col_botoes_add_rem_1.button("➕ Adicionar Documento", key="ui_add_doc_autor_btn", help="Adiciona um novo campo para listar um documento."):
        st.session_state.num_documentos_autor += 1
        st.rerun()

    if st.session_state.num_documentos_autor > 0:
        if col_botoes_add_rem_2.button("➖ Remover Último Documento", key="ui_rem_doc_autor_btn", help="Remove o último campo de documento da lista."):
            st.session_state.num_documentos_autor -= 1
            if st.session_state.form_data["documentos_autor"]:
                st.session_state.form_data["documentos_autor"].pop()
            if f"doc_{st.session_state.num_documentos_autor}" in st.session_state.ia_generated_content_flags.get("documentos_autor_descricoes", {}):
                del st.session_state.ia_generated_content_flags["documentos_autor_descricoes"][f"doc_{st.session_state.num_documentos_autor}"]
            st.rerun()
    st.markdown("---")

    with st.form("form_documentos_autor_nav_ui"):
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.form_submit_button("⬅ Voltar (Natureza da Ação)"):
                st.session_state.current_form_step_index = FORM_STEPS.index("natureza_acao")
                st.rerun()
        with col_nav2:
            if st.form_submit_button("Próximo (Revisar e Simular) ➡"):
                # Filtra documentos vazios ou incompletos antes de prosseguir
                docs_filtrados = []
                for doc_item in st.session_state.form_data.get("documentos_autor", []):
                    tipo_valido = doc_item.get("tipo") and doc_item.get("tipo") != TIPOS_DOCUMENTOS_COMUNS[0]
                    descricao_presente = doc_item.get("descricao","").strip()
                    if (tipo_valido and descricao_presente) or \
                       (doc_item.get("tipo") == TIPOS_DOCUMENTOS_COMUNS[0] and descricao_presente): # "Nenhum..." com descrição factual é válido
                        docs_filtrados.append(doc_item)
                
                st.session_state.form_data["documentos_autor"] = docs_filtrados
                st.session_state.num_documentos_autor = len(docs_filtrados)
                
                st.session_state.current_form_step_index += 1 
                st.rerun()

def exibir_revisao_e_iniciar_simulacao():
    idx_etapa = FORM_STEPS.index('revisar_e_simular')
    st.subheader(f"{idx_etapa + 1}. Revisar Dados e Iniciar Simulação")
    form_data_local = st.session_state.form_data
    st.info(f"**ID do Processo (Gerado):** `{form_data_local.get('id_processo', 'N/A')}`")

    with st.expander("Qualificação do Autor", expanded=False): 
        st.text_area("Revisão - Autor", value=form_data_local.get("qualificacao_autor", "Não preenchido"), height=100, disabled=True, key="ui_rev_autor_area")
    with st.expander("Qualificação do Réu", expanded=False): 
        st.text_area("Revisão - Réu", value=form_data_local.get("qualificacao_reu", "Não preenchido"), height=100, disabled=True, key="ui_rev_reu_area")
    with st.expander("Fatos", expanded=True): 
        st.text_area("Revisão - Fatos", value=form_data_local.get("fatos", "Não preenchido"), height=200, disabled=True, key="ui_rev_fatos_area")
    with st.expander("Fundamentação Jurídica", expanded=False): 
        st.text_area("Revisão - Direito", value=form_data_local.get("fundamentacao_juridica", "Não preenchido"), height=200, disabled=True, key="ui_rev_dir_area")
    with st.expander("Pedidos", expanded=False): 
        st.text_area("Revisão - Pedidos", value=form_data_local.get("pedidos", "Não preenchido"), height=200, disabled=True, key="ui_rev_ped_area")
    with st.expander("Natureza da Ação", expanded=False): 
        st.text_input("Revisão - Natureza da Ação", value=form_data_local.get("natureza_acao", "Não preenchido"), disabled=True, key="ui_rev_nat_input")
    
    with st.expander("Documentos Juntados pelo Autor", expanded=True):
        documentos_autor_revisao = form_data_local.get("documentos_autor", [])
        if documentos_autor_revisao:
            for i, doc in enumerate(documentos_autor_revisao):
                st.markdown(f"**Documento {i+1}:** {doc.get('tipo', 'N/A')}")
                st.text_area(f"Descrição Doc. {i+1}", value=doc.get('descricao', 'Sem descrição'), height=75, disabled=True, key=f"ui_rev_doc_autor_{i}")
        else:
            st.write("Nenhum documento foi listado pelo autor.")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Voltar (Documentos do Autor)", use_container_width=True, key="ui_btn_voltar_revisao"):
            st.session_state.current_form_step_index = FORM_STEPS.index("documentos_autor")
            st.rerun()
    with col2:
        campos_obrigatorios = ["qualificacao_autor", "qualificacao_reu", "natureza_acao", "fatos", "fundamentacao_juridica", "pedidos"]
        todos_preenchidos = all(form_data_local.get(campo, "").strip() for campo in campos_obrigatorios)
        
        if st.button("🚀 Iniciar Simulação com estes Dados", type="primary", disabled=not todos_preenchidos, use_container_width=True, key="ui_btn_iniciar_sim"):
            st.session_state.simulation_running = True
            current_pid = form_data_local.get('id_processo')
            # Limpa resultados para este ID para forçar nova simulação se ID for o mesmo
            if current_pid in st.session_state.get('simulation_results', {}):
                del st.session_state.simulation_results[current_pid] 
            st.rerun()
        elif not todos_preenchidos:
            st.warning("Campos essenciais (Autor, Réu, Fatos, Direito, Pedidos, Natureza da Ação) devem ser preenchidos.")

def rodar_simulacao_principal(dados_coletados: dict):
    """Executa a simulação do processo jurídico e exibe o progresso."""
    st.markdown(f"--- INICIANDO SIMULAÇÃO PARA O CASO: **{dados_coletados.get('id_processo','N/A')}** ---")
    
    if not dados_coletados or not dados_coletados.get('id_processo'):
        st.error("Erro: Dados do caso incompletos para iniciar a simulação.")
        st.session_state.simulation_running = False
        if st.button("Retornar ao formulário"): st.rerun()
        return

    documentos_autor_formatado = "\n\n--- Documentos Juntados pelo Autor (formulário) ---\n"
    docs_autor_lista = dados_coletados.get("documentos_autor", [])
    if docs_autor_lista:
        for i, doc in enumerate(docs_autor_lista):
            documentos_autor_formatado += f"{i+1}. Tipo: {doc.get('tipo', 'N/A')}\n   Descrição: {doc.get('descricao', 'N/A')}\n"
    else:
        documentos_autor_formatado += "Nenhum documento listado pelo autor no formulário.\n"

    conteudo_processo_texto = f"""
ID do Processo: {dados_coletados.get('id_processo')}
Qualificação do Autor:\n{dados_coletados.get('qualificacao_autor')}
Qualificação do Réu:\n{dados_coletados.get('qualificacao_reu')}
Natureza da Ação: {dados_coletados.get('natureza_acao')}
Dos Fatos:\n{dados_coletados.get('fatos')}
Da Fundamentação Jurídica:\n{dados_coletados.get('fundamentacao_juridica')}
Dos Pedidos:\n{dados_coletados.get('pedidos')}
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
    
    retriever_do_caso = None
    placeholder_rag = st.empty() 
    with placeholder_rag.status("⚙️ Inicializando sistema RAG...", expanded=True):
        st.write("Carregando modelos e criando índice vetorial com dados do caso...")
        try:
            retriever_do_caso = criar_ou_carregar_retriever(
                dados_coletados.get('id_processo',''), 
                documento_caso_atual=documento_do_caso_atual, 
                recriar_indice=True # Sempre recria para garantir que os dados do formulário atual sejam usados
            )
            if retriever_do_caso:
                st.write("✅ Retriever RAG pronto!")
            else:
                st.error("⚠️ Falha ao inicializar o retriever RAG.") # st.error dentro do status
        except Exception as e_rag:
            st.error(f"Erro crítico na inicialização do RAG: {e_rag}")
            retriever_do_caso = None 

    if not retriever_do_caso:
        placeholder_rag.empty() 
        st.error("Falha crítica ao criar o retriever. A simulação não pode continuar.")
        st.session_state.simulation_running = False
        if st.button("Tentar Novamente (Recarregar Formulário)"):
            st.session_state.current_form_step_index = FORM_STEPS.index("revisar_e_simular") 
            st.rerun()
        return

    placeholder_rag.success("🚀 Sistema RAG inicializado e pronto!")
    time.sleep(1.5) 
    placeholder_rag.empty()

    estado_inicial = EstadoProcessual(
        id_processo=dados_coletados.get('id_processo',''),
        retriever=retriever_do_caso,
        nome_do_ultimo_no_executado=None, etapa_concluida_pelo_ultimo_no=None,
        proximo_ator_sugerido_pelo_ultimo_no=ADVOGADO_AUTOR, 
        documento_gerado_na_etapa_recente=None, historico_completo=[],
        pontos_controvertidos_saneamento=None, manifestacao_autor_sem_provas=False,
        manifestacao_reu_sem_provas=False, # etapa_a_ser_executada_neste_turno="", (removido do EstadoProcessual)
        dados_formulario_entrada=dados_coletados,
        documentos_juntados_pelo_reu=None,
        sentimento_peticao_inicial=None,
        sentimento_contestacao=None
    )

    st.subheader("⏳ Acompanhamento da Simulação:")
    if 'expand_all_steps' not in st.session_state: st.session_state.expand_all_steps = True
    
    # Checkbox para expandir/recolher passos da simulação
    # A função on_change deve ser um callable, não uma atribuição direta.
    def toggle_expand_all():
        st.session_state.expand_all_steps = not st.session_state.expand_all_steps

    st.session_state.expand_all_steps = st.checkbox("Expandir todos os passos da simulação", value=st.session_state.get('expand_all_steps', True), key="cb_expand_all_sim_steps_ui")


    progress_bar_placeholder = st.empty()
    steps_container = st.container()
    max_passos_simulacao = 15 # Aumentado devido à complexidade e possíveis re-roteamentos
    passo_atual_simulacao = 0
    estado_final_simulacao = None

    try:
        for s_idx, s_event in enumerate(app.stream(input=estado_inicial, config={"recursion_limit": max_passos_simulacao})):
            passo_atual_simulacao += 1
            if not s_event or not isinstance(s_event, dict) or not list(s_event.keys()):
                print(f"AVISO: Evento de stream inesperado ou vazio no passo {s_idx}: {s_event}")
                continue

            nome_do_no_executado = list(s_event.keys())[0]
            
            if nome_do_no_executado == "__end__":
                # O valor associado a "__end__" é o estado final completo.
                estado_final_simulacao = list(s_event.values())[0] 
                nome_do_no_executado = END # Para consistência no log
            else:
                estado_parcial_apos_no = s_event[nome_do_no_executado]
                if not isinstance(estado_parcial_apos_no, dict): 
                    print(f"AVISO: Formato de estado inesperado do nó {nome_do_no_executado}. Pode afetar o estado final.")
                    # Tentamos usar o último estado completo conhecido se houver, senão o parcial (problemático)
                    estado_final_simulacao = estado_final_simulacao if estado_final_simulacao else estado_parcial_apos_no
                else:
                    estado_final_simulacao = estado_parcial_apos_no # Atualiza com o estado mais recente

            etapa_concluida_log = estado_final_simulacao.get('etapa_concluida_pelo_ultimo_no', 'N/A')
            doc_gerado_completo = str(estado_final_simulacao.get('documento_gerado_na_etapa_recente', ''))
            prox_ator_sug_log = estado_final_simulacao.get('proximo_ator_sugerido_pelo_ultimo_no', 'N/A')

            expander_title = f"Passo {passo_atual_simulacao}: Nó '{nome_do_no_executado}' concluiu etapa '{etapa_concluida_log}'"
            if nome_do_no_executado == END: expander_title = f"🏁 Passo {passo_atual_simulacao}: Fim da Simulação"
            
            with steps_container.expander(expander_title, expanded=st.session_state.get('expand_all_steps', True)):
                st.markdown(f"**Nó Executado:** `{nome_do_no_executado}`")
                st.markdown(f"**Etapa Concluída:** `{etapa_concluida_log}`")
                if "ERRO" not in etapa_concluida_log and doc_gerado_completo:
                    st.text_area("Documento Gerado:", value=doc_gerado_completo, height=200, key=f"ui_doc_step_sim_{passo_atual_simulacao}", disabled=True)
                elif doc_gerado_completo: 
                    st.error(f"Detalhe do Erro/Documento: {doc_gerado_completo}")
                st.markdown(f"**Próximo Ator Sugerido (pelo nó):** `{prox_ator_sug_log}`")
            
            # Estimativa de progresso
            # O número de etapas no mapa_tarefa_no_atual pode ser uma boa base.
            num_total_etapas_estimadas = 7 # Estimativa baseada no fluxo típico (PI, Despacho, Contest, Saneamento, Manifest Autor, Manifest Reu, Sentença)
            progress_val = min(1.0, passo_atual_simulacao / num_total_etapas_estimadas ) 
            progress_bar_placeholder.progress(progress_val, text=f"Simulando... {int(progress_val*100)}% (Passo {passo_atual_simulacao})")

            if nome_do_no_executado == END or prox_ator_sug_log == ETAPA_FIM_PROCESSO:
                steps_container.success("🎉 Fluxo da simulação concluído!")
                break 
            if "ERRO_FLUXO" in etapa_concluida_log or "ERRO_ETAPA" in etapa_concluida_log:
                steps_container.error(f"❌ Erro crítico no fluxo em '{nome_do_no_executado}'. Simulação interrompida.")
                break
            if passo_atual_simulacao >= max_passos_simulacao:
                steps_container.warning(f"Simulação atingiu o limite máximo de {max_passos_simulacao} passos e foi interrompida.")
                break
        
        progress_bar_placeholder.progress(1.0, text="Simulação Concluída!")
        if estado_final_simulacao:
            st.session_state.simulation_results[dados_coletados.get('id_processo','')] = estado_final_simulacao
            exibir_resultados_simulacao(estado_final_simulacao) # Chama a função de exibição
        else:
            st.warning("A simulação terminou, mas não foi possível obter o estado final completo.")

    except Exception as e_sim:
        st.error(f"ERRO INESPERADO DURANTE A EXECUÇÃO DA SIMULAÇÃO: {e_sim}")
        import traceback
        st.text_area("Stack Trace do Erro:", traceback.format_exc(), height=300)
    finally:
        progress_bar_placeholder.empty()

def exibir_resultados_simulacao(estado_final_simulacao: dict):
    """Exibe os resultados detalhados da simulação, incluindo linha do tempo e funcionalidades adicionais."""
    
    doc_completo_placeholder_res = st.empty() # Para visualização de docs da timeline

    st.subheader("📊 Resultados da Simulação")

    # Análise de Sentimentos
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
        st.markdown("---")

    # Linha do Tempo Interativa
    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        st.markdown("#### Linha do Tempo Interativa do Processo")
        historico = estado_final_simulacao["historico_completo"]
        icon_map = {
            ADVOGADO_AUTOR: "🙋‍♂️", JUIZ: "⚖️", ADVOGADO_REU: "🙋‍♀️",
            ETAPA_PETICAO_INICIAL: "📄", ETAPA_DESPACHO_RECEBENDO_INICIAL: "➡️",
            ETAPA_CONTESTACAO: "🛡️", ETAPA_DECISAO_SANEAMENTO: "🛠️",
            ETAPA_MANIFESTACAO_SEM_PROVAS_AUTOR: "🗣️", ETAPA_MANIFESTACAO_SEM_PROVAS_REU: "🗣️",
            ETAPA_SENTENCA: "🏁", "DEFAULT_ACTOR": "👤", "DEFAULT_ETAPA": "📑",
            "ERRO_FLUXO": "❌", "ERRO_ETAPA": "❓" # Simplificando chaves de erro
        }
        num_etapas = len(historico)
        if num_etapas > 0 :
            cols = st.columns(min(num_etapas, 8)) # Ajuste o número de colunas conforme necessário
            for i, item_hist in enumerate(historico):
                ator_hist = item_hist.get('ator', 'N/A')
                etapa_hist = item_hist.get('etapa', 'N/A')
                doc_completo_hist = str(item_hist.get('documento', 'N/A'))
                
                ator_icon = icon_map.get(ator_hist, icon_map["DEFAULT_ACTOR"])
                # Para etapas de erro, use um ícone genérico de erro se a etapa específica não estiver no icon_map
                etapa_icon_key = etapa_hist if not "ERRO" in etapa_hist else "ERRO_FLUXO" # Agrupa ícones de erro
                etapa_icon = icon_map.get(etapa_icon_key, icon_map["DEFAULT_ETAPA"])
                cor_fundo = "rgba(255, 0, 0, 0.1)" if "ERRO" in etapa_hist else "rgba(0, 0, 0, 0.03)"

                with cols[i % len(cols)]:
                    container_style = f"border: 1px solid #ddd; border-radius: 5px; padding: 10px; text-align: center; background-color: {cor_fundo}; height: 130px; display: flex; flex-direction: column; justify-content: space-around; margin-bottom: 5px;"
                    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 28px;'>{ator_icon}{etapa_icon}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 11px; margin-bottom: 3px;'><b>{ator_hist}</b><br>{etapa_hist[:30]}{'...' if len(etapa_hist)>30 else ''}</div>", unsafe_allow_html=True)
                    # Chave única para o botão incluindo ID do processo para evitar conflitos entre simulações
                    btn_key = f"ui_btn_timeline_doc_{i}_{estado_final_simulacao.get('id_processo', 'pid')}"
                    if st.button(f"Ver Doc {i+1}", key=btn_key, help=f"Visualizar: {etapa_hist}", use_container_width=True):
                        st.session_state.doc_visualizado = doc_completo_hist
                        st.session_state.doc_visualizado_titulo = f"Doc. Linha do Tempo (Passo {i+1}): {ator_hist} - {etapa_hist}"
                        st.rerun() 
                    st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True) 
    else:
        st.warning("Nenhum histórico completo para exibir na linha do tempo.")

    # Visualização do Documento da Timeline
    if st.session_state.get('doc_visualizado') is not None: 
        with doc_completo_placeholder_res.container():
            st.subheader(st.session_state.get('doc_visualizado_titulo', "Visualização de Documento"))
            st.text_area("Conteúdo do Documento:", st.session_state.doc_visualizado, height=350, key="ui_doc_view_sim_area_results", disabled=True)
            if st.button("Fechar Visualização do Documento", key="ui_close_doc_view_sim_btn_results", type="primary"):
                st.session_state.doc_visualizado = None
                st.session_state.doc_visualizado_titulo = ""
                doc_completo_placeholder_res.empty()
                st.rerun()

    # Funcionalidades Adicionais da Sentença
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
            if st.button("📄 Gerar Ementa (Padrão CNJ)", key="ui_btn_gerar_ementa", use_container_width=True):
                if sentenca_texto_completo:
                    with st.spinner("Gerando ementa no padrão CNJ..."):
                        st.session_state.ementa_cnj_gerada = gerar_ementa_cnj_padrao(sentenca_texto_completo, id_proc)
                        st.session_state.show_ementa_popup = True # Controla exibição do "popup"
                        st.rerun()
                else:
                    st.warning("Texto da sentença não encontrado para gerar ementa.")
        
        with col_verificador:
            # A variável global search_tool é importada de llm_models
            from llm_models import search_tool as imported_search_tool 
            if imported_search_tool:
                if st.button("🔍 Verificar Sentença com Jurisprudência", key="ui_btn_verificar_sentenca", use_container_width=True):
                    if sentenca_texto_completo:

                        st.session_state.verificacao_sentenca_resultado = "Processando verificação..." # Feedback imediato
                        st.session_state.show_verificacao_popup = True
                        st.rerun() # Permite que o popup apareça com a mensagem de processamento


                    else:
                        st.warning("Texto da sentença não encontrado para verificação.")
            else:
                col_verificador.info("Verificação com Google desabilitada (API não configurada).")

    # Exibição dos "Pop-ups" (simulados com containers)
    if st.session_state.get('show_ementa_popup', False) and st.session_state.get('ementa_cnj_gerada'):
        with st.container():
            st.markdown("---")
            st.subheader("📄 Ementa Gerada (Padrão CNJ)")
            st.markdown(st.session_state.ementa_cnj_gerada)
            if st.button("Fechar Ementa", key="ui_close_ementa_popup"):
                st.session_state.show_ementa_popup = False
                st.session_state.ementa_cnj_gerada = None # Limpa para a próxima vez
                st.rerun()
            st.markdown("---")

    if st.session_state.get('show_verificacao_popup', False):
        with st.container():
            st.markdown("---")
            st.subheader("🔍 Verificação da Sentença com Jurisprudência")
            # Se o resultado ainda não foi calculado (primeiro rerun após clicar no botão)
            if st.session_state.verificacao_sentenca_resultado == "Processando verificação..." and sentenca_texto_completo:
                 with st.spinner("Buscando e analisando jurisprudência... Isso pode levar alguns instantes."):
                    st.session_state.verificacao_sentenca_resultado = verificar_sentenca_com_jurisprudencia(sentenca_texto_completo)
                    st.rerun() # Re-run para exibir o resultado calculado
            
            if st.session_state.verificacao_sentenca_resultado and st.session_state.verificacao_sentenca_resultado != "Processando verificação...":
                st.markdown(st.session_state.verificacao_sentenca_resultado)
            elif not sentenca_texto_completo and st.session_state.verificacao_sentenca_resultado == "Processando verificação...":
                st.warning("Texto da sentença não disponível para verificação.") # Caso raro

            if st.button("Fechar Verificação", key="ui_close_verif_popup"):
                st.session_state.show_verificacao_popup = False
                st.session_state.verificacao_sentenca_resultado = None # Limpa
                st.rerun()
            st.markdown("---")
    
    # --- INÍCIO DA NOVA SEÇÃO: Funcionalidades Planejadas ---
    st.markdown("---")
    st.markdown("#### 🚀 Funcionalidades Planejadas (Roadmap)")
    st.caption("Recursos que podem ser adicionados em futuras versões para enriquecer a simulação:")

    col_planejadas1, col_planejadas2 = st.columns(2)

    with col_planejadas1:
        st.button("⚖️ Iniciar Fase Recursal *", disabled=True, use_container_width=True,
                  help="EM BREVE: Simular a interposição de recursos (ex: Apelação) e contrarrazões.")
        st.button("📄 Exportar Peças (PDF/DOCX) *", disabled=True, use_container_width=True,
                  help="EM BREVE: Permitir o download dos documentos gerados pela simulação.")
        st.button("📂 Meus Modelos RAG *", disabled=True, use_container_width=True,
                  help="EM BREVE: Permitir que o usuário adicione seus próprios modelos de documentos para o RAG.")

    with col_planejadas2:
        st.button("🧾 Calcular Custas e Prazos *", disabled=True, use_container_width=True,
                  help="EM BREVE: Simular o cálculo fictício de custas processuais e a contagem de prazos.")
        st.button("🧠 Sugerir Argumentos Avançados *", disabled=True, use_container_width=True,
                  help="EM BREVE: Assistência da IA para desenvolver teses jurídicas específicas com base no caso.")
        st.button("🏆 Modo Desafio/Avaliação *", disabled=True, use_container_width=True,
                  help="EM BREVE: Testar seus conhecimentos em cenários processuais com feedback da IA.")

    # --- FIM DA NOVA SEÇÃO ---

    # Histórico Detalhado
    st.markdown("#### Histórico Detalhado (Conteúdo Completo das Etapas)")
    st.session_state.expand_all_history = st.checkbox("Expandir todo o histórico detalhado", value=st.session_state.get('expand_all_history', False), key="cb_expand_all_hist_detail_ui")


    if estado_final_simulacao and estado_final_simulacao.get("historico_completo"):
        for i, item_hist in enumerate(estado_final_simulacao["historico_completo"]):
            ator_hist = item_hist.get('ator', 'N/A'); etapa_hist = item_hist.get('etapa', 'N/A')
            doc_completo_hist = str(item_hist.get('documento', 'N/A'))
            with st.expander(f"Detalhe {i+1}: Ator '{ator_hist}' | Etapa '{etapa_hist}'", expanded=st.session_state.get('expand_all_history', False)):
                st.text_area(f"Documento Completo (Passo {i+1}):", value=doc_completo_hist, height=200, key=f"ui_doc_hist_detail_sim_{i}", disabled=True)
    
    # Exibir documentos juntados pelo Réu
    if estado_final_simulacao and estado_final_simulacao.get("documentos_juntados_pelo_reu"):
        st.markdown("#### Documentos Juntados pelo Réu (Gerados pela IA)")
        with st.expander("Ver Documentos do Réu", expanded=False):
            for i, doc_reu in enumerate(estado_final_simulacao.get("documentos_juntados_pelo_reu", [])):
                st.markdown(f"**Documento {i+1} (Réu):** {doc_reu.get('tipo', 'N/A')}")
                st.text_area(f"Descrição Doc. Réu {i+1}", value=doc_reu.get('descricao', 'Sem descrição'), height=75, disabled=True, key=f"ui_res_doc_reu_{i}")

    st.markdown("--- FIM DA EXIBIÇÃO DOS RESULTADOS ---")


if __name__ == '__main__':
    st.title("Testando Componentes da UI (ui_components.py)")
    st.write("Este arquivo é destinado a ser importado pelo `main_app.py`.")
    st.write("Para testar os componentes individualmente, você precisaria simular o `st.session_state` e as dependências.")

    st.info("Exemplo de como `inicializar_estado_formulario` seria chamada:")


    st.warning("As funções de exibição de formulário (exibir_formulario_autor, etc.) e de simulação "
               "(rodar_simulacao_principal, exibir_resultados_simulacao) são projetadas para serem "
               "chamadas dentro de um fluxo de aplicação Streamlit gerenciado pelo `main_app.py`.")