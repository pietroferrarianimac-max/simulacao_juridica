# main_app.py

import streamlit as st
import os
import time # Para o ID único de processo ao reiniciar

# Importar configurações e constantes
from settings import (
    GOOGLE_API_KEY,
    LANGCHAIN_TRACING_V2,
    LANGCHAIN_PROJECT,
    FORM_STEPS # Necessário para a lógica de navegação dos formulários
)

# Importar componentes da UI e lógica de estado
from ui_components import (
    inicializar_estado_formulario,
    exibir_formulario_qualificacao_autor,
    exibir_formulario_qualificacao_reu,
    exibir_formulario_fatos,
    exibir_formulario_direito,
    exibir_formulario_pedidos,
    exibir_formulario_natureza_acao,
    exibir_formulario_documentos_autor,
    exibir_revisao_e_iniciar_simulacao,
    rodar_simulacao_principal,
    exibir_resultados_simulacao
)

# --- Bloco Principal de Execução do Streamlit ---
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="IA-Mestra: Simulação Jurídica Avançada", page_icon="⚖️")
    st.title("IA-Mestra: Simulação Jurídica Avançada ⚖️")
    st.caption("Uma ferramenta para simular o fluxo processual com assistência de IA, utilizando LangGraph e RAG.")

    # Verificação Crítica da API Key do Google
    if not GOOGLE_API_KEY:
        st.error("🔴 ERRO CRÍTICO: A variável de ambiente GOOGLE_API_KEY não foi definida. A aplicação não pode funcionar sem ela.")
        st.stop() # Impede a execução do restante da aplicação

    # Inicializa o estado da sessão para formulários e simulação
    inicializar_estado_formulario()

    # --- Barra Lateral (Sidebar) ---
    st.sidebar.title("Painel de Controle 🕹️")
    if st.sidebar.button("🔄 Nova Simulação (Limpar Formulário)", key="main_nova_sim_btn", type="primary", use_container_width=True):
        st.session_state.current_form_step_index = 0
        novo_id_processo = f"caso_sim_{int(time.time())}"

        # Reset completo do form_data e flags associadas
        st.session_state.form_data = {
            "id_processo": novo_id_processo, "qualificacao_autor": "", "qualificacao_reu": "",
            "fatos": "", "fundamentacao_juridica": "", "pedidos": "",
            "natureza_acao": "", "documentos_autor": []
        }
        # Recria as flags de IA com base no novo form_data
        st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
        st.session_state.ia_generated_content_flags["documentos_autor_descricoes"] = {}
        st.session_state.num_documentos_autor = 0

        # Reset do estado da simulação
        st.session_state.simulation_running = False
        st.session_state.doc_visualizado = None
        st.session_state.doc_visualizado_titulo = ""
        st.session_state.ementa_cnj_gerada = None
        st.session_state.verificacao_sentenca_resultado = None
        st.session_state.show_ementa_popup = False
        st.session_state.show_verificacao_popup = False
        st.success(f"Formulário limpo. Novo ID de processo: {novo_id_processo}")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.info(
        "ℹ️ Preencha os formulários sequenciais para definir os parâmetros do caso. "
        "A IA pode auxiliar no preenchimento com dados fictícios ou sugestões jurídicas contextuais."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🚀 Funcionalidades Planejadas:")
    st.sidebar.button("💾 Salvar Simulação *", disabled=True, use_container_width=True,
                      help="EM BREVE: Salvar o estado atual da simulação para continuar depois.")
    st.sidebar.button("📂 Carregar Simulação *", disabled=True, use_container_width=True,
                      help="EM BREVE: Carregar uma simulação salva anteriormente.")
    st.sidebar.selectbox("📜 Selecionar Rito Processual *",
                         options=["Rito Comum Ordinário (Atual)", "Juizado Especial Cível (Em breve)", "Execução (Em breve)"],
                         index=0,
                         disabled=True,
                         help="EM BREVE: Simular diferentes ritos processuais.",
                         key="main_select_rito_futuro")
    st.sidebar.markdown("---")

    # Link para LangSmith
    if LANGCHAIN_TRACING_V2 == "true" and LANGCHAIN_PROJECT:
        project_name = LANGCHAIN_PROJECT
        # Tenta obter um ID de organização/tenant, se disponível
        org_id_ou_tenant = os.getenv('LANGCHAIN_TENANT_ID', os.getenv('LANGCHAIN_ORGANIZATION_ID')) # Tenta algumas vars comuns

        langsmith_url_base = "https://smith.langchain.com/"
        if org_id_ou_tenant:
            langsmith_url = f"{langsmith_url_base}o/{org_id_ou_tenant}/projects/{project_name}"
        else:
            # Fallback para um link mais genérico se o ID da organização não estiver disponível
            # Este link pode levar à página de login ou à lista de projetos se já estiver logado.
            langsmith_url = f"{langsmith_url_base}projects?search={project_name}"
        st.sidebar.markdown(f"🔍 [Monitorar no LangSmith ({project_name})]({langsmith_url})", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("LangSmith tracing não está habilitado ou configurado.")

    # --- Lógica Principal de Exibição da UI (Conteúdo da Página) ---
    if st.session_state.get('simulation_running', False):
        id_processo_atual = st.session_state.form_data.get('id_processo')
        if id_processo_atual and id_processo_atual not in st.session_state.get('simulation_results', {}):
            # Se a simulação está marcada como rodando, mas não há resultados para o ID atual,
            # então execute a simulação.
            rodar_simulacao_principal(st.session_state.form_data)
        elif id_processo_atual and st.session_state.get('simulation_results', {}).get(id_processo_atual):
            # Se a simulação está marcada como rodando E já existem resultados para o ID atual,
            # apenas exiba esses resultados.
            st.info(f"📖 Exibindo resultados da simulação para o ID: {id_processo_atual}")
            exibir_resultados_simulacao(st.session_state.simulation_results[id_processo_atual])
            # Botão para iniciar uma nova simulação a partir da tela de resultados
            if st.button("Iniciar uma Nova Simulação (Limpar Tudo)", key="main_nova_sim_btn_results"):
                # Reutiliza a mesma lógica do botão da sidebar para consistência
                st.session_state.current_form_step_index = 0
                novo_id_processo = f"caso_sim_{int(time.time())}"
                st.session_state.form_data = {
                    "id_processo": novo_id_processo, "qualificacao_autor": "", "qualificacao_reu": "",
                    "fatos": "", "fundamentacao_juridica": "", "pedidos": "",
                    "natureza_acao": "", "documentos_autor": []
                }
                st.session_state.ia_generated_content_flags = {key: False for key in st.session_state.form_data.keys()}
                st.session_state.ia_generated_content_flags["documentos_autor_descricoes"] = {}
                st.session_state.num_documentos_autor = 0
                st.session_state.simulation_running = False
                st.session_state.doc_visualizado = None
                st.session_state.doc_visualizado_titulo = ""
                st.session_state.ementa_cnj_gerada = None
                st.session_state.verificacao_sentenca_resultado = None
                st.session_state.show_ementa_popup = False
                st.session_state.show_verificacao_popup = False
                st.rerun()
        else:
            # Caso de segurança: simulation_running é True, mas algo deu errado com o ID ou resultados.
            st.warning("⚠️ A simulação anterior não produziu resultados ou houve um problema de estado. Por favor, inicie uma nova simulação.")
            st.session_state.simulation_running = False # Reseta para evitar loop
            if st.button("Ir para o início do formulário", key="main_goto_form_start_error"):
                st.session_state.current_form_step_index = 0
                st.rerun()
    else: # Exibir os formulários
        passo_atual_idx = st.session_state.current_form_step_index

        # --- Indicador de Progresso do Formulário ---
        if 0 <= passo_atual_idx < len(FORM_STEPS):
            nome_passo_atual = FORM_STEPS[passo_atual_idx]
            total_passos_preenchimento = len(FORM_STEPS) -1 # Exclui a etapa de revisão

            if nome_passo_atual != "revisar_e_simular":
                progresso_percentual = (passo_atual_idx) / total_passos_preenchimento if total_passos_preenchimento > 0 else 0
                st.progress(progresso_percentual)
                titulo_passo_formatado = nome_passo_atual.replace('_', ' ').title()
                st.markdown(f"#### Etapa de Preenchimento: **{titulo_passo_formatado}** (Passo {passo_atual_idx + 1} de {total_passos_preenchimento})")
            else: # Etapa de Revisão
                st.markdown(f"#### Etapa Final: **Revisar Dados e Iniciar Simulação** (Passo {len(FORM_STEPS)} de {len(FORM_STEPS)})")
            st.markdown("---")
        # --- Fim do Indicador de Progresso ---

            # Seleciona qual função de formulário exibir com base no índice atual
            if passo_atual_idx < len(FORM_STEPS):
                current_step_key = FORM_STEPS[passo_atual_idx]
                if current_step_key == "autor": exibir_formulario_qualificacao_autor()
                elif current_step_key == "reu": exibir_formulario_qualificacao_reu()
                elif current_step_key == "fatos": exibir_formulario_fatos()
                elif current_step_key == "direito": exibir_formulario_direito()
                elif current_step_key == "pedidos": exibir_formulario_pedidos()
                elif current_step_key == "natureza_acao": exibir_formulario_natureza_acao()
                elif current_step_key == "documentos_autor": exibir_formulario_documentos_autor()
                elif current_step_key == "revisar_e_simular": exibir_revisao_e_iniciar_simulacao()
                else:
                    st.error(f"🔴 ERRO INTERNO: Etapa do formulário desconhecida '{current_step_key}'.")
                    st.warning("Por favor, tente reiniciar a simulação a partir do menu lateral.")
            else:
                st.error("🔴 ERRO INTERNO: Índice da etapa do formulário inválido. Tentando reiniciar...")
                st.session_state.current_form_step_index = 0
                st.rerun()