# IA-Mestra: Simulação Jurídica Avançada ⚖️✨
Projeto Desenvolvido para a Imersão IA Alura + Google (12-16 de Maio de 2025)

Este projeto é um protótipo/MVP que explora o potencial da Inteligência Artificial no domínio jurídico, com foco na simulação de fluxos processuais e na assistência inteligente a profissionais e estudantes de Direito.


# Olá, Comunidade Alura e Google! 👋
644.606 tokens gastos e 356 runs depois, é com grande entusiasmo que apresento o IA-Mestra, unindo Google, agentes e o universo do Direito. Desenvolvido com o auxílio fundamental da IA Gemini e aplicando os conceitos transformadores sobre agentes inteligentes que exploramos durante esta Imersão IA, o IA-Mestra vai além da teoria.

Este simulador é um protótipo que demonstra, na prática, como a Inteligência Artificial, incluindo o poder do Gemini e recursos do Google como o Buscador para pesquisa de jurisprudência, pode revolucionar a forma como interagimos com o Direito. Ele não apenas gera documentos, mas simula um ecossistema onde diferentes "atores" de IA (advogados, juiz) interagem, tomam decisões estratégicas e impulsionam um processo judicial do início ao fim.

Trouxe um pouco da minha experiência a serviço do Tribunal de Justiça de São Paulo, espero que gostem.

# Vote no IA-Mestra! Seu Apoio Faz a Diferença! 🚀
Se você acredita no potencial do IA-Mestra e na importância de inovar no campo jurídico com Inteligência Artificial, seu voto é muito importante!

Este projeto visa ser uma ferramenta que possa, de fato, auxiliar no desenvolvimento de futuros e atuais profissionais do Direito. 

Vejamos.

# O Desafio que o IA-Mestra Busca Endereçar 🎯
O Direito é complexo. Para estudantes, visualizar o trâmite de um processo e a interconexão das peças pode ser abstrato. Para advogados, especialmente os em início de carreira, desenvolver o raciocínio estratégico, prever desdobramentos e redigir peças iniciais de qualidade consome tempo e exige prática. Como a IA pode auxiliar?

A Solução: IA-Mestra - Seu Assistente Jurídico Inteligente 💡
O IA-Mestra é um simulador processual avançado que utiliza um sistema de agentes inteligentes para:

## Simular o Procedimento: 
Desde a petição inicial, passando pela contestação, despacho saneador, manifestações, até a sentença.
## Gerar Peças Processuais: 
Com base nos dados do caso fornecidos pelo usuário e em modelos consultados via RAG (Retrieval Augmented Generation).
## Demonstrar Casos de Uso Diversificados da IA:
### Geração de Conteúdo Jurídico:
Criação de petições, despachos, sentenças, etc.
### Análise de Sentimento:
Avaliação do tom da petição inicial e da contestação.
### Sumarização e Estruturação de Informações:
Geração de Ementas no padrão CNJ.
### Pesquisa e Análise Contextual:
Verificação da sentença com jurisprudência (utilizando Google Search e análise por LLM).
### Tomada de Decisão Estratégica Simulada: 
Agentes de IA decidem os próximos passos baseados no estado do processo.
### Análise de Dados (Implícita):
A IA analisa os inputs do usuário e o histórico processual para agir.

# Para Quem? 👥
## Estudantes Universitários de Direito:
Uma ferramenta dinâmica para treinar a redação de peças, compreender o fluxo processual e exercitar o raciocínio jurídico de forma interativa.
## Advogados e Profissionais da Área:
Um ambiente para exercitar o raciocínio sobre casos hipotéticos, explorar diferentes estratégias, obter auxílio na redação de minutas iniciais e até mesmo realizar análises preditivas básicas sobre os próximos passos processuais.

# Tecnologias Utilizadas 🛠️
Python
Streamlit: Para a interface web interativa.
LangChain & LangGraph: Orquestração dos agentes e do fluxo processual.
Google Gemini (via API): Como o cérebro por trás da geração de texto, análise e tomada de decisão dos agentes.
Google GenerativeAI Embeddings: Para a vetorização de documentos no RAG.
FAISS: Para a criação do vector store local (RAG).
Google Search API (Custom Search JSON API): Para a funcionalidade de busca de jurisprudência.
Dotenv: Gerenciamento de variáveis de ambiente.

## Arquitetura Modular Inteligente 🧠
Para garantir a organização, manutenibilidade e escalabilidade do projeto, o IA-Mestra foi desenvolvido com uma arquitetura modular:

main_app.py: Ponto de entrada da aplicação Streamlit, orquestra a UI.
ui_components.py: Define todos os componentes visuais e interativos do Streamlit (formulários, exibição de resultados).
settings.py: Centraliza configurações, constantes e o carregamento de variáveis de ambiente.
llm_models.py: Inicializa o modelo LLM (Gemini) e a ferramenta de busca (Google Search).
rag_utils.py: Funções para carregamento de documentos e criação/gerenciamento do RAG (FAISS).
agent_helpers.py: Funções utilitárias compartilhadas pelos agentes.
agents.py: Define a lógica e o comportamento de cada agente (Advogado Autor, Juiz, Advogado Réu).
graph_definition.py: Define o estado processual (EstadoProcessual), o mapa de fluxo (mapa_tarefa_no_atual), o roteador e constrói o grafo LangGraph.
judicial_features.py: Implementa funcionalidades jurídicas específicas, como geração de ementa e verificação de sentença.
Comece a Simular! (Instalação e Execução) 🚀
# Siga os passos abaixo para rodar o IA-Mestra em sua máquina local:

## Pré-requisitos
Python 3.9 ou superior
Pip (gerenciador de pacotes Python)
Git (para clonar o repositório)
## Passos para Instalação 
### Clone o Repositório:



git clone https://[URL_DO_SEU_REPOSITORIO_GIT_AQUI]
cd [NOME_DA_PASTA_DO_PROJETO]
### Crie um Ambiente Virtual (Recomendado):


python -m venv .venv
Para ativar no Windows:


.venv\Scripts\activate
Para ativar no macOS/Linux:

source .venv/bin/activate

### Instale as Dependências:
Crie um arquivo requirements.txt com todas as bibliotecas Python necessárias (ex: streamlit, langchain, langchain-google-genai, langchain-community, langgraph, faiss-cpu ou faiss-gpu, python-dotenv, docx2txt). E então execute:


pip install -r requirements.txt
(Se você não tiver um requirements.txt pronto, pode gerar um no seu ambiente de desenvolvimento com pip freeze > requirements.txt)

### IMPORTANTE: Configure as Chaves de API (.env):
Crie um arquivo chamado .env na raiz do projeto e adicione suas chaves:
Sem as chaves, o projeto não vai rodar como esperado.
Lembrando que o gemini 1.5 foi utilizado dentro da camada free tier, para não
gerar gastos. No mais, as chamadas do Google Search são limitadas na camada
free tier e o buscador de jurisprudência que é usada para avaliação da sentença ao final não funcionará corretamente se extrapolar o limite.

GOOGLE_API_KEY="SUA_GOOGLE_API_KEY_AQUI"

# Para LangSmith Tracing (Opcional, mas recomendado para debug)
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="SUA_LANGSMITH_API_KEY_AQUI"
LANGCHAIN_PROJECT="SimulacaoJuridicaDebug" # Ou o nome que preferir

# Para a funcionalidade de Busca no Google (Jurisprudência)
GOOGLE_API_KEY_SEARCH="SUA_GOOGLE_API_KEY_PARA_CUSTOM_SEARCH_AQUI"
GOOGLE_CSE_ID="SEU_CUSTOM_SEARCH_ENGINE_ID_AQUI"
Você precisará habilitar a "Custom Search JSON API" no Google Cloud Console e criar um "Programmable Search Engine" para obter as duas últimas chaves.
Estrutura de Pastas para RAG (Modelos):
Certifique-se de ter a seguinte estrutura de pastas na raiz do projeto (ou ajuste os caminhos em settings.py):

simulacao_juridica_data/
    modelos_peticoes/
        (coloque aqui arquivos .docx de modelos de petições)
    modelos_juiz/
        (coloque aqui arquivos .docx de modelos de despachos, sentenças)
# Rodando a Aplicação
Com o ambiente virtual ativado e as dependências instaladas, execute:



streamlit run main_app.py
A aplicação deverá abrir automaticamente no seu navegador!

Estrutura do Projeto (Módulos) 📂
# Como mencionado, o projeto é modular:

main_app.py: Orquestrador da UI Streamlit.
ui_components.py: Funções de renderização dos formulários e resultados.
settings.py: Configurações globais e chaves.
llm_models.py: Inicialização do LLM (Gemini) e Search Tool.
rag_utils.py: Utilitários para Retrieval Augmented Generation (FAISS).
agent_helpers.py: Funções de apoio para os agentes.
agents.py: Lógica dos agentes (Advogado Autor, Juiz, Advogado Réu).
graph_definition.py: Definição do estado, mapa de fluxo e construção do grafo LangGraph.
judicial_features.py: Funções como geração de ementa e verificação de sentença.

# Visão de Futuro (Roadmap) ✨
Este MVP é apenas o começo! O IA-Mestra tem potencial para evoluir com funcionalidades como:

Simulação da Fase Recursal (Apelação, Contrarrazões).
Módulo de Produção de Provas Detalhado (Testemunhal, Pericial).
Salvamento e Carregamento de Simulações.
Exportação de Peças para PDF/.docx.
Personalização de Modelos RAG pelo usuário.
Outros Ritos Processuais (Juizados Especiais, Execução).
Modo Desafio com avaliação de desempenho.
E muito mais, conforme explorado em nossa análise MoSCoW! (Os placeholders * na UI já indicam alguns desses planos)


# Agradecimentos Especiais 🙏
À Alura e ao Google pela incrível oportunidade da Imersão IA, que foi fundamental para a concepção e desenvolvimento deste projeto.
À IA Gemini, que não só é o motor deste simulador, mas também foi uma ferramenta de auxílio valiosa durante todo o processo de desenvolvimento.
A todos os instrutores e colegas da Imersão pelo aprendizado compartilhado.

# Vote no IA-Mestra! Peço novamente, pois seu apoio faz a diferença! 🚀

Muito obrigado!