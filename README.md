# Agentic RAG com Qdrant e LangGraph

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pedro-hsprado-dataengineer)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Dev-PPrado)
[![Portfólio](https://img.shields.io/badge/Portf%C3%B3lio-4B5563?style=for-the-badge&logo=googlechrome&logoColor=white)](https://dev-pprado.github.io/)

API de *Retrieval-Augmented Generation* (RAG) com arquitetura agêntica, construída com [FastAPI](https://fastapi.tiangolo.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Qdrant](https://qdrant.tech/documentation/) e [Langfuse](https://langfuse.com/docs). O sistema permite ingerir documentos PDF e TXT em coleções vetoriais e consultá-los por meio de um agente que decide, a cada mensagem, se precisa ou não buscar contexto na base de conhecimento.

## Sobre o projeto

Este é um **projeto de aprendizado**, desenvolvido como parte dos estudos da **trilha de Engenharia de IA da Jornada de Dados**. Ele corresponde ao projeto final do módulo de RAG da trilha e tem como objetivo aplicar, em um único sistema, os conceitos estudados ao longo do curso:

- ingestão e *chunking* de documentos;
- geração de *embeddings* e armazenamento em banco vetorial;
- orquestração de agentes com *tool calling*;
- exposição do sistema via API REST;
- observabilidade de aplicações com LLMs.

Por se tratar de um projeto de estudo, o foco está na clareza da arquitetura e na experimentação com as ferramentas, e não em requisitos completos de produção (autenticação, testes automatizados, *deploy*, entre outros).

## Arquitetura

Em um RAG tradicional, o fluxo é fixo: toda pergunta dispara uma busca no banco vetorial, o contexto recuperado é anexado ao prompt e o resultado é enviado ao LLM.

Neste projeto, o fluxo é controlado por um agente implementado com um `StateGraph` do LangGraph. O LLM recebe a ferramenta `retrieve_documents` e decide quando utilizá-la:

- **Resposta direta**: saudações e conversas que não dependem da base documental são respondidas sem consulta ao banco vetorial.
- **Recuperação de contexto**: perguntas sobre o acervo acionam a ferramenta, que busca os trechos mais relevantes no Qdrant e os devolve ao agente para compor a resposta, com citação da fonte (`[Fonte: nome_arquivo, pagina]`).

Essa abordagem reduz chamadas desnecessárias ao banco vetorial e consumo de tokens, mantendo buscas direcionadas quando o contexto é necessário.

```text
            +---------+
 START ---> |  agent  | ---(sem tool call)---> END
            +---------+
                 |  ^
      (tool call)|  |
                 v  |
            +---------+
            |  tools  |  retrieve_documents -> Qdrant
            +---------+
```

## Stack tecnológica

| Componente | Tecnologia | Função |
|---|---|---|
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Endpoints REST assíncronos |
| Orquestração | [LangGraph](https://langchain-ai.github.io/langgraph/) / [LangChain](https://python.langchain.com/) | Grafo de estado do agente e *tool calling* |
| Banco vetorial | [Qdrant](https://qdrant.tech/documentation/) (Docker) | Armazenamento e busca de *embeddings* |
| LLM | OpenAI `gpt-4.1-mini` | Agente de conversação e extração de metadados |
| Embeddings | OpenAI `text-embedding-3-large` (3072 dimensões) | Representação semântica dos trechos |
| Observabilidade | [Langfuse](https://langfuse.com/docs) | Rastreamento de chamadas, tokens e latência |
| Leitura de documentos | PyPDF2 / PyMuPDF | Extração de texto de PDFs |
| Gerenciamento de dependências | [uv](https://docs.astral.sh/uv/) | Ambiente virtual e instalação de pacotes |

## Funcionalidades

### Ingestão de documentos

- Upload de um ou mais arquivos PDF ou TXT para uma coleção específica.
- Estratégia de *chunking* configurável por requisição: divisor recursivo (`RecursiveCharacterTextSplitter`) ou por caractere (`CharacterTextSplitter`).
- Tamanho de *chunk*, *overlap* e contagem por tokens (`tiktoken`) parametrizáveis.
- Documentos já existentes na coleção são ignorados, evitando duplicidade; também é possível substituir ou remover um documento.

### Extração automática de metadados

Durante a ingestão, a primeira página do documento é enviada ao LLM com saída estruturada (`with_structured_output`), que gera:

- `classificacao`: tipo do documento em até três palavras;
- `descricao`: resumo do documento em até duas frases.

Esses campos são armazenados como metadados de cada vetor, junto com nome do arquivo e número da página.

### Chat com múltiplas coleções

- O endpoint de chat recebe o nome da coleção, que é repassado à ferramenta de busca via configuração de *runtime* do LangGraph. Assim, a mesma aplicação consulta diferentes bases sem alteração de código.
- Suporte a histórico de conversa enviado pelo cliente.
- Possibilidade de anexar um arquivo à pergunta, usado como contexto adicional e cruzado com o conteúdo recuperado da base.

### Observabilidade

Cada requisição ao chat cria um `CallbackHandler` do Langfuse, permitindo acompanhar a execução do grafo, as chamadas à ferramenta, o consumo de tokens e o tempo de resposta.

## Estrutura do projeto

```text
rag-production/
├── index.html               # Interface web para testes (gestão de coleções, upload e chat)
├── main.py                  # Ponto de entrada da aplicação
├── pyproject.toml           # Dependências do projeto
├── .env.example             # Modelo de variáveis de ambiente
└── src/
    ├── api.py               # Instância do FastAPI, CORS e registro dos routers
    ├── models.py            # Modelos Pydantic de requisição, resposta e metadados
    ├── settings.py          # Configurações e variáveis de ambiente
    ├── pdf_utils.py         # Extração de texto de PDFs em memória (PyMuPDF)
    ├── customlogger.py      # Configuração padronizada de logs
    ├── chat/
    │   ├── chat.py          # Grafo do agente, ferramenta de busca e serviço de chat
    │   ├── llm_models.py    # Instâncias de LLMs e embeddings
    │   └── qdrant.py        # Cliente assíncrono do Qdrant
    ├── embedder/
    │   ├── client.py        # Operações no Qdrant (coleções, upsert, remoção)
    │   └── processor.py     # Extração de texto, chunking, metadados e embeddings
    └── routers/
        ├── chat.py          # Endpoint de chat
        └── qdrant.py        # Endpoints de gestão de coleções e documentos
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Verificação de disponibilidade da API |
| `POST` | `/qdrant/collection?collection_name={nome}` | Cria uma coleção |
| `DELETE` | `/qdrant/collection/{collection_name}` | Remove uma coleção |
| `POST` | `/qdrant/collection/{collection_name}/document` | Ingere um ou mais documentos |
| `PUT` | `/qdrant/collection/{collection_name}/document/{filename}` | Substitui um documento existente |
| `DELETE` | `/qdrant/collection/{collection_name}/document/{filename}` | Remove um documento |
| `POST` | `/chat/ask` | Envia uma pergunta ao agente |

Parâmetros de ingestão (form-data): `files`, `chunk_size` (padrão `1000`), `overlap_chunk` (padrão `100`), `splitter_type` (`recursive` ou `character`) e `use_tiktoken` (padrão `false`).

Parâmetros do chat (form-data): `query`, `collection_name`, `history` (JSON com mensagens `role`/`content`), `file` (opcional) e `clear_history`.

A documentação interativa completa fica disponível em `http://localhost:8000/docs` com a aplicação em execução.

## Como executar

### Pré-requisitos

- Python 3.13 ou superior
- [uv](https://docs.astral.sh/uv/)
- [Docker](https://docs.docker.com/)
- Chave de API da OpenAI
- Projeto no Langfuse (opcional, para observabilidade)

### 1. Configurar as variáveis de ambiente

Copie o arquivo de exemplo e preencha as chaves:

```bash
cp .env.example .env
```

| Variável | Descrição |
|---|---|
| `OPENAI_API_KEY` | Chave de API da OpenAI |
| `QDRANT_HOST` | Host do Qdrant (padrão `localhost`) |
| `QDRANT_PORT` | Porta do Qdrant (padrão `6333`) |
| `LANGFUSE_PUBLIC_KEY` | Chave pública do Langfuse |
| `LANGFUSE_SECRET_KEY` | Chave secreta do Langfuse |
| `LANGFUSE_BASE_URL` | URL do Langfuse (padrão `https://cloud.langfuse.com`) |

### 2. Iniciar o Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

O painel do Qdrant fica disponível em `http://localhost:6333/dashboard`.

### 3. Instalar as dependências

```bash
uv sync
```

### 4. Iniciar a API

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Testar pela interface web

Abra o arquivo `index.html` no navegador. A interface se conecta à API em `http://localhost:8000`.

1. No painel esquerdo, crie uma coleção (por exemplo, `base_estudo`) e envie arquivos PDF ou TXT.
2. No painel direito, informe a mesma coleção e converse com o agente.

## Possíveis evoluções

- Respostas em *streaming* no endpoint de chat (o serviço já possui o método `astream_resposta`).
- Busca híbrida (densa e esparsa) e filtros por metadados.
- Persistência do histórico de conversas no servidor.
- Testes automatizados e avaliação da qualidade das respostas.
- Containerização da API e orquestração com Docker Compose.
- Autenticação e restrição de CORS.

## Créditos

Projeto desenvolvido por Pedro Prado como parte da trilha de Engenharia de IA da Jornada de Dados.
