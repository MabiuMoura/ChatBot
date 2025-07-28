# 🤖 Chatbot com LlamaIndex, LangChain, FastAPI e Streamlit

Este projeto é um chatbot que utiliza **modelos de linguagem local via Ollama**, combinando **LlamaIndex** para indexação de documentos e **LangChain** para orquestração da resposta com base no conteúdo de arquivos PDF, TXT, DOCX e CSV.

---

## 📦 Tecnologias Utilizadas

- [LlamaIndex](https://www.llamaindex.ai/)
- [LangChain](https://www.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.com/)
- Embeddings: [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE)

---

## 📁 Estrutura do Projeto

├── indexador.py # Indexa documentos no diretório data/

├── main.py # API FastAPI com recuperação e resposta via LLM

├── app.py # Interface front-end com Streamlit

├── storage_labse/ # Armazena índice vetorial persistente

├── data/ # Coloque aqui seus arquivos PDF, TXT, CSV, DOCX


---

## ⚙️ Pré-requisitos

Antes de tudo, instale:

- Python 3.10 ou superior
- [Ollama](https://ollama.com/download) instalado e configurado

---

## 📥 Instalação

1. Clone o projeto:

bash
git clone https://github.com/seu-usuario/seu-repo-chatbot.git
cd seu-repo-chatbot

2. Crie um ambiente virtual:

python -m venv venv

source venv/bin/activate  # Linux/macOS

venv\Scripts\activate     # Windows

3. Instale as dependências:

pip install -r requirements.txt

## 🧠 Baixando e Rodando o Modelo com Ollama

1. Instale o Ollama
Acesse: https://ollama.com/download

Siga as instruções para seu sistema operacional. Após instalar, verifique se está funcionando com:

ollama list

2.  Baixe um modelo (ex: LLaMA 3)

 ollama pull llama3 

3. Rode o Ollama como um servidor local (se ainda não estiver rodando)


ollama serve

4. Altere o nome do modelo no código (opcional)
No main.py, o nome do modelo está como:

OLLAMA_MODEL = "mymodel"

Troque para:

OLLAMA_MODEL = "llama3"

Ou qualquer outro modelo que tenha sido baixado com ollama pull.

## 📑 Etapa 1 – Indexar os Documentos
Coloque seus arquivos PDF, DOCX, TXT ou CSV dentro da pasta ./data.

Depois, rode:

python  .py

## 🚀 Etapa 2 – Iniciar a API com FastAPI
Rode o servidor backend com:

uvicorn main:app --reload

## 💬 Etapa 3 – Rodar a Interface de Chat com Streamlit
Abra outro terminal (com o ambiente virtual ativado) e execute:

streamlit run app.py

