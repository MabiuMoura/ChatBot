import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from threading import Thread
from queue import Queue
from typing import Generator

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.schema import Document, BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_core.messages import HumanMessage, SystemMessage


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""

Contexto:
{context}

Pergunta:
{question}

Resposta:"""
)


# Configurações iniciais
STORAGE_DIR = "./storage_labse"
OLLAMA_MODEL = "mymodel"

# FastAPI app
app = FastAPI()

# Classe para entrada da requisição
class PerguntaRequest(BaseModel):
    pergunta: str


# Callback que envia tokens em tempo real
class StreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = Queue()
        self.done = False

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs):
        self.done = True

    def stream(self) -> Generator[str, None, None]:
        while not self.done or not self.queue.empty():
            token = self.queue.get()
            yield token


# Adaptador do retriever do LlamaIndex para LangChain
class LangChainCompatibleRetriever(BaseRetriever):
    def __init__(self, llamaindex_retriever):
        super().__init__()
        self._llamaindex_retriever = llamaindex_retriever

    def _get_relevant_documents(self, query: str):
        nodes = self._llamaindex_retriever.retrieve(query)
        return [Document(page_content=node.text, metadata=node.metadata) for node in nodes]

    async def _aget_relevant_documents(self, query: str):
        nodes = await self._llamaindex_retriever.aretrieve(query)
        return [Document(page_content=node.text, metadata=node.metadata) for node in nodes]


# Pré-carrega o índice e o retriever
print("🔧 Carregando embedding e índice...")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/LaBSE")
storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
index = load_index_from_storage(storage_context)
retriever = LangChainCompatibleRetriever(index.as_retriever(similarity_top_k=4))
print("✅ Índice carregado.")


# Função para instanciar o modelo com handler de streaming
def get_llm(handler: StreamHandler) -> OllamaLLM:
    return OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=0.7,
        top_p=0.9,
        num_ctx=4096,
        streaming=True,
        callbacks=[handler]
    )


# Endpoint FastAPI
@app.post("/perguntar")
async def perguntar(request: PerguntaRequest):
    handler = StreamHandler()
    llm = get_llm(handler)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=False
    )

    def gerar_streaming() -> Generator[str, None, None]:
        # Exibe "loading" logo de início
        yield "[🔄 Pensando... por favor, aguarde]\n\n"

        def rodar_chain():
            try:
                print("🔍 Recuperando documentos relevantes...")
                context_docs = retriever.get_relevant_documents(request.pergunta)
                context = "\n\n".join(doc.page_content for doc in context_docs)

                print("🧠 Formatando prompt...")
                # Formata o prompt dinâmico
                human_prompt = prompt_template.format(
                    context=context,
                    question=request.pergunta
                )

                # Prepara as mensagens do chat
                messages = [
                    SystemMessage(content="Você é um assistente de IA que sempre responde em português brasileiro, de forma clara e educada. Responda à pergunta com base apenas nas informações fornecidas no contexto repassado do llamaindex. Se a resposta não estiver presente no contexto, responda apenas essa frase genérica: 'Desculpe, não tenho informações suficientes para responder a essa pergunta.', sem nenhum adicional de texto."),
                    HumanMessage(content=human_prompt)
                ]

                print("🚀 Enviando para o modelo LLM...")
                # Envia para o modelo
                llm.invoke(messages)
                print("✅ Resposta recebida do modelo.")
            except Exception as e:
                print("❌ Erro durante execução:", e)
                handler.queue.put(f"\n[❌ Erro] {str(e)}")
            finally:
                handler.done = True

        # Inicia o processamento em background
        Thread(target=rodar_chain).start()

        # Vai emitindo os tokens conforme chegam
        for token in handler.stream():
            yield token

    return StreamingResponse(gerar_streaming(), media_type="text/plain")


# Execução direta (opcional com uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
