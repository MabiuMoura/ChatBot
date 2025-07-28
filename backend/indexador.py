import os
from datetime import datetime
from typing import List
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# Importa os loaders do LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)

# Configurações iniciais
text_splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=100,
    separator="\n\n",
    paragraph_separator="\n\n"
)

def carregar_documento(caminho_do_arquivo: str):
    """
    Detecta a extensão de um arquivo e usa o Document Loader apropriado do LangChain.
    Retorna uma lista de Documentos LangChain.
    """
    nome_arquivo, extensao = os.path.splitext(caminho_do_arquivo)
    extensao = extensao.lower()
    loader = None
    if extensao == '.pdf':
        print(f"Detectado arquivo PDF. Usando PyPDFLoader...")
        loader = PyPDFLoader(caminho_do_arquivo)
    elif extensao == '.docx':
        print(f"Detectado arquivo DOCX. Usando Docx2txtLoader...")
        loader = Docx2txtLoader(caminho_do_arquivo)
    elif extensao == '.txt':
        print(f"Detectado arquivo TXT. Usando TextLoader...")
        loader = TextLoader(caminho_do_arquivo, encoding='utf-8')
    elif extensao == '.csv':
        print(f"Detectado arquivo CSV. Usando CSVLoader...")
        loader = CSVLoader(file_path=caminho_do_arquivo, encoding='utf-8')
    else:
        return None
    try:
        documentos = loader.load()
        return documentos
    except Exception as e:
        print(f"Ocorreu um erro ao carregar o arquivo {caminho_do_arquivo}: {e}")
        return None

def load_documents(data_dir: str) -> List[Document]:
    """
    Carrega todos os arquivos suportados do diretório usando os loaders do LangChain
    e converte para Documentos do LlamaIndex.
    """
    documentos_finais = []
    for nome_arquivo in os.listdir(data_dir):
        caminho_completo = os.path.join(data_dir, nome_arquivo)
        if os.path.isfile(caminho_completo):
            print("\n" + "="*50)
            print(f"Processando arquivo: {nome_arquivo}")
            documentos_carregados = carregar_documento(caminho_completo)
            if documentos_carregados:
                print(f"Sucesso! '{nome_arquivo}' carregado em {len(documentos_carregados)} parte(s).")
                # Converte para Document do LlamaIndex
                for doc in documentos_carregados:
                    documentos_finais.append(Document(
                        text=doc.page_content,
                        metadata={**doc.metadata, "file_name": nome_arquivo}
                    ))
        else:
            print(f"\nIgnorando '{nome_arquivo}', pois é um diretório.")
    return documentos_finais


def main():
    print("🚀 Sistema de Indexação Aprimorado 🚀")
    
    # Configurações
    DATA_DIR = "../data"
    STORAGE_DIR = "./storage_labse"
    
    Settings.llm = None
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/LaBSE"
    )
    Settings.text_splitter = text_splitter

    # Carregar documentos usando os loaders do LangChain
    print(f"\n📂 Carregando documentos de {DATA_DIR}...")
    documents = load_documents(DATA_DIR)
    if not documents:
        raise ValueError("Nenhum documento válido encontrado!")
    print(f"✅ {len(documents)} documentos/partes processados")

    # Criar/recarregar índice
    if not os.path.exists(STORAGE_DIR):
        print("\n🛠️ Criando novo índice...")
        index = VectorStoreIndex.from_documents(
            documents,
            show_progress=True
        )
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"💾 Índice salvo em {STORAGE_DIR}")
    else:
        print("\n🔍 Recarregando índice existente...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)



if __name__ == "__main__":
    main()