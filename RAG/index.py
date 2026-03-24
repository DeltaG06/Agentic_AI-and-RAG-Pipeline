from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(file_path = pdf_path)
docs = loader.load()

#split the docs into smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)


chunks= text_splitter.split_documents(documents = docs)


#vector embeddings

embedding_model = OllamaEmbeddings(
    model = "nomic-embed-text"
)

vector_store  = QdrantVectorStore.from_documents(
    documents = chunks ,
    embedding = embedding_model,
    url = "http://localhost:6333",
    collection_name = "learning rag"
    
)


print("Indexing done...")