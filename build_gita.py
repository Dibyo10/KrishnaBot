import os
import glob
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Load Gita files
pdf_files = glob.glob("gita_data/*.pdf")


documents = []
for pdf in pdf_files:
    documents.extend(PyPDFLoader(pdf).load())


# Debug
print(f"📄 Loaded {len(documents)} Gita documents.")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

print(f"🔍 Split into {len(docs)} chunks for embedding.")

# Build FAISS index
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector index
vectorstore.save_local("gita_faiss_index")
print("✅ FAISS index saved to 'gita_faiss_index/'")