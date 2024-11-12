import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

index_name = 'legal-chatbot'
embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
vectorstores = PineconeVectorStore(
  index_name=index_name,
  embedding=embedding,
  pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

# doc_search = PineconeVectorStore.from_documents(documents=docs, embedding=embedding, index_name=index_name)

def add_case(docs):
  loader = CSVLoader(file_path="sample.csv", encoding="utf-8")
  docs = loader.load()
  return vectorstores.aadd_documents(docs)

def case_search(query):
  return vectorstores.similarity_search(query, k=3)

if __name__ == '__main__':
  add_case()