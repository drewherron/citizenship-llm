import os
import sys
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever

# Get environment variables
api_key = os.environ.get("API_KEY")

def load_documents(directory_path):
    loader = DirectoryLoader(directory_path, glob='**/*.txt')
    documents = loader.load()
    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    return splits

# Create Chroma vector database
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

# Create retriever with query transformations
def create_retriever(vectorstore):
    retriever = vectorstore.as_retriever()

    # Prompt for generating alternative queries
    prompt_template = """You are an AI assistant. Generate three different versions of the given user question to retrieve relevant documents from a vector database.

Original question: {question}
"""
    multi_query_prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(temperature=0)

    multi_query_retriever = MultiQueryRetriever(
        retriever=retriever,
        llm=llm,
        prompt=multi_query_prompt
    )

    return multi_query_retriever

def main():
    # Load documents
    directory_path = "documents"
    print("Loading documents...")
    documents = load_documents(directory_path)

    # Split the documents
    print("Splitting documents...")
    splits = split_documents(documents)

    # Create vectorstore
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(splits)

    # Create retriever
    print("Creating retriever...")
    retriever = create_retriever(vectorstore)

if __name__ == '__main__':
    main()
