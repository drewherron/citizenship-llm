import os
import sys
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# Get environment variables
api_key = os.environ.get("API_KEY")

# Load documents
def load_documents(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
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

# Create the conversational retrieval chain
def create_conversational_chain(retriever):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversational_chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': ChatPromptTemplate.from_template("""
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know; don't try to make up an answer.

{context}

Question: {question}
""")}
    )
    return conversational_chain


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

    # Create conversational chain
    print("Setting up conversational chain...")
    chain = create_conversational_chain(retriever)

if __name__ == '__main__':
    main()
