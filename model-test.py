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


#print(os.getenv("OPENAI_API_KEY"))

# Load documents
def load_documents(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    documents = loader.load()
    if not documents:
        raise ValueError("No documents were loaded. Check the directory path and ensure PDF files are present.")
    print(f"Loaded {len(documents)} documents.")
    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("No splits were created. Check the document contents and text splitting parameters.")
    print(f"Split into {len(splits)} chunks.")
    return splits

# Create the vectorstore using OpenAI embeddings and Chroma
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./citizenship_data/.chromadb"
    )
    print("Vectorstore created successfully.")
    return vectorstore

def main():

    # Load documents
    directory_path = './documents'
    print("Loading documents...")
    documents = load_documents(directory_path)

    # Split the documents
    print("Splitting documents...")
    splits = split_documents(documents)

    # Create vectorstore
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(splits)

    # Set up the retriever and prompt
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define document formatting function
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Introduction, list loaded document sources
    print("\nWelcome to the Citizenship Study Assistant. Ask me a question and I will answer it from the documents loaded.\n")
    document_data_sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        document_data_sources.add(doc_metadata['source'])
    for doc in document_data_sources:
        print(f"  {doc}")
    print()

    # Start the REPL
    while True:
        try:
            user_input = input("llm>> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye!")
                break

            # Invoke the RAG chain with user input
            result = rag_chain.invoke(user_input)
            print("Assistant:", result)
        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break

if __name__ == '__main__':
    main()
