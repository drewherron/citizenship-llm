import os
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader

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

# Define document formatting function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create the vectorstore using OpenAI embeddings and Chroma
def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=".chromadb"
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

    # Initialize LLM (you could use GoogleGenerativeAI or OpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Provide introduction and list loaded document sources
    print("\nWelcome to the Citizenship Study Assistant. Ask me a question and I will answer it using both the base LLM and the RAG-enhanced LLM.\n")
    document_data_sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        document_data_sources.add(doc_metadata['source'])
    for doc in document_data_sources:
        print(f"  {doc}")

    # Start the REPL
    while True:
        try:
            user_input = input("llm>> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye!")
                break

            # Invoke both the base LLM and the RAG chain with user input
            llm_response = llm.invoke(user_input)
            rag_response = rag_chain.invoke(user_input)

            # Extract content from base LLM response
            llm_result = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

            # Print both results for comparison
            print("\n=== Results Comparison ===")
            print("Base LLM Response:")
            print(llm_result)
            print("\nRAG LLM Response:")
            print(rag_response)
            print("=========================\n")

        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break


if __name__ == '__main__':
    main()
