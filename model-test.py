import os
import warnings
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
#from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import LLMChain


# I know we're getting a LangChainDeprecationWarning, it doesn't matter for this project. Adding this line to suppress the warning for cleaner output.
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Please see the migration guide")

# Load documents
def load_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    if not documents:
        raise ValueError("No documents were loaded. Check the directory path and ensure PDF files are present.")
    print(f"Loaded {len(documents)} documents.")
    #i = 1
    #for document in documents:
    #    print(f"Document {i}:")
    #    print(document)
    #    i += 1
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

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Initialize memory for base LLM
    base_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize memory for RAG LLM
    rag_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create prompt templates
    base_prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping with U.S. citizenship, naturalization, and the citizenship exam.

{chat_history}
User: {user_input}
Assistant:""")

    rag_prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping with U.S. citizenship, naturalization, and the citizenship exam.

Context:
{context}

{chat_history}
User: {user_input}
Assistant:""")

    # Create base LLM chain
    base_llm_chain = LLMChain(
        llm=llm,
        prompt=base_prompt_template,
        memory=base_memory
    )

    # Function to get chat history
    def get_chat_history(_):
        return rag_memory.load_memory_variables({})["chat_history"]

    # Function to get context
    def get_context(inputs):
        return format_docs(
            retriever.get_relevant_documents(inputs["user_input"])
        )

    # Create RAG chain
    rag_chain = (
        RunnableMap(
            {
                "user_input": RunnablePassthrough(),
                "chat_history": get_chat_history,
                "context": get_context,
            }
        )
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )

    # Provide introduction and list loaded document sources
    print("\nWelcome to the Citizenship Study Assistant with memory.\n")
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

            # For base LLM: base_llm_chain with user_input
            base_response = base_llm_chain.predict(user_input=user_input)

            # For RAG LLM: rag_chain with user_input
            rag_response = rag_chain.invoke({"user_input": user_input})

            # Update the memory for the RAG LLM
            rag_memory.save_context({"user_input": user_input}, {"output": rag_response})

            # Print both results for comparison
            print("\n=== Results Comparison ===")
            print("Base LLM Response:")
            print(base_response)
            print("\nRAG LLM Response:")
            print(rag_response)
            print("=========================\n")

        except (KeyboardInterrupt, EOFError):
            print("\nAssistant: Goodbye!")
            break


if __name__ == '__main__':
    main()
