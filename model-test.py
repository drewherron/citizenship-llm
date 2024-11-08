import os
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    #prompt = hub.pull("rlm/rag-prompt")

    # Initialize LLM (you could use GoogleGenerativeAI or OpenAI)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Initialize memory for base LLM
    base_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize memory for RAG LLM
    rag_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # A prompt for the base LLM
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

    # Create RAG chain
    rag_chain = (
        {
            "user_input": RunnablePassthrough(),
            "chat_history": rag_memory
        }
        | RunnableMap(
            {
                "context": lambda inputs: format_docs(
                    retriever.get_relevant_documents(inputs["user_input"])
                ),
                "user_input": lambda inputs: inputs["user_input"],
                "chat_history": lambda inputs: inputs["chat_history"],
            }
        )
        | rag_prompt_template
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
    print()

    # Start the REPL
    while True:
        try:
            user_input = input("llm>> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye!")
                break

            # For base LLM, we use the base_llm_chain with memory
            base_response = base_llm_chain.predict(user_input=user_input)

            # Invoke both the base LLM and the RAG chain with user input
            llm_response = llm.invoke(base_prompt)
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
