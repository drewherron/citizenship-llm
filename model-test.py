import os
import uuid
import warnings
from datetime import datetime
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# I know we're getting a LangChainDeprecationWarning, it doesn't really matter for this project.
# This line suppresses the warning for cleaner output.
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Please see the migration guide")

def load_pdf_documents(directory_path):
    """
    Loads all PDF documents from the specified directory and returns them as a list of Document objects.

    Args:
        directory_path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of Document objects loaded from the PDFs.

    Raises:
        ValueError: If no documents are loaded from the specified directory.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: '{directory_path}'. Skipping TXT document loading.")
        return []

    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    if not documents:
        raise ValueError("No documents were loaded. Check the directory path and ensure PDF files are present.")
    print(f"Loaded {len(set(doc.metadata['source'] for doc in documents))} PDF documents.")
    return documents

def load_txt_documents(directory_path):
    """
    Loads all TXT documents from the specified directory and returns them as a list of Document objects.

    Args:
        directory_path (str): The path to the directory containing TXT files.

    Returns:
        list: A list of Document objects loaded from the TXT files.
    """
    if not os.path.exists(directory_path):
        print(f"Directory not found: '{directory_path}'. Skipping TXT document loading.")
        return []

    documents = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf8"}
    ).load()

    if not documents:
        print("No TXT documents were loaded. Check the directory path and ensure TXT files are present.")
    else:
        # Adjust the source paths to include './' at the beginning
        for doc in documents:
            if not doc.metadata['source'].startswith('./'):
                doc.metadata['source'] = './' + doc.metadata['source']
        print(f"Loaded {len(documents)} TXT documents.")
    return documents

def split_documents(documents):
    """
    Splits the provided documents into smaller chunks using a recursive character text splitter.

    Args:
        documents (list): A list of Document objects to be split into chunks.

    Returns:
        list: A list of Document chunks after splitting.

    Raises:
        ValueError: If no splits are created from the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    if not splits:
        raise ValueError("No splits were created. Check the document contents and text splitting parameters.")
    print(f"Split into {len(splits)} chunks.")
    return splits

def format_docs(docs):
    """
    Formats a list of Document objects into a single string, separating each document's content with two newlines.

    Args:
        docs (list): A list of Document objects to format.

    Returns:
        str: A formatted string containing the content of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_vectorstore(splits, model_choice):
    """
    Creates a vectorstore from the provided document splits using embeddings and Chroma.

    Args:
        splits (list): A list of Document chunks to be embedded and stored.
        model_choice (str): The selected LLM model.

    Returns:
        Chroma: An instance of the Chroma vectorstore containing the document embeddings.
    """
    if model_choice == "1":
        embeddings = OpenAIEmbeddings()
    elif model_choice == "2":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    elif model_choice == "3":
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError("Invalid model choice for embeddings.")

    # Generate a unique collection name using UUID
    collection_name = str(uuid.uuid4())

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=collection_name,
    )
    print("Vectorstore created successfully.")
    return vectorstore

def create_header(model_name, total_width=60):
    """
    Creates a formatted header string with a specified model name, centered within a line of equal signs.

    Args:
        model_name (str): The name of the model to include in the header.
        total_width (int): The total width of the header line, including padding and the model name. Defaults to 60.

    Returns:
        str: A formatted header string with the model name centered and padded with equal signs.
    """
    model_name_str = f" {model_name} "
    padding_width = (total_width - len(model_name_str)) // 2
    header = "=" * padding_width + model_name_str + "=" * (total_width - padding_width - len(model_name_str)) + "\n"
    return header

def create_footer(model_name, total_width=60):
    """
    Creates a formatted footer string with a specified model name, centered within a line of equal signs.

    Args:
        model_name (str): The name of the model to include in the footer.
        total_width (int): The total width of the footer line, including padding and the model name. Defaults to 60.

    Returns:
        str: A formatted footer string with the model name centered and padded with equal signs.
    """
    model_name_str = f" {model_name} "
    padding_width = (total_width - len(model_name_str)) // 2
    footer = "=" * padding_width + model_name_str + "=" * (total_width - padding_width - len(model_name_str)) + "\n"
    return footer

def main():

    output_filename = "logfile.txt"
    with open(output_filename, "a") as log_file:
        while True:
            # Initialize model selection
            model_names = {
                "1": "gpt-3.5-turbo",
                "2": "gemini-1.5-pro",
                "3": "claude-3-sonnet-20240229"
            }

            # LLM selection menu
            llm = None
            while llm is None:
                print("Select LLM model:")
                for key, name in model_names.items():
                    print(f"{key}. {name}")
                choice = input("Choose the model you want to use: ")

                if choice == "1":
                    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                elif choice == "2":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
                elif choice == "3":
                    llm = ChatAnthropic(model="claude-3-sonnet-20240229")
                else:
                    print("Invalid choice. Try again.")

            model_name = model_names.get(choice)
            header = create_header(model_name)
            footer = create_footer(model_name)
            log_file.write(f"{header}\n")

            # Mode selection menu
            mode = None
            while mode not in ["1", "2", "3"]:
                print("\nSelect mode:")
                print("1. Base LLM")
                print("2. RAG LLM")
                print("3. Both LLMs")
                mode = input("Choose the mode you want to use: ")
                if mode not in ["1", "2", "3"]:
                    print("Invalid choice. Try again.")

            # Document selection menu
            if mode != "1":
                documents = []
                while not documents:
                    print("\nSelect document types to load:")
                    print("1. PDF only")
                    print("2. PDF and TXT files")
                    doc_choice = input("Choose the documents you want to load: ")

                    if doc_choice == "1":
                        print("\nLoading PDF documents...")
                        documents = load_pdf_documents('./documents')
                    elif doc_choice == "2":
                        print("\nLoading PDF documents...")
                        documents = load_pdf_documents('./documents')
                        print("Loading TXT documents...")
                        documents.extend(load_txt_documents('./documents'))
                    else:
                        print("Invalid choice. Try again.")

                if not documents:
                    print("No documents were loaded. Exiting program.")
                    return

                # Split the documents
                print("Splitting documents...")
                splits = split_documents(documents)

                # Create vectorstore
                print("Creating vectorstore...")
                vectorstore = create_vectorstore(splits, choice)

                # Set up the retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                # List loaded document sources
                document_data_sources = set()
                for doc_metadata in retriever.vectorstore.get()['metadatas']:
                    source = doc_metadata['source']
                    if not source.startswith('./'):
                        source = './' + source
                    document_data_sources.add(source)
                print("\nDocuments loaded:")
                for doc in document_data_sources:
                    print(f"  {doc}")
            else:
                # When in Base LLM mode, set retriever to None
                retriever = None
                documents = []
                doc_choice = None

            # Initialize memory for base LLM
            base_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Initialize memory for RAG LLM
            rag_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Create prompt templates
            base_prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping with U.S. citizenship, naturalization, and the citizenship exam. Be a critical tutor and be fair but not so agreeable that your feedback is incorrect. The user wants to learn from their mistakes, and wants honest feedback.

{chat_history}
User: {user_input}
Assistant:""")

            rag_prompt_template = ChatPromptTemplate.from_template("""
You are an assistant helping with U.S. citizenship, naturalization, and the citizenship exam. Be a critical tutor and be fair but not so agreeable that your feedback is incorrect. The user wants to learn from their mistakes, and wants honest feedback.

Context:
{context}

{chat_history}
User: {user_input}
Assistant:""")

            # Function to get chat history for base LLM
            def get_chat_history_base(_):
                return base_memory.load_memory_variables({})["chat_history"]

            # Function to get chat history for RAG LLM
            def get_chat_history_rag(_):
                return rag_memory.load_memory_variables({})["chat_history"]

            # Function to get context
            def get_context(inputs):
                return format_docs(
                    retriever.invoke(inputs["user_input"])
                )

            # Welcome message
            print("\nWelcome to the Citizenship Study Assistant.")
            if mode == "1":
                print("Mode: Base LLM")
            elif mode == "2":
                print("Mode: RAG LLM")
            elif mode == "3":
                print("Mode: Both LLMs")
            print("Type 'menu' to return to the main menu.")
            print("Type 'exit' or 'quit' to end the session.\n")

            # Create base LLM chain
            base_llm_chain = RunnableSequence(
                base_prompt_template,
                llm
            )

            # Create RAG chain
            rag_chain = (
                RunnableMap(
                    {
                        "user_input": RunnablePassthrough(),
                        "chat_history": get_chat_history_rag,
                        "context": get_context,
                    }
                )
                | rag_prompt_template
                | llm
                | StrOutputParser()
            )

            # Start the REPL
            while True:
                try:
                    user_input = input("Query >> ")
                    if user_input.lower() in ['exit', 'quit']:
                        print("Assistant: Goodbye!")
                        return
                    elif user_input.lower() == 'menu':
                        print("Returning to the main menu...\n")
                        break
                    else:
                        # Base LLM only
                        if mode == "1":
                            base_response = base_llm_chain.invoke({
                                "user_input": user_input,
                                "chat_history": get_chat_history_base(None)
                            })

                            base_response_content = base_response.content if hasattr(base_response, 'content') else str(base_response)

                            base_memory.save_context({"user_input": user_input}, {"output": base_response_content})

                            print(f"\n============== {model_name} - Base LLM ==============")
                            print("Response:")
                            print(base_response_content)
                            print("======================================================\n")

                            # Append to log file
                            log_file.write(f"####  User:\n{user_input}\n")
                            log_file.write(f"\n####  Base LLM Response:\n{base_response_content}\n")
                            log_file.write(f"\n{footer}\n")

                        # RAG LLM only
                        elif mode == "2":
                            rag_response = rag_chain.invoke({"user_input": user_input})

                            rag_memory.save_context({"user_input": user_input}, {"output": rag_response})

                            print(f"\n============== {model_name} - RAG LLM ==============")
                            print("Response:")
                            print(rag_response)
                            print("======================================================\n")

                            # Append to log file
                            log_file.write(f"####  User:\n{user_input}\n")
                            if doc_choice == "1":
                                log_file.write(f"\n####  RAG LLM Response (PDF only):\n{rag_response}\n")
                            elif doc_choice == "2":
                                log_file.write(f"\n####  RAG LLM Response (PDF + TXT):\n{rag_response}\n")
                            log_file.write(f"\n{footer}\n")

                        # Both LLMs
                        elif mode == "3":
                            base_response = base_llm_chain.invoke({
                                "user_input": user_input,
                                "chat_history": get_chat_history_base(None)
                            })

                            base_response_content = base_response.content if hasattr(base_response, 'content') else str(base_response)

                            base_memory.save_context({"user_input": user_input}, {"output": base_response_content})

                            rag_response = rag_chain.invoke({"user_input": user_input})

                            rag_memory.save_context({"user_input": user_input}, {"output": rag_response})

                            print(f"\n============== {model_name} ==============")
                            print("Base LLM Response:")
                            print(base_response_content)
                            print("\nRAG LLM Response:")
                            print(rag_response)
                            print("======================================================\n")

                            # Append to log file
                            log_file.write(f"####  User:\n{user_input}\n")
                            log_file.write(f"\n####  Base LLM Response:\n{base_response_content}\n")
                            if doc_choice == "1":
                                log_file.write(f"\n####  RAG LLM Response (PDF only):\n{rag_response}\n")
                            elif doc_choice == "2":
                                log_file.write(f"\n####  RAG LLM Response (PDF + TXT):\n{rag_response}\n")
                            log_file.write(f"\n{footer}\n")

                except (KeyboardInterrupt, EOFError):
                    print("\nAssistant: Goodbye!")
                    return


if __name__ == '__main__':
    main()

