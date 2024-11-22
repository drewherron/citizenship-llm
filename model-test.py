import os
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
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic


# I know we're getting a LangChainDeprecationWarning, it doesn't really matter for this project.
# This line suppresses the warning for cleaner output.
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Please see the migration guide")

# Load documents
def load_documents(directory_path):
    """
    Loads all PDF documents from the specified directory and returns them as a list of Document objects.

    Args:
        directory_path (str): The path to the directory containing PDF files.

    Returns:
        list: A list of Document objects loaded from the PDFs.

    Raises:
        ValueError: If no documents are loaded from the specified directory.
    """
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            documents.extend(docs)
    if not documents:
        raise ValueError("No documents were loaded. Check the directory path and ensure PDF files are present.")
    print(f"Loaded {len(set(doc.metadata['source'] for doc in documents))} documents.")
    return documents

# Split documents into chunks
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

# Define document formatting function
def format_docs(docs):
    """
    Formats a list of Document objects into a single string, separating each document's content with two newlines.

    Args:
        docs (list): A list of Document objects to format.

    Returns:
        str: A formatted string containing the content of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# Create the vectorstore using OpenAI embeddings and Chroma
def create_vectorstore(splits, model_choice):
    """
    Creates a vectorstore from the provided document splits using embeddings and Chroma.

    Args:
        splits (list): A list of Document chunks to be embedded and stored.
        model_choice (str): The selected LLM model.

    Returns:
        Chroma: An instance of the Chroma vectorstore containing the document embeddings.

    Prints:
        str: A message confirming successful creation of the vectorstore.
    """
    if model_choice == "1":
        embeddings = OpenAIEmbeddings()
    elif model_choice == "2":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_query")
    elif model_choice == "3":
        embeddings = OpenAIEmbeddings()
    else:
        raise ValueError("Invalid model choice for embeddings.")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
    )
    print("Vectorstore created successfully.")
    return vectorstore

def main():

    model_names = {
        "1": "gpt-3.5-turbo",
        "2": "gemini-1.5-pro",
        "3": "claude-3-sonnet-20240229"
    }

    llm = None
    while llm is None:
        print("Select LLM model:")
        for key, name in model_names.items():
            print(f"{key}. {name}")
        choice = input("Enter the number of the model you want to use: ")

        if choice == "1":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        elif choice == "2":
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        elif choice == "3":
            llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        else:
            print("Invalid choice. Try again.")

    model_name = model_names.get(choice)
    output_filename = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"

    # Load documents
    directory_path = './documents'
    print("Loading documents...")
    documents = load_documents(directory_path)

    # Split the documents
    print("Splitting documents...")
    splits = split_documents(documents)

    # Create vectorstore
    print("Creating vectorstore...")
    vectorstore = create_vectorstore(splits, choice)

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
        """
        Retrieves the conversation history from the base language model's memory.

        Args:
            _ : Placeholder argument (not used).

        Returns:
            list: A list of messages representing the chat history from the base model.
        """
        return base_memory.load_memory_variables({})["chat_history"]

    # Function to get chat history for RAG LLM
    def get_chat_history_rag(_):
        """
        Retrieves the conversation history from the RAG-enhanced language model's memory.

        Args:
            _ : Placeholder argument (not used).

        Returns:
            list: A list of messages representing the chat history from the RAG model.
        """
        return rag_memory.load_memory_variables({})["chat_history"]

    # Function to get context
    def get_context(inputs):
        """
        Retrieves relevant context documents based on the user's input by querying the retriever.

        Args:
            inputs (dict): A dictionary containing the user's input under the key "user_input".

        Returns:
            str: A formatted string of relevant documents to be used as context in the prompt.
        """
        return format_docs(
            retriever.invoke(inputs["user_input"])
        )

    # List loaded document sources
    document_data_sources = set()
    for doc_metadata in retriever.vectorstore.get()['metadatas']:
        document_data_sources.add(doc_metadata['source'])
    print("\nDocuments loaded:")
    for doc in document_data_sources:
        print(f"  {doc}")

    print("\nWelcome to the Citizenship Study Assistant.")
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
    with open(output_filename, "a") as log_file:
        while True:
            try:
                user_input = input("Query >> ")
                if user_input.lower() in ['exit', 'quit']:
                    print("Assistant: Goodbye!")
                    break

                # Invoke base_llm_chain
                base_response = base_llm_chain.invoke({
                    "user_input": user_input,
                    "chat_history": get_chat_history_base(None)
                })

                # Extract Base LLM response
                base_response_content = base_response.content if hasattr(base_response, 'content') else str(base_response)

                # Update the memory for the Base LLM
                base_memory.save_context({"user_input": user_input}, {"output": base_response_content})

                # Invoke rag_chain
                rag_response = rag_chain.invoke({"user_input": user_input})

                # Update the memory for the RAG LLM
                rag_memory.save_context({"user_input": user_input}, {"output": rag_response})

                # Print both results for comparison
                print(f"\n============== {model_name} ==============")
                print("Base LLM Response:")
                print(base_response_content)
                print("\nRAG LLM Response:")
                print(rag_response)
                print("============================================\n")

                # Append user input and responses to the log file
                log_file.write(f"  ####  User:               ####\n{user_input}\n")
                log_file.write(f"\n####  Base LLM Response:  #### \n{base_response_content}\n")
                log_file.write(f"\n####  RAG LLM Response:   ####\n{rag_response}\n")
                log_file.write("============================================\n\n")

            except (KeyboardInterrupt, EOFError):
                print("\nAssistant: Goodbye!")
                break


if __name__ == '__main__':
    main()
