import os
import sys
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Get environment variables
api_key = os.environ.get("API_KEY")

def load_documents(directory_path):
    loader = DirectoryLoader(directory_path, glob='**/*.txt')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    return splits

def create_vectorstore(splits):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(splits, embeddings)
    return vectorstore

def main():
    print("Hi.")

if __name__ == '__main__':
    main()
