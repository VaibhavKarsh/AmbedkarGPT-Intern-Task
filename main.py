import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate

database_dir = "database"
filepath = "speech.txt"

#### Document Ingestion Start ####

print("Starting data ingestion")
print("Loading document")
    
loader = TextLoader(filepath)
documents = loader.load()
    
if not documents:
    print("No documents were loaded. Check the file path and try again.")
    exit(1)

print(f"Loaded {len(documents)} document(s).")
print("Splitting document into chunks")
    
embedding_model = OllamaEmbeddings(
    model='embeddinggemma:latest'
)
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=25
)
    
texts = text_splitter.split_documents(documents)
    
print(f"Split into {len(texts)} text chunks.")
print("Creating embeddings and storing in Chroma DB")
    
vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=database_dir,
)
    
print(f"Vector database created with {vector_db._collection.count()} documents.")

#### Document Ingestion Complete ####

#### RAG Q&A Start ####

retriever = vector_db.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.1
)

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions based on the provided context.
    Context: {context}
    Question: {question}
    Answer based only on the context provided above.If the answer is not in the context,say "I don't have that information 
    in the provided context"."""
)

chain = prompt_template | llm

print("\n You can ask your questions now!! (type 'exit' to close):\n")
    
while True:
    question = input("Your Question: ").strip()
        
    if not question:
        continue
        
    if question.lower() in ["exit"]:
        print("Thank You!")
        break
    
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    result = chain.invoke({"context": context, "question": question})
        
    print(f"ANSWER: {result.content}")

#### RAG Q&A End ####