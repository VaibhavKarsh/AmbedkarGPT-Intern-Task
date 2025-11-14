# AmbedkarGPT

A RAG Q&A system that answers questions about Dr. B.R. Ambedkar's speech using LangChain, ChromaDB and Ollama.

## Setup

### Download Ollama and start the server.

Then Download gemma3:4b model using : 
```bash 
ollama run gemma3:4b 
```

### Then create a virtual env to keep a separate environment and install all the required libraries

```bash
python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

## To use this system

Run the code using : " python main.py"

Ask your question :
```
Your Question: What does Ambedkar say about the shastras?
```

Type `quit` to exit.

## How It Works

1. It firsts loads `speech.txt`
2. Then it splits the loaded documents into chunks using RecursiveCharacterTextSplitter
3. Then these embeddings are converted into vector embeddings using "embeddinggemma:latest" and stored in the vector database
4. It uses gemma3:4b to generate answers as it is a relatively smaller and efficient model.


## Files

- `main.py` - RAG pipeline
- `speech.txt` - Provided external document
- `requirements.txt` - Dependencies
- `chroma_db/` - Vector database

