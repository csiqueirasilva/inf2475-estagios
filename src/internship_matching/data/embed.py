from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = "http://localhost:11434"

def get_embed_func(base_url : str = OLLAMA_URL):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url = base_url)
    embed_func = embeddings.embed_documents
    return embed_func
    