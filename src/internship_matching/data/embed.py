from langchain_ollama import OllamaEmbeddings, OllamaLLM

OLLAMA_URL = "http://localhost:11434"

from langchain_ollama import ChatOllama

OLLAMA_LLM = OllamaLLM(
    model="gemma3:12b",
    base_url = OLLAMA_URL, 
    request_timeout=5,
    temperature=0,
    repeat_penalty=1.5,
    num_predict=500
)

def get_embed_func(base_url : str = OLLAMA_URL):
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url = base_url)
    embed_func = embeddings.embed_documents
    return embed_func
    
