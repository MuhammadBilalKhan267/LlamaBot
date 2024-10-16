from langchain_community.chat_models import ChatOllama
from .embedfunction import Embedder

llm = ChatOllama(model="llama3.1", top_k=5)

embed = Embedder(
    model='bge-m3'
    )