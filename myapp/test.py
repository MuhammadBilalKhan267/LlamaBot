from colorama import Fore
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
import chromadb
from embedfunction import Embedder
from langchain.prompts import PromptTemplate

if __name__ == "__main__":
    
    llm = ChatOllama(model="llama3", top_k=5)

    prompt_template_doc = """
        Use the following pieces of context to answer the question at the end.
        {context}
        If the the context is irrelevant ignore the context.
        You can also look into chat history.
        {chat_history}
        Question: {question}
        Answer:
        """
    
    
    prompt_doc = PromptTemplate(
        template=prompt_template_doc, input_variables=["context", "question","chat_history"]
    )
    
    memory = ConversationSummaryBufferMemory(llm= llm,memory_key="chat_history", return_messages=True,output_key='answer')
    memory.save_context({"input": "I am Bilal"}, {"answer": "Hello Bilal, How can I help you?"})
    
    client = chromadb.PersistentClient(path="indexed_documents")  # or HttpClient()
    db = Chroma(client=client, collection_name='c_bilal1',embedding_function=Embedder("llama3"))
    
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0, "k": 5})
    
    support_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=False,
        memory = memory,
        combine_docs_chain_kwargs={"prompt": prompt_doc}
    )
    
    response = support_qa.invoke({"question": "what is my name"})
    print(response)

    
