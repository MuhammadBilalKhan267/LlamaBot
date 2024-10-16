from django.shortcuts import render, HttpResponse, HttpResponseRedirect
from django.http import JsonResponse
from .models import Chat, Messages
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from datetime import datetime
import pytz
from tzlocal import get_localzone
import chromadb
from io import BytesIO
from .embedfunction import Embedder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .pdf_loader import load_pdf_from_file
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain_experimental.text_splitter import SemanticChunker
from .__init__ import llm, embed

# Create your views here.
def home(request):
    if request.user.is_authenticated:
        print("hi")
        chats = request.user.chats.all().order_by("-updated_at")
        return render(request, "home.html", {"chats": chats})
    else:
        print("hi22")
        return HttpResponseRedirect("/login/")
    

def base(request):
    return render(request, "base.html")

def chat(request, id):
    if not request.user.is_authenticated:
        return HttpResponseRedirect("/login/")
    if request.method == "GET":
        try:
            chat = request.user.chats.get(id=id)
            messages = chat.messages_set.all()
        except Chat.DoesNotExist:
            return HttpResponseRedirect("/")
        
        local_tz = get_localzone()
        for message in messages:

            utc_time = message.created_at

            # Set the timezone to UTC
            utc_time = utc_time.replace(tzinfo=pytz.UTC)

            # Convert to local time
            message.created_at = utc_time.astimezone()


        return render(request, "chat.html", {"chat": chat, "messages": messages})

    
def add_message(request):
    if not request.user.is_authenticated:
        return HttpResponseRedirect("/login/")
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        try:
            chat = request.user.chats.get(id=data["chat_id"])
            history = chat.messages_set.all().order_by("-created_at")[:10]
        except Chat.DoesNotExist:
            return HttpResponseRedirect("/")
        message = data["message"]

        client = chromadb.PersistentClient(path="indexed_documents")  # or HttpClient()
        db = Chroma(client=client, collection_name='c_'+data['user']+data['chat_id'],embedding_function=Embedder("bge-m3"))
        retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0., "k": 5})
        print(retriever.invoke(message))
        
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key='chat_history',
            output_key="answer",
            return_messages=True)
        
        print(history)
        for i in range(len(history)-1, 0, -2):
            memory.save_context({"input": history[i].text}, {"answer": history[i-1].text})

        print(memory.load_memory_variables({}))
        prompt_template_doc = """
            Use the following pieces of context and chat history to answer the questions.
            Respond in a conversational manner. You are an asssistant talking to a user.
            Talk in a friendly conversational manner without giving out any unnecessary details.
            Context:
            {context}
            Chat History:
            {chat_history}
            Question: {question}
            Answer:
            """
        
        
        prompt_doc = PromptTemplate(
            template=prompt_template_doc, input_variables=["context", "question","chat_history"]
        )
        print("here")
        crc = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever,
            verbose=False,
            combine_docs_chain_kwargs={"prompt": prompt_doc}
            )

        response = crc.invoke({"question": message})
        print(response)
        
        message = Messages(chat=chat, text=data["message"], type=0)
        message.save()
        ai = Messages(chat=chat, text=response["answer"], type=1)
        ai.save()

        response_data = {
            "human": {
                'id': message.id,
                'text': message.text,
                'created_at': message.created_at,
                'type': message.type
            },
            "assistant": {
                'id': ai.id,
                'text': ai.text,
                'created_at': ai.created_at,
                'type': ai.type
            }
        }

        return JsonResponse(response_data, safe=False)
    else:
        return HttpResponseRedirect("/")
    

def add_file(request):
    if not request.user.is_authenticated:
        return HttpResponseRedirect("/login/")
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')
        chat_id = request.POST.get('chat_id')
        user = request.POST.get('user')
        file_object = BytesIO(uploaded_file.read())

        # Load the PDF content into documents
        doc = load_pdf_from_file(file_object)

        try:
            chat = request.user.chats.get(id=chat_id)
        except Chat.DoesNotExist:
            return HttpResponseRedirect("/")
        
        print(doc.page_content[:10000])
        text_splitter = SemanticChunker(embed, breakpoint_threshold_amount=70, buffer_size=6)
        chunked_docs = text_splitter.split_documents([doc])

        print(chunked_docs)
        for doc in chunked_docs:
            print(doc.page_content)

        chunked_docs_json = [
            doc.page_content for doc  in chunked_docs
        ]
        metadatas = [{"no": i} for i in range(len(chunked_docs))]
        ids = [uploaded_file.name + str(i) for i in range(len(chunked_docs_json))]
        client = chromadb.PersistentClient(path="indexed_documents")  # or HttpClient()
        col = client.get_or_create_collection('c_'+user+chat_id, embedding_function=embed, metadata={"hnsw:space": "cosine"})
        try:
            col.add(
                documents=chunked_docs_json,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            return JsonResponse({"status": 500, "error": str(e)}, safe=False)

        return JsonResponse({"status": 200}, safe=False)
    else:
        return HttpResponseRedirect("/")
    
def add_chat(request):
    if request.user.is_authenticated:
        if request.method=="POST":
            print(request.POST)
            chat = Chat(user=request.user, name=request.POST['name'])
            chat.save()
            return HttpResponseRedirect("/chat/"+str(chat.id))
        else:
            return HttpResponseRedirect("/")
    else:
        return HttpResponseRedirect("/login/")
    

def delete_chat(request):
    if request.user.is_authenticated:
        if request.method == "POST":
            data = json.loads(request.body.decode("utf-8"))
            id = data["chat_id"]
            print(id)
            try:
                chat = request.user.chats.get(id=id)
                print(chat)
                chat.delete()
            except Chat.DoesNotExist:
                return HttpResponseRedirect("/")
            try:
                collection_name = 'c_'+request.user.username+str(id)
                client = chromadb.PersistentClient(path="indexed_documents")  # or HttpClient()
                client.delete_collection(collection_name)
            except Exception as e:
                print(f"Error deleting collection: {e}")

            return HttpResponse(status=200)
        else:
            return HttpResponseRedirect("/")
        return HttpResponseRedirect("/")
    else:
        return HttpResponseRedirect("/login/")