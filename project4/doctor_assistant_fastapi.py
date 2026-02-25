import os
from pydantic import BaseModel
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversation.base import ConversationChain
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

app = FastAPI(title="Doctor Assistant API")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)

# her kullanıcı için ayrı bir bellek oluşturmak için bir sözlük kullanıyoruz
# dicitonery de her değerin key i str vlause da ConversationBufferMemory olacak şekilde tanımlıyoruz
user_memories : Dict[str, ConversationBufferMemory] = {}

#basemodel ile sabit bir şema oluşturuyoruz. Bu şema user_id ve message içerecek şekilde tanımlanıyor
class ChatRequest(BaseModel):
    user_id: str
    age: int
    message: str

class ChatResponse(BaseModel):
    response: str

#sohbet enpoint
#sunuya post request /chat adresine gönderildiğinde
# aşağıda fonk çalışacak ve ChatResponse modelini döndürecek şekilde tanımlanıyor
@app.post("/chat", response_model=ChatResponse)
def chat_with_doctor(request: ChatRequest):
    try:
        if request.user_id not in user_memories:
            user_memories[request.user_id] = ConversationBufferMemory(return_messages=True)
        
        user_memory = user_memories[request.user_id]
        if len(user_memory.chat_memory.messages) == 0:
            intro = (f"sen bir doktor asistanısın.hastanın adı {request.user_id} ve yaşı {request.age}.hastaya yaşına uygun ismiyle hitap ederek sorular sorarak hastanın sağlık durumunu anlamaya çalışıyorsun. Hastanın verdiği cevaplara göre ona uygun tavsiyelerde bulunuyorsun.")
            user_memory.chat_memory.add_user_message(intro)

        conversation_chain = ConversationChain(llm=llm, memory=user_memory)
        response = conversation_chain.predict(input=request.message)

        print(f"ai response: {response}")  # AI yanıtını kontrol etmek için ekledim
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
