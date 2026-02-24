import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversation.base import ConversationChain

load_dotenv()
api_key = os.getenv("API_KEY")
print("API KEY:", api_key)  # API anahtarını kontrol etmek için ekledim
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory)

name = input("What is your name? ") 
age = input("What is your age? ")
intro = (f"sen bir doktor asistanısın.hastanın adı {name} ve yaşı {age}.hastaya yaşına uygun ismiyle hitap ederek sorular sorarak hastanın sağlık durumunu anlamaya çalışıyorsun. Hastanın verdiği cevaplara göre ona uygun tavsiyelerde bulunuyorsun.")

memory.chat_memory.add_user_message(intro)

print("Merhaba, ben doktor asistanınızım. Size nasıl yardımcı olabilirim?")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Görüşmek üzere!")
        break
    response = conversation.predict(input=user_input)
    print(f"Doctor Assistant: {response}")