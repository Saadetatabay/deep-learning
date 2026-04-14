import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)

name = input("What is your name? ") 
age = input("What is your age? ")

template = f"""Sen profesyonel bir doktor asistanısın. 
Hastanın adı {name} ve yaşı {age}. 
Hastaya yaşına uygun bir şekilde ({name} Bey/Hanım veya sadece {name} diyerek) hitap etmelisin.
Görevin: Sorular sorarak hastanın sağlık durumunu anlamaya çalışmak ve verdiği cevaplara göre uygun tavsiyelerde bulunmaktır.

Önemli Kurallar:
- Tıbbi bir tanı koyma, sadece tavsiye ver.
- Her zaman nazik ve empatik ol.
- Gerektiğinde bir doktora görünmesini hatırlat.

Geçmiş Konuşmalar:
{{history}}

Hasta: {{input}}
Doktor Asistanı:"""

prompt = PromptTemplate(input_variables=["history","input"],template=template)
memory = ConversationBufferMemory(ai_prefix="Doktor Asistanı",human_prefix="Hasta")
conversation = ConversationChain(prompt=prompt,llm=llm, memory=memory)

print("Merhaba, ben doktor asistanınızım. Size nasıl yardımcı olabilirim?")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Görüşmek üzere!")
        break
    response = conversation.predict(input=user_input)
    print(f"Doctor Assistant: {response}")