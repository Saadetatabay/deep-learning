import requests

api_url = "http://localhost:8000/chat"

name = input("Lütfen adınızı girin: ")
age = int(input("Lütfen yaşınızı girin: "))
print("sohbet başladı, çıkmak için 'exit' yazın.")

while True:
    user_message = input("Soru: ")
    if user_message.lower() == "exit":
        print("Sohbet sonlandırıldı.")
        break

    payload = {
        "user_id": name,
        "age": age,
        "message": user_message
    }

    try:
        response = requests.post(api_url, json=payload,timeout=30)
        if response.status_code == 200:
            ai_response = response.json().get("response", "")
            print(f"AI Cevap: {ai_response}")
        else:
            print(f"Hata: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"İstek hatası: {e}")
        