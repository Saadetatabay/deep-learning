import numpy as np
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import text_to_word_sequence

max_features = 10000  # en sık kullanılan 10.000 kelime
maxlen = 500  # maksimum kelime sayısı  

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

word_index = imdb.get_word_index() # kelime indeksini al "the":1
index_to_word = {k+3: v for v,k in word_index.items()}
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

# kelime->indeks dönüşümü
word_to_index = {v: k for k,v in index_to_word.items()}


# modeli yükle
model = load_model('rnn_model.h5')
print("Model loaded successfully.")

def predict_review(review):
    """
    kullanıcan gelen metni temizler
    ve modelin anlayacağı formata dönüştürür
    tahmin yapar ve sonucu döndürür
    """
    #yorumu 
    words = text_to_word_sequence(review) # metni kelimelere ayır
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

    #kelimeleri indekslere dönüştür
    encoded_review = [word_to_index.get(word,2) for word in words]  # 2: <UNK>
    #veriyi pad et
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    #tahmin yap
    prediction = model.predict(padded_review)
    prediction = prediction[0][0]  # tahmin sonucunu al
    print(f"Prediction: {prediction:.4f}")
    if prediction >= 0.5:
        print("Positive review")
    else:
        print("Negative review")

user_input = input("Enter a movie review: ")
predict_review(user_input)