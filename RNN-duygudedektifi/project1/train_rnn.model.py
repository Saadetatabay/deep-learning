import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb

# stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the IMDB dataset
# x_train, y_train: eğitim verisi ve etiketleri
max_features = 10000  # en sık kullanılan 10.000 kelime
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) # sayısallaştırılmış veriyi yükle

# sayıları kelimelere dönüştür
word_index = imdb.get_word_index() # kelime indeksini al "the":1 
index_word = {v: k for k, v in word_index.items()} # indeks kelimeye dönüştür "1":"the"

# 3 çıkarmamızın sebebi, 0, 1, 2'nin özel karakterler için ayrılmış olmasıdır.
decoded = " ".join([index_word.get(i-3,"?") for i in x_train[0]])

print("Decoded review:", decoded)

# veri ön işleme

# sayıları kelimelere dönüştür
def preprocess_review(encoded_review):
    words =  [index_word.get(i-3,"?") for i in encoded_review if i > 3]
    # stopword'leri kaldır
    words = [word.lower() for word in words if word not in stop_words and word.isalpha() ]
    # kelimeleri indekslere dönüştür
    return [word_index.get(word, 0) for word in words]

#veriyi temizle
X_train = [preprocess_review(review) for review in x_train]
X_test = [preprocess_review(review) for review in x_test]

# RNN sabit uzunlukta girdi bekler, bu yüzden veriyi pad ediyoruz
maxlen = 200  # maksimum kelime sayısı
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)