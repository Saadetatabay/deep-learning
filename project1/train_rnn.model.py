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
maxlen = 500  # maksimum kelime sayısı
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

#model oluşturma
model = Sequential() # sıralı model oluştur

#max_feauters kaç farklı kelime olduğunu belirtir, output_dim gömülü vektörün boyutunu belirtir her kelime 32 boyutlu vektör
#input_length ise her girdi dizisinin uzunluğunu belirtir
#keliemleri vektörlere dönüştürmek için embedding katmanı ekle
model.add(Embedding(input_dim=max_features, output_dim=32, input_length=maxlen)) # embedding katmanı ekle

# basit bir RNN katmanı ekle
model.add(SimpleRNN(32)) # 32 nöronlu RNN katmanı

# çıktı katmanı ekle sigmoid 0-1 arasında değer döndürür
model.add(Dense(1, activation='sigmoid')) # ikili sınıflandırma için

# modeli derle
# optimizer: ağırlakları güncellemek için kullanılan algoritma loss hesaplandıkça ağırlıkları hatayı mninimize edecek şekilde günceller, adam genellikle iyi sonuç verir
# loss: kayıp fonksiyonu, 
# metrics: modelin performansını değerlendirmek için kullanılan metrikler
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

