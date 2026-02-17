import numpy as np
import nltk
import matplotlib.pyplot as plt
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
original_word_index = imdb.get_word_index() # kelime indeksini al "the":1 
index_to_word = {k+3: v for v,k in original_word_index.items()} # indeks kelimeye dönüştür "1":"the"
index_to_word[0] = "<PAD>"
index_to_word[1] = "<START>"
index_to_word[2] = "<UNK>"

# 3 çıkarmamızın sebebi, 0, 1, 2'nin özel karakterler için ayrılmış olmasıdır.
def decode_review(encoded_review):
    return " ".join([index_to_word.get(i, "?") for i in encoded_review])

print("Encoded review:", x_train[0])  # ilk yorumu sayısal olarak yazdır
print("Decoded review:", decode_review(x_train[0]))  # ilk yorumu çözümlendir ve yazdır

# veri ön işleme

# sayıları kelimelere dönüştür
word_to_index = {v: k for k,v in index_to_word.items()} # kelime indeksine dönüştür "the":1

def preprocess_review(encoded_review):
    #sayıları kelimelere dönüştür
    words =  [index_to_word.get(i, "?") for i in encoded_review if i > 3]
    # stopword'leri kaldır
    cleaned_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    # kelimeleri indekslere dönüştür
    return [word_to_index.get(word, 2) for word in cleaned_words]
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

#modeli eğit
# epochs: tüm eğitim verisi üzerinde kaç kez geçileceği batch_size: 50.000 yorum 64lük gruplara bölünür ve her grup model tarafından işlenir validation_split: eğitim verisinin %20'si doğrulama için ayrılır
history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.2)
#history bir python nesnesidir ve modelin eğitimi sırasında kaydedilen metrikleri içerir.
# history.history['accuracy'] eğitim doğruluğu
# history.history['val_accuracy'] görülmemiş verideki doğruluk
# history.history['loss'] eğitim versinde hata
# history.history['val_loss'] görülmemiş verideki hata

# modeli değerlendir
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_history(history)

#test verisi üzerinde modeli değerlendir
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

#modeli kaydet
model.save('rnn_model.h5')
print("Model saved as rnn_model.h5")