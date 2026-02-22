import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle # tokenizer'ı kaydetmek ve yüklemek için

from sklearn.model_selection import train_test_split #veriyi eğitim ve test setlerine bölmek için
from sklearn.preprocessing import MinMaxScaler #veriyi ölçeklendirmek için

from tensorflow.keras.preprocessing.text import Tokenizer #metni sayısal dizilere dönüştürmek için
from tensorflow.keras.preprocessing.sequence import pad_sequences #dizileri aynı uzunluğa getirmek için, padding işlemi için
from tensorflow.keras.models import Sequential #model oluşturmak için
from tensorflow.keras.layers import Embedding, LSTM, Dense #model katmanları
from tensorflow.keras.losses import MeanSquaredError #regresyon için kayıp fonksiyonu    
from tensorflow.keras.metrics import MeanAbsoluteError #regresyon için metrik


splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet', 'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/Yelp/yelp_review_full/" + splits["train"])
print(df.head())

# lablları 0-4 aralığından 1-5 aralığına çekiyoruz
df['label'] = df['label'] + 1

texts = df['text'].values
print(texts[:5])
labels = df['label'].values

# en sık kullanılan 10.000 kelimeyi dikkate al
# oov_token, eğitim sırasında görülmeyen kelimeler için kullanılacak özel bir token'dır.
# Bu token, modelin eğitildiği kelime dağarcığında olmayan kelimeleri temsil eder.
# Böylece model, eğitim sırasında karşılaşmadığı kelimelerle başa çıkabilir ve tahmin yaparken bu tür kelimeleri oov_token ile temsil eder.
# anlam kaybını azaltmak için 10.000 kelimeyi dikkate alıyoruz
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")

tokenizer.fit_on_texts(texts) # metinleri sayısal dizilere dönüştürmek için tokenizer'ı eğit
with open('tokenizer.pickle', 'wb') as handle: # tokenizer'ı kaydet
    pickle.dump(tokenizer, handle)
sequences = tokenizer.texts_to_sequences(texts) # metinleri sayısal dizilere dönüştür
padded_sequences = pad_sequences(sequences, maxlen = 100, padding='post') # dizileri aynı uzunluğa getirmek için padding işlemi yap

# labelı 0--1 aralığına ölçeklendir
scaler = MinMaxScaler()
# reshape(-1, 1) ile labels'ı 2 boyutlu hale getiriyoruz çünkü MinMaxScaler 2 boyutlu veri bekler
labels_scaled = scaler.fit_transform(labels.reshape(-1, 1))

# veriyi eğitim ve test setlerine böl   
# yüzde 20'sini test seti olarak ayırıyoruz
# random_state=42, bölmenin rastgeleliğini kontrol eder ve aynı sonucu verir
X_train, X_test, y_train, y_test = train_test_split(padded_sequences,labels_scaled, test_size=0.2, random_state=42)

print("Eğitim seti boyutu:", len(X_train))
print("Test seti boyutu:", len(X_test))
print("Örnek eğitim verisi (padded):", X_train[0])
print("Örnek eğitim etiketi (ölçeklendirilmiş):", y_train[0])


model = Sequential()
#input_dim kaç tane eşsiz kelime olduğunu belirtir
#output_dim gömülü vektörün boyutunu belirtir her kelime 64 boyutlu vektör
#input_length ise her girdi dizisinin uzunluğunu belirtir padded_sequences ile 100 yaptık
#keliemleri vektörlere dönüştürmek için embedding katmanı ekle
model.add(Embedding(input_dim=10000, output_dim=64, input_length=100)) # embedding katmanı ekle

# bunun sonucunda 128 tane sayıdan oluşan bir özet vektör elde ederiz
# return_sequences=True yaparsak her kelime için bir özet vektör döndürür 
model.add(LSTM(128)) # 64 nöronlu LSTM katmanı ek

# 64 nöronlu bir çıktı katmanı ekle
# her bir nöron farklı bir özellik öğrenir ve bu özelliklerin kombinasyonu bize tahmin yaparken yardımcı olur
# relu aktivasyon fonksiyonu, negatif değerleri sıfırlar ve pozitif değerleri olduğu gibi bırakır, bu da modelin daha karmaşık ilişkileri öğrenmesine yardımcı olur
# relu hızlıdır ve vanishing gradient problem'ini azaltır, bu da derin ağlarda daha iyi performans sağlar
# ara katman olarak eklediğimiz için modelin öğrenme kapasitesini artırır ve daha karmaşık ilişkileri öğrenmesine yardımcı olur
model.add(Dense(64, activation='relu')) # çıktı katmanı ekle

# relu tanh ara katman
# softmax (çok sınıflı) ve sigmoid (iki sınıflı) sınıflandırma


# tek bir nöronlu çıktı katmanı ekle
model.add(Dense(1)) # tek bir nöronlu çıktı katmanı ekle

# biz sınıflandırma yapmıyoruz, regresyon yapıyoruz, bu yüzden kayıp fonksiyonu olarak MeanSquaredError kullanıyoruz
# metric de accuracy değil MeanAbsoluteError kullanıyoruz çünkü regresyon problemlerinde doğruluk yerine hata ölçümleri daha anlamlıdır
model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

# epochs: tüm eğitim verisi üzerinde kaç kez geçileceği batch_size: 50.000 yorum 64lük gruplara bölünür ve her grup model tarafından işlenir validation_split: eğitim verisinin %20'si doğrulama için ayrılır
# her 64 yorumda bir modelin doğruluğunu ve kaybını hesaplar
history = model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
# history nesnesinin içinde :
#    loss her epoch daki hata değeri(MeanSquaredError)
#    val_loss her epoch daki doğrulama hata değeri(MeanSquaredError) modedlin hiç görmediği veri

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# plotlama ile overfitting olup olmadığını görebiliriz
# eğer eğitim kaybı sürekli azalırken doğrulama kaybı artmaya başlarsa model overfitting yapıyor demektir
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss mean squared error')
plt.legend()
plt.show()
model.save('lstm_model.h5')

