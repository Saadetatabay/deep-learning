import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = [
"bugün hava top oynamak için çok güzel",
"sabah erken kalkmak beni mutlu ediyor",
"kahve içmeden güne başlayamıyorum",
"bugün biraz yorgun hissediyorum",
"ders çalışmam gerekiyor ama üşeniyorum",
"akşam yürüyüşe çıkmak istiyorum",
"annem bugün çok güzel yemek yaptı",
"trafik yüzünden okula geç kaldım",
"arkadaşlarımla sinemaya gitmek istiyorum",
"bu hafta sonu dinlenmek istiyorum",
"kitap okumak beni rahatlatıyor",
"spor yaptıktan sonra kendimi iyi hissediyorum",
"bugün çok yoğun bir gündü",
"yarın önemli bir sınavım var",
"telefonumun şarjı bitmek üzere",
"müzik dinlerken ders çalışıyorum",
"bugün kendime zaman ayırdım",
"yağmur yağınca evde kalmak istiyorum",
"yeni bir şeyler öğrenmek beni heyecanlandırıyor",
"akşam erken uyumam gerekiyor",
"sabah alarm çalınca uyanmak zor oluyor",
"arkadaşımla kahve içmeye gittim",
"bugün motivasyonum biraz düşük",
"yemek yaptıktan sonra mutfağı topladım",
"ders arasında kısa bir mola verdim",
"bugün kendimi çok enerjik hissediyorum",
"hafta sonu ailemi ziyaret edeceğim",
"alışverişe çıkmam gerekiyor",
"film izlerken patlamış mısır yedim",
"spor salonuna yazılmayı düşünüyorum",
"bugün hava biraz soğuk",
"toplantı beklediğimden uzun sürdü",
"ödevimi son dakikada bitirdim",
"bugün erken yatmayı planlıyorum",
"yeni bir projeye başladım",
"bilgisayarım çok yavaş çalışıyor",
"arkadaşım bana sürpriz yaptı",
"bugün çok fazla kahve içtim",
"sabah koşu yapmak istiyorum",
"kendimi geliştirmek için çabalıyorum",
"bugün biraz stresliyim",
"yarın erken kalkmam gerekiyor",
"ders notlarımı tekrar ediyorum",
"akşam dışarıda yemek yiyeceğiz",
"bugün çok güzel bir haber aldım",
"markete gitmeyi unuttum",
"telefonla uzun süre konuştum",
"bugün planlarım değişti",
"yeni bir hobi edinmek istiyorum",
"kendime küçük hedefler koyuyorum",
"bugün erken kalktım ve kahvaltı yaptım",
"otobüsü kaçırdım ve işe geç kaldım",
"yeni aldığım kitabı okumaya başladım",
"öğle arasında biraz yürüyüş yaptım",
"akşam yemeğinde makarna yedik",
"telefonumu evde unuttum",
"bugün biraz başım ağrıyor",
"yarın arkadaşımın doğum günü",
"çamaşırları makineye attım",
"ev temizliği yapmam gerekiyor",
"ders çalışırken müzik açtım",
"marketten süt almayı unuttum",
"akşam haberleri izledim",
"bugün dışarı çıkmak istemiyorum",
"yeni bir diziye başladım",
"spor yapmaya tekrar başlayacağım",
"sabah kahvemi balkonda içtim",
"bugün hava çok rüzgarlı",
"toplantı iptal edildi",
"ödevimi bitirmek için uğraşıyorum",
"arkadaşımla uzun uzun konuştuk",
"bugün çok fazla mesaj aldım",
"yeni bir uygulama indirdim",
"akşam erken uyudum",
"yarın alışverişe gideceğim",
"bugün biraz dalgındım",
"yolda eski bir arkadaşıma rastladım",
"hafta sonu şehir dışına çıkacağız",
"bilgisayarımı güncelledim",
"akşamüstü kahve molası verdim",
"bugün kendime sağlıklı yemek yaptım",
"sabah koşuya çıktım",
"evde tek başıma film izledim",
"yeni bir tarif denedim",
"bugün işler planladığım gibi gitmedi",
"ders notlarımı düzenledim",
"telefonum sürekli çalıyor",
"akşam arkadaşlarımla buluştum",
"bugün biraz huzursuz hissediyorum",
"yarın önemli bir görüşmem var",
"spor yaptıktan sonra duş aldım",
"evde küçük bir tamirat yaptım",
"bugün motivasyonum yüksekti",
"alışveriş listesini hazırladım",
"akşam yürüyüş yaparken yağmura yakalandım",
"bugün erken saatlerde uyandım",
"yeni hedefler belirledim",
"arkadaşım bana mesaj attı",
"bugün çok fazla su içtim",
"yarın dinlenmeyi planlıyorum"
]

tokenizer = Tokenizer()
# her kelimeye benzersiz bir indeks atar
tokenizer.fit_on_texts(data) #sözlük
eşsiz_kelimeler = len(tokenizer.word_index)
print(eşsiz_kelimeler) # kelime indeksini yazdırır {"bugün":1, "hava":2, ...}
# metinleri sayısal dizilere dönüştürür burada oluşturulan sözlükten her kelimenin idsine bakılır
sequences = tokenizer.texts_to_sequences(data)

# n-gram oluşturma

# her cümlenin ilk kelimesiyle başlayarak n-gram'ler oluşturulur
n_grams = []
for seq in sequences:
    for i in range(1, len(seq)):
       # [seq[:i+1]]  # seq[:2] -> [1, 2], seq[:3] -> [1, 2, 3
        n_grams.append(seq[:i+1])

# padding işlemi
max_seq_length = max(len(seq) for seq in n_grams)
padded_n_grams = pad_sequences(n_grams, maxlen=max_seq_length,padding='pre') #baştan padding yapar

print("Padded n-grams shape:", padded_n_grams) # (n_samples, max_seq_length)
# girdi ve çıktı verilerini oluşturma
X = padded_n_grams[:,:-1] # son kelime hariç tüm kelimeler girdi olarak kullanılır
y = padded_n_grams[:,-1]  # son kelime çıktı olarak kullanılır

# hedef kelimeleri one-hot encode etme
# modelin bulması gereken y değişkenini işaretliyrouz aslında her değere
# ayrı ayrı bir sütun açarak o sütunda sadece o kelime varsa 1 yoksa 0 yapar
y = tensorflow.keras.utils.to_categorical(y, num_classes=eşsiz_kelimeler+1) # kelime sayısı + 1 (0 için)

# ben bugün hastayım -> [1,2,3]
# ben bugün -> [1,2]
# ben -> [1]
# girdi: [1,2] -> çıktı: 3
# y ye one hot encode yaparak [0,0,0,1,0,...] yaparız

# model oluşturma
model = Sequential()
# input_dim kaç tane eşsiz kelime olduğunu belirtir0
# output_dim gömülü vektörün boyutunu belirtir her kelime 50 boyutlu vektör
# input_length ise her girdi dizisinin uzunluğunu belirtir
model.add(Embedding(input_dim=eşsiz_kelimeler+1, output_dim=50, input_length=X.shape[1])) # embedding katmanı ekle

#lstm e her cümle kelimeleri vektörlere dönüştürülmüş olarak gelir ve sırayla işler
# unutma giriş çıkış olarak çalışır her cümle için 100 boyutlu bir vektör döndürür
model.add(LSTM(100)) # 100 nöronlu LSTM katmanı
#cümlenin anlamını 100 boyutlu bir vektörle özetler
# bu 100 boyutlu özet vektörü alır softmax algoritmasıyla her kelime için bir olasılık döndürür

model.add(Dense(eşsiz_kelimeler+1, activation='softmax')) # çıktı

#loss da hata hesaplanır ve modelin çıktısı ile gerçek çıktı arasındaki farkı minimize etmeye çalışırız adam optimizasyon algoritmasıyla ağırlıkları güncelleriz
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
