import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model

# eğitmeyeceğimiz için compile=False yapıyoruz
model = load_model('lstm_model.h5', compile=False)
print("Model loaded successfully.")


# tokenizer'ı yükle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

input_text = ["This place is absolute garbage..." \
" Half of the tees are not available, including all the grass tees." \
" It is cash only, and they sell the last bucket at 8, despite having lights." \
" And if you finish even a minute after 8, don't plan on getting a drink." \
" The vending machines are sold out (of course) and they sell drinks inside, but close the drawers at 8 on the dot." \
" There are weeds grown all over the place. I noticed some sort of batting cage, but it looks like those are out of order as well." \
" Someone should buy this place and turn it into what it should be.","Old school.....traditional \"mom 'n pop\" quality and perfection."
" The best fish and chips you'll ever enjoy and equally superb fried shrimp. A great out of the way, non-corporate, vestige of Americana."
" You will love it."]

sequences = tokenizer.texts_to_sequences(input_text)
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')
predictions = model.predict(padded_sequences)

predictions = predictions*5

for i,content in enumerate(input_text):
    print(f"Review: {content}")
    print(f"Predicted rating: {predictions[i][0]:.2f}\n")