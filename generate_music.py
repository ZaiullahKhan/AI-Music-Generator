import numpy as np
import pickle
from keras.models import load_model

model = load_model("music_model.h5")
X = np.load("input.npy")

with open("mappings.pkl", "rb") as f:
    pitchnames, note_to_int = pickle.load(f)

int_to_note = {num: note for note, num in note_to_int.items()}
n_vocab = len(pitchnames)

start = np.random.randint(0, len(X) - 1)
pattern = X[start]
pattern = pattern.tolist()

prediction_output = []

for note_index in range(500):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)

    pattern.append([index / float(n_vocab)])
    pattern = pattern[1:]

pickle.dump(prediction_output, open("generated_notes.pkl", "wb"))
