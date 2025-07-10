import pickle
import numpy as np
from keras.utils import to_categorical

sequence_length = 100

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(pitchnames)}

network_input = []
network_output = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))
network_output = to_categorical(network_output)

np.save("input.npy", network_input)
np.save("output.npy", network_output)
pickle.dump((pitchnames, note_to_int), open("mappings.pkl", "wb"))
