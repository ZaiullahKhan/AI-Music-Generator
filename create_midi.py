from music21 import note, chord, stream
import pickle

with open("generated_notes.pkl", "rb") as f:
    prediction_output = pickle.load(f)

offset = 0
output_notes = []

for pattern in prediction_output:
    if '.' in pattern or pattern.isdigit():
        notes_in_chord = [int(n) for n in pattern.split('.')]
        new_chord = chord.Chord(notes_in_chord)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        output_notes.append(new_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write("midi", fp="output.mid")
