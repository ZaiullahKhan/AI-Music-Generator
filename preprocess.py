from music21 import converter, instrument, note, chord
import glob
import pickle

notes = []

for file in glob.glob("midi_dataset/*.mid"):
    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)

    elements = parts.parts[0].recurse() if parts else midi.flat.notes

    for element in elements:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)
