from music21 import converter, instrument, note, chord
import numpy as np
from keras.utils import np_utils
import glob
import pickle
from typing import Any, List, Tuple


class PrepareData():
    def __init__(self, note_data_path: str = "midi_songs/*.mid"):
        self.notes = self.extract_notes(note_data_path)
        self.num_unique_notes = self.get_num_unique_notes(self.notes)
        self.pitch_names = self.get_pitchnames(self.notes)
        self.note_to_int = dict((note, number)
                                for number, note in enumerate(self.pitch_names))

    def get_num_unique_notes(self, notes: List) -> int:
        return len(set(notes))

    def get_pitchnames(self, notes: List) -> List:
        return sorted(set(item for item in notes))

    def get_notes(self) -> List:
        return self.notes

    def extract_notes(self, path: str) -> List[Any]:
        """Extract all the notes from the data MIDI folder. Place all notes
        into an array

        Returns:
            List[Any]: [description]
        """
        notes = []

        try:
            pickle_in = open('data/notes', 'rb')
            notes = pickle.load(pickle_in)
            pickle_in.close()
        except:
            for file in glob.glob(path):
                midi = converter.parse(file)

                notes_to_parse = None
                parts = instrument.partitionByInstrument(midi)
                if parts:  # file has instrument parts
                    notes_to_parse = parts.parts[0].recurse()
                else:  # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n)
                                              for n in element.normalOrder))

            with open('data/notes', 'wb') as filepath:
                pickle.dump(notes, filepath)

        return notes

    def generate_training_data(self, sequence_length: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
        """Take all extracted notes and convert into training data. 

        Network input/training data: Many sequences of 100 notes
        Network output/label: Note following the 100th, the 101st note

        Args:
            notes (List[Any]): [description]
        """
        network_input = []
        network_output = []

        # Generate sequence of 100nth input notes, followed by the 101nth note.
        # First 100n notes are inputs, 101n note is the predicted output
        for i in range(0, len(self.notes) - sequence_length, 1):
            sequence_in = self.notes[i:i + sequence_length]
            sequence_out = self.notes[i + sequence_length]
            network_input.append([self.note_to_int[char]
                                  for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])

        num_samples = len(network_input)

        # Reshape the input into a format compatible with LSTM layers
        # Array with num_samples of notes. Each sample has 100 notes and they are flat.
        # i.e. [ [note, ..., 100th note], [note, ..., 100th note], ..., num_samples[note, ..., 100th note]]
        network_input = np.reshape(
            network_input, (num_samples, sequence_length, 1))
        num_unique_notes = self.get_num_unique_notes(self.notes)
        network_input = network_input / float(num_unique_notes)

        # perform one-hot encoding
        # network_output contains 158 different pitches
        network_output = np_utils.to_categorical(network_output)
        print('network_input shape: ', network_input.shape)
        print('network_output shape: ', network_output.shape)

        return network_input, network_output, num_samples
