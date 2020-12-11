from music21 import converter, instrument, note, chord
import numpy as np
from keras.utils import np_utils
import glob
from typing import Any, List, Optional


def extract_notes() -> List[Any]:
    """Extract all the notes from the data MIDI folder. Place all notes
    into an array

    Returns:
        List[Any]: [description]
    """
    notes = []

    for file in glob.glob("midi_songs/*.mid"):
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
                notes.append('.'.join(str(n) for n in element.normalOrder))
    return notes


def generate_training_data(notes: List[Any], sequence_length: int = 100):
    """Take all extracted notes and convert into training data. 

    Network input/training data: Many sequences of 100 notes
    Network output/label: Note following the 100th, the 101st note

    Args:
        notes (List[Any]): [description]
    """

    # get all pitch names - sort all notes and create a list of them
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []

    # this code is getting lots of sequences of notes, and then getting the next note that follows
    # note that follows is likely the 'label', the proceeding sequence are the inputs
    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        # sequence of notes, like 100 notes
        sequence_in = notes[i:i + sequence_length]
        # note following the input note sequence
        sequence_out = notes[i + sequence_length]

        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    num_samples = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    # Array with num_samples of notes. Each sample has 100 notes and they are flat.
    # i.e. [ [note, ..., 100th note], [note, ..., 100th note], ..., num_samples[note, ..., 100th note]]
    network_input = np.reshape(
        network_input, (num_samples, sequence_length, 1))

    # normalize input
    # need to divide by the
    network_input = network_input / float(num_samples)

    # perform one-hot encoding
    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


notes = extract_notes()
network_input, network_output = generate_training_data(notes)
