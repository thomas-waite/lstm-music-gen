from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import numpy as np


def model_factory(network_input: np.ndarray, num_samples: int, load_trained=False) -> Sequential:
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(num_samples))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if load_trained:
        model.load_weights('weights/weights.hdf5')
    return model
