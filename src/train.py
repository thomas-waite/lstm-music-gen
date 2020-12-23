from src.prepare_data import PrepareData
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
import numpy as np
from src.model import model_factory


def execute_training():
    data_preparation = PrepareData()
    network_input, network_output = data_preparation.generate_training_data()
    num_pitches = data_preparation.get_num_pitches()
    print('Generated training data')
    lstm_model = model_factory(network_input, num_pitches)
    print('Created lstm model')
    print('Training')
    train(network_input, network_output, lstm_model)
    print('Trained')


def train(network_input: np.ndarray, network_output: np.ndarray, model: Sequential):
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=200,
              batch_size=128, callbacks=callbacks_list)


execute_training()
