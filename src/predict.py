from src.model import model_factory
from src.prepare_data import PrepareData


def generate():
    data_preparation = PrepareData()
    network_input, network_output = data_preparation.generate_training_data()
    num_pitches = data_preparation.get_num_pitches()

    load_trained = True
    model = model_factory(network_input, num_pitches, load_trained)
