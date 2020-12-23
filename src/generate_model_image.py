from keras.utils.vis_utils import plot_model
from src.model import model_factory
import numpy as np

num_samples = 1
dummy_data = np.array([[[1, 100, 1], [1, 100, 1]]])
model = model_factory(dummy_data, num_samples)
plot_model(model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
