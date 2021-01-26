from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

import Network as ntw



model = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
model.model = load_model('saved_model.h5')

ex = ntw.get_example('Codebook_cell8x8\\0person01118-30-30.npy')

ex = np.array([ex])
print(model.predict(ex))

