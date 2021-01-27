from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

import Codebook as cb
import Network as ntw
import face_extractor


test_person = '15'
net = ntw.Network('saved_model_test15.h5')


codebook_path = 'Codebook_test15'
train_set, test_set = ntw.prepare_train_and_test_data(codebook_path, test_person)
train_set_v = train_set.values
test_set_v = test_set.values

net.test_model(test_set_v)

# net.get_orientation_from_saved_image('faces\\Person02\\person02113-60+90.jpg')
# net.get_orientation_from_saved_hog('Codebook_cell16_block8_v2\\0person01118-30-30.npy')


