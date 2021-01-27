from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

import Codebook as cb
import Network as ntw
import face_extractor


net = ntw.Network('saved_model_v8_block8.h5')
net.get_orientation_from_saved_image('faces\\Person02\\person02113-60+90.jpg')

# net.get_orientation_from_saved_hog('Codebook_cell16_block8_v2\\0person01118-30-30.npy')


# # load network model
# model = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
# model.model = load_model('saved_model_v3.h5')
#
#
#
# # prepare face extractor
# extractor = face_extractor.FaceExtractor('', '')
# faces = extractor.preprocess_image_from_path('faces\\Person02\\person02118-30-30.jpg')
#
# # calculate hog of the image
# codebook = cb.Codebook()
# if faces:
#     # hog = codebook.Calc_descriptors_Cimg(faces[0])
#     hog = codebook.get_descriptors_for_network(faces[0])
#
#     # list_hog = [x[0] for x in hog]
#
#     # print(model.predict(np.array([list_hog])))
#     print(model.predict(hog))
#
# else:
#     print("No face found")

