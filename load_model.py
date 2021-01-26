from keras.models import load_model
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

import Codebook as cb
import Network as ntw
import face_extractor

# load network model
model = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
model.model = load_model('saved_model.h5')

# prepare face extractor
extractor = face_extractor.FaceExtractor('', '')
faces = extractor.preprocess_image_from_path('examples\\3d_model.jpg')

# calculate hog of the image
codebook = cb.Codebook()
if faces:
    hog = codebook.Calc_descriptors_Cimg(faces[0])

print(model.predict(hog))



ex = ntw.get_example('Codebook_cell8x8\\0person01118-30-30.npy')

ex = np.array([ex])
print(ex.shape)
print(model.predict(ex))

