import face_extractor
import Codebook as cb
import Network as ntw
import numpy as np

from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score

# extractor = face_extractor.FaceExtractor('faces', 'cropped_images')
# extractor.preprocess_images()

codebook_path = 'Codebook_cell16_block8_v3'
codebook = cb.Codebook()
codebook.create_codebook('cropped_images', codebook_path)
codebook.Load_codebook_to_mem(codebook_path)
# v_angle, h_angle = codebook.Estimate_angles_for_img('cropped_images/Person15/0person15115-30-75.jpg')
# print("Estimated orientation: Vertical= {}, Horizontal= {}".format(v_angle, h_angle))

dataset = ntw.prepare_data(codebook_path)
train_set, test_set = ntw.prepare_train_and_test_data(codebook_path, '15')
dataset_v = dataset.values
train_set_v = train_set.values
test_set_v = test_set.values

X = train_set_v[:, :-2]
Y = train_set_v[:, -2:]
iSize = X.shape[1]
print(X.shape)
model = ntw.create_model()
estimator = KerasRegressor(build_fn=ntw.create_model, epochs=70, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, Y)
estimator.model.save('saved_model_v9_block8.h5')
prediction = estimator.predict(X)

ntw.test_model(train_set_v)
# ntw.loss_history_model(model, dataset)

print(prediction)

ex = ntw.get_example('{}\\0person01118-30-30.npy'.format(codebook_path))

ex = np.array([ex])
print(estimator.predict(ex))