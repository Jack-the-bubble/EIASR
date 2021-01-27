import face_extractor
import Codebook as cb
import Network as ntw

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# extractor = face_extractor.FaceExtractor('faces', 'cropped_images')
# extractor.preprocess_images()

test_person = '15'
codebook_path = 'Codebook_test15'
codebook = cb.Codebook()
codebook.create_codebook('cropped_images', codebook_path)
codebook.Load_codebook_to_mem(codebook_path)

train_set, test_set = ntw.prepare_train_and_test_data(codebook_path, test_person)
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
estimator.model.save('saved_model_test15.h5')
prediction = estimator.predict(X)