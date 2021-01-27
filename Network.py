import warnings

warnings.filterwarnings('ignore')

import os
import numpy
import re
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.initializers import RandomUniform
from keras.callbacks import EarlyStopping

from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import face_extractor
import Codebook as cb


def get_sample_data(template_path):
    filename = template_path.split(os.sep)[-1]
    data = re.findall("\d*\d", filename)
    signs = re.findall('[+ -]', filename)
    vertical = int(data[2]) if signs[0] == '+' else -int(data[2])
    horizontal = int(data[3]) if signs[1] == '+' else -int(data[3])

    return vertical, horizontal


def _get_samples_paths(codebook_path):
    dir_list = os.listdir(codebook_path)
    template_paths = []

    for directory in dir_list:
        directory = os.path.join(codebook_path, directory)
        template_paths.append(directory)

    return template_paths

def prepare_train_and_test_data(codebook_path, test_target):
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    final_train = []
    final_test = []

    list_of_names = _get_samples_paths(codebook_path)
    for sample in list_of_names:
        vertical, horizontal = get_sample_data(sample)
        hogs = numpy.load(sample)
        train_list = []
        test_list = []
        if '0person'+test_target in sample:
            for x in hogs:
                test_list.append(x[0])

            test_list.append(horizontal)
            test_list.append(vertical)
            final_test.append(test_list)

        else:
            for x in hogs:
                train_list.append(x[0])

            train_list.append(horizontal)
            train_list.append(vertical)
            final_train.append(train_list)

    test_set = test_set.append(final_test, ignore_index=True)
    train_set = train_set.append(final_train, ignore_index=True)

    return train_set, test_set



def prepare_data(codebook_path='Codebook'):
    # columns = [*range(4, 1770, 1)]
    # dataset = pd.DataFrame(columns=columns, index=[])
    dataset = pd.DataFrame()
    final = []

    list_of_names = _get_samples_paths(codebook_path)
    for sample in list_of_names:
        vertical, horizontal = get_sample_data(sample)
        hogs = numpy.load(sample)
        list_of_hogs = []
        for x in hogs:
            list_of_hogs.append(x[0])

        list_of_hogs.append(horizontal)
        list_of_hogs.append(vertical)
        final.append(list_of_hogs)

    dataset = dataset.append(final, ignore_index=True)
    print(dataset)
    return dataset


def get_example(example_path):
    vertical, horizontal = get_sample_data(example_path)
    hogs = numpy.load(example_path)
    list_of_hogs = []
    for x in hogs:
        list_of_hogs.append(x[0])

    list_of_hogs.append(horizontal)
    list_of_hogs.append(vertical)
    example = list_of_hogs[:-2]
    return example


def create_model():
    codebook = cb.Codebook()
    extractor = face_extractor.FaceExtractor('', '')
    input_dim = ((extractor.final_size/codebook.cell_size[0])**2)*codebook.nbins
    model = Sequential()
    model.add(Dense(144, input_dim=int(input_dim), kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def loss_history_model(model, dataset):
    early_stopping_monitor = EarlyStopping(patience=10)

    X = dataset[:, :-2]
    Y = dataset[:, -2]
    scalar = MinMaxScaler()
    scalar.fit(X)
    X = scalar.transform(X)
    history = model.fit(X, Y, validation_split=0.3, epochs=200, batch_size=1, verbose=1,
                        callbacks=[early_stopping_monitor])

    scores = model.evaluate(X, Y, verbose=1)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return None


class Network:
    def __init__(self, saved_model_path):
        '''
        constructor function for the class
        '''
        self.model = KerasRegressor(build_fn=create_model, epochs=70,
                                    batch_size=5, verbose=0)

        self.model.model = load_model(saved_model_path)
        self.extractor = face_extractor.FaceExtractor('', '')
        self.codebook = cb.Codebook()


    def get_orientation_from_saved_image(self, image_path):
        faces = self.extractor.preprocess_image_from_path(image_path)
        if faces:
            hog = self.codebook.get_descriptors_for_network(faces[0])

            self._print_formatted_orientation(hog)

    def get_orientation_from_saved_hog(self, hog_path):
        hog = np.array([get_example(hog_path)])
        self._print_formatted_orientation(hog)

    def _print_formatted_orientation(self, hog):
        orientation = self.model.predict(hog)
        # print(orientation)
        vertical = ("up", abs(orientation[0][1])) if orientation[0][1] > 0 else \
            ("down", abs(orientation[0][1]))
        horizontal = ("right", abs(orientation[0][0])) if orientation[0][0] > 0 else \
            ("left", abs(orientation[0][0]))

        print("Head is turned {} by {:.1f} degress and {} by {:.1f} degrees.".format(
            vertical[0], vertical[1], horizontal[0], horizontal[1]
        ))

    def test_model(self, test_array):
        X = test_array[:, :-2]
        Y = test_array[:, -2:]
        passed = 0
        train_size = 0

        for sample, reference in zip(X, Y):
            orientation = self.model.predict(np.array([sample]))
            print("\nEstimated: {:.2f}    {:.2f}".format(orientation[0][0], orientation[0][1]))
            print("Real {}".format(reference))
            horizontal = 15*round(orientation[0][0]/15)
            vertical = 15*round(orientation[0][1]/15)

            train_size = train_size+1
            if horizontal == reference[0] and vertical == reference[1]:
                passed = passed + 1

        print("Accuracy of {:.2f}".format(passed/train_size))



