import numpy as np
import pandas as pd

from tkinter.filedialog import askopenfilename, askdirectory
from tkinter import *

from sklearn.model_selection import KFold, train_test_split

class DataManager():
    def __init__(self, batch_size=20, n_folds=1, use_additional_data=False):
        self.batch_size = batch_size
        self.n_folds = n_folds
        self.reset_pointer()
        self.load_training_data(use_additional_data)
        self.setup_folds(n_folds)

    def load_training_data(self, use_additional_data):
        print('Reading training data...    ', end='\r')
        train_data = np.loadtxt('data/training.csv', delimiter=',', skiprows=1)

        if use_additional_data is True:
            print('Reading additional data...    ', end='\r')
            additional_data = np.loadtxt('data/additional_training.csv', delimiter=',', skiprows=1)
            
            # Set NaN values to zero and concatenate to normal data
            additional_data = np.nan_to_num(additional_data) 
            train_data = np.concatenate((train_data, additional_data), axis=0)

        # Extract features (Assert first row to be indices)
        self.features = train_data[:,1:-1]  
        self.features_size = self.features.shape[1]
        self.n_features = self.features.shape[0]

        # Extract labels and turn into one-hot vectors
        labels = train_data[:,-1]
        self.classes = int(max(labels)+1)
        self.labels = self.labels_to_one_hot(labels, self.classes)

    def load_test_data(self):
        print('Reading testing data...     ', end='\r')
        test_data = np.loadtxt('data/testing.csv', delimiter=',', skiprows=1)
        test_data = test_data[:,1:]
        return test_data

    def write_prediction(self, predictions, name, directory):
        output = pd.DataFrame(predictions)
        output.columns = ['prediction']
        output.index +=1
        output.index.name='ID'  
        output.to_csv(directory + '/' + name, sep=',')

    def labels_to_one_hot(self, labels, classes):
        one_hot = np.zeros((len(labels), classes))

        for i in range(len(labels)): 
            one_hot[i, int(labels[i])] += 1
        return one_hot

    def get_training_size(self):
        return self.features_size, self.classes

    def setup_folds(self, n_folds):
        self.train_index, self.test_index = [], []
        
        if n_folds == 1:
            split = int(self.n_features * 0.8)
            self.train_index.append(np.arange(split))
            self.test_index.append(np.arange(split, self.n_features))
            np.random.shuffle(self.train_index[0])
            np.random.shuffle(self.test_index[0])
        else:
            k_fold = KFold(n_splits=n_folds)
            for train, test in k_fold.split(range(self.n_features)):
                np.random.shuffle(train)
                np.random.shuffle(test)
                self.train_index.append(train)
                self.test_index.append(test)

    def reset_pointer(self):
        self.current_pointer = np.zeros(self.n_folds)

    def get_next_batch(self, fold):
        start_pointer = self.current_pointer[fold] * self.batch_size
        end_pointer = start_pointer + self.batch_size
        self.current_pointer[fold] += 1

        start_pointer, end_pointer = int(start_pointer), int(end_pointer)
        indices = self.train_index[fold][start_pointer:end_pointer]
        
        done = end_pointer >= len(self.train_index[fold])

        if done: 
            self.current_pointer[fold] = 0
            np.random.shuffle(self.train_index[fold])

        return done, self.features[indices,:], self.labels[indices,:]

    def get_test(self, fold):
        indices = self.test_index[fold]
        return self.features[indices], self.labels[indices]

    

        
