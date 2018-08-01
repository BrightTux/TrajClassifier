import numpy as np
import keras
from processor import process_image
import glob
import os
from keras.preprocessing.image import img_to_array, load_img


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=10, dim=(None,480,640), n_channels=3,
                 n_classes=9, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.image_shape = (480,640, n_channels)
        #print ("ID and labels:" , list_IDs, labels)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[g] for g in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp)
        print( "X, y shapes:", X.shape, ",", y.shape)

        return X, y

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(frames, self.image_shape)]
        #return [process_image(x, self.image_shape) for x in frames]




    def __data_generation(self, list_IDs_temp, list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        #print("list_IDs_temp ", list_IDs_temp)
        #print("list_labels_temp ", list_labels_temp)


        for i, ID in enumerate(list_labels_temp):
            # Store class
            y[i] = ID

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            lst = list(self.dim)
            lst[0] = len(ID)
            self.dim = tuple(lst)
            #print("X (in i): ", X.shape)
            for j, valID in enumerate(ID):
                # Store sample
                #print("X (in j): ", X.shape)
                X[i,:] = self.build_image_sequence(valID)
                #X.append(self.build_image_sequence(valID))


        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
