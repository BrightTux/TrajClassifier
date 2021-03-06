import numpy as np
import keras
from processor import process_image
import glob
import os
import csv
from keras.preprocessing.image import img_to_array, load_img


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, shape_h, shape_w, seq_length, batch_size=10, dim=(None,480,640), n_channels=3,
                 n_classes=9, shuffle=True, bool_addnoise=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.image_shape = (shape_h,shape_w, n_channels)
        self.seq_length = seq_length

        self.bool_addnoise = bool_addnoise
        #print ("ID and labels:" , list_IDs, labels)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.data = self.get_data()

        # print out x and y size:
        print("X size: ", len(list_IDs))
        print("y size: ", len(labels))

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open('train_random.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

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

        add_noise = self.bool_addnoise
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[g] for g in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_labels_temp, add_noise)
        #print( "X, y shapes:", X.shape, ",", y.shape)


        return X, y


    def build_image_sequence(self, frames, add_noise):
        """Given a set of frames (filenames), build our sequence."""
        bool_addnoise = add_noise
        return [process_image(frames, self.image_shape, bool_addnoise)]
        #return [process_image(x, self.image_shape) for x in frames]




    def rescale_list(self, input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        # assert len(input_list) >= size
        if (len(input_list) < size):
            #print(input_list)
            output = [x for pair in zip(input_list,input_list) for x in pair]

        else:
            # Get the number to skip between iterations.
            skip = len(input_list) // size
            # Build our new output.
            output = [input_list[i] for i in range(0, len(input_list), skip)]


        # Cut off the last one if needed.
        return output[:size]

    def __data_generation(self, list_IDs_temp, list_labels_temp, add_noise):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        bool_addnoise = add_noise

        #print("list_IDs_temp ", list_IDs_temp)
        #print("list_labels_temp ", list_labels_temp)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            lst = list(self.dim)
            #lst[0] = len(ID)
            #self.dim = tuple(lst)
            #print("X (in i): ", X.shape)
            frames = self.rescale_list(ID, lst[0])

            for j, valID in enumerate(frames):
                # Store sample
                X[i,:] = self.build_image_sequence(valID, bool_addnoise)



        # change to the following line if its the lstmconv2d network
        #y = np.empty((self.batch_size, lst[0]), dtype=int)

        #change to the following line if its the conv3d network
        y = np.empty((self.batch_size, 40), dtype=int)
        for i, ID in enumerate(list_labels_temp):
            # Store class
            y[i] = ID


        return X,  keras.utils.to_categorical(y, num_classes=self.n_classes)
