# https://github.com/farquasar1/ConvLSTM.git

import matplotlib
matplotlib.use('agg')

import os
from os import walk
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
K.set_image_dim_ordering('tf')
#from processor import process_image

from keras.layers import (ConvLSTM2D, BatchNormalization, Conv3D, Conv2D, Flatten, LSTM, Reshape,
TimeDistributed, MaxPooling2D, MaxPooling3D, UpSampling2D, Input, merge, Dense, Activation, Dropout)
from keras.models import Sequential, Model
from keras import losses
import random
from keras.utils import plot_model

from dataclass import DataGenerator

import matplotlib.pyplot as plt
import numpy as np

import cv2
import glob

cwd = os.getcwd()
print(cwd)
data_format='channels_last'

# clear the console.
#os.system('cls' if os.name == 'nt' else 'clear')

target_height, target_width, channel_size = 240, 360, 1


seq_length = 40


model = '3DConv2Dpool-graylarge-withMask'

# Parameters
params = {'shape_h': target_height, 'shape_w': target_width,
          'seq_length': seq_length,
          'dim': (seq_length,target_height,target_width),
          'batch_size': 1,
          'n_classes': 9,
          'n_channels': channel_size,
          'shuffle': True,
          'bool_addnoise': True
          }


# ------------------------------------------------------------------------------------------------------------------------

def class_convLstm_clare(input_shape):
    c = 32
    activation_fn = 'relu'
    kernal_size = (2,2)
    num_classes = 9

    return_sequences_setting = True

    input_img = Input(input_shape, name='input')

    # ------------------- NOT USING -------------------------------------------------------------------------------------------------------
    # # normal conv network to resize the img
    # # input input_shape = batch_size x rows x cols x channel
    # # input input_shape = batch_size x 640 x 480 x 3
    #
    # x = TimeDistributed(Conv2D(128, kernal_size, activation='relu', padding='same',data_format='channels_last'))(input_img)
    # c0 = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(x)
    # # output size = (batch, new_rows, new_cols, filters)
    # # output size = (batch_size, 320, 240, 128)
    # ------------------- NOT USING -------------------------------------------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------------------

    # (samples,time, rows, cols, channels)
    # X: (1, 52, 15, 128, 1) means that you have only one sample that is a sequence of 52 images.

    # start of convlstm network
    print("input_img size: ", input_img)
    x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn, return_sequences=return_sequences_setting)(input_img)
    #x = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    c1 = ConvLSTM2D(nb_filter=c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = Dropout(0.25)(x)

    x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    #x = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    c2 = ConvLSTM2D(nb_filter=2 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = Dropout(0.25)(x)

    x = ConvLSTM2D(nb_filter=3 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    #x = ConvLSTM2D(nb_filter=3 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    c3 = ConvLSTM2D(nb_filter=3 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
    x = Dropout(0.25)(x)

    x = ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    #x = ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)
    c4 = ConvLSTM2D(nb_filter=4 * c, nb_row=3, nb_col=3, border_mode='same', activation=activation_fn,return_sequences=return_sequences_setting)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c4)
    c5 = Dropout(0.25)(x)

    x = TimeDistributed(Flatten(), name='flatten')(c5)
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    c6 = Dropout(0.25)(x)

    #output = TimeDistributed(Dense(num_classes, activation='softmax'), name='output')(c6)

    output = Dense(num_classes, activation='softmax', name='output')(c6)

    model = Model(input_img, output=[output])

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # model.summary()

    return model

# ----------------------------------------------------------------------------------------------------------------------------------


def class_3dconv_clare(input_shape):
    c = 4
    activation_fn = 'relu'
    kernal_size = 2
    num_classes = 9
    dropout_val = 0.15


    return_sequences_setting = True

    input_img = Input(input_shape, name='input')

    # start of 3dconv network
    print("input_img size: ", input_img)
    x = Conv3D(kernel_size=kernal_size ,filters=4*c, padding='same', activation=activation_fn)(input_img)
    x = Conv3D(kernel_size=kernal_size ,filters=4*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=4*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=4*c, padding='same', activation=activation_fn)(x)


    c1 = Conv3D(kernel_size=kernal_size ,filters=4*c, padding='same', activation=activation_fn)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c1)
    x = Dropout(dropout_val)(x)

    x = Conv3D(kernel_size=kernal_size ,filters=8*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=8*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=8*c, padding='same', activation=activation_fn)(x)
    c2 = Conv3D(kernel_size=kernal_size ,filters=8*c, padding='same', activation=activation_fn)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c2)
    x = Dropout(dropout_val)(x)


    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    c3 = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)

    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c3)
    x = Dropout(dropout_val)(x)

    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    x = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)
    c4 = Conv3D(kernel_size=kernal_size ,filters=16*c, padding='same', activation=activation_fn)(x)



    x = TimeDistributed(MaxPooling2D((2, 2), (2, 2)))(c4)
    #x = MaxPooling3D()(x)

    # c5 = Dropout(0.25)(x)
    #
    # x = TimeDistributed(Flatten(), name='flatten')(c5)
    # x = TimeDistributed(Dense(256, activation='relu'))(x)
    # c6 = Dropout(0.25)(x)
    #
    # #output = TimeDistributed(Dense(num_classes, activation='softmax'), name='output')(c6)
    #
    # output = Dense(num_classes, activation='softmax', name='output')(c6)
    #
    # model = Model(input_img, output=[output])
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #
    c5 = Dropout(dropout_val)(x)

    x = TimeDistributed(Flatten(), name='flatten')(c5)


    #x = LSTM(512, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)

    #x = TimeDistributed(Dense(128, activation='relu'))(x)
    c6 = Dropout(dropout_val)(x)



    output = TimeDistributed(Dense(num_classes, activation='softmax'), name='output')(c6)
    #reshape = Reshape((128))(c6)
    #output = Dense(num_classes, activation='softmax', name='output')(c6)




    model = Model(input_img, output=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # model.summary()

    return model
# ------------------------------------------------------------------------------------------------------------------------

  def lrcn(input_shape):
    """Build a CNN into RNN.
    Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
    steering-models/community-models/chauffeur/models.py
    Heavily influenced by VGG-16:
        https://arxiv.org/abs/1409.1556
    Also known as an LRCN:
        https://arxiv.org/pdf/1411.4389.pdf
    """

    num_classes = 8
    input_img = Input(input_shape, name='input')

    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
            activation='relu', padding='same'), input_shape=input_img))
    model.add(TimeDistributed(Conv2D(32, (3,3),
            kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3),
            padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False, dropout=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model



# ------------------------------------------------------------------------------------------------------------------------



def net_summary(net):
    import sys
    from io import StringIO
    # Temporarily redirect stdout, print net summary and then restore stdout
    msg = StringIO()
    out = sys.stdout
    sys.stdout = msg
    net.summary()
    sys.stdout = out
    return msg.getvalue()




def train_model(data_type, image_shape, class_limit, model, batch_size, network=None, nb_epochs=100, train_list=None, test_list=None, jitter=None, output_dir=None):
    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath=os.path.join('data', 'weights', model + '-' + data_type + \
            '.{epoch:03d}-{val_acc:.3f}.hdf5'),
        verbose=1,
        save_best_only=True, monitor='val_acc')

    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model + '-tensorbard.log'), write_images=True)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(monitor='val_acc',
                                  min_delta=0,
                                  patience=200,
                                  verbose=0,
                                  mode='auto')

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))

    fread_train = train_list.readlines()
    fread_test = test_list.readlines()

    x_train_input = []
    y_train_label = []

    y_classes = ["NoActivity", "Traj1", "Traj2", "Traj3", "Traj5", "Traj6", "Traj7", "Traj8", "Traj9"]


    for x in fread_train:
        a,b = x.split(";")
        x_train_input.append(a)
        y_train_label.append(y_classes.index(b.strip()))



    x_test_input = []
    y_test_label = []

    for x in fread_test:
        a,b = x.split(";")
        x_test_input.append(a)
        y_test_label.append(y_classes.index(b.strip()))



    # (samples,time, rows, cols, channels)
    # X: (1, 52, 15, 128, 1) means that you have only one sample that is a sequence of 52 images.
    # ------- Training data: ----------------------------------------------------
    input_dir = "train/"
    sequences = [os.path.join(input_dir, f) for f in x_train_input]

    seq_train_x = []
    f1_train = []
    for index, i in enumerate(sequences):
        for (dirpath, dirnames, filenames) in walk(i):
            for x in filenames:
            # f1_train.extend(filenames)
                f1_train.append(os.path.join(dirpath, x))
        seq_train_x.append(f1_train)
        f1_train = []

    seq_length = [len(f) for f in seq_train_x]

    # testing to see if it works:
    # print(seq_train_x[0])
    # print(y_train_label[0])
    # print(seq_length[0])
    # print(sequences[0])

    # ------- Training data: ----------------------------------------------------

    # ------- Testing data: -----------------------------------------------------
    input_dir = "test/"
    sequences_test = [os.path.join(input_dir, f) for f in x_test_input]

    seq_test_x = []
    f1_test = []
    for index, i in enumerate(sequences_test):
        for (dirpath, dirnames, filenames) in walk(i):
            for x in filenames:
            # f1_test.extend(filenames)
                f1_test.append( os.path.join(dirpath, x))
        seq_test_x.append(f1_test)
        f1_test = []

    seq_length_test = [len(f) for f in seq_test_x]

    # for i, v0 in enumerate(seq_test_x):
    #     for j, value in enumerate(seq_test_x[i]):
    #         seq_test_x[i][j] = os.path.join(input_dir, value)

    # ------- Testing data: -----------------------------------------------------




    # Generators
    training_generator = DataGenerator(seq_train_x, y_train_label, **params)
    validation_generator = DataGenerator(seq_test_x, y_test_label, **params)


    # Setup model and train
    # (samples,time, rows, cols, channels)
    # X: (1, 52, 15, 128, 1) means that you have only one sample that is a sequence of 52 images.

    input_shape = (None, target_height, target_width, channel_size)
    model = network(input_shape)
    print(net_summary(model))

    # print ("(---------------------------- DEBUG ----------------------------)")
    # print("generator: ", generator)
    model.fit_generator(training_generator, epochs=nb_epochs, validation_data=validation_generator, use_multiprocessing=True, workers=5,
                     callbacks=[tb, checkpointer, early_stopper, csv_logger])





# ------------------------------------------------------------------------------------------------------------------------
#                        TESTING FEATURES
# ------------------------------------------------------------------------------------------------------------------------


    # for x in x_test_input:
    #     print(x)


    ## try to display the input
    ## imread = second flag 1 = normal(rgb), 0 = gray
    ## mypath = "train/"+x_test_input[0]
    # f = []
    # for (dirpath, dirnames, filenames) in walk(mypath):
    #     f.extend(filenames)
    #     break
    # print(f)
    #
    # for filename in f:
    #     img = cv2.imread("train/"+x_test_input[0]+ "/" +filename,1)
    #     cv2.imshow('image',img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

if __name__ == '__main__':
    # Load data split

    train_list = patch_path('train_random.csv')
    test_list = patch_path('test_random.csv')

    f_train= open(train_list,"r")
    f_test= open(test_list,"r")


    #model = 'convlstm2d'
    saved_model = None  # None or weights file
    class_limit = 9  # int, can be 1-101 or None

    load_to_memory = False  # pre-load the sequences into memory

    data_type = 'images'
    #height, width, depth = 480, 640, 3 # input image size
    image_shape = (target_height, target_width, 1)

     # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

     # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
        str(timestamp) + '.log'))



    network = class_3dconv_clare  # input_shape = (None, 96, 108, 3)
    #network = class_convLstm_clare  # input_shape = (None, 96, 108, 3)
    #plot_model(network, to_file=model+'_model.png', show_shapes=True)

    batch_size = 1
# def train_model(data_type, image_shape, class_limit, model, batch_size,network=None, nb_epochs=100, train_list=None, test_list=None, jitter=None, output_dir=None):

    train_model(data_type, image_shape, 9, model, batch_size, network, nb_epochs=5000, train_list=f_train, test_list=f_test, output_dir='tmp1')
