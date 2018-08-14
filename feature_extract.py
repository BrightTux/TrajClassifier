"""
This script generates extracted features for each video, which other
models make use of.
You can change you sequence length and limit to a set number of classes
below.
class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np

import os
from os import walk
# from data import DataSet
from dataclass import DataGenerator
from extractor import Extractor
from tqdm import tqdm



def patch_path(path):
    return os.path.join(os.path.dirname(__file__), path)

# Set defaults.
seq_length = 40
class_limit = None  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.

target_height, target_width, channel_size = 299, 299, 3


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


train_list = patch_path('train_random.csv')
test_list = patch_path('test_random.csv')

f_train= open(train_list,"r")
f_test= open(test_list,"r")


fread_train = f_train.readlines()


x_train_input = []
y_train_label = []

y_classes = ["NoActivity", "Traj1", "Traj2", "Traj3", "Traj5", "Traj6", "Traj7", "Traj8", "Traj9"]


for x in fread_train:
    a,b = x.split(";")
    x_train_input.append(a)
    y_train_label.append(y_classes.index(b.strip()))




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





data = DataGenerator(seq_train_x, y_train_label, **params)
# data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data))

# sequences = [os.path.join(input_dir, f) for f in x_train_input]

#for video in data.data:
for index, i in enumerate(sequences):

    # Get the path to the sequence for this video.
    #path = os.path.join('train', video[0])  # numpy will auto-append .npy

    # Check if we already have it.
    if os.path.isfile(path + '.npy'):
        pbar.update(1)
        continue

    # Get the frames for this video.
    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)

    # Now loop through and extract features to build the sequence.
    sequence = []
    for image in frames:
        features = model.extract(image)
        sequence.append(features)

    # Save the sequence.
    np.save(path, sequence)

    pbar.update(1)

pbar.close()
