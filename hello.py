# hello.py

import numpy as np
from numpy  import array
np.set_printoptions(precision=2)

batch_size = 2
dim = (1, 2,2)
n_channels = 1

X = np.zeros((batch_size, *dim, n_channels))
insert_this = np.empty((*dim, n_channels))
print("shape befre ",insert_this.shape)

print(insert_this)

insert_this = array([[[[1][2]][[3][4]]]])
print("shape after ",insert_this.shape)
print ("X shape: ", X.shape)
print ("X: ")
print (X)


print ()
print()
print(insert_this)
print(insert_this.shape)

X[0:] = insert_this


print(X)
