# hello.py

import numpy as np
from numpy  import array
np.set_printoptions(precision=2)


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    assert len(input_list) >= size

    # Get the number to skip between iterations.
    skip = len(input_list) // size

    # Build our new output.
    output = [input_list[i] for i in range(0, len(input_list), skip)]

    # Cut off the last one if needed.
    return output[:size]


if __name__ == '__main__':

    print("hello world")
    size = 4

    input_list = [1,2,3,4,5]

    input_list2 = ['train/931\\0001.jpg', 'train/931\\0002.jpg', 'train/931\\0003.jpg', 'train/931\\0004.jpg', 'train/931\\0005.jpg', 'train/931\\0006.jpg', 'train/931\\0007.jpg', 'train/931\\0008.jpg', 'train/931\\0009.jpg', 'train/931\\0010.jpg', 'train/931\\0011.jpg', 'train/931\\0012.jpg', 'train/931\\0013.jpg', 'train/931\\0014.jpg',
    'train/931\\0015.jpg', 'train/931\\0016.jpg', 'train/931\\0017.jpg',
 'train/931\\0018.jpg', 'train/931\\0019.jpg', 'train/931\\0020.jpg', 'train/931\\0021.jpg', 'train/931\\0022.jpg', 'train/931\\0023.jpg', 'train/931\\0024.jpg', 'train/931\\0025.jpg', 'train/931\\0026.jpg', 'train/931\\0027.jpg', 'train/931\\0028.jpg', 'train/931\\0029.jpg', 'train/931\\0030.jpg', 'train/931\\0031.jpg', 'train/931\\0032.jpg', 'train/931\\0033.jpg', 'train/931\\0034.jpg',
 'train/931\\0035.jpg', 'train/931\\0036.jpg', 'train/931\\0037.jpg', 'train/931\\0038.jpg', 'train/931\\0039.jpg', 'train/931\\0040.jpg', 'train/931\\0041.jpg', 'train/931\\0042.jpg', 'train/931\\0043.jpg',
'train/931\\0044.jpg', 'train/931\\0045.jpg', 'train/931\\0046.jpg', 'train/931\\0047.jpg', 'train/931\\0048.jpg', 'train/931\\0049.jpg', 'train/931\\0050.jpg', 'train/931\\0051.jpg', 'train/931\\0052.jpg',
'train/931\\0053.jpg', 'train/931\\0054.jpg', 'train/931\\0055.jpg', 'train/931\\0056.jpg', 'train/931\\0057.jpg', 'train/931\\0058.jpg', 'train/931\\0059.jpg', 'train/931\\0060.jpg', 'train/931\\0061.jpg',
'train/931\\0062.jpg', 'train/931\\0063.jpg', 'train/931\\0064.jpg', 'train/931\\0065.jpg', 'train/931\\0066.jpg', 'train/931\\0067.jpg', 'train/931\\0068.jpg', 'train/931\\0069.jpg', 'train/931\\0070.jpg',
'train/931\\0071.jpg', 'train/931\\0072.jpg', 'train/931\\0073.jpg', 'train/931\\0074.jpg', 'train/931\\0075.jpg', 'train/931\\0076.jpg', 'train/931\\0077.jpg', 'train/931\\0078.jpg',
 'train/931\\0079.jpg', 'train/931\\0080.jpg', 'train/931\\0081.jpg', 'train/931\\0082.jpg', 'train/931\\0083.jpg', 'train/931\\0084.jpg', 'train/931\\0085.jpg', 'train/931\\0086.jpg', 'train/931\\0087.jpg',
 'train/931\\0088.jpg', 'train/931\\0089.jpg', 'train/931\\0090.jpg', 'train/931\\0091.jpg', 'train/931\\0092.jpg', 'train/931\\0093.jpg', 'train/931\\0094.jpg', 'train/931\\0095.jpg', 'train/931\\0096.jpg',
 'train/931\\0097.jpg', 'train/931\\0098.jpg', 'train/931\\0099.jpg', 'train/931\\0100.jpg', 'train/931\\0101.jpg', 'train/931\\0102.jpg', 'train/931\\0103.jpg', 'train/931\\0104.jpg'
, 'train/931\\0105.jpg', 'train/931\\0106.jpg', 'train/931\\0107.jpg', 'train/931\\0108.jpg']

    input_list2 = ['train/931\\0001.jpg', 'train/931\\0002.jpg', 'train/931\\0003.jpg']

    print(rescale_list(input_list, size))


    output = [x for pair in zip(input_list,input_list) for x in pair]

    print(output)
