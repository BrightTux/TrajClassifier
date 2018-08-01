# % randomly order test/train list
f= open("test_random.csv","w+")
import random

with open('test.csv') as f_input:
    lines = f_input.read().splitlines()
    random.shuffle(lines)
    # print ('\n'.join(lines))
    f.write('\n'.join(lines))
