import numpy as np

class Encode:
    def __init__(self, myLBC):
        myLBC.x = np.mod(np.matmul(myLBC.generator, np.transpose(myLBC.m)), 2)