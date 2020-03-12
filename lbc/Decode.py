import numpy as np
import itertools

class Decode:
    def __init__(self, myLBC, decoder_type):
        self.myLBC = myLBC
        if decoder_type == 'ML':    # ML decoder using syndromes
            if myLBC.syndromes == None:  # create hashmap for the possible syndromes
                self.find_syndromes()
            self.ML_decode()

    def ML_decode(self):
        # use syndromes to choose the min distance codeword
        y = np.array(self.myLBC.likelihoods > 0, dtype=int)  # hard decision
        syndrome_vec = np.mod(np.matmul(self.myLBC.parity_matrix, y), 2)
        pred_error_vec = self.myLBC.syndromes[syndrome_vec.tostring()]
        self.myLBC.r = np.mod(y + pred_error_vec, 2)
        self.myLBC.m_bar = self.myLBC.r[:self.myLBC.K]

    def find_syndromes(self):
        enum_bin_vecs = list(itertools.product([0, 1], repeat=self.myLBC.N))    # generate all binary vectors
        self.myLBC.syndromes = {}
        for i in range(len(enum_bin_vecs)):
            error_vec = np.array(enum_bin_vecs[i], dtype=int)
            syndrome_vec = np.mod(np.matmul(self.myLBC.parity_matrix, np.transpose(error_vec)), 2)
            self.myLBC.syndromes[syndrome_vec.tostring()] = error_vec