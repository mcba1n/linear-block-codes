from lbc.LBC import LBC
from lbc.Encode import Encode
from lbc.Decode import Decode
import numpy as np
import itertools

myLBC = LBC(8, 3, 'hadamard')
myLBC.m = np.array([0,1,0])
Encode(myLBC)
myLBC.y = myLBC.x #perfect channel, need AWGN + BPSK
Decode(myLBC, 'ML')
print(myLBC.m_bar)
# Plot class