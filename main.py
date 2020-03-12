from lbc.LBC import LBC
from lbc.Encode import Encode
from lbc.Decode import Decode
from lbc.AWGN import AWGN
from lbc.Plot import Plot
import numpy as np

myLBC = LBC(8, 3, 'hadamard')
Plot(myLBC, 10000, np.arange(0, 10, 1), 'fer')