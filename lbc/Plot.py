from lbc.Encode import Encode
from lbc.Decode import Decode
from lbc.AWGN import AWGN
import numpy as np
import matplotlib.pyplot as plt

class Plot:
    def __init__(self, myLBC, iterations, Eb_No_vals, plot_type):
        self.myLBC = myLBC
        self.Eb_No_vals = Eb_No_vals
        self.iterations = iterations
        self.plot_type = plot_type

        self.MC_simulation()
        self.plot_simulation()

    def MC_simulation(self):
        np.random.seed(1764)    # initialise seed
        FER_records = np.zeros(self.Eb_No_vals.shape[0], dtype=np.float)
        BER_records = np.zeros(self.Eb_No_vals.shape[0], dtype=np.float)

        for j in range(self.Eb_No_vals.shape[0]):
            frame_error_count = 0
            bit_error_count = 0

            for i in range(self.iterations):
                # simulate a random message
                self.myLBC.m = np.random.randint(2, size=self.myLBC.K)
                Encode(self.myLBC)
                AWGN(self.myLBC, self.Eb_No_vals[j])
                Decode(self.myLBC, 'ML')

                # count the errors
                error_vec = self.myLBC.m ^ self.myLBC.m_bar
                num_errors = sum(error_vec)
                frame_error_count += (num_errors > 1)
                bit_error_count += num_errors

            FER_records[j] = frame_error_count / self.iterations
            BER_records[j] = bit_error_count / (self.myLBC.K * self.iterations)
            print('Simulation progress: (', j+1, '/', self.Eb_No_vals.shape[0], ')')

        self.myLBC.FER_simulation = FER_records
        self.myLBC.BER_simulation = BER_records

    def plot_simulation(self):
        if self.plot_type == 'fer':
            data = self.myLBC.FER_simulation
        elif self.plot_type == 'ber':
            data = self.myLBC.BER_simulation
        else:
            return

        fig = plt.figure()
        new_plot = fig.add_subplot(111)
        new_plot.cla()
        new_plot.plot(self.Eb_No_vals, data, '-o', markersize=6, linewidth=3)
        new_plot.set_title(str(self.myLBC))
        new_plot.set_ylabel(self.plot_type.upper())
        new_plot.set_xlabel("$E_b/N_o$ (dB)")
        new_plot.grid(linestyle='-')
        new_plot.set_yscale('log')
        fig.show()