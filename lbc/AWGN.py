import matplotlib.pyplot as plt
import numpy as np

class AWGN:
    def __init__(self, myLBC, Eb_No, plot_noise = False):
        """
        :param myLBC: a linear block code object created using the :class:`LBC` class
        :param Eb_No: the design SNR in decibels
        :param plot_noise: a flag to view the modeled noise
        :type myLBC: :class:`LBC`
        :type Eb_No: float
        :type plot_noise: bool
        """

        self.myLBC = myLBC
        self.Es = myLBC.get_normalised_SNR(Eb_No)
        self.No = 1
        self.plot_noise = plot_noise

        tx = self.modulation(self.myLBC.x)
        rx = tx + self.noise(self.myLBC.N)
        self.myLBC.likelihoods = np.array(self.get_likelihoods(rx), dtype=np.float64)

    def LLR(self, y):
        """
        Finds the log-likelihood ratio of a received signal.
        LLR = Pr(y=0)/Pr(y=1).
        :param y: a received signal from a gaussian-distributed channel
        :type y: float
        :return: log-likelihood ratio for the input signal ``y``
        :rtype: float
        """

        return -2 * y * np.sqrt(self.Es) / self.No

    def get_likelihoods(self, y):
        """
        Finds the log-likelihood ratio of an ensemble of received signals using :func:`LLR`.
        :param y: an ensemble of received signals
        :type y: ndarray<float>
        :return: log-likelihood ratios for the input signals ``y``
        :rtype: ndarray<float>
        """

        return [self.LLR(y[i]) for i in range(len(y))]

    def modulation(self, x):
        """
        BPSK modulation for a bit field.
        "1" maps to +sqrt(E_s) and "0" maps to -sqrt(E_s).
        :param x: an ensemble of information to send
        :type x: ndarray<int>
        :return: Modulated signal with the information from ``x``
        :rtype: ndarray<float>
        """

        return 2 * (x - 0.5) * np.sqrt(self.Es)

    def noise(self, N):
        """
        Generate gaussian noise with a specified noise power.
        For a noise power N_o, the double-side noise power is N_o/2.
        :param N: the noise power
        :type N: float
        :return: white gaussian noise vector
        :rtype: ndarray<float>
        """

        # gaussian RNG vector
        s = np.random.normal(0, np.sqrt(self.No / 2), size=N)

        # display RNG values with ideal gaussian pdf
        if self.plot_noise:
            num_bins = 1000
            count, bins, ignored = plt.hist(s, num_bins, density=True)
            plt.plot(bins, 1 / (np.sqrt(np.pi * self.No)) * np.exp(- (bins) ** 2 / self.No),
                        linewidth=2, color='r')
            plt.title('AWGN')
            plt.xlabel('Noise, n')
            plt.ylabel('Density')
            plt.legend(['Theoretical', 'RNG'])
            plt.draw()
        return s

    def show_noise(self):
        """
        Trigger showing the gaussian noise. Only works if ``plot_noise`` is True.
        """
        plt.show()