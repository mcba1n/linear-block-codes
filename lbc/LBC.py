import numpy as np

class LBC:
    def __init__(self, N, K, code):
        self.N = N
        self.K = K
        self.syndromes = None
        self.r = None
        self.x = None
        self.y = None
        self.m_bar = None
        print("Linear block code!")

        if code == 'hadamard':
            self.generator = np.transpose(self.canonical_form(self.hadamard()))
            self.parity_matrix = self.to_parity_matrix(np.transpose(self.generator))
            self.d = int(N / 2)
            self.t = int(np.floor((self.d - 1) / 2))

    def to_parity_matrix(self, A):
        """
        Convert a generator matrix to its dual matrix, namely the parity check matrix.
        :param G: An n x m matrix where m > n.
        :return: [-A^T_{n}, I_{m-n}]
        """
        m = A.shape[1]
        n = A.shape[0]
        P = A[:, np.array(range(n, m), dtype=int)]
        I = np.eye(m - n, dtype=int)
        H = np.mod(np.concatenate([-np.transpose(P), I], axis=1), 2)
        return H

    def canonical_form(self, A):
        """
        Transform A into row canonical form.
        Use this to turn a non-systematic to a systematic counterpart with the same Hamming distance.
        :param A: An n x m matrix where m > n.
        :return: [I|P]
        """
        m = A.shape[1]
        n = A.shape[0]
        j = 0  # pivot row
        i = 0  # current column
        p = []  # column permutation for standard canonical form

        # gauss-jordan algorithm
        while j < n and i < m:
            # find non-zero element in column i, starting in row j
            pivot_ind = -1
            for k in range(j, n):
                if A[k, i] == 1:
                    pivot_ind = k
                    break

            if pivot_ind != -1:  # if the row is not a zero vector
                # swap rows j and pivot_ind in A, but do not change the value of i
                # now A[j,j] will contain the pivot value of 1
                A[[j, pivot_ind]] = A[[pivot_ind, j]]

                # set the rows above and below row j to zero
                for u in range(n):
                    if u == j:
                        continue
                    # If A[u,i] is one, make it zero by adding row j to row u
                    if A[u, i] == 1:
                        A[u, i] = np.mod(A[u, i] + A[j, i], 2)
                # record the identity sub-matrix indices
                p.append(i)
                j = j + 1
            # get next column
            i = i + 1

        # add non-identity columns to end of p
        for i in range(m):
            if i not in list(p):
                p.append(i)
        A = A[:, np.transpose(p)]  # now A is of the form [I|P]
        return A

    def hadamard(self):
        """
        Create a Hadamard generator matrix using the Hadamard matrix -- Sylvester construction method.
        The generator matrix is not unique, since the basis vectors can be chosen differently.
        Hamming distance, d = N/2.
        :return:
        """
        H_0 = np.array([[1, 1], [1, -1]], dtype=int)
        H_n = H_0
        n = int(np.log2(self.N))
        for i in range(n - 1):
            H_n = np.kron(H_0, H_n)
        G = np.mod(H_n + 2*np.ones(self.N, dtype=int), 3)
        return G[-self.K:, :]     # return a Hadamard basis; any K vectors excluding the zero vector

    def get_normalised_SNR(self, design_SNR):
        """
        Normalise E_b/N_o so that the message bits have the same energy for any code rate.
        :param design_SNR: E_b/N_o in decibels
        :type design_SNR: float
        :return: normalised E_b/N_o in linear units
        :rtype: float
        """

        Eb_No_dB = design_SNR
        Eb_No = 10 ** (Eb_No_dB / 10)  # convert dB scale to linear
        Eb_No = Eb_No * (self.K / self.N)  # normalised message signal energy by R=K/N
        return Eb_No