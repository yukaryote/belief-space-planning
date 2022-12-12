import numpy as np
import scipy.stats as stats


class HistogramFilter:
    """
    Histogram filter with N bins. Call update() at each time step.
    p (N, 1) is initially uniform across all states.
    move function takes in robot arm motion (y-axis value), returns p_bar for all bins
    sense function takes in the beam measurement (cm), returns p for all bins (final probability of some bin being true)
    """
    def __init__(self, N, field, noise_std, lower_bound, upper_bound):
        # N is the number of pins
        self.N = N
        # p is the probability distribution over the N bins
        self.p = np.ones((N,)) / N
        self.field = field
        self.bin_size = (upper_bound - lower_bound) / (1.0 * N)
        self.noise_std = noise_std

    def calc_p_obs(self, x_idx, measurement):
        """
        field is an N-d array of distances, where N is the number of bins in the histogram filter.
        measurement is the measurement collected from the laser sensor (cm) with Gaussian noise.
        Returns p(measurement | state = x).
        """
        x_min = self.field[x_idx] - self.bin_size / 2
        x_max = self.field[x_idx] + self.bin_size / 2
        p_min = stats.norm.cdf(x_min, measurement, scale=self.noise_std)
        p_max = stats.norm.cdf(x_max, measurement, scale=self.noise_std)

        return p_max - p_min

    def move(self, motion):
        """
        Updates p according to robot end effector movement.
        motion is the amount (cm) in the y-axis that the robot moved. We assume no motion noise.
        Returns p_bar
        """
        q = np.zeros_like(self.p)
        for x in range(len(self.p)):
            # we're assuming no motion, noise, so we don't sum over all possible states, just the previous one.
            s = self.p[x + motion] * self.p[x]
            q[x] = s

        q = q / np.linalg.norm(q)
        return q

    def sense(self, measurement):
        """
        Updates p according to measurement (the laser beam reading at time t).
        p(measurement | state = x_t) is given by function calc_p_obs
        """
        q = np.zeros_like(self.p)
        for x in range(len(self.p)):
            pZGivenX = self.calc_p_obs(x, measurement)
            q[x] = self.p[x] * pZGivenX

        q = q / np.linalg.norm(q)
        return q

    def update(self, motion, measurement):
        self.p = self.move(motion)
        self.p = self.sense(measurement)
        return self.p
