import numpy as np
import scipy.stats as stats


class HistogramFilter:
    """
    Histogram filter with N bins. Call update() at each time step.
    p (N, 1) is initially uniform across all states.
    move function takes in robot arm motion (y-axis value), returns p_bar for all bins
    sense function takes in the beam measurement (cm), returns p for all bins (final probability of some bin being true)
    """
    def __init__(self, N, field, noise_std, motion_noise_std, lower_bound, upper_bound, time_step=0.02):
        # N is the number of pins
        self.N = N
        # p is the probability distribution over the N bins
        self.p = np.ones((N,)) / N
        self.field = field
        self.bin_size = (upper_bound - lower_bound) / (1.0 * N)
        self.noise_std = noise_std
        self.time_step = time_step
        self.process_noise = motion_noise_std

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
        # if measurement > 0.05:
        #     print("calc p obs", x_idx, measurement, p_max - p_min)
        return p_max - p_min

    def move(self, motion):
        """
        Updates p according to robot end effector movement.
        motion is the velocity (m) in the y-axis. We assume no motion noise.
        Returns p_bar
        """
        q = np.zeros_like(self.p)
        noise = np.random.normal(0, 0.01)
        for x in range(len(self.p)):
            q[x] = self.p[int(x + (motion + noise) * self.time_step)]

        q = q / np.sum(q)
        return q

    def sense(self, measurement):
        """
        Updates p according to measurement (the laser beam reading at time t).
        p(measurement | state = x_t) is given by function calc_p_obs
        """
        q = np.zeros_like(self.p)
        for x in range(len(self.p)):
            pZGivenX = self.calc_p_obs(x, measurement)
            q[x] = pZGivenX * self.p[x]

        q = q / np.sum(q)
        return q

    def update(self, motion, measurement):
        self.p = self.move(motion)
        self.p = self.sense(measurement)
        return self.p

    def reset(self):
        self.p = np.ones((self.N,)) / self.N
