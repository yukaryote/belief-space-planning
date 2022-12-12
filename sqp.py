import numpy as np

from pydrake.all import (
    MathematicalProgram, Solve, eq
)

from pydrake.symbolic import Variable, arctan, sin, pow
from utils.histogram_filter import HistogramFilter
from scipy.special import kl_div

from utils.add_bodies import BOX_SIZE, GAP


def h(x):
    return 2 * 0.1 / np.pi * arctan(sin(2 * np.pi * x * 2) / 0.01)


def H(x):
    return h(x).Differentiate(x)


class SolveSQP:
    def __init__(self, x_g, T=100, alpha=0.0085, N=100, noise_std=0.1, r=0.0025):
        self.N = N
        self.T = T
        self.alpha = alpha
        self.x_g = x_g
        self.r = r

        # create the h(x) function
        self.lower_bound = -BOX_SIZE[1] - GAP / 2
        self.upper_bound = BOX_SIZE[1] + GAP / 2

        # y_range is the list of bin edges
        self.y_range = np.linspace(self.lower_bound, self.upper_bound, self.N, endpoint=False)
        self.h = np.zeros((self.N,))
        for i in range(self.N):
            if -GAP / 2 < self.y_range[i] < GAP / 2:
                self.h[i] = 0.25
            else:
                self.h[i] = 0.05

        self.H = np.zeros_like(self.h)
        prev = self.h[0]
        # h is a square wave, so its derivative is two impulses of opposite sign
        for i in range(len(self.h)):
            if prev < self.h[i]:
                self.H[i] = 1
            elif prev > self.h[i]:
                self.H[i] = -1
        self.Q = np.random.normal(loc=0, scale=noise_std)
        self.histogram_filter = HistogramFilter(self.N, self.h, noise_std, self.lower_bound, self.upper_bound)

        # initialize state at a random location along the y-axis
        self.x = np.random.uniform(self.lower_bound, self.upper_bound)
        self.time_step = 0.02

    def dirtran(self, x_samples, use_goal=False):
        # Discrete-time approximation of the double integrator.
        K = len(x_samples)

        prog = MathematicalProgram()

        # Create decision variables

        w = np.empty((K, self.T, 1), dtype=Variable)
        u = np.empty((self.T - 1), dtype=Variable)
        # x is k-dim positions in the y-axis from our k samples
        x = np.empty((K, self.T, 1), dtype=Variable)
        for t in range(self.T):
            for k in range(K):
                w[k, t] = prog.NewContinuousVariables(1, 'w' + str(k) + str(t))
                x[k, t] = prog.NewContinuousVariables(1, 'x' + str(k) + str(t))
            if t < self.T - 1:
                u[t] = prog.NewContinuousVariables(1, 'u' + str(t))

        # Add costs and constraints
        x0 = np.vstack([x_samples] * self.T).T
        # J = np.mean([self.calc_weights(x_samples[0], k, x_samples, x, self.T) ** 2 for k in range(K)])
        J = np.mean([w[k, self.T - 1].dot(w[k, self.T - 1]) for k in range(K)])
        cost2 = self.alpha * u.dot(u)
        prog.AddCost(J)
        prog.AddConstraint(eq(x0, x[:, :, 0]))
        for t in range(self.T - 1):
            for k in range(K):
                prog.AddConstraint(eq(x[k, t + 1], self.f(x[k, t], u[t])))
                prog.AddConstraint(eq(w[k, t + 1], w[k, t] * pow(np.e, self.phi(x[k, t][0], x[0, t][0]))))
                prog.AddBoundingBoxConstraint(self.lower_bound, self.upper_bound, x[k, t][0])
        if use_goal:
            xf = self.x_g
            prog.AddBoundingBoxConstraint(xf, xf, x[0, self.T - 1])

        result = Solve(prog)
        assert (result.is_success()), "Optimization failed"

        u_sol = result.GetSolution(u)
        return u_sol

    def calc_weights(self, x_1, i, x_samples, x, T):
        weight = 1
        for t in range(1, T):
            weight *= pow(np.e, self.phi(x[i, t][0], x[0, t][0]))
            # print("weiht", weight)
        return weight

    def phi(self, x, y):
        """
        Weighting function
        x and y are two continuous states. We have to bin them.
        """
        # print(x, y)
        return 1 / 2 * (h(x) - h(y)) * 1 / (2 * self.Q + H(x) * H(x) + H(y) * H(y)) * (h(x) - h(y))

    def f(self, x, u):
        """
        Returns next state if we are in state x and take action u
        """
        # print("u", u)
        state = x + u[0] * self.time_step
        # print("f state", state)
        return x + u * self.time_step

    def F(self, x, u):
        """
        Returns next state if we are in state x and take actions u (a T-dim vector of actions)
        """
        state = x
        # print("prev state", state)
        for i in range(len(u)):
            state += self.f(state, u[i])
        # print("after state", state[0])
        return state[0]

    def theta_cap(self, belief_state):
        """
        Probability that we are in a ball of radius r around x_g
        """
        cdf = np.cumsum(belief_state)
        # calculate which bin x_g +/- r is in.
        print(self.x_g - self.r)
        lower_bound = np.digitize(self.x_g - self.r, self.y_range)
        upper_bound = np.digitize(self.x_g + self.r, self.y_range)
        print(lower_bound, upper_bound)
        return cdf[upper_bound] - cdf[lower_bound]

    def J(self, x_samples, u, t):
        return np.mean([self.calc_weights(x_samples[0], k, x_samples, u, t) ** 2 for k in range(len(x_samples))])

    def create_plan(self, x_samples, omega=0.5):
        belief_state = self.histogram_filter.p[:]
        u = self.dirtran(x_samples, use_goal=True)
        belief_states = np.ndarray(shape=(belief_state.shape[0], self.T))
        belief_states[0] = belief_state
        for t in range(self.T - 1):
            belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        if self.theta_cap(belief_state) <= omega:
            u = self.dirtran(x_samples)
            belief_states[0] = belief_state
            for t in range(self.T - 1):
                belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        return belief_states, u

    def re_plan(self, omega=0.5, K=15, thresh=0.25, kl_thresh=0.5, rho=0.5):
        belief_state = self.histogram_filter.p[:]
        while self.theta_cap(belief_state) <= omega:

            x_samples = [np.argmax(self.histogram_filter.p)]
            k = 1
            while k < K:
                sample = np.random.choice(len(belief_state), 1, p=belief_state)
                if sample[0] > thresh:
                    x_samples.append(sample[0])
                    k += 1

            belief_states, u = self.create_plan(x_samples, omega)
            belief_states[0] = belief_state
            t_break = 0

            for t in range(self.T - 1):
                u_t = u[t]
                # take action u_t, observe z_next
                self.x = self.f(self.x, u_t)
                # choose correct bin index for state x
                z_next = self.h[int(self.x - self.lower_bound)]
                belief_state = self.histogram_filter.update(u_t, z_next)

                if kl_div(belief_state, belief_state) > kl_thresh and self.J(x_samples, u, t) < 1 - rho:
                    t_break = t
                    break
                t_break += 1
            belief_state = belief_states[t_break]


if __name__ == "__main__":
    sqp = SolveSQP(0.)
    sqp.re_plan()
