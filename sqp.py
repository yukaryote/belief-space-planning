import numpy as np

from pydrake.all import (
    MathematicalProgram, Solve, eq
)

from pydrake.symbolic import Variable, arctan, sin, pow
from utils.histogram_filter import HistogramFilter
from scipy.special import kl_div

import matplotlib.pyplot as plt

from utils.add_bodies import BOX_SIZE, GAP


def h(x):
    c = 0.5
    a = -0.025
    b = 0.025
    A = 0.02
    L = 1000
    z = 3. / 4 + (b - a) / (2 * c)
    h = 0.05 + -A * c / (a - b) * \
           1. / (1 + pow(np.e, L * (np.sin(2 * np.pi * ((x - a) / c + z)) - np.sin(2 * np.pi * z))))
    return h


def H(x):
    # print(h(x).Differentiate(x))
    return h(x).Differentiate(x)


class SolveSQP:
    def __init__(self, x_g, x_start, T=10, alpha=0.0085, N=10, noise_std=0.025, r=0.025):
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
        x = np.linspace(self.lower_bound, self.upper_bound, num=self.N, endpoint=False)
        # self.h = 0.1 * np.sin(4 * np.pi * x + np.pi/(4*0.1)) + 0.15
        self.h = np.zeros((self.N,))
        for i in range(self.N):
            if -GAP / 2 < self.y_range[i] < GAP / 2:
                self.h[i] = 0.25
            else:
                self.h[i] = 0.05

        self.Q = np.random.normal(loc=0, scale=noise_std)
        self.histogram_filter = HistogramFilter(self.N, self.h, noise_std, 0.01, self.lower_bound, self.upper_bound)

        # initialize state at a random location along the y-axis
        # self.x = np.random.uniform(self.lower_bound, self.upper_bound)
        self.x = x_start
        self.time_step = 0.02

        self.p_graph = []
        self.x_graph = []

    def dirtran(self, x_samples, use_goal=False):
        # Discrete-time approximation of the double integrator.
        K = len(x_samples)

        prog = MathematicalProgram()

        # Create decision variables

        # x is k-dim positions in the y-axis from our k samples
        x = np.empty((K, self.T), dtype=Variable)
        w = np.empty((K, self.T), dtype=Variable)
        u = np.empty((self.T - 1, 1), dtype=Variable)
        for t in range(self.T):
            for k in range(K):
                x[k, t] = prog.NewContinuousVariables(1, 'x' + str(k) + "_" + str(t))
                w[k, t] = prog.NewContinuousVariables(1, 'w' + str(k) + "_" + str(t))
            if t < self.T - 1:
                u[t] = prog.NewContinuousVariables(1, 'u' + str(t))

        # Add costs and constraints
        J = 1. / K * np.sum(w[:, self.T - 1].T @ w[:, self.T - 1])
        action_cost = self.alpha * (u.T @ u)[0][0]
        prog.AddCost(J + action_cost)
        for t in range(self.T - 1):
            # prog.AddBoundingBoxConstraint(-6, 6, u[t])
            for k in range(K):
                prog.AddConstraint(eq(x[k, t + 1], self.f(x[k, t], u[t])))
                prog.AddConstraint(eq(w[k, t + 1], w[k, t] * pow(np.e, -self.phi(x[k, t][0], x[0, t][0]))))
                prog.AddConstraint(eq(w[k, 0], 1))
                prog.AddConstraint(eq(x[k, 0], x_samples[k]))
                prog.AddBoundingBoxConstraint(self.lower_bound, self.upper_bound, x[k, t][0])
        if use_goal:
            prog.AddConstraint(eq(x[0, self.T - 1], self.x_g))

        init_guess = np.zeros((2 * K * self.T + self.T - 1,))
        init_guess[2 * K * self.T:] = np.random.uniform(-0.5, 0.5, self.T - 1)
        result = Solve(prog, initial_guess=init_guess)
        assert (result.is_success()), "Optimization failed"

        u_sol = result.GetSolution(u)
        return u_sol

    def calc_weights(self, x_1, i, x_samples, x, T):
        weight = 1
        for t in range(1, T):
            weight *= pow(np.e, self.phi(x[i, t][0], x[0, t][0]))
        return weight

    def phi(self, x, y):
        """
        Weighting function
        x and y are two continuous states. We have to bin them.
        """
        return 1 / 2 * (h(x) - h(y)) * 1 / (2 * self.Q + H(x) * H(x) + H(y) * H(y)) * (h(x) - h(y))

    def f(self, x, u):
        """
        Returns next state if we are in state x and take action u
        """
        return x + u * self.time_step

    def F(self, x, u):
        """
        Returns next state if we are in state x and take actions u (a T-dim vector of actions)
        """
        state = x
        for i in range(len(u)):
            state += self.f(state, u[i])
        return state[0]

    def theta_cap(self, belief_state):
        """
        Probability that we are in a ball of radius r around x_g
        """
        cdf = np.cumsum(belief_state)
        # calculate which bin x_g +/- r is in.
        lower_bound = np.digitize(self.x_g - self.r, self.y_range) - 1
        upper_bound = np.digitize(self.x_g + self.r, self.y_range) - 1
        res = cdf[upper_bound] - cdf[lower_bound]
        return res

    def J(self, x_samples, u, t):
        return np.mean([self.calc_weights(x_samples[0], k, x_samples, u, t) ** 2 for k in range(len(x_samples))])

    def create_plan(self, x_samples, omega=0.5):
        belief_state = self.histogram_filter.p[:]
        """
        plt.title("Initial belief state")
        plt.xlabel("end effector position along y axis")
        plt.ylabel("probability")
        plt.plot(self.y_range, belief_state)
        plt.show()
        """
        u = self.dirtran(x_samples, use_goal=True)
        belief_states = np.ndarray(shape=(self.T, belief_state.shape[0],))
        belief_states[0] = belief_state
        x_T = np.zeros((self.T,))
        x_T[0] = x_samples[0]
        for t in range(self.T - 1):
            x_T[t + 1] = self.f(x_T[t], u[t])
            belief_states[t + 1] = self.histogram_filter.update(u[t], self.h[
                max(np.digitize(x_T[t + 1], self.y_range) - 1, 0)])
            """
            plt.title("Belief state after measurement")
            plt.xlabel("end effector position along y axis")
            plt.ylabel("probability")
            plt.plot(self.y_range, belief_states[t+1])
            plt.show()
            """
        if self.theta_cap(belief_state) <= omega:
            u = self.dirtran(x_samples)
            belief_states[0] = belief_state
            self.histogram_filter.reset()
            for t in range(self.T - 1):
                belief_states[t + 1] = self.histogram_filter.update(u[t], self.h[
                    max(np.digitize(x_T[t + 1], self.y_range) - 1, 0)])
                x_T[t + 1] = self.f(x_T[t], u[t])
        return belief_states, u

    def re_plan(self, omega=0.5, K=15, thresh=0.25, kl_thresh=0.5, rho=0.5):
        belief_state = self.histogram_filter.p[:]
        while self.theta_cap(belief_state) <= omega:
            x_samples = [self.y_range[np.argmax(self.histogram_filter.p)]]
            k = 1
            while k < K:
                sample = np.random.choice(len(belief_state), 1, p=belief_state)
                if sample[0] > thresh:
                    x_samples.append(self.y_range[sample[0]])
                    k += 1

            belief_states, u = self.create_plan(x_samples, omega)
            belief_states[0] = belief_state
            t_break = 0

            for t in range(self.T - 1):
                u_t = u[t]
                # take action u_t, observe z_next
                self.x = self.f(self.x, u_t)
                self.x_graph.append(self.x)
                # choose correct bin index for state x
                z_next = self.h[max(np.digitize(self.x, self.y_range) - 1, 0)]
                belief_state = self.histogram_filter.update(u_t, z_next)
                self.p_graph.append(belief_state)

                if sum(kl_div(belief_state, belief_state)) > kl_thresh and self.J(x_samples, u, t) < 1 - rho:
                    t_break = t
                    break
                t_break += 1
            belief_state = belief_states[t_break]
        return self.x_graph


if __name__ == "__main__":
    sqp = SolveSQP(0., -0.1)
    sqp.re_plan()

    lower_bound = -BOX_SIZE[1] - GAP / 2
    upper_bound = BOX_SIZE[1] + GAP / 2
    n = 10
    trials = 100

    # y_range is the list of bin edges
    y_range = np.linspace(lower_bound, upper_bound, n, endpoint=False)

    print(sqp.x)
    plt.title("End effector trajectory (horizon T=10)")
    plt.xlabel("time")
    plt.ylabel("end effector position along y axis")
    plt.plot(np.linspace(0, len(sqp.x_graph), len(sqp.x_graph), endpoint=False), sqp.x_graph)
    plt.show()

    plt.title("Belief states over time (horizon T=10)")
    plt.ylabel("end effector position along y axis")
    plt.xlabel("probability")
    alpha_step = 1./len(sqp.p_graph)
    for i in range(len(sqp.p_graph)):
        plt.plot(np.linspace(0, len(sqp.y_range), len(sqp.y_range), endpoint=False), sqp.p_graph[i], color="blue", alpha = alpha_step * i)
    plt.show()
    """
    success = np.zeros_like(y_range)
    for i in range(len(y_range)):
        failures = 0
        for t in range(trials):
            sqp = SolveSQP(0., y_range[i])
            try:
                sqp.re_plan()
            except:
                print("optimization failed")
                failures += 1
                break
            if not(-0.005 <= sqp.x < 0.005):
                failures += 1
            del sqp
        success[i] = (trials - failures)/(1. * trials)

    plt.title("Success rate of starting end effector positions with goal position of x = 0.")
    plt.xlabel("starting end effector position")
    plt.ylabel("success rate")
    plt.plot(y_range, success)
    plt.show()
    print("success rate", np.mean(success))
    """
