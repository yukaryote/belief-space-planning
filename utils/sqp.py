import numpy as np

from pydrake.all import (
    MathematicalProgram, Solve, eq
)

from pydrake.symbolic import Variable
from utils.histogram_filter import HistogramFilter
from scipy.special import kl_div


class SolveSQP:
    def __init__(self, T, alpha, N, field, noise_std):
        self.T = T
        self.alpha = alpha
        self.h = field
        self.H = np.zeros_like(self.h)
        prev = self.h[0]
        # h is a square wave, so its derivative is two impulses of opposite sign
        for i in range(len(self.h)):
            if prev < self.h[i]:
                self.H[i] = 1
            elif prev > self.h[i]:
                self.H[i] = -1
        self.Q = np.random.normal(loc=0, scale=noise_std)
        self.histogram_filter = HistogramFilter(N, field, noise_std)

    def dirtran(self, x_samples, x_g=None):
        # Discrete-time approximation of the double integrator.
        K = len(x_samples)

        prog = MathematicalProgram()

        # Create decision variables
        u = np.empty((1, self.T - 1), dtype=Variable)
        x = np.empty((1, K, self.T), dtype=Variable)
        for t in range(self.T - 1):
            u[:, t] = prog.NewContinuousVariables(1, 'u' + str(t))
            for k in range(K):
                x[:, k, t] = prog.NewContinuousVariables(1, 'x' + str(k) + str(t))

        for k in range(K):
            x[:, k, self.T - 1] = prog.NewContinuousVariables(1, 'x' + str(k) + str(self.T))

        # Add costs and constraints
        x0 = x_samples
        J = np.mean([self.calc_weights(x_samples[0], k, x_samples, u, self.T) ** 2 for k in range(K)])
        cost2 = self.alpha * sum([np.linalg.norm(u[:, t]) ** 2 for t in range(self.T)])
        prog.AddCost(J + cost2)
        prog.AddBoundingBoxConstraint(x0, x0, x[:, :, 0])
        for t in range(self.T - 1):
            for k in range(K):
                w_next = self.calc_weights(x_samples[0], k, x_samples, u, t + 1)
                w_cur = self.calc_weights(x_samples[0], k, x_samples, u, t)
                prog.AddConstraint(eq(x[:, k, t + 1], self.f(x[:, k, t], u[:, t])))
                prog.AddConstraint(eq(w_next, w_cur * np.e ** self.phi(x[:, k, t], x_samples[0])))
        if x_g is not None:
            xf = x_g
            prog.AddBoundingBoxConstraint(xf, xf, x[:, 0, self.T - 1])

        result = Solve(prog)

        u_sol = result.GetSolution(u)
        assert (result.is_success()), "Optimization failed"
        return u_sol

    def calc_weights(self, x_1, i, x_samples, u, T):
        weight = 1
        for t in range(T):
            weight *= np.e ** (self.phi(self.F(x_samples[i], u[:t]), self.F(x_1, u[:t])))
        return weight

    def phi(self, x, y):
        """
        Weighting function
        """
        return 1 / 2 * (self.h[x] - self.h[y].T) @ np.linalg.inv(2 * self.Q +
                        self.H[x] @ self.H[x].T + self.H[y] @ self.H[y].T) @ (
                        self.h[x] - self.h[y])

    def F(self, x, u):
        return

    def f(self, x, u):
        """
        Returns next state if we are in state x and take action u
        """
        return

    def theta_cap(self, belief_state, r, x_g):
        """
        Probability that we are in a ball of radius r around x_g
        """
        return 0

    def J(self, x_samples, u, t):
        return np.mean([self.calc_weights(x_samples[0], k, x_samples, u, t) ** 2 for k in range(len(x_samples))])

    def create_plan(self, x_samples, x_g, omega=0.5, r=0.5):
        belief_state = self.histogram_filter.p[:]
        u = self.dirtran(x_samples, x_g=x_g)
        belief_states = np.ndarray(shape=(belief_state.shape[0], self.T))
        belief_states[0] = belief_state
        for t in range(self.T - 1):
            belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        if self.theta_cap(belief_state, r, x_g) <= omega:
            u = self.dirtran(x_samples, self.T)
            belief_states[0] = belief_state
            for t in range(self.T - 1):
                belief_state[t + 1] = self.histogram_filter.update(u[t], self.h[x_samples[0][t]])
        return belief_states, u

    def re_plan(self, x_g, omega=0.5, r=0.5, K=15, thresh=0.25, kl_thresh=0.5, rho=0.5):
        belief_state = self.histogram_filter.p[:]
        while self.theta_cap(belief_state, r, x_g) <= omega:
            x_samples = [np.argmax(self.histogram_filter.p)]
            k = 1
            while k < K:
                sample = np.random.choice(len(belief_state), 1, p=belief_state)
                if sample > thresh:
                    x_samples.append(sample)
                    k += 1
            belief_states, u = self.create_plan(x_samples, x_g, omega, r)
            belief_states[0] = belief_state
            t_break = 0
            for t in range(self.T - 1):
                u_t = u[t]
                # TODO: get z_next from MultiBody plant
                z_next = None
                belief_state = self.histogram_filter.update(u_t, z_next)
                if kl_div(belief_state, belief_state) > kl_thresh and self.J(x_samples, u, t) < 1 - rho:
                    t_break = t
                    break
                t_break += 1
            belief_state = belief_states[t_break]
