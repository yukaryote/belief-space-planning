import numpy as np

from pydrake.all import (
    MathematicalProgram, Solve, eq, le, ge
)

from pydrake.symbolic import Variable
from histogram_filter import histogram_filter
from scipy.special import kl_div


def dirtran(x_samples, T, x_g=None, alpha=0.0085):
    # Discrete-time approximation of the double integrator.
    K = len(x_samples)
    dt = 0.01
    A = np.eye(2) + dt * np.mat('0 1; 0 0')
    B = dt * np.mat('0; 1')

    prog = MathematicalProgram()

    # Create decision variables
    u = np.empty((2, T - 1), dtype=Variable)
    x = np.empty((4, K, T), dtype=Variable)
    for t in range(T - 1):
        u[:, t] = prog.NewContinuousVariables(2, 'u' + str(t))
        for k in range(K):
            x[:, k, t] = prog.NewContinuousVariables(4, 'x' + str(k) + str(t))

    for k in range(K):
        x[:, k, T - 1] = prog.NewContinuousVariables(4, 'x' + str(k) + str(T))

    # Add costs and constraints
    x0 = x_samples
    J = np.mean([calc_weights(x_samples[0], k, x_samples, u, T) ** 2 for k in range(K)])
    cost2 = alpha * sum([np.linalg.norm(u[:, t]) ** 2 for t in range(T)])
    prog.AddCost(J + cost2)
    prog.AddBoundingBoxConstraint(x0, x0, x[:, :, 0])
    for t in range(T - 1):
        for k in range(K):
            w_next = calc_weights(x_samples[0], k, x_samples, u, t+1)
            w_cur = calc_weights(x_samples[0], k, x_samples, u, t)
            prog.AddConstraint(eq(x[:, k, t + 1], f(x[:, k, t], u[:, t])))
            prog.AddConstraint(eq(w_next, w_cur * np.e**phi(x[:, k, t], x_samples[0])))
    if x_g is not None:
        xf = x_g
        prog.AddBoundingBoxConstraint(xf, xf, x[:, 0, T - 1])

    result = Solve(prog)

    u_sol = result.GetSolution(u)
    assert (result.is_success()), "Optimization failed"
    return u_sol


def calc_weights(x_1, i, x_samples, u, T):
    weight = 1
    for t in range(T):
        weight *= np.e ** (phi(F(x_samples[i], u[:t]), F(x_1, u[:t])))
    return weight


def phi(x, y):
    return 1 / 2 * (h(x) - h(y).T) @ np.linalg.inv(2 * np.identity(1) + H(x) @ H(x).T + H(y) @ H(y).T) @ (h(x) - h(y))


def F(x, u):
    return


def f(x, u):
    return


def h(x):
    return


def H(x):
    return


def theta(belief_state, r, x_g):
    return 0


def J(x_samples, u, t):
    return np.mean([calc_weights(x_samples[0], k, x_samples, u, t) ** 2 for k in range(len(x_samples))])


def create_plan(belief_state, x_samples, x_g, T, omega=0.5, r=0.5):
    u = dirtran(x_samples, T, x_g=x_g)
    # TODO: update belief state
    if theta(belief_state, r, x_g) <= omega:
        u = dirtran(x_samples, T)
        # TODO: update belief state
    return belief_state, u


def re_plan(belief_state, x_g, T, G, omega=0.5, r=0.5, K=15, thresh=0.25, kl_thresh=0.5, rho=0.5):
    while theta(belief_state, r, x_g) <= omega:
        x_samples = [np.argmax(belief_state)]
        k = 1
        while k < K:
            sample = np.random.choice(belief_state, 1, p=belief_state)
            if sample > thresh:
                x_samples.append(sample)
                k += 1
        belief_state, u = create_plan(belief_state, x_samples, x_g, T, omega, r)
        b_1 = belief_state
        for t in range(T-1):
            # TODO execute action u_t, perceive observation z_{t+1}
            u_t = None
            z_next = None
            belief_state = G(belief_state, u_t, z_next)
            if kl_div(belief_state, belief_state) > kl_thresh and J(x_samples, u, t) < 1 - rho:
                break
        # TODO update belief state
