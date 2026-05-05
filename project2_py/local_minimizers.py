import numpy as np

def adam_optimizer(f, g, x0, n, count, alpha, gamma, gamma_v, gamma_s, epsilon):
    path = [x0]
    x_best = x0
    x_len = len(x0)
    sqr = np.zeros(x_len)
    vel = np.zeros(x_len)
    k = 0
    while count() + (3 + x_len) < n:
        k += 1
        gradient = g(x_best)
        vel = gamma_v*vel + (1-gamma_v)*gradient
        sqr = gamma_s*sqr + (1-gamma_s)*(gradient**2)
        vel_hat = vel / (1 - gamma_v**k)
        sqr_hat = sqr / (1 - gamma_s**k)
        x_best = x_best - alpha * vel_hat / (epsilon + np.sqrt(sqr_hat))
        path.append(x_best)
        alpha *= gamma
    return path[-1]