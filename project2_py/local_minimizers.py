import numpy as np

def adam_optimizer(f, g, x0, n, count, alpha, gamma, gamma_v, gamma_s, epsilon, k=0, vel=None, sqr=None, cost_g=None):
    path = [x0]
    x_best = x0
    x_len = len(x0)
    if sqr is None: sqr = np.zeros(x_len)
    if vel is None: vel = np.zeros(x_len)
    if cost_g is None: cost_g = 3 + x_len

    while count() + cost_g <= n:
        k += 1
        gradient = g(x_best) # Costs 
        vel = gamma_v*vel + (1-gamma_v)*gradient
        sqr = gamma_s*sqr + (1-gamma_s)*(gradient**2)
        vel_hat = vel / (1 - gamma_v**k)
        sqr_hat = sqr / (1 - gamma_s**k)
        step = alpha * vel_hat / (epsilon + np.sqrt(sqr_hat))
        x_best = x_best - step
        path.append(x_best)
        alpha *= gamma
        if np.linalg.norm(step) < epsilon:
            break
    return path, alpha, k, vel, sqr