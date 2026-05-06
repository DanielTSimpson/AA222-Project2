#
# File: project2.py
#

## top-level submission file

'''
Note: Do not import any other modules here.
        To import from another file xyz.py here, type
        import project2_py.xyz
        However, do not import any modules except numpy in those files.
        It's ok to import modules only in files that are
        not imported here (e.g. for your plotting code).
'''
import numpy as np

try:
    from .helpers import Simple1, Simple2, Simple3
    from .local_minimizers import adam_optimizer
except ImportError:
    from helpers import Simple1, Simple2, Simple3
    from local_minimizers import adam_optimizer


def augmented_lagrange_ADAM(f, g, c, x0, n, count, path, params=None):
    #Hyper terms
    x_length = len(x0)

    # ALM Terms
    if count() < n:
        c0 = np.array(c(x0)).flatten() # Costs 1
        mu = np.zeros(len(c0))
    else:
        mu = np.array([])
    rho = params.get('rho', 1.0)
    gamma_rho = params.get('gamma_rho', 1.01)
    num_inner_loops = params.get('num_inner_loops', 12)

    #ADAM Terms
    alpha = params.get('alpha', 0.32)
    gamma = params.get('gamma', 0.95)     
    gamma_v = params.get('gamma_v', 0.9)    
    gamma_s = params.get('gamma_s', 0.999)    
    epsilon = params.get('epsilon', 1e-8)

    def penalty(x, mu, rho, c): # Each penalty call costs 1
        if len(mu) == 0:
            return 0.0
        constraints = np.array(c(x)).flatten()
        p_Lagrange = 0
        for i in range(len(constraints)): # For every constraint, sum the maxes to the penalty function
            p_Lagrange += (np.maximum(mu[i] + rho*constraints[i], 0)**2 - mu[i]**2) / (2*rho) # Penalty term
        return float(p_Lagrange)
    
    def penalty_g(x, mu, rho, c, h = 1e-5): # Each penalty_g call costs 1 + x_length
        x_length = len(x)
        penalty_gh = np.zeros(x_length)
        if len(mu) == 0:
            return penalty_gh
        identity = np.identity(x_length)
        p0 = penalty(x, mu, rho, c)
        for idx, _ in enumerate(identity): # Perform a forward difference method to get the approx gradient of the pentalty
            penalty_gh[idx] = penalty(x + identity[idx]*h, mu, rho, c) - p0
        return penalty_gh / h

    vel = np.zeros(x_length)
    sqr = np.zeros(x_length)
    k = 0
    while count() < n:
        
        x = path[-1]
        
        augmented_f = lambda x: f(x) + penalty(x, mu, rho, c) # Each augmented_f call costs 2
        augmented_g = lambda x: g(x) + penalty_g(x, mu, rho, c) # Each augmented_g call costs 3 + x_length

        cost_g = 3 + x_length if len(mu) > 0 else 2 
        evals_for_mu = 1 if len(mu) > 0 else 0
        augmented_n = np.minimum(count() + cost_g * num_inner_loops, n - evals_for_mu)
        
        adam_path, _, k, vel, sqr = adam_optimizer(
            augmented_f, augmented_g, x, augmented_n, count, 
            alpha, gamma, gamma_v, gamma_s, epsilon, 
            k, vel, sqr, cost_g
        )
        
        x_new = adam_path[-1]
        if len(mu) > 0 and count() < n:
            mu = np.maximum(mu + rho * np.array(c(x_new)).flatten(), 0) # Update the Lagrange multiplier (This costs 1)          
        path.append(x_new)
        rho = rho * gamma_rho # Slowly increase rho to penalize the constraint term more as we approach the optimum 
        #alpha = alpha / np.sqrt(rho) # Decay alpha as rho increases so ADAM takes smaller steps with higher penalties on the constraint terms
        alpha = params.get('alpha', 0.32) # Restore sub-problem step size
    return path


def optimize_with_history(f, g, c, x0, n, count, prob):
    """
    Args:
        f (function): Function to be optimized
        g (function): Gradient function for `f`
        c (function): Function evaluating constraints
        x0 (np.array): Initial position to start from
        n (int): Number of evaluations allowed. Remember `f` and `c` cost 1 and `g` costs 2
        count (function): takes no arguments are reutrns current count
        prob (str): Name of the problem. So you can use a different strategy 
                 for each problem. `prob` can be `simple1`,`simple2`,`simple3`,
                 `secret1` or `secret2`
    Returns:
        x_best (np.array): best selection of variables found
    """

    path = [x0]
    if prob == 'simple1':  ### Augmented Lagrange Method (ALM) ###
        simple1_params = {
            'rho': 1.0,
            'gamma_rho': 1.50,
            'num_inner_loops': 12,
            'alpha': 0.32,
            'gamma': 0.75,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, simple1_params)

    if prob == 'simple2':
        simple2_params = {
            'rho': 1.0,
            'gamma_rho': 1.5,
            'num_inner_loops': 12,
            'alpha': 0.32,
            'gamma': 0.75,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, simple2_params)

    if prob == 'simple3':
        simple3_params = {
            'rho': 1.0,
            'gamma_rho': 1.50,
            'num_inner_loops': 12,
            'alpha': 0.32,
            'gamma': 0.75,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, simple3_params)

    if prob == 'secret1':
        secret1_params = {
            'rho': 1.0,
            'gamma_rho': 1.50,
            'num_inner_loops': 4,
            'alpha': 0.32,
            'gamma': 0.75,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, secret1_params)
    
    if prob == 'secret2':
        secret2_params = {
            'rho': 1.0,
            'gamma_rho': 1.50,
            'num_inner_loops': 4,
            'alpha': 0.32,
            'gamma': 0.75,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, secret2_params)

    else:
        default_params = {
            'rho': 1.0,
            'gamma_rho': 1.02,
            'num_inner_loops': 12,
            'alpha': 0.32,
            'gamma': 0.95,
            'gamma_v': 0.9,
            'gamma_s': 0.999,
            'epsilon': 1e-8
            }
        path = augmented_lagrange_ADAM(f, g, c, x0, n, count, path, default_params)

    return path


def optimize(f, g, c, x0, n, count, prob):
    """
    Wrapper for autograder that only returns the final point.
    """
    path = optimize_with_history(f, g, c, x0, n, count, prob)
    return path[-1]

if __name__ == '__main__':
    try:
        from .plotters import plot_problem
    except ImportError:
        from plotters import plot_problem
    current_problem = Simple1()
    if current_problem.xdim == 2:
        print(f"Rendering 2D optimization paths for {current_problem.prob}...")
        plot_problem(current_problem, plot_size=3)
