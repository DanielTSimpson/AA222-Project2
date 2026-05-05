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
        #print(f"x0: {x0}")
        #Hyper terms
        x_length = len(x0)

        # ALM Terms
        mu = np.zeros(len(c(x0)))
        rho = 1.0
        gamma_rho = 1.02
        k = 0

        #ADAM Terms
        alpha = 0.32     
        gamma = 0.95     
        gamma_v = 0.9    
        gamma_s = 0.999    
        epsilon = 1e-8

        def penalty(x, mu, rho, c):
            constraints = c(x)
            p_Lagrange = 0
            for i in range(len(constraints)): # For every constraint, sum the maxes to the penalty function
                p_Lagrange += (np.maximum(mu[i] + rho*constraints[i], 0)**2 - mu[i]**2) / (2*rho) # Penalty term

            return p_Lagrange
        
        def penalty_g(x, mu, rho, c, h = 1e-5):
            x_length = len(x)
            penalty_gh = np.zeros(x_length)
            identity = np.identity(x_length)
            p0 = penalty(x, mu, rho, c)
            for idx, _ in enumerate(identity):
                penalty_gh[idx] = penalty(x + identity[idx]*h, mu, rho, c) - p0
            return penalty_gh / h

        cost_per_iter = 5 + x_length
        num_inner_loops = 12
        while count() < n:
            #print(f"Outer loop start: {count()}")
            x = path[-1]

            augmented_f = lambda x: f(x) + penalty(x, mu, rho, c)
            augmented_g = lambda x: g(x) + penalty_g(x, mu, rho, c)
        
            x_new = adam_optimizer(augmented_f, augmented_g, x, np.minimum(count()+cost_per_iter*num_inner_loops, n - 1), count, alpha, gamma, gamma_v, gamma_s, epsilon)
            #print(f"x_new: {x_new}, c(x_new): {c(x_new)}")
            #print(f"After ADAM: {count()}") 

            mu = np.maximum(mu + rho * c(x_new), 0) # Update the Lagrange multiplier           
            #print(f"After mu update: {count()}")

            path.append(x_new)
            rho = rho*gamma_rho
            k += 1
#    print(f"Final mu: {mu}, Final rho: {rho}")
#    print(f"Final f value: {f(path[-1])}")
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
