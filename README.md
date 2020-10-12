# Noise-Tolerant Quasi-Newton Methods for Noisy Unconstrained Optimization
Authors: Hao-Jun Michael Shi (Northwestern University) and Yuchen Xie (Northwestern University)

## Introduction
The module `ntqn.py` contain both extensions of the BFGS and L-BFGS methods for the minimization of a nonlinear function 
subject to errors. The work is motivated by applications that contain computational noise, employ low-precision 
arithmetic, or are subject to statistical noise. We only assume that the noise is additive and bounded. 
The optimizer is provided potentially noisy evaluations of the function and gradient, and seeks to find an approximate minimizer. 
The curvature information is stabilized through a lengthening procedure that spaces out the points at which gradient differences 
are computed. The implementation is written in Python 3. Please see our paper at <https://arxiv.org/abs/2010.04352> for more details on our work.

## Usage
To use our code, import `ntqn.py` and call the function. 
The function signature is 
`bfgs_e(func, grad, x0, eps_f=0., eps_g=0., callback=None, options=None)`
where
- `func` (callable): (Noisy) objective function to be minimized with signature `func(x)` where `x` is an ndarray with 
shape `(n,)`. 
- `grad` (callable): (Noisy) gradient of objective function with signature `grad(x)` where `x` is an ndarray with shape 
`(n,)`. 
- `x0` (ndarray): Initial iterate with shape `(n,)`.
- `eps_f` (float or callable): Noise level of the function. If callable, has signature `eps_f(x)` and returns float. (Default: 0)
- `eps_g` (float, ndarray, or callable): Noise level of the gradient. If ndarray, then corresponds to noise level with 
respect to each coordinate. If callable, has signature `eps_g(x)` that returns float or ndarray. (Default: 0)
- `callback` (callable): Callback function.
- `options` (dict): dictionary containing options for algorithm:
    - `max_iter` (int): Maximum number of iterations. (Default: 1000)
    - `max_feval` (int): Maximum number of function evaluations. (Default: max_iter * n)
    - `max_geval` (int): Maximum number of gradient evaluations. (Default: 3000)
    - `max_ls_iter` (int): Maximum number of line search trials in split phase. (Default: 20)
    - `split_iter` (int): Maximum number of line search trials before triggering split phase. (Default: 30)
    - `term_iter` (int): Maximum number of iterations attempting to reduce observed objective value before
    terminating. (Default: 5)
    - `c1` (float): Constant for Armijo condition. (Default: 1e-4)
    - `c2` (float): Constant for Wolfe condition. (Default: 0.9)
    - `c3` (float): Constant for noise control condition. (Default: 0.5)
    - `tol` (float): Desired gradient tolerance acceptable for convergence. (Default: 1e-5)
    - `alpha_init` (float): Initial steplength. (Default: 1)
    - `beta_init` (float): Initial lengthening parameter. (Default: 1)
    - `H_init` (ndarray): Initial BFGS matrix as 2-D ndarray with shape `(n,n)`. (Default: np.eye(n))
    - `qn_hist_size` (int): Length of history for curvature pairs for L-BFGS. If np.inf, uses BFGS. (Default: 10)
    - `f_hist_size` (int): Length of history for tracking function values. (Default: 10)
    - `mu_hist_size` (int): Length of history for strong convexity parameter heuristic. (Default: 10)
    - `display` (int): Display level. 
        - `0`: no display
        - `1`: display summary
        - `2`: display iterations
        - `3`: display inner iterations from line search
    - `terminate` (int): Termination criteria.
        - `0`: do not terminate until iteration, function evaluations, or gradient evaluations limit is reached.
        - `1`: in addition to the above, terminate when no more progress is made due to numerical error or no more 
        progress is made in the observed function value.
        - `2`: in addition to the above, terminate when one reaches the noise level.
        - `3`: in addition to the above, terminate when one has satisfied the desired gradient tolerance.

The function outputs:
- `x_k` (ndarray): Final solution.
- `f_k` (float): Final objective value.
- `iter` (int): Number of iterations.
- `func_evals` (int): Number of function evaluations.
- `grad_evals` (int): Number of gradient evaluations.
- `flag` (int): Flag for termination.
    - `0`: Converged to desired gradient tolerance.
    - `1`: Reached maximum number of iterations.
    - `2`: Reached maximum number of function evaluations.
    - `3`: Reached maximum number of gradient evaluations.
    - `4`: Reached noise level of the function.
    - `5`: Reached noise level of the gradient.
    - `6`: No more progress made after `term_count` iterations.
    - `7`: No more progress due to numerical issues.
- `results` (dict): Dictionary containing:
    - `f_ks` (ndarray): History of function values from each iteration with shape `(iter,)`.
    - `norm_gks` (ndarray): History of gradient norms from each iteration with shape `(iter,)`.
    - `func_evals` (ndarray): History of cumulative function evaluations from each iteration with shape `(iter,)`. 
    - `grad_evals` (ndarray): History of cumulative gradient evaluations from each iteration with shape `(iter,)`.
    - `alphas` (ndarray): History of steplengths with shape `(iter,)`.
    - `betas` (ndarray): History of lengthening parameters with shape `(iter,)`.
    - `mus` (ndarray): History of strong convexity parameter estimates with shape `(iter,)`. 

A sample usage for this code on the Rosenbrock problem with synthetic stochastic noise is:
```buildoutcfg
import numpy as np
import scipy.optimize
import ntqn

n = 2
x0 = np.array([-1.25, 1.])

def func(x):
    return scipy.optimize.rosen(x) + np.random.uniform(-1e-5, 1e-5)
def grad(x):
    return scipy.optimize.rosen_der(x) + np.random.uniform(-1e-3, 1e-3, size=x0.shape)

x_opt, f_opt, iters, f_evals, g_evals, flag, results = ntqn.bfgs_e(func, grad, x0, eps_f=1e-5, eps_g=np.sqrt(n) * 1e-3)
```

## Citation

If you use this code or our results in your research, please cite our two works:

    @article{shi2020noise,
      title={A Noise-Tolerant Quasi-Newton Method for Unconstrained Optimization},
      author={Shi, Hao-Jun Michael and Xie, Yuchen and Byrd, Richard H and Nocedal, Jorge},
      journal={arXiv preprint arXiv:2010.04352},
      year={2020}
    }
    	
    @article{xie2020analysis,
      title={Analysis of the BFGS Method with Errors},
      author={Xie, Yuchen and Byrd, Richard H and Nocedal, Jorge},
      journal={SIAM Journal on Optimization},
      volume={30},
      number={1},
      pages={182--209},
      year={2020},
      publisher={SIAM}
    }