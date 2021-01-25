import numpy as np


def bfgs_e(func, grad, x0, eps_f=0., eps_g=0., callback=None, options=None):
    """BFGS-E: Noise-Tolerant BFGS/BFGS with Errors

    A noise-tolerant (L-)BFGS algorithm that minimizes an objective function with error in its function and gradient
    evaluations. Requires knowledge of the noise level in both the function and gradient. Method defaults to using
    limited memory.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.

            ``func(x) -> float``

        where ``x`` is a 1-D array with shape (n,).
    grad : callable
        Method for computing the gradient vector.

            ``grad(x) -> array_like, shape (n,)``

        where ``x`` is a 1-D array with shape (n,).
    x0 : ndarray, shape (n,)
        Initial guess.
    eps_f : {float, callable}, optional
        Noise level of function. If callable, should return float. (Default: 0)
    eps_g : {float, ndarray, callable}, optional
        Noise level of the gradient. Required if grad is callable. If float or callable that returns float, assumes
        2-norm bound on the gradient noise. If ndarray or callable that returns ndarray, eps_g must be have shape (n,)
        and assumes bounds on the noise with respect to each component. (Default: 0)
    callback : callable, optional
        An optional user-supplied function, called after each iteration. Called as ``callback(x)``, where ``x`` is the
        current iterate.
    options : dict, optional
        Options passed to the solver. It optionally contains:
            max_iter : int, optional
                Maximum number of iterations. (Default: 1000)
            max_feval : int, optional
                Maximum number of function evaluations. (Default: max_iter * n)
            max_geval : int, optional
                Maximum number of gradient evaluations. (Default: 3000)
            max_ls_iter : int, optional
                Maximum number of line search trials in split phase. (Default: 20)
            split_iter : int, optional
                Maximum number of line search trials before triggering split phase. (Default: 30)
            term_iter : int, optional
                Maximum number of iterations attempting to reduce observed objective value before terminating.
                (Default: 5)
            c1 : float, optional
                Constant for Armijo condition. (Default: 1e-4)
            c2 : float, optional
                Constant for Wolfe condition. (Default: 0.9)
            c3 : float, optional
                Constant for noise control condition. (Default: 0.5)
            tol : float, optional
                Desired gradient tolerance acceptable for convergence. (Default: 1e-5)
            alpha_init : float, optional
                Initial steplength. (Default: 1)
            beta_init : float, optional
                Initial lengthening parameter. (Default: 1)
            H_init: ndarray, optional
                Initial BFGS matrix as 2-D ndarray with shape (n,n). (Default: I)
            qn_hist_size : int, optional
                Length of history for curvature pairs for L-BFGS. (Default: 10)
            f_hist_size : int, optional
                Length of history for tracking function value. (Default: 10)
            mu_hist_size : int, optional
                Length of history for strong convexity parameter heuristic. (Default: 10)
            display : int, optional
                Display level. (Default: 2)
                    0 : no display
                    1 : display summary
                    2 : display iterations
                    3 : display inner iterations (from line search) and outer iterations
            terminate : int, optional
                Determines termination criteria. (Default: 3)
                    0 : do not terminate until iteration, function evaluation, or gradient evaluation
                        is reached
                    1 : terminate when one no more progress is made due to numerical error or no more progress
                        is made on the observed function value
                    2 : in addition to the above, terminate when one reaches the noise level
                    3 : in addition to the above, terminate when one satisfies the desired gradient tolerance

    Returns
    -------
    x_k : ndarray
        Final solution.
    f_k : float
        Final objective value `func(x_k)`.
    iter : int
        Number of iterations.
    func_evals : int
        Number of function evaluations.
    grad_evals : int
        Number of gradient evaluations.
    flag : int
        Flag for termination.
            0 : Converged to desired gradient tolerance.
            1 : Reached maximum number of iterations.
            2 : Reached maximum number of function evaluations.
            3 : Reached maximum number of gradient evaluations.
            4 : Reached noise level of the function.
            5 : Reached noise level of the gradient.
            6 : No more progress made after term_count iterations.
            7 : No more progress due to numerical issues.
    results : dict
        Dictionary containing:
            f_ks : ndarray
                History of function values from each iteration with shape (iter,).
            norm_gks : ndarray
                History of gradient norms from each iteration with shape (iter,).
            func_evals : ndarray
                History of cumulative function evaluations from each iteration with shape (iter,).
            grad_evals : ndarray
                History of cumulative gradient evaluations from each iteration with shape (iter,).
            alphas : ndarray
                History of steplengths with shape (iter,).
            betas : ndarray
                History of lengthening parameters with shape (iter,).
            mus : ndarray
                History of strong convexity parameter estimates with shape (iter,).

    Notes
    -----
    . The signature is designed to be similar to the `minimize` function in SciPy.
    . Uses a generalized form of the noise-tolerant Armijo-Wolfe line search as designed in [1], consisting of an
      initial and split phase.
    . The noise control condition employed in the line search can also handle component-wise noise levels of the
      gradient, which generalizes the work done in [1].
    . Can also handle changing levels of noise or iterate-dependent noise levels by passing callable noise level
      functions.

    Examples
    --------
    >>> from scipy.optimize import rosen
    >>> from ntqn import bfgs_e
    >>> x0 = np.array([-1.25, 1.])
    >>> x_opt, f_opt, iters, f_evals, g_evals, flag, results = bfgs_e(rosen, x0)

    References
    ----------
    .. [1] Hao-Jun Michael Shi, Yuchen Xie, Richard Byrd, and Jorge Nocedal.
       "A Noise-Tolerant Quasi-Newton Algorithm for Unconstrained Optimization."
    .. [2] Yuchen Xie, Richard Byrd, and Jorge Nocedal.
       "Analysis of the BFGS Method with Errors."
       SIAM Journal on Optimization 30.1 (2020): 182-209.

    Authors
    -------
    Hao-Jun Michael Shi (hjmshi@u.northwestern.edu)
    Yuchen Xie (ycxie@u.northwestern.edu)
    Department of Industrial Engineering and Management Sciences
    Northwestern University

    """

    # initialize values
    x_k = np.copy(x0)
    n = len(x_k)
    iter = 0
    func_evals = 0
    grad_evals = 0
    armijo_fails = 0
    wolfe_fails = 0
    alpha = 0.
    beta = 0.

    # construct options dict if not given
    if not options:
        options = {}

    # set options to default if not given
    if 'max_iter' not in options.keys():
        options['max_iter'] = 1000
    if 'max_feval' not in options.keys():
        options['max_feval'] = options['max_iter'] * n
    if 'max_geval' not in options.keys():
        options['max_geval'] = 3000
    if 'max_ls_iter' not in options.keys():
        options['max_ls_iter'] = 20
    if 'split_iter' not in options.keys():
        options['split_iter'] = 30
    if 'term_iter' not in options.keys():
        options['term_iter'] = 5
    if 'c1' not in options.keys():
        options['c1'] = 1e-4
    if 'c2' not in options.keys():
        options['c2'] = 0.9
    if 'c3' not in options.keys():
        options['c3'] = 0.5
    if 'tol' not in options.keys():
        options['tol'] = 1e-5
    if 'alpha_init' not in options.keys():
        options['alpha_init'] = 1.0
    if 'beta_init' not in options.keys():
        options['beta_init'] = 1.0
    if 'qn_hist_size' not in options.keys():
        options['qn_hist_size'] = 10
    if 'f_hist_size' not in options.keys():
        options['f_hist_size'] = 10
    if 'mu_hist_size' not in options.keys():
        options['mu_hist_size'] = 10
    if 'display' not in options.keys():
        options['display'] = 2
    if 'terminate' not in options.keys():
        options['terminate'] = 3

    verbose = (options['display'] >= 3)

    # initialize BFGS matrix or L-BFGS memory
    if options['qn_hist_size'] < np.inf:
        qn_hist = []
        qn_init = 1.
    else:
        if 'H_init' in options.keys():
            H = np.copy(options['H_init'])
        else:
            H = np.identity(n)

    # assumes gradient is not noisy if passed without specifying eps_g or eps_f = 0
    if (not np.any(eps_g) and callable(grad)) or eps_f == 0.:
        eps_g = 0.

    # track average function over window
    f_avg = np.inf
    f_hist = np.array([])

    # track strong convexity parameter for heuristic
    mu = np.inf
    mu_hist = np.array([])

    # initialize results dict
    results = {'f_ks': np.array([]), 'norm_gks': np.array([]), 'func_evals': np.array([]), 'grad_evals': np.array([]),
               'alphas': np.array([]), 'betas': np.array([]), 'mus': np.array([]), 'eps_fs': np.array([]),
               'eps_gs': np.array([]), 'norm_pks': np.array([])}

    # flag for determining success
    flag = 1
    term_count = 0

    # get machine precision
    eps_m = np.finfo(np.float).eps

    # calculate initial function and gradient values
    f_k = func(x_k)
    f_hist = np.append(f_hist, f_k)
    func_evals += 1

    # compute gradient
    g_k = grad(x_k)
    grad_evals += 1

    # norm_gk = np.linalg.norm(g_k, ord=np.inf)
    norm_gk = np.linalg.norm(g_k)
    norm_pk = np.copy(norm_gk)

    # compute function and gradient error bound
    if callable(eps_f):
        eps_fk = eps_f(x_k)
    else:
        eps_fk = np.copy(eps_f)

    if callable(eps_g):
        eps_gk = eps_g(x_k)
    else:
        eps_gk = np.copy(eps_g)

    # print initial header
    if options['display'] >= 2:
        print('=======================================================================================================================')
        print('Solving with Noise-Tolerant BFGS Method')

    # main loop
    while iter < options['max_iter']:

        # perform callback
        if callback is not None:
            callback(x_k)

        # print header
        if options['display'] >= 2 and iter % 25 == 0:
            print('-----------------------------------------------------------------------------------------------------------------------')
            print('    Iter      |      F       |    ||g||     |   F Evals    |   G Evals    |    alpha     |     beta     |      mu      ')
            print('-----------------------------------------------------------------------------------------------------------------------')

        # print iteration info
        if options['display'] >= 2:
            print('  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                  % (iter, f_k, norm_gk, func_evals, grad_evals, alpha, beta, mu))

        # store values
        results['f_ks'] = np.append(results['f_ks'], f_k)
        results['norm_gks'] = np.append(results['norm_gks'], norm_gk)
        results['func_evals'] = np.append(results['func_evals'], func_evals)
        results['grad_evals'] = np.append(results['grad_evals'], grad_evals)
        results['alphas'] = np.append(results['alphas'], alpha)
        results['betas'] = np.append(results['betas'], beta)
        results['mus'] = np.append(results['mus'], mu)
        results['eps_fs'] = np.append(results['eps_fs'], eps_fk)
        results['eps_gs'] = np.append(results['eps_gs'], eps_gk)
        results['norm_pks'] = np.append(results['norm_pks'], norm_pk)

        # compute (L-)BFGS search direction
        if options['qn_hist_size'] < np.inf:
            q = - g_k

            m = len(qn_hist)
            a = np.zeros(m)

            for i in reversed(range(m)):
                s, y, rho = qn_hist[i]
                a[i] = rho * np.dot(s, q)
                q = q - a[i] * y

            r = qn_init * q

            for i in range(m):
                s, y, rho = qn_hist[i]
                b = rho * np.dot(y, r)
                r = r + (a[i] - b) * s

            p_k = r

        else:
            p_k = -np.dot(H, g_k)
        norm_pk = np.linalg.norm(p_k)

        # perform line search
        f_old = f_k
        alpha, beta, mu_hat, f_k, g_new, eps_fk, eps_gk, ls_fevals, ls_gevals, armijo_flag, wolfe_flag, split_flag = \
            _line_search_nt_wolfe(func, grad, x_k, p_k, eps_f=eps_f, eps_g=eps_g, f_k=f_k, g_k=g_k,
                                  alpha_init=options['alpha_init'], beta_init=options['beta_init'], mu=mu,
                                  c1=options['c1'], c2=options['c2'], c3=options['c3'],
                                  split_iter=options['split_iter'], max_ls_iter=options['max_ls_iter'], verbose=verbose)

        func_evals += ls_fevals
        grad_evals += ls_gevals

        x_old = np.copy(x_k)

        # if Armijo condition is satisfied, perform update
        if armijo_flag:
            x_k = x_k + alpha * p_k

            # track history of function values and compute average
            f_hist = np.append(f_hist, f_k)
            if f_hist.shape[0] > options['f_hist_size']:
                f_hist = f_hist[1:]
                f_avg = np.mean(f_hist)

        else:
            armijo_fails += 1

        # if Wolfe condition is satisfied, perform BFGS update
        if wolfe_flag:
            y_k = g_new - g_k
            s_k = beta * p_k

            if options['qn_hist_size'] < np.inf:
                # update L-BFGS matrix
                rho = 1 / np.dot(s_k, y_k)
                qn_init = 1 / (rho * np.dot(y_k, y_k))
                qn_hist.append((s_k, y_k, rho))
                if len(qn_hist) > options['qn_hist_size']:
                    qn_hist.pop(0)

            else:
                # heuristic at first iteration to capture scaling
                if iter == 0:
                    H = np.dot(y_k, s_k) / np.dot(y_k, y_k) * np.identity(n)

                # update BFGS matrix
                rho = 1. / np.dot(s_k, y_k)
                mat = np.identity(n) - rho * np.outer(s_k, y_k)
                H = np.matmul(np.matmul(mat, H), mat.transpose()) + rho * np.outer(s_k, s_k)

            # update mu
            mu_hist = np.append(mu_hist, mu_hat)
            if mu_hist.shape[0] > options['mu_hist_size']:
                mu_hist = mu_hist[1:]
            mu = np.min(mu_hist)

        else:
            wolfe_fails += 1

        # iterate counter
        iter += 1

        # check if made progress
        if f_k < f_avg:
            term_count = 0
        else:
            term_count += 1

        # set new gradient at new iterate
        # if Armijo condition holds, not split, and grad provided, sets gradient to be g_new
        if armijo_flag and not split_flag and callable(grad):
            g_k = g_new

        # otherwise, evaluates gradient at new point
        else:
            g_k = grad(x_k)
            grad_evals += 1

            if callable(eps_g):
                eps_gk = eps_g(x_k)
            else:
                eps_gk = np.copy(eps_g)

        # norm_gk = np.linalg.norm(g_k, ord=np.inf)
        norm_gk = np.linalg.norm(g_k)

        # check if desired tolerance is reached
        if options['terminate'] >= 3 and norm_gk <= options['tol']:
            flag = 0
            break

        # check if maximum number of function evaluations is reached
        elif func_evals > options['max_feval']:
            flag = 2
            break

        # check if maximum number of gradient evaluations is reached
        elif grad_evals > options['max_geval']:
            flag = 3
            break

        # check if function noise level is reached
        elif options['terminate'] >= 2 and np.abs(f_k - f_avg) < 2 * eps_fk:
            flag = 4
            break

        # check if gradient noise level is reached
        elif (options['terminate'] >= 2 and isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and
              eps_gk.shape[0] > 1 and np.all(np.abs(g_k) < eps_gk)):
            flag = 5
            break

        elif (options['terminate'] >= 2 and (not isinstance(eps_gk, np.ndarray) or eps_gk.ndim == 0 or
                                             eps_gk.shape[0] == 0) and norm_gk < eps_gk):
            flag = 5
            break

        # check if no more progress is made
        elif options['terminate'] >= 1 and term_count >= options['term_iter']:
            flag = 6
            break

        # check numerics
        elif options['terminate'] >= 1 and eps_f == 0 and eps_g == 0 and \
                (np.linalg.norm(x_k - x_old, ord=np.inf) < eps_m * np.maximum(1., np.linalg.norm(x_k, ord=np.inf)) or
                 f_old - f_k < eps_m * np.maximum(1., np.abs(f_k))):
            flag = 7
            break

    # print iteration info
    if options['display'] >= 2:
        print('  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
              % (iter, f_k, norm_gk, func_evals, grad_evals, alpha, beta, mu))

    # store values
    results['f_ks'] = np.append(results['f_ks'], f_k)
    results['norm_gks'] = np.append(results['norm_gks'], norm_gk)
    results['func_evals'] = np.append(results['func_evals'], func_evals)
    results['grad_evals'] = np.append(results['grad_evals'], grad_evals)
    results['alphas'] = np.append(results['alphas'], alpha)
    results['betas'] = np.append(results['betas'], beta)
    results['mus'] = np.append(results['mus'], mu)
    results['eps_fs'] = np.append(results['eps_fs'], eps_fk)
    results['eps_gs'] = np.append(results['eps_gs'], eps_gk)
    results['norm_pks'] = np.append(results['norm_pks'], norm_pk)

    # print results
    if options['display'] >= 1:
        np.set_printoptions(threshold=5)
        print('=======================================================================================================================')
        print('Summary of Run')
        print('=======================================================================================================================')
        if flag == 0:
            print('Success! Converged to desired gradient tolerance.')
        elif flag == 1:
            print('Terminated! Reached maximum number of iterations.')
        elif flag == 2:
            print('Terminated! Reached maximum number of function evaluations.')
        elif flag == 3:
            print('Terminated! Reached maximum number of gradient evaluations.')
        elif flag == 4:
            print('Terminated! Reached noise level of the function.')
        elif flag == 5:
            print('Terminated! Reached noise level of the gradient.')
        elif flag == 6:
            print('Terminated! No more progress made after term_count iterations.')
        elif flag == 7:
            print('Terminated! No more progress due to numerical issues.')
        print('Total Number of Iterations:', iter)
        print('Final Solution:', x_k)
        print('Final Objective Value:', f_k)
        print('Final Norm of Gradient:', norm_gk)
        print('Number of Function Evaluations:', func_evals)
        print('Number of Gradient Evaluations:', grad_evals)
        print('=======================================================================================================================')

    return x_k, f_k, iter, func_evals, grad_evals, flag, results


def _line_search_nt_wolfe(func, grad, x_k, p_k, eps_f=0., eps_g=0., f_k=None, g_k=None, alpha_init=1., beta_init=1.,
                          mu=None, c1=1e-4, c2=0.9, c3=0.5, split_iter=30, max_ls_iter=20, verbose=False):
    """Noise-Tolerant Armijo-Wolfe Line Search

    Function for performing noise-tolerant Armijo-Wolfe line search for noise-tolerant BFGS and L-BFGS methods.
    Line search reverts back to bisectioning weak Armijo-Wolfe line search if both eps_f and eps_g = 0.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.

            ``func(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,).
    grad : callable, optional
        Method for computing the gradient vector.

            ``grad(x, *args) -> array_like, shape (n,)``

        where ``x`` is a 1-D array with shape (n,).
    x_k : ndarray
        Current iterate with shape (n,).
    p_k : ndarray
        Search direction with shape (n,).
    eps_f : {float, callable}, optional
        Noise level of function. If callable, should return float. (Default: 0)
    eps_g : {float, ndarray, callable}, optional
        Noise level of the gradient. If float or callable that returns float, assumes 2-norm bound on the gradient
        noise. If ndarray or callable that returns ndarray, eps_g must be have shape (n,) and assumes bounds on the
        noise with respect to each component.
    f_k : float, optional
        (Noisy) objective value `func` at current iterate x_k. If None, will evaluate with `func(x_k)`.
    g_k : float, optional
        (Noisy) gradient `grad` at current iterate x_k. If None, will evaluate with `grad(x_k)`.
    alpha_init : float, optional
        Initial steplength. (Default: 1)
    beta_init : float, optional
        Initial lengthening parameter. (Default: 1)
    mu : float, optional
        Estimate of strong convexity parameter.
    c1 : float, optional
        Constant for Armijo condition. (Default: 1e-4)
    c2 : float, optional
        Constant for Wolfe condition. (Default: 0.9)
    c3 : float, optional
        Constant for noise control condition. (Default: 0.5)
    split_iter : int, optional
        Maximum number of line search trials before triggering split phase. (Default: 30)
    max_ls_iter : int, optional
        Maximum number of line search trials/iterations in split phase. (Default: 20)
    verbose : bool, optional
        Print line search information. (Default: False)

    Returns
    -------
    alpha : float
        Steplength.
    beta : float
        Lengthening parameter.
    mu : float
        Estimate of strong convexity parameter.
    f_new : float
        Function value at new iterate.
    g_new : ndarray
        Gradient at new iterate.
    eps_fp : float
        Noise level of function at x_k + alpha * p_k.
    eps_gp : float
        Noise level of gradient at x_k + beta * p_k.
    func_evals : int
        Number of function evaluations.
    grad_evals : int
        Number of gradient evaluations.
    armijo_flag : bool
        Flag for Armijo condition success.
    wolfe_flag : bool
        Flag for Wolfe and noise control condition success.
    split_flag : bool
        Flag for when steplength and lengthening parameter are not equal.

    Authors
    -------
    Hao-Jun Michael Shi (hjmshi@u.northwestern.edu)
    Yuchen Xie (ycxie@u.northwestern.edu)
    Department of Industrial Engineering and Management Sciences
    Northwestern University

    """

    # print initial header for line search
    if verbose:
        print(
            '====================================================================================================================================================='
            )

    # initialize line search
    armijo_flag = False
    wolfe_flag = False
    split_flag = False
    ls_iter = 0
    func_evals = 0
    grad_evals = 0

    # define upper and lower brackets
    upper_bracket = np.inf
    lower_bracket = 0

    # compute f_k and g_k if necessary
    if f_k is None:
        f_k = func(x_k)
        func_evals += 1

    # compute gradient at x_k, if not available
    if g_k is None:
        g_k = grad(x_k)
        grad_evals += 1

    # compute error bounds
    if callable(eps_f):
        eps_fk = eps_f(x_k)
    else:
        eps_fk = np.copy(eps_f)

    if callable(eps_g):
        eps_gk = eps_g(x_k)
    else:
        eps_gk = np.copy(eps_g)

    # compute dot product and norm
    gtp = np.dot(g_k, p_k)
    norm_pk = np.linalg.norm(p_k)

    # initialize quantities
    g_new = np.inf
    gtp_new = np.inf

    # define both steplength and lengthening parameter
    alpha = np.copy(alpha_init)
    beta = np.copy(beta_init)

    # store best steplength, strong convexity parameter
    alpha_best = 0.
    f_best = np.inf
    eps_fbest = np.inf
    rhs = None

    # check if steplength and lengthening parameter are different
    if alpha != beta:
        split_flag = True

        # evaluate gradient
        g_new = grad(x_k + beta * p_k)
        grad_evals += 1
        gtp_new = np.dot(g_new, p_k)

    # compute initial step function value
    f_new = func(x_k + alpha * p_k)
    func_evals += 1

    if callable(eps_f):
        eps_fp = eps_f(x_k + alpha * p_k)
    else:
        eps_fp = np.copy(eps_f)

    # set sufficient decrease
    if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
        etp = eps_gk.dot(np.abs(p_k))
    else:
        etp = eps_gk * norm_pk

    # print initial header for line search
    if verbose:
        print(
            '-----------------------------------------------------------------------------------------------------------------------------------------------------'
            )
        print('Entering Initial Phase')

    # main loop
    while ls_iter < split_iter and not split_flag:

        # print header
        if verbose and ls_iter % 25 == 0:
            print(
                '-----------------------------------------------------------------------------------------------------------------------------------------------------'
                )
            print(
                '   LS Iter    |    alpha     |     F_new    |    F_old     |  (g_old)Tp   |  (g_new)Tp   |    eps_fk    |    eps_fp    |    eps_gk    |    eps_gp    '
                )
            print(
                '-----------------------------------------------------------------------------------------------------------------------------------------------------'
                )

        # enforce sufficient decrease if condition satisfied
        if gtp < -etp:
            suff_dec = c1 * alpha * gtp

        # otherwise, enforce simple decrease
        else:
            suff_dec = 0

        # relax Armijo after 1st trial point
        if ls_iter > 0:
            relax = eps_fk + eps_fp
        else:
            relax = 0

        # check Armijo condition
        if f_new > f_k + suff_dec + relax:

            # armijo failure
            armijo_flag = False

            # print result
            if verbose:
                print(
                    '  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |              |  % 8.3e  |  % 8.3e  |  % 8.3e  |              '
                    % (ls_iter, alpha, f_k, f_new, gtp, eps_fk, eps_fp, eps_gk))

            # bisect
            upper_bracket = alpha
            alpha = (lower_bracket + alpha) / 2.0
            beta = alpha

            # evaluate function
            f_new = func(x_k + alpha * p_k)
            func_evals += 1

            if callable(eps_f):
                eps_fp = eps_f(x_k + alpha * p_k)
            else:
                eps_fp = np.copy(eps_f)

        else:

            # armijo success
            armijo_flag = True

            # track best steplength
            if f_new < f_best:
                alpha_best = alpha
                f_best = f_new
                eps_fbest = eps_fp

            # evaluate gradient
            g_new = grad(x_k + beta * p_k)
            grad_evals += 1
            gtp_new = np.dot(g_new, p_k)

            if callable(eps_g):
                eps_gp = eps_g(x_k + beta * p_k)
            else:
                eps_gp = np.copy(eps_g)

            if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
                rhs = (eps_gk + eps_gp).dot(np.abs(p_k))
            else:
                rhs = (eps_gk + eps_gp) * norm_pk

            # check noise control condition
            if np.abs(gtp_new - gtp) < (1 + c3) * rhs:

                # print result
                if verbose:
                    print(
                        '  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                        % (ls_iter, alpha, f_k, f_new, gtp, gtp_new, eps_fk, eps_fp, eps_gk, eps_gp))

                # split
                split_flag = True
                beta = 2 * alpha

                continue

            # check Wolfe condition
            if gtp_new < c2 * gtp:

                # wolfe failure
                wolfe_flag = False

                # print result
                if verbose:
                    print(
                        '  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                        % (ls_iter, alpha, f_k, f_new, gtp, gtp_new, eps_fk, eps_fp, eps_gk, eps_gp))

                # bisect
                lower_bracket = alpha
                if upper_bracket == np.inf:
                    alpha = 2 * alpha
                else:
                    alpha = (upper_bracket + alpha) / 2
                beta = alpha

                # evaluate function
                f_new = func(x_k + alpha * p_k)
                func_evals += 1

                if callable(eps_f):
                    eps_fp = eps_f(x_k + alpha * p_k)
                else:
                    eps_fp = np.copy(eps_f)

            else:
                # wolfe success
                wolfe_flag = True

        # iterate
        ls_iter += 1

        # break if both conditions satisfied
        if armijo_flag and wolfe_flag:

            # print result
            if verbose:
                print(
                    '  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                    % (ls_iter, alpha, f_k, f_new, gtp, gtp_new, eps_fk, eps_fp, eps_gk, eps_gp))
            break

    # change to split mode
    if not armijo_flag or not wolfe_flag or split_flag:
        split_flag = True

        # re-initialize beta if necessary
        if rhs:
            beta_hist = (1 + c3) * rhs / (mu * norm_pk ** 2)
        else:
            if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
                beta_hist = 2 * (1 + c3) * eps_gk.dot(np.abs(p_k)) / (mu * norm_pk ** 2)
            else:
                beta_hist = 2 * (1 + c3) * eps_gk / (mu * norm_pk)
        beta = np.maximum(beta, beta_hist)

        # evaluate gradient
        g_new = grad(x_k + beta * p_k)
        grad_evals += 1
        gtp_new = np.dot(g_new, p_k)

        # perform backtracking and lengthening
        # use best alpha if available
        if f_best < np.inf:
            alpha = alpha_best
            f_new = f_best
            eps_fp = eps_fbest
            armijo_flag = True
        else:
            alpha, f_new, eps_fp, fevals_back, armijo_flag = \
                _backtracking(func, x_k, p_k, g_k, alpha=alpha, eps_f=eps_f, eps_g=eps_g, f_k=f_k, eps_fk=eps_fk,
                              eps_fp=eps_fp, eps_gk=eps_gk, gtp=gtp, f_new=f_new, etp=etp, norm_pk=norm_pk, c1=c1,
                              max_ls_iter=max_ls_iter, verbose=verbose)
            func_evals += fevals_back
        if not wolfe_flag:
            beta, g_new, gtp_new, eps_gp, fevals_length, gevals_length, wolfe_flag = \
                _lengthening(grad, x_k, p_k, beta=beta, eps_g=eps_g, g_k=g_k, eps_gk=eps_gk, eps_gp=eps_gp, gtp=gtp,
                             g_new=g_new, gtp_new=gtp_new, norm_pk=norm_pk, c3=c3, max_ls_iter=max_ls_iter,
                             verbose=verbose)
            func_evals += fevals_length
            grad_evals += gevals_length

    # if Armijo fails, then set to initial point
    if not armijo_flag:
        alpha = 0.0
        f_new = f_k
        eps_fp = eps_fk

    # if Wolfe fails, set gradient to initial point
    if not wolfe_flag:
        mu = np.inf
        beta = 0.0
        g_new = g_k
        eps_gp = eps_gk

    # compute mu if beta satisfies conditions
    else:
        mu = (gtp_new - gtp) / (beta * norm_pk ** 2)

        # print inner iterations
    if verbose and armijo_flag is True and wolfe_flag is True:
        print(
            '-----------------------------------------------------------------------------------------------------------------------------------------------------'
            )
        print('Success! Steplength:', alpha, 'Lengthening:', beta)
        print(
            '====================================================================================================================================================='
            )

    elif verbose:
        print(
            '-----------------------------------------------------------------------------------------------------------------------------------------------------'
            )
        print('Line search failed! Steplength:', alpha, 'Lengthening:', beta)
        print(
            '====================================================================================================================================================='
            )

    return alpha, beta, mu, f_new, g_new, eps_fp, eps_gp, func_evals, grad_evals, armijo_flag, wolfe_flag, split_flag


def _backtracking(func, x_k, p_k, g_k, alpha=1., eps_f=0., eps_g=0., f_k=None, eps_fk=None, eps_fp=None,
                  eps_gk=None, gtp=None, f_new=None, etp=None, norm_pk=None, c1=1e-4, max_ls_iter=20, verbose=False):
    """Backtracking Line Search

    Function for performing backtracking Armijo line search with relaxation. Designed for split phase of noise-tolerant
    two-phase Armijo-Wolfe line search. Reverts back to Armijo backtracking line search if both eps_f and eps_g = 0.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.

            ``func(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,).
    x_k : ndarray
        Current iterate with shape (n,).
    p_k : ndarray
        Search direction with shape (n,).
    g_k : ndarray
        (Noisy) gradient `grad` at current iterate x_k with shape (n,).
    alpha : float
        Initial steplength. (Default: 1)
    eps_f : {float, callable}, optional
        Noise level of function. If callable, should return float. (Default: 0)
    eps_g : {float, ndarray, callable}, optional
        Noise level of the gradient. If float or callable that returns float, assumes 2-norm bound on the gradient
        noise. If ndarray or callable that returns ndarray, eps_g must be have shape (n,) and assumes bounds on the
        noise with respect to each component.
    f_k : float, optional
        (Noisy) objective value `func` at current iterate x_k. If None, will evaluate with `func(x_k)`.
    eps_fk : float, optional
        Noise level of function at x_k. If None, will evaluate with `eps_f(x_k)`. (Default: None)
    eps_fp : float, optional
        Noise level of function at x_k + alpha * p_k. If None, will evaluate with `eps_f(x_k + alpha * p_k)`.
        (Default: None)
    eps_gk : {float, ndarray}, optional
        Noise level of gradient at x_k. If None, will evaluate with `eps_g(x_k)`. (Default: None)
    gtp : float, optional
        Directional derivative at x_k along p_k. If None, will evaluate with `g_k.dot(p_k)`. (Default: None)
    f_new : float, optional
        (Noisy) objective value `func` at iterate x_k + alpha * p_k. If None, will evaluate with
        `func(x_k + alpha * p_k)`.
    etp : float, optional
        Noise level of directional derivative along p_k. If None, will evaluate with `eps_gk * np.linalg.norm(p_k)` if
        eps_gk is a float; otherwise, `eps_gk.dot(np.abs(p_k))`. (Default: None)
    norm_pk : float, optional
        Norm of p_k. If None, will evaluate with `np.linalg.norm(p_k)`. (Default: None)
    c1 : float, optional
        Constant for Armijo condition. (Default: 1e-4)
    max_ls_iter : int, optional
        Maximum number of line search trials/iterations. (Default: 20)
    verbose : bool, optional
        Print line search information. (Default: False)

    Returns
    -------
    alpha : float
        Steplength.
    f_new : float
        Function value at new iterate x_k + alpha * p_k.
    eps_fp : float
        Noise level of function at x_k + alpha * p_k.
    func_evals : int
        Number of function evaluations.
    armijo_flag : bool
        Flag for Armijo condition success.

    Authors
    -------
    Hao-Jun Michael Shi (hjmshi@u.northwestern.edu)
    Yuchen Xie (ycxie@u.northwestern.edu)
    Department of Industrial Engineering and Management Sciences
    Northwestern University

    """

    # print initial header for lengthening
    if verbose:
        print(
            '-----------------------------------------------------------------------------------------------------------------------------------------------------'
            )
        print('Entering Split Phase: Backtracking')

    # initialize counter and flags
    func_evals = 0
    ls_iter = 0
    armijo_flag = False

    # initialize directional derivatives and function value (if necessary)
    if gtp is None:
        gtp = np.dot(g_k, p_k)
    if f_k is None:
        f_k = func(x_k)
        func_evals += 1
    if f_new is None:
        f_new = func(x_k + alpha * p_k)
        func_evals += 1
    if norm_pk is None:
        norm_pk = np.linalg.norm(p_k)

    # initialize noise levels (if necessary)
    if eps_fk is None:
        if callable(eps_f):
            eps_fk = eps_f(x_k)
        else:
            eps_fk = np.copy(eps_f)
    if eps_fp is None:
        if callable(eps_f):
            eps_fp = eps_f(x_k + alpha * p_k)
        else:
            eps_fp = np.copy(eps_f)
    if eps_gk is None:
        if callable(eps_g):
            eps_gk = eps_g(x_k)
        else:
            eps_gk = np.copy(eps_g)

    # set sufficient decrease
    if etp is None:
        if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
            etp = eps_gk.dot(np.abs(p_k))
        else:
            etp = eps_gk * norm_pk

    # enforce sufficient decrease if condition satisfied
    if gtp < -etp:
        suff_dec = c1 * alpha * gtp

    # otherwise, enforce simple decrease
    else:
        suff_dec = 0

    # relax Armijo
    relax = eps_fk + eps_fp

    # main loop
    while ls_iter < max_ls_iter:

        # print header
        if verbose and ls_iter % 25 == 0:
            print(
            '-----------------------------------------------------------------------------------------------------------------------'
            )
            print(
            '   LS Iter    |    alpha     |     F_new    |    F_old     |  (g_old)Tp   |    eps_f     |    eps_fp    |     rhs      '
            )
            print(
            '-----------------------------------------------------------------------------------------------------------------------'
            )

        # print result
        if verbose:
            print('  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                  % (ls_iter, alpha, f_new, f_k, gtp, eps_fk, eps_fp, f_k + suff_dec + relax))

        # check Armijo condition
        if f_new > f_k + suff_dec + relax:

            # backtrack if split
            alpha = alpha / 10.0
            f_new = func(x_k + alpha * p_k)
            func_evals += 1

            # reevaluate noise level
            if callable(eps_f):
                eps_fp = eps_f(x_k + alpha * p_k)
            else:
                eps_fp = np.copy(eps_f)

            # enforce sufficient decrease if condition satisfied
            if gtp < -etp:
                suff_dec = c1 * alpha * gtp

            # otherwise, enforce simple decrease
            else:
                suff_dec = 0

            # relax Armijo
            relax = eps_fk + eps_fp

            # iterate
            ls_iter += 1

        else:
            armijo_flag = True
            break

    return alpha, f_new, eps_fp, func_evals, armijo_flag


def _lengthening(grad, x_k, p_k, beta=1., eps_g=0., g_k=None, eps_gk=None, eps_gp=None, gtp=None,
                 g_new=None, gtp_new=None, norm_pk=None, c3=0.5, max_ls_iter=20, verbose=False):
    """Lengthening Procedure

    Function for performing lengthening. Designed for split phase of noise-tolerant two-phase Armijo-Wolfe line search.

    Parameters
    ----------
    grad : callable
        Method for computing the gradient vector. It should be a function that returns the gradient
        vector:

            ``grad(x) -> array_like, shape (n,)``

        where ``x`` is a 1-D array with shape (n,).
    x_k : ndarray
        Current iterate with shape (n,).
    p_k : ndarray
        Search direction with shape (n,).
    beta : float
        Initial lengthening parameter. (Default: 1)
    eps_g : {float, ndarray, callable}, optional
        Noise level of the gradient. If float or callable that returns float, assumes 2-norm bound on the gradient
        noise. If ndarray or callable that returns ndarray, eps_g must be have shape (n,) and assumes bounds on the
        noise with respect to each component.
    g_k : ndarray, optional
        (Noisy) gradient `grad` at current iterate x_k. If None, will evaluate with `grad(x_k)`.
    eps_gk : {float, ndarray}, optional
        Noise level of gradient at x_k. If None, will evaluate with `eps_g(x_k)`. (Default: None)
    eps_gp : {float, ndarray}, optional
        Noise level of gradient at x_k + beta * p_k. If None, will evaluate with `eps_g(x_k + beta * p_k)`.
        (Default: None)
    gtp : float, optional
        Directional derivative at x_k along p_k. If None, will evaluate with `g_k.dot(p_k)`. (Default: None)
    g_new : ndarray, optional
        (Noisy) objective value `func` at iterate x_k + beta * p_k. If None, will evaluate with
        `grad(x_k + beta * p_k)`. (Default: None)
    gtp_new : float, optional
        Directional derivative at x_k + beta * p_k along p_k. If None, will evaluate with `g_new.dot(p_k)`.
        (Default: None)
    norm_pk : float, optional
        Norm of p_k. If None, will evaluate with `np.linalg.norm(p_k)`. (Default: None)
    c3 : float, optional
        Constant for noise control condition. (Default: 0.5)
    max_ls_iter : int, optional
        Maximum number of line search trials/iterations. (Default: 20)
    verbose : bool, optional
        Print line search information. (Default: False)

    Returns
    -------
    beta : float
        Lengthening parameter.
    g_new : ndarray
        Gradient at new iterate.
    gtp_new : float
        Directional derivative at new iterate along p_k.
    eps_gp : float
        Noise level of gradient at new iterate.
    func_evals : int
        Number of function evaluations.
    grad_evals : int
        Number of gradient evaluations.
    wolfe_flag : bool
        Flag for Wolfe and noise control condition success.

    Authors
    -------
    Hao-Jun Michael Shi (hjmshi@u.northwestern.edu)
    Yuchen Xie (ycxie@u.northwestern.edu)
    Department of Industrial Engineering and Management Sciences
    Northwestern University

    """

    # print initial header for lengthening
    if verbose:
        print(
            '-----------------------------------------------------------------------------------------------------------------------------------------------------'
            )
        print('Entering Split Phase: Lengthening')

    # initialize counter and flags
    func_evals = 0
    grad_evals = 0
    ls_iter = 0
    wolfe_flag = False

    # initialize gradient and directional derivatives (if necessary)
    if g_k is None:
        g_k = grad(x_k)
        grad_evals += 1
    if gtp is None:
        gtp = np.dot(g_k, p_k)
    if g_new is None:
        g_new = grad(x_k + beta * p_k)
        grad_evals += 1
    if gtp_new is None:
        gtp_new = np.dot(g_new, p_k)
    if norm_pk is None:
        norm_pk = np.linalg.norm(p_k)

    # initialize noise levels (if necessary)
    if eps_gk is None:
        if callable(eps_g):
            eps_gk = eps_g(x_k)
        else:
            eps_gk = np.copy(eps_g)
    if eps_gp is None:
        if callable(eps_g):
            eps_gp = eps_g(x_k + beta * p_k)
        else:
            eps_gp = np.copy(eps_g)

    # compute rhs of noise control condition
    if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
        rhs = (eps_gk + eps_gp).dot(np.abs(p_k))
    else:
        rhs = (eps_gk + eps_gp) * norm_pk

    # main loop
    while ls_iter < max_ls_iter:

        # print header
        if verbose and ls_iter % 25 == 0:
            print(
            '--------------------------------------------------------------------------------------------------------'
            )
            print(
            '   LS Iter    |     beta     |  (g_old)Tp   |  (g_new)Tp   |    eps_g     |    eps_gp    |     rhs      '
            )
            print(
            '--------------------------------------------------------------------------------------------------------'
            )

        # print result
        if verbose:
            print('  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  |  % 8.3e  '
                  % (ls_iter, beta, gtp, gtp_new, eps_g, eps_gp, rhs))

        # check noise control condition
        if gtp_new - gtp < (1 + c3) * rhs:

            # lengthen
            beta = 2 * beta

            # evaluate gradient and gradient error
            g_new = grad(x_k + beta * p_k)
            grad_evals += 1
            gtp_new = np.dot(g_new, p_k)

            if callable(eps_g):
                eps_gp = eps_g(x_k + beta * p_k)
            else:
                eps_gp = np.copy(eps_g)

            if isinstance(eps_gk, np.ndarray) and eps_gk.ndim > 0 and eps_gk.shape[0] > 1:
                rhs = (eps_gk + eps_gp).dot(np.abs(p_k))
            else:
                rhs = (eps_gk + eps_gp) * norm_pk

            # iterate
            ls_iter += 1

        else:
            wolfe_flag = True
            break

    return beta, g_new, gtp_new, eps_gp, func_evals, grad_evals, wolfe_flag


if __name__ == '__main__':
    import scipy.optimize

    n = 2
    x0 = np.array([-1.25, 1.])

    print('Test on Rosenbrock Function with no noise with BFGS')
    func = scipy.optimize.rosen
    grad = scipy.optimize.rosen_der
    options = {'qn_hist_size': np.inf}
    x_opt, f_opt, iters, f_evals, g_evals, flag, results = bfgs_e(func, grad, x0, options=options)

    print('Test on Rosenbrock Function with no noise with L-BFGS (t = 10)')
    func = scipy.optimize.rosen
    grad = scipy.optimize.rosen_der
    options = {'qn_hist_size': 10}
    x_opt, f_opt, iters, f_evals, g_evals, flag, results = bfgs_e(func, grad, x0, options=options)

    print('Test on Rosenbrock Function with noise with BFGS')
    def func(x):
        return scipy.optimize.rosen(x) + np.random.uniform(-1e-5, 1e-5)
    def grad(x):
        return scipy.optimize.rosen_der(x) + np.random.uniform(-1e-5, 1e-5, size=x0.shape)
    options = {'qn_hist_size': np.inf}
    x_opt, f_opt, iters, f_evals, g_evals, flag, results = bfgs_e(func, grad, x0, eps_f=1e-5, eps_g=np.sqrt(2) * 1e-5, options=options)

    print('Test on Rosenbrock Function with noise with L-BFGS (t = 10)')
    def func(x):
        return scipy.optimize.rosen(x) + np.random.uniform(-1e-5, 1e-5)
    def grad(x):
        return scipy.optimize.rosen_der(x) + np.random.uniform(-1e-5, 1e-5, size=x0.shape)
    options = {'qn_hist_size': 10}
    x_opt, f_opt, iters, f_evals, g_evals, flag, results = bfgs_e(func, grad, x0, eps_f=1e-5, eps_g=np.sqrt(2) * 1e-5, options=options)
