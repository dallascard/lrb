from optparse import OptionParser

import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

import lrb


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--elim', action="store_true", dest="elimination", default=False,
                      help='Do heuristic variable elimination: default=%default')
    parser.add_option('--lower', dest='lower', default=None,
                      help='Lower limit for weights: default=%default')
    parser.add_option('--upper', dest='upper', default=None,
                      help='Upper limit for weights: default=%default')
    parser.add_option('--intercept', action="store_true", dest="intercept", default=False,
                      help='Fit an intercept: default=%default')
    parser.add_option('--tol', dest='tol', default=1e-5,
                      help='Tolerance for convergence (relative change in sum(abs(coef_))): default=%default')
    parser.add_option('--max_iter', dest='max_iter', default=200,
                      help='Maximum number of iterations: default=%default')
    parser.add_option('-n', dest='n', default=1000,
                      help='Number of instances: default=%default')
    parser.add_option('-p', dest='p', default=50,
                      help='Number of features: default=%default')
    parser.add_option('--sparsity_X', dest='sparsity_X', default=0.5,
                      help='Expected proportion of zero entries in X: default=%default')
    parser.add_option('--sparsity_beta', dest='sparsity_beta', default=0.5,
                      help='Expected proportion of zero entries in beta: default=%default')
    parser.add_option('--weights', action="store_true", dest="weights", default=False,
                      help='Generate random sample weights: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')
    parser.add_option('--skl', action="store_true", dest="skl", default=False,
                      help='Use sklearn implementation: default=%default')
    parser.add_option('--both', action="store_true", dest="both", default=False,
                      help='Run both implementations and compare: default=%default')
    parser.add_option('-v', dest='verbose', default=2,
                      help='Verbosity level: default=%default')

    (options, args) = parser.parse_args()

    do_elimination = options.elimination
    lower = options.lower
    if lower is not None:
        lower = float(lower)
    upper = options.upper
    if upper is not None:
        upper = float(upper)
    print(lower, upper)
    fit_intercept = options.intercept
    tol = float(options.tol)
    max_iter = int(options.max_iter)
    n = int(options.n)
    p = int(options.p)
    sparsity_X = float(options.sparsity_X)
    sparsity_beta = float(options.sparsity_beta)
    weights = options.weights
    seed = options.seed
    use_skl = options.skl
    use_both = options.both
    verbose = int(options.verbose)

    if seed is not None:
        np.random.seed(int(seed))

    #X = np.array(np.random.randint(low=0, high=2, size=(n, p)), dtype=np.float64)
    X = np.array(np.random.binomial(p=1-sparsity_X, n=1, size=(n, p)), dtype=np.float64)
    beta_mask = np.array(np.random.binomial(p=1-sparsity_beta, n=1, size=p), dtype=np.float64)
    beta = np.array(np.random.randn(p), dtype=np.float64) * beta_mask
    if fit_intercept:
        bias = np.random.randn()
    else:
        bias = 0
    if verbose > 0:
        print(beta)
        print(bias)

    ps = expit(np.dot(X, beta) + bias)
    y = np.random.binomial(p=ps, n=1, size=n)

    X = sparse.csc_matrix(X)

    if weights:
        sample_weights=np.random.rand(n) + 0.5
    else:
        sample_weights=None

    if use_skl or use_both:
        model = LogisticRegression(C=1.0, penalty='l1', fit_intercept=fit_intercept, solver='liblinear', tol=tol, max_iter=max_iter, verbose=verbose)
        model.fit(X, y, sample_weight=sample_weights)
        if verbose > 0:
            print()
            print(model.coef_)
            print(model.intercept_)
        pred = model.predict(X)
        if verbose > 0:
            print(np.sum(np.abs(y - pred)) / float(n))

    if (not use_skl) or use_both:
        y2 = y.copy()
        y2[y == 0] = -1

        solver = lrb.LogisticRegressionBounded(C=1.0, fit_intercept=fit_intercept, lower=lower, upper=upper, do_elimination=do_elimination)
        #solver.fit(X, y2, tol=1e-4, init_w=model.coef_[0], min_epochs=0, max_epochs=200, randomize=False, verbose=verbose)
        solver.fit(X, y2, sample_weight=sample_weights, tol=tol, max_epochs=max_iter, randomize=True, verbose=verbose)
        if verbose > 0:
            print(solver.coef_)
            print(solver.intercept_)

        pred_probs = solver.predict_proba(X)
        pred = np.argmax(pred_probs, axis=1)
        if verbose > 0:
            print(np.sum(np.abs(y - pred)) / float(n))

    if use_both:
        diff = np.abs(model.coef_[0] - solver.coef_[0])
        print("Maximum weight difference from skl:", np.max(diff))
        print("Intercept diff = ", np.abs(model.intercept_ - solver.intercept_))
        print(model.predict(np.zeros((1, p))))
        print(solver.predict(np.zeros((1, p))))


if __name__ == '__main__':
    main()
