from optparse import OptionParser

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

class CDN:

    def __init__(self, C=1.0, beta=0.5, sigma=0.01, lower=None, upper=None):
        self._C = C
        self._beta = beta
        self._sigma = sigma
        self._w = None
        self._R = None
        self._exp_nyXw = None
        self._expits = None
        self._f_val = None
        self._g = None
        self._H = None
        self._lower = lower
        self._upper = upper

    def fit(self, X, y, tol=1e-5, min_epochs=2, max_epochs=200, init_w=None, verbose=0, randomize=False):
        """
        Coordinate descent with Newton directions for L1-regularized logistic regression
        :param X: n x p feature matrix
        :param y: vector of labels in {-1, +1}
        :param max_iter:
        :return:
        """

        n_items, n_features = X.shape
        yX = y.reshape((n_items, 1)) * X
        if init_w is None:
            #self._w = np.random.randn(n_features)
            self._w = np.zeros(n_features)
        else:
            self._w = init_w
        self._R = np.sum(np.abs(self._w))
        self._expits = expit(np.dot(yX, self._w))
        self._exp_nyXw = np.exp(-np.dot(yX, self._w))
        self._f_val = self._compute_f(self._exp_nyXw, self._w)
        self._g = self._compute_gradients(yX)

        for k in range(max_epochs):
            delta, ls_steps = self._update(yX, randomize=randomize)
            w_sum = np.sum(np.abs(self._w))
            if w_sum > 0:
                rel_change = delta / w_sum
            else:
                rel_change = 0.0
            if verbose > 1:
                print("epoch %d, delta=%0.5f, rel_change=%0.5f, ls_steps=%d" % (k, delta, rel_change, ls_steps))
            if rel_change < tol and k >= min_epochs - 1:
                if verbose > 0:
                    print("relative change below tolerance; stoppping after %d epochs" % k)
                return

        if verbose > 0:
            print("Maximum epochs exceeded; stopping after %d epochs" % k)

    def pred_probs(self, X):
        n, p = X.shape
        probs = np.zeros([n, 2])
        prob_pos = expit(np.dot(X, self._w))
        probs[:, 1] = prob_pos
        probs[:, 0] = 1.0 - prob_pos
        return probs

    def _update(self, yX, randomize=False):
        n_items, n_features = yX.shape
        running_abs_change = 0
        running_ls_steps = 0
        order = np.arange(n_features)
        if randomize:
            np.random.shuffle(order)
        for j in order:
            change, ls_steps = self._update_one_coordinate(yX, j)
            running_abs_change += np.abs(change)
            running_ls_steps += ls_steps
        return running_abs_change, running_ls_steps

    def _update_one_coordinate(self, yX, j):
        h = self._compute_hessian_element(yX, j)

        if self._g[j] + 1.0 <= h * self._w[j]:
            d = -(self._g[j] + 1.0) / h
        elif self._g[j] - 1.0 >= h * self._w[j]:
            d = -(self._g[j] - 1.0) / h
        else:
            d = -self._w[j]

        a = 1.0  # alternate name for lambda
        i = 0
        if np.abs(d) > 0:
            # check upper and lower limits, and set max step accordingly
            if self._lower is not None:
                if self._w[j] + d < self._lower:
                    diff = self._lower - self._w[j]
                    a = diff / float(d)
            if self._upper is not None:
                if self._w[j] + d > self._upper:
                    diff = self._upper - self._w[j]
                    a = diff / float(d)

            # unless we've hit a bound, use line search to find how far to move in this direction
            if a > 0:
                # set up the threshold for convergence
                thresh = self._sigma * (self._g[j] * d + np.abs(self._w[j] + d) - np.abs(self._w[j]))
                # remove the current weight from the stored 1-norm of weights
                R_minus_w_j = self._R - np.abs(self._w[j])
                # do line search
                f_new, a, i, exp_nyXw = self._line_search(yX[:, j], j, d, self._w[j], R_minus_w_j, a, thresh)
                # store the updated values
                self._f_val = f_new
                self._w[j] += a * d
                # add the 1-norm of the new weight back into the stored sum
                self._R = R_minus_w_j + np.abs(self._w[j])
                self._exp_nyXw = exp_nyXw
                # recompute the stored probabilities and gradient
                self._expits = self._compute_probs(yX)
                self._g = self._compute_gradients(yX)
        return a * d, i

    def _line_search(self, yX_j, j, d, prev_w_j, base_R, a, thresh):
        i = 0

        step = a * d
        w_j = prev_w_j + step
        exp_nyXw = self._exp_nyXw * np.exp(-step * yX_j)

        L = self._compute_L(exp_nyXw)
        R = base_R + np.abs(w_j)
        f_new = L + R
        while f_new - self._f_val > a * thresh:
            a *= self._beta
            step = a * d
            w_j = prev_w_j + step
            exp_nyXw = self._exp_nyXw * np.exp(-step * yX_j)
            L = self._compute_L(exp_nyXw)
            R = base_R + np.abs(w_j)
            f_new = L + R
            i += 1
        return f_new, a, i, exp_nyXw

    def _compute_probs(self, yX):
        return expit(np.dot(yX, self._w))

    def _compute_f(self, exp_nyXw, w):
        return self._compute_L(exp_nyXw) + self._compute_R(w)

    def _compute_L(self, exp_nyXw):
        return self._C * np.sum(np.log(1.0 + exp_nyXw), axis=0)

    def _compute_R(self, w):
        return np.sum(np.abs(w))

    def _compute_gradients(self, yX):
        return self._C * np.dot((self._expits - 1.0), yX)

    def _compute_hessian_element(self, yX, j):
        return self._C * np.dot(yX[:, j] ** 2, self._expits * (1.0 - self._expits))

    def get_w(self):
        return self._w.copy()


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n', default=1000,
                      help='Number of instances: default=%default')
    parser.add_option('-p', dest='p', default=50,
                      help='Number of features: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed: default=%default')
    parser.add_option('--skl', action="store_true", dest="skl", default=False,
                      help='Use sklearn implementation: default=%default')
    parser.add_option('-v', dest='verbose', default=0,
                      help='Verbosity level: default=%default')

    (options, args) = parser.parse_args()

    seed = options.seed
    use_skl = options.skl
    n = int(options.n)
    p = int(options.p)
    verbose = int(options.verbose)

    if seed is not None:
        np.random.seed(int(seed))

    #X = np.array(np.random.randint(low=0, high=2, size=(n, p)), dtype=np.float64)
    X = np.random.randn(n, p)
    beta = np.array(np.random.randn(p), dtype=np.float64) * np.random.randint(low=0, high=2, size=p)
    print(beta)

    X2 = X**2
    beta2 = np.array(np.random.randn(p), dtype=np.float64) * np.random.randint(low=0, high=2, size=p)
    ps = expit(np.dot(X, beta) + np.dot(X2, beta2))

    #ps = expit(np.dot(X, beta))
    y = np.random.binomial(p=ps, n=1, size=n)

    if use_skl:
        model = LogisticRegression(C=1.0, penalty='l1', fit_intercept=False)
        model.fit(X, y)
        print(model.coef_)
        pred = model.predict(X)
        print(np.sum(np.abs(y - pred)) / float(n))

    else:
        y_float = np.array(y, dtype=np.float64)
        y_float[y == 0] = -1.0

        solver = CDN(C=1.0)
        #w = np.array(model.coef_)
        #w = w.reshape((p, ))
        solver.fit(X, y_float, max_epochs=200, randomize=True, verbose=verbose)
        print(solver.get_w())

        pred_probs = solver.pred_probs(X)
        pred = np.argmax(pred_probs, axis=1)
        print(np.sum(np.abs(y - pred)) / float(n))



if __name__ == '__main__':
    main()
