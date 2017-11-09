import sys
from optparse import OptionParser

import numpy as np
from scipy import sparse
from scipy.special import expit

from libc.math cimport exp
from libc.math cimport log
from libc.math cimport abs as c_abs


class LogisticRegressionBounded:

    def __init__(self, C=1.0, fit_intercept=False, lower=None, upper=None, do_elimination=True):
        self._C = C
        if lower is None:
            self._lower = -np.inf
        else:
            self._lower = lower
        if upper is None:
            self._upper = np.inf
        else:
            self._upper = upper
        self._fit_intercept = fit_intercept
        self._do_elimination = do_elimination
        self._w = None          # model weights
        self.coef_ = None
        self.intercept_ = 0.0        
        self._L = None          # loss
        self._R = None          # regularization penalty 
        self._exp_nyXw = None   # stored vector of exp(-yXw) values
        self._probs = None      # stored vector of 1/(1+exp(-yXw)) values        
        self._active = None     # vector of active variables
        self._v = None          
        self._M = 0

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_

    def fit(self, X, y, sample_weight=None, beta=0.9, sigma=0.01, tol=1e-5, min_epochs=2, max_epochs=200, init_w=None, verbose=0, randomize=False):
        """
        Coordinate descent with Newton directions for L1-regularized logistic regression
        :param X: n x p feature matrix
        :param y: vector of labels in {0, 1}
        :param max_iter:
        :return:
        """

        n_items, n_features = X.shape
        assert sparse.issparse(X)

        # add an intercept term if desired
        if self._fit_intercept:
            X = sparse.hstack([np.ones((n_items, 1)), X])
            n_features += 1

        if sample_weight is None:
            sample_weights = np.ones(n_items)
        else:
            sample_weights = sample_weight
            assert len(sample_weights) == n_items

        # change labels to {-1, +1}
        y = np.array(y, dtype=np.int32)        
        y[y==0] = -1
        # premultiply y * X
        yX = X.multiply(y.reshape((n_items, 1))).tocsc()

        # convert sparse matrix to a set of vectors and indices
        yX_j_starts = np.zeros(n_features, dtype=np.int32)
        yX_j_lengths = np.zeros(n_features, dtype=np.int32)
        yX_rows = []
        yX_vals = []
        index = 0
        for j in range(n_features):                
            yX_j_coo = yX[:, j].tocoo()
            yX_j_length = len(yX_j_coo.data)
            yX_j_starts[j] = index
            yX_j_lengths[j] = yX_j_length
            for row, val in zip(yX_j_coo.row, yX_j_coo.data):
                yX_rows.append(row)
                yX_vals.append(val)
            index += yX_j_length
        yX_rows = np.array(yX_rows, dtype=np.int32)
        yX_vals = np.array(yX_vals)

        # initialize coefficients
        if init_w is None:
            self._w = np.zeros(n_features)
        else:
            self._w = init_w.copy()
            assert len(self._w) == n_features

        # initialize all remaining variables
        self._v = np.zeros(n_features)
        self._active = np.ones(n_features, dtype=np.int32)
        self._M = 0
        self._exp_nyXw = np.exp(-yX.dot(self._w))
        self._probs = 1.0 / (1.0 + self._exp_nyXw)
        self._R = np.sum(np.abs(self._w))
        self._L = self._C * np.sum(sample_weights * np.log(1.0 + self._exp_nyXw))

        order = np.array(np.arange(n_features), dtype=np.int32)

        for k in range(max_epochs):
            if randomize:
                np.random.shuffle(order)

            delta, ls_steps, L, R = sparse_update(n_items, n_features, self._C, beta, sigma, self._L, self._R, self._probs, self._exp_nyXw, self._w, self._lower, self._upper, yX_j_starts, yX_j_lengths, yX_rows, yX_vals, sample_weights, order, self._M, self._v, self._active)
            self._L = L
            self._R = R

            # update the threshold for eliminating variables on the next iteration
            if self._do_elimination and k > 0:
                M = np.max(self._v / k)

            w_sum = np.sum(np.abs(self._w))
            if w_sum > 0:
                rel_change = delta / w_sum
            else:
                rel_change = 0.0
            if verbose > 1:
                print("epoch %d, delta=%0.5f, rel_change=%0.5f, ls_steps=%d" % (k, delta, rel_change, ls_steps))
            if rel_change < tol and k >= min_epochs - 1:
                if verbose > 0:
                    print("relative change below tolerance; stopping after %d epochs" % k)
                break

        if k == max_epochs:
            print("Maximum epochs exceeded; stopping after %d epochs" % k)

        if self._fit_intercept:
            self.intercept_ = self._w[0]
            self.coef_ = self._w[1:].reshape((1, n_features-1))
        else:
            self.intercept_ = 0
            self.coef_ = self._w.reshape((1, n_features))


    def predict(self, X):
        return np.array(X.dot(self.coef_[0]) + self.intercept_ > 0, dtype=int)

    def predict_proba(self, X):
        n_items, n_features = X.shape
        if self._fit_intercept:
            X = sparse.hstack([np.ones((n_items, 1)), X])
            n_features += 1
        probs = np.zeros([n_items, 2])
        prob_pos = expit(X.dot(self._w))
        probs[:, 1] = prob_pos
        probs[:, 0] = 1.0 - prob_pos
        return probs


cdef sparse_update(int n_items, int n_features, double C, double beta, double sigma, double L, double R, double[:] probs, double[:] exp_nyXw, double[:] w, double lower, double upper, int[:] yX_j_starts, int[:] yX_j_lengths, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, int[:] order, double M, double[:] v, int[:] active):

        cdef int i = 0
        cdef int j
        cdef double running_abs_change = 0
        cdef int running_ls_steps = 0
        cdef double[:] exp_nyXw_new = np.zeros(n_items)
        cdef int ls_steps
        cdef double change
        cdef int is_active
        while i < n_items:
            exp_nyXw_new[i] = exp_nyXw[i]
            i += 1

        i = 0
        while i < n_features:
            j = order[i]
            if active[j] > 0:
                ls_steps, change, L, R, v_j, is_active = sparse_update_one_coordinate(C, beta, sigma, L, R, probs, exp_nyXw, exp_nyXw_new, w[j], lower, upper, yX_j_starts[j], yX_j_lengths[j], yX_rows, yX_vals, sample_weights, M)
                if c_abs(change) > 0:
                    w[j] += change
                v[j] = v_j
                active[j] = is_active

                running_abs_change += np.abs(change)
                running_ls_steps += ls_steps
            i += 1

        return running_abs_change, running_ls_steps, L, R



cdef sparse_update_one_coordinate(double C, double beta, double sigma, double L, double R, double[:] probs, double[:] exp_nyXw, double[:] exp_nyXw_new, double w_j, double lower, double upper, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double M):

    cdef int index
    cdef int i
    cdef double f_val = L + R
    cdef double thresh

    cdef double d  # base Newton step
    cdef double a = 1.0  # my name for lambda
    cdef double line_steps = 0
    cdef double v_j
    cdef int active = 1
    cdef double base_L = L
    cdef double base_R = R

    # compute the gradient and Hessian elements
    cdef double g = compute_grad_j(C, probs,  yX_j_start, yX_j_length, yX_rows, yX_vals, sample_weights)
    cdef double h = compute_hessian_element(C, probs, yX_j_start, yX_j_length, yX_rows, yX_vals ,sample_weights)

    if M > 0:
        # if w is 0 and the gradient is small, eliminate the variable from the active set
        if w_j == 0 and -1 + M < g < 1 - M:
            #print("Eliminating %d" % j)                
            active = 0
            v_j = 0
            a = 0.0
            print("Eliminating a variable")
            return line_steps, a, L, R, v_j, active

    # compute a new value for updating self._M
    if w_j > 0:
        v_j = c_abs(g + 1)
    elif w_j < 0:
        v_j = c_abs(g - 1)
    else:
        v_j = g - 1
        if -1 - g > v_j:
            v_j = -1 - g
        if 0 > v_j:
            v_j = 0

    # do soft-thresholding
    if g + 1.0 <= h * w_j:
        d = -(g + 1.0) / h
    elif g - 1.0 >= h * w_j:
        d = -(g - 1.0) / h
    else:
        d = -w_j

    # check upper and lower limits, and set max step accordingly
    if w_j + d < lower:
        diff = lower - w_j
        a = diff / d
    if w_j + d > upper:
        diff = upper - w_j
        a = diff / d

    # unless we've hit a bound, use line search to find how far to move in this direction
    if a > 0 and c_abs(d) > 0:
        # set up the threshold for convergence
        thresh = sigma * (g * d + c_abs(w_j + d) - c_abs(w_j))

        # remove the current weight from the stored 1-norm of weights
        base_R = base_R - c_abs(w_j)

        # also remove the influence of the relevant parts of exp(-yXw)
        i = yX_j_start
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            base_L = base_L - C * sample_weights[index] * log(1.0 + exp_nyXw[index])
            i += 1

        # do line search
        #f_new, a, i, exp_nyXw = self._line_search(yX_j, d, self._w[j], R_minus_w_j, a, thresh)
        line_steps = sparse_line_search(C, g, h, f_val, exp_nyXw, exp_nyXw_new, yX_j_start, yX_j_length, yX_rows, yX_vals, sample_weights, d, w_j, base_L, base_R, a, beta, thresh)
        a = a * (beta ** line_steps)

        w_j += a * d

        # update the objective pieces        
        base_R = base_R + c_abs(w_j + a * d)

        i = yX_j_start
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            base_L = base_L + C * sample_weights[index] * log(1.0 + exp_nyXw_new[index])
            i += 1
            # also update the relevant values of exp(-yXw) and 1/(1+exp(-yXw))
            exp_nyXw[index] = exp_nyXw_new[index]
            probs[index] = 1.0 / (1.0 + exp_nyXw_new[index])

    return line_steps, a * d, base_L, base_R, v_j, active

cdef double compute_grad_j(double C, double[:] probs, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights):
    cdef double g = 0.0
    cdef int i = yX_j_start
    cdef int index
    while (i < yX_j_start + yX_j_length):
        index = yX_rows[i]
        g += sample_weights[index] * yX_vals[i] * (probs[index] - 1.0)
        i += 1
    return g * C

cdef double compute_hessian_element(double C, double[:] probs, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights):
    cdef double h = 0.0
    cdef int i = yX_j_start
    cdef int index
    while (i < yX_j_start + yX_j_length):
        index = yX_rows[i]
        h += sample_weights[index] * probs[index] * (1.0 - probs[index]) * yX_vals[i] * yX_vals[i]
        i += 1
    return h * C

cdef double sparse_line_search(double C, double g, double h, double f_val, double[:] exp_nyXw_orig, double[:] exp_nyXw, int yX_j_start, int yX_j_length, int[:] yX_rows, double[:] yX_vals, double[:] sample_weights, double d, double prev_w_j, double base_L, double base_R, double a, double beta, double thresh):
    cdef line_steps = 0
    cdef double orig_a = a
    cdef double step = a * d
    cdef double w_j = prev_w_j + step

    cdef double L = base_L
    cdef double R = base_R

    cdef int index
    cdef int i = yX_j_start
    while i < yX_j_start + yX_j_length:
        index = yX_rows[i]
        exp_nyXw[index] = exp_nyXw_orig[index] * exp(-step * yX_vals[i])
        L = L + C * sample_weights[index] * log(1.0 + exp_nyXw[index])
        i += 1

    R = base_R + abs(w_j)
    cdef double f_new = L + R
    cdef int count = 0
    # check for convergence (and also set an upper limit)
    while f_new - f_val > a * thresh and count < 2000:
        line_steps += 1        
        a = a * beta
        step = a * d
        w_j = prev_w_j + step

        i = yX_j_start
        L = base_L
        while i < yX_j_start + yX_j_length:
            index = yX_rows[i]
            exp_nyXw[index] = exp_nyXw_orig[index] * exp(-step * yX_vals[i])
            L = L + C * sample_weights[index] * log(1.0 + exp_nyXw[index])
            i += 1

        R = base_R + abs(w_j)
        f_new = L + R
        count += 1

    return line_steps
