from optparse import OptionParser

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import cycdn


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    n = 1000
    p = 6
    np.random.seed(42)
    X = np.array(np.random.randint(low=0, high=2, size=(n, p)), dtype=np.float64)
    beta = np.array(np.random.randn(p), dtype=np.float64) * np.random.randint(low=0, high=2, size=p)
    print(beta)
    ps = expit(np.dot(X, beta))
    y = np.random.binomial(p=ps, n=1, size=n)

    model = LogisticRegression(C=1.0, penalty='l1', fit_intercept=False)
    model.fit(X, y)
    print(model.coef_)
    pred = model.predict(X)
    print(np.sum(np.abs(y - pred)) / float(n))
    #print(model.predict_proba(X))

    y_float = np.array(y, dtype=np.float64)
    y_float[y == 0] = -1.0

    solver = cycdn.CDN(C=1.0)
    w = np.array(model.coef_)
    w = w.reshape((p, ))
    solver.fit(X, y_float, randomize=True)
    print(solver.get_w())

    pred_probs = solver.pred_probs(X)
    pred = np.argmax(pred_probs, axis=1)
    print(np.sum(np.abs(y - pred)) / float(n))


    #print(model.coef_)
    #w = np.array(model.coef_)
    #w = w.reshape((p, ))
    #solver.w = w
    #solver.test(X, y)
    #print(solver.w)

if __name__ == '__main__':
    main()


