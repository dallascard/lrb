## Logistic Regression with bounded coefficients

This repo implements a projected coordinate-descent Newton (CDN) solver for L1-regularized logistic regression in Cython. 

The basic CDN algorithm is presented in Yuan et al (2010) and Yuan et al (2012). The only modification here is to add projection to allow for imposing bounds on the model coefficients, such as constraining them to be positive. 

The API closely resembles the scikit-learn implementation of [Logistic Regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), which calls [liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/), which in turn uses newGLMNET (Yuan et al, 2012). This cython implementation of CDN is perhaps 10 times slower than scikit-learn's call to liblinear, but it allows for placing constraints on the weights, which was an option in the original [GLMNET](https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html) package, but not in liblinear or scikit-learn. 

### Requirements

* numpy
* scipy 
* cython

### Installation

To use this code, first compile it with cython by using

`> python setup.py build_ext --inplace`

You can then run a test using `test_lrb.py` (add `--both` to compare to the scikit-learn results or `-h` to see more options).

### Usage

To embed in python code, the API is essentially the same as scikit-learn, e.g.

```
import lrb
model = lrb.LogisticRegressionBounded()
model.fit(X, y)
predictions = model.predict(X)
```

To add bounds on the weights, use the `lower` or `upper` keywords, e.g.

```
model = lrb.LogisticRegression(lower=0)
```

Note that at the moment, the intercept is being regularized (as is the case in liblinear and scikit-learn). It is also being restricted by the bounds in the same way as the coefficients. I may change one or both of these in the future...

### References


* Yuan et al. A Comparison of Optimization Methods and Software for
Large-scale L1-regularized Linear Classification. JMLR 11 (2010).
* Yuan et al. An Improved GLMNET for L1-regularized Logistic
Regression. JLMR 13 (2012).