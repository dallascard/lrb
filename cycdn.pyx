import numpy as np
from scipy.special import expit



cdef double compute_hessian_element(double C, int N, double [:] expits, double [:, ::1] X, int j):
    cdef double h = 0.0
    cdef int i = 0

    while (i < N):
        h += expits[i] * (1.0 - expits[i]) * X[i, j] * X[i, j]
        i += 1
    return h * C

