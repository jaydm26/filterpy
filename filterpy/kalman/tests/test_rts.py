# -*- coding: utf-8 -*-
"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter


DO_PLOT = False


def test_rts():
    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([[-1.0, 1.0]]).T  # initial state (location and velocity)

    fk.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition matrix

    fk.H = np.array([[1.0, 0.0]])  # Measurement function
    fk.P = np.eye(fk.dim_x) * 0.01  # covariance matrix
    fk.R = np.eye(fk.dim_z) * 5  # state uncertainty
    fk.Q = np.eye(fk.dim_x) * 0.0001  # process uncertainty

    zs = [np.array([[t + random.randn()]]) * 4 for t in range(40)]

    mu, cov, _, _ = fk.batch_filter(zs)
    mus = [x[0] for x in mu]

    M, P, _, _ = fk.rts_smoother(mu, cov)

    if DO_PLOT:
        (p1,) = plt.plot(zs, "cyan", alpha=0.5)
        (p2,) = plt.plot(M[:, 0], c="b")
        (p3,) = plt.plot(mus, c="r")
        (p4,) = plt.plot([0, len(zs)], [0, len(zs)], "g")  # perfect result
        plt.legend([p1, p2, p3, p4], ["measurement", "RKS", "KF output", "ideal"], loc=4)

        plt.show()


if __name__ == "__main__":
    DO_PLOT = False
    test_rts()
