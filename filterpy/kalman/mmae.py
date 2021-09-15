# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,too-many-instance-attributes

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
from __future__ import absolute_import, division
from filterpy.common.helpers import check_input
from typing import Iterable, Optional

from copy import deepcopy
import numpy as np
from filterpy.kalman.kalman_filter import KalmanFilter
from filterpy.common import pretty_str


class MMAEFilterBank(object):
    """
    Implements the fixed Multiple Model Adaptive Estimator (MMAE). This
    is a bank of independent Kalman filters. This estimator computes the
    likelihood that each filter is the correct one, and blends their state
    estimates weighted by their likelihood to produce the state estimate.

    Parameters
    ----------

    filters : list of Kalman filters
        List of Kalman filters.

    p : list-like of floats
       Initial probability that each filter is the correct one. In general
       you'd probably set each element to 1./len(p).

    dim_x : float
        number of random variables in the state X

    H : Measurement matrix

    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : ndarray
        Last measurement used in update(). Read only.

    filters : list of Kalman filters
        List of Kalman filters.

    Examples
    --------

    ..code:
        ca = make_ca_filter(dt, noise_factor=0.6)
        cv = make_ca_filter(dt, noise_factor=0.6)
        cv.F[:,2] = 0 # remove acceleration term
        cv.P[2,2] = 0
        cv.Q[2,2] = 0

        filters = [cv, ca]
        bank = MMAEFilterBank(filters, p=(0.5, 0.5), dim_x=3)

        for z in zs:
            bank.predict()
            bank.update(z)

    Also, see my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    References
    ----------

    Zarchan and Musoff. "Fundamentals of Kalman filtering: A Practical
    Approach." AIAA, third edition.

    """

    def __init__(
        self,
        filters: Iterable[KalmanFilter],
        p: Iterable[float],
        dim_x: int,
        H: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        if len(filters) != len(p):
            raise ValueError("length of filters and p must be the same")

        if dim_x < 1:
            raise ValueError("dim_x must be >= 1")

        self.filters = filters
        self.p = np.asarray(p)
        self.dim_x = dim_x
        self.dim_z = filters[0].dim_z
        self.dim_u = filters[0].dim_u

        if H is None:
            self.H = None
        else:
            exp_shape = (self.dim_z, self.dim_x)
            assert check_input(H, exp_shape, "H")
            self.H = np.copy(H)

        if R is None:
            self.R = np.array([[0]])
        else:
            exp_shape = (self.dim_z, self.dim_z)
            assert check_input(R, exp_shape, "R")
            self.R = np.copy(R)

        # try to form a reasonable initial values, but good luck!
        try:
            self.z = np.copy(filters[0].z)
            self.x = np.copy(filters[0].x)
            self.P = np.copy(filters[0].P)

        except AttributeError:
            self.z = None
            self.x = None
            self.P = None

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u: np.ndarray = None):
        """
        Predict next position using the Kalman filter state propagation
        equations for each filter in the bank.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.
        """
        if u is None:
            u = np.array([[0]])
        dim_u = u.shape[0]
        exp_shape = (dim_u, 1)
        assert check_input(u, exp_shape, "u")

        for f in self.filters:
            f.predict(u)

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.
        """

        exp_shape = (self.dim_z, 1)
        assert check_input(z, exp_shape, "z")

        if R is None:
            R = self.R
        exp_shape = (self.dim_z, self.dim_z)
        assert check_input(R, exp_shape, "R")

        if H is None:
            H = self.H
        exp_shape = (self.dim_z, self.dim_x)
        assert check_input(H, exp_shape, "H")

        # new probability is recursively defined as prior * likelihood
        for i, f in enumerate(self.filters):
            f.update(z, R, H)
            self.p[i] *= f.likelihood

        self.p /= sum(self.p)  # normalize

        # compute estimated state and covariance of the bank of filters.
        self.P = np.zeros(self.filters[0].P.shape)

        # state can be in form [x,y,z,...] or [[x, y, z,...]].T
        is_row_vector = self.filters[0].x.ndim == 1
        if is_row_vector:
            self.x = np.zeros((self.dim_x, 1))
            for f, p in zip(self.filters, self.p):
                self.x += f.x * p
        else:
            self.x = np.zeros((self.dim_x, 1))
            for f, p in zip(self.filters, self.p):
                self.x += f.x * p

        for x, f, p in zip(self.x, self.filters, self.p):
            y = f.x - x
            self.P += p * y @ y.T + f.P

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def __repr__(self):
        return "\n".join(
            [
                "MMAEFilterBank object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("log-p", self.p),
            ]
        )
