# -*- coding: utf-8 -*-
# pylint: disable=too-many-instance-attributes, too-many-locals, invalid-name
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
from typing import Iterable, List, Optional

import numpy as np
from scipy.linalg import inv
from filterpy.common import pretty_str, check_input


class FixedLagSmoother(object):
    """Fixed Lag Kalman smoother.

    Computes a smoothed sequence from a set of measurements based on the
    fixed lag Kalman smoother. At time k, for a lag N, the fixed-lag smoother
    computes the state estimate for time k-N based on all measurements made
    between times k-N and k. This yields a pretty good smoothed result with
    O(N) extra computations performed for each measurement. In other words,
    if N=4 this will consume about 5x the number of computations as a
    basic Kalman filter. However, the loops contain only 3 dot products, so it
    will be much faster than this sounds as the main Kalman filter loop
    involves transposes and inverses, as well as many more matrix
    multiplications.

    Implementation based on Wikipedia article as it existed on
    November 18, 2014.


    Examples
    --------

    .. code-block:: Python

        from filterpy.kalman import FixedLagSmoother
        fls = FixedLagSmoother(dim_x=2, dim_z=1)

        fls.x = np.array([[0.],
                          [.5]])

        fls.F = np.array([[1.,1.],
                          [0.,1.]])

        fls.H = np.array([[1.,0.]])

        fls.P *= 200
        fls.R *= 5.
        fls.Q *= 0.001

        zs = [...some measurements...]
        xhatsmooth, xhat = fls.smooth_batch(zs, N=4)


    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    Wikipedia http://en.wikipedia.org/wiki/Kalman_filter#Fixed-lag_smoother

    Simon, Dan. "Optimal State Estimation," John Wiley & Sons pp 274-8 (2006).

    |
    |

    """

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 1, N: Optional[int] = None):
        """Create a fixed lag Kalman filter smoother. You are responsible for
        setting the various state variables to reasonable values; the defaults
        below will not give you a functional filter.

        Parameters
        ----------

        dim_x : int
            Number of state variables for the Kalman filter. For example, if
            you are tracking the position and velocity of an object in two
            dimensions, dim_x would be 4.

            This is used to set the default size of P, Q, and u

        dim_z : int
            Number of of measurement inputs. For example, if the sensor
            provides you with position in (x,y), dim_z would be 2.

        N : int, optional
            If provided, the size of the lag. Not needed if you are only
            using smooth_batch() function. Required if calling smooth()
        """

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.N = N

        self.__x = np.zeros((dim_x, 1))  # state
        self.__x_s = np.zeros((dim_x, 1))  # smoothed state
        self.__P = np.eye(dim_x)  # uncertainty covariance
        self.__Q = np.eye(dim_x)  # process uncertainty
        self.__F = np.eye(dim_x)  # state transition matrix
        self.__H = np.eye(dim_z, dim_x)  # Measurement function
        self.__R = np.eye(dim_z)  # state uncertainty
        self.K = np.zeros((dim_x, 1))  # kalman gain
        self.__y = np.zeros((dim_z, 1))
        self.__B = np.zeros((dim_x, 1))
        self.S = np.zeros((dim_z, dim_z))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self.count = 0

        if N is not None:
            self.xSmooth = []

    @property
    def x(self):  # pylint: disable=missing-function-docstring
        return self.__x

    @x.setter
    def x(self, vec: np.ndarray):
        exp_shape = (self.dim_x, 1)
        if check_input(vec, exp_shape, "x"):
            self.__x = vec

    @property
    def x_s(self):  # pylint: disable=missing-function-docstring
        return self.__x_s

    @x_s.setter
    def x_s(self, vec: np.ndarray):
        exp_shape = (self.dim_x, 1)
        if check_input(vec, exp_shape, "x_s"):
            self.__x_s = vec

    @property
    def P(self):  # pylint: disable=missing-function-docstring
        return self.__P

    @P.setter
    def P(self, mat: np.ndarray):
        exp_shape = (self.dim_x, self.dim_x)
        if check_input(mat, exp_shape, "P"):
            self.__P = mat

    @property
    def Q(self):  # pylint: disable=missing-function-docstring
        return self.__Q

    @Q.setter
    def Q(self, mat: np.ndarray):
        exp_shape = (self.dim_x, self.dim_x)
        if check_input(mat, exp_shape, "Q"):
            self.__Q = mat

    @property
    def B(self):  # pylint: disable=missing-function-docstring
        return self.__B

    @B.setter
    def B(self, mat: np.ndarray):
        if mat is not None:
            exp_shape = (self.dim_x, self.dim_u)
            if check_input(mat, exp_shape, "B"):
                self.__B = mat
        else:
            self.__B = mat

    @property
    def F(self):  # pylint: disable=missing-function-docstring
        return self.__F

    @F.setter
    def F(self, mat: np.ndarray):
        exp_shape = (self.dim_x, self.dim_x)
        if check_input(mat, exp_shape, "F"):
            self.__F = mat

    @property
    def H(self):  # pylint: disable=missing-function-docstring
        return self.__H

    @H.setter
    def H(self, mat: np.ndarray):
        exp_shape = (self.dim_z, self.dim_x)
        if check_input(mat, exp_shape, "H"):
            self.__H = mat

    @property
    def R(self):  # pylint: disable=missing-function-docstring
        return self.__R

    @R.setter
    def R(self, mat: np.ndarray):
        exp_shape = (self.dim_z, self.dim_z)
        if check_input(mat, exp_shape, "R"):
            self.__R = mat

    @property
    def M(self):  # pylint: disable=missing-function-docstring
        return self.__M

    @M.setter
    def M(self, mat: np.ndarray):
        exp_shape = (self.dim_x, self.dim_z)
        if check_input(mat, exp_shape, "M"):
            self.__M = mat

    @property
    def z(self):  # pylint: disable=missing-function-docstring
        return self.__z

    @z.setter
    def z(self, vec: np.ndarray):
        exp_shape = (self.dim_z, 1)
        if check_input(vec, exp_shape, "z"):
            self.__z = vec

    @property
    def y(self):  # pylint: disable=missing-function-docstring
        return self.__y

    @y.setter
    def y(self, vec: np.ndarray):
        exp_shape = (self.dim_z, 1)
        if check_input(vec, exp_shape, "y"):
            self.__y = vec

    def smooth(self, z: np.ndarray, u: Optional[np.ndarray] = None):
        """Smooths the measurement using a fixed lag smoother.

        On return, self.xSmooth is populated with the N previous smoothed
        estimates,  where self.xSmooth[k] is the kth time step. self.x
        merely contains the current Kalman filter output of the most recent
        measurement, and is not smoothed at all (beyond the normal Kalman
        filter processing).

        self.xSmooth grows in length on each call. If you run this 1 million
        times, it will contain 1 million elements. Sure, we could minimize
        this, but then this would make the caller's code much more cumbersome.

        This also means that you cannot use this filter to track more than
        one data set; as data will be hopelessly intermingled. If you want
        to filter something else, create a new FixedLagSmoother object.

        Parameters
        ----------

        z : ndarray or scalar
            measurement to be smoothed


        u : ndarray, optional
            If provided, control input to the filter
        """

        if u is not None:
            exp_shape = (self.dim_u, 1)
            assert check_input(u, exp_shape, "u")

        exp_shape = (self.dim_z, 1)
        assert check_input(z, exp_shape, "z")

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        B = self.B
        N = self.N

        k = self.count

        # predict step of normal Kalman filter
        x_pre = F @ x
        if u is not None:
            x_pre += B @ u

        P = F @ P @ F.T + Q

        # update step of normal Kalman filter
        self.y = z - H @ x_pre

        self.S = H @ P @ H.T + R
        SI = inv(self.S)

        K = P @ H.T @ SI

        x = x_pre + K @ self.y

        I_KH = self._I - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        self.xSmooth.append(x_pre.copy())

        # compute invariants
        HTSI = H.T @ SI
        F_LH = (F - K @ H).T

        if k >= N:
            PS = P.copy()  # smoothed P for step i
            for i in range(N):
                K = PS @ HTSI  # smoothed gain
                PS = PS @ F_LH  # smoothed covariance

                si = k - i
                self.xSmooth[si] = self.xSmooth[si] + K @ self.y
        else:
            # Some sources specify starting the fix lag smoother only
            # after N steps have passed, some don't. I am getting far
            # better results by starting only at step N.
            self.xSmooth[k] = x.copy()

        self.count += 1
        self.x = x
        self.P = P

    def smooth_batch(self, zs: np.ndarray, N: int, us: Optional[List[np.ndarray]] = None):
        """batch smooths the set of measurements using a fixed lag smoother.
        I consider this function a somewhat pedalogical exercise; why would
        you not use a RTS smoother if you are able to batch process your data?
        Hint: RTS is a much better smoother, and faster besides. Use it.

        This is a batch processor, so it does not alter any of the object's
        data. In particular, self.x is NOT modified. All date is returned
        by the function.

        Parameters
        ----------


        zs : ndarray of measurements

            iterable list (usually ndarray, but whatever works for you) of
            measurements that you want to smooth, one per time step.

        N : int
           size of fixed lag in time steps

        us : ndarray, optional

            If provided, control input to the filter for each time step


        Returns
        -------

        (xhat_smooth, xhat) : ndarray, ndarray

            xhat_smooth is the output of the N step fix lag smoother
            xhat is the filter output of the standard Kalman filter
        """

        n = len(zs)
        if isinstance(zs, Iterable):
            exp_shape = (self.dim_z, 1)
            assert all([check_input(z, exp_shape, "z") for z in zs])
            assert len(zs) == n, f"Length of zs is not the same as zs. Expected {n}, got {len(zs)}."
        else:
            raise TypeError("zs must be an iterable.")

        if not isinstance(N, int):
            raise TypeError("N must be an integer.")

        if us is not None:
            if isinstance(us, Iterable):
                exp_shape = (self.dim_u, 1)
                assert all([check_input(u, exp_shape, "u") for u in us])
                assert len(us) == n, f"Length of us must be the same as zs. Expected {n}, got {len(us)}."

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        B = self.B

        if x.ndim == 1:
            xSmooth = np.zeros((len(zs), self.dim_x))
            xhat = np.zeros((len(zs), self.dim_x))
        else:
            xSmooth = np.zeros((len(zs), self.dim_x, 1))
            xhat = np.zeros((len(zs), self.dim_x, 1))
        for k, z in enumerate(zs):

            # predict step of normal Kalman filter
            x_pre = F @ x
            if us is not None:
                x_pre += B @ us[k]

            P = F @ P @ F.T + Q

            # update step of normal Kalman filter
            y = z - H @ x_pre

            S = H @ P @ H.T + R
            SI = inv(S)

            K = P @ H.T @ SI

            x = x_pre + K @ y

            I_KH = self._I - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T

            xhat[k] = x.copy()
            xSmooth[k] = x_pre.copy()

            # compute invariants
            HTSI = H.T @ SI
            F_LH = (F - K @ H).T

            if k >= N:
                PS = P.copy()  # smoothed P for step i
                for i in range(N):
                    K = PS @ HTSI  # smoothed gain
                    PS = PS @ F_LH  # smoothed covariance

                    si = k - i
                    xSmooth[si] = xSmooth[si] + K @ y
            else:
                # Some sources specify starting the fix lag smoother only
                # after N steps have passed, some don't. I am getting far
                # better results by starting only at step N.
                xSmooth[k] = xhat[k]

        return xSmooth, xhat

    def __repr__(self):
        return "\n".join(
            [
                "FixedLagSmoother object",
                pretty_str("dim_x", self.x),
                pretty_str("dim_z", self.x),
                pretty_str("N", self.N),
                pretty_str("x", self.x),
                pretty_str("x_s", self.x_s),
                pretty_str("P", self.P),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("B", self.B),
            ]
        )
