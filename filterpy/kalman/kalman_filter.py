# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-branches,
# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-lines

"""
This module implements the linear Kalman filter in both an object
oriented and procedural form. The KalmanFilter class implements
the filter by storing the various matrices in instance variables,
minimizing the amount of bookkeeping you have to do.

All Kalman filters operate with a predict->update cycle. The
predict step, implemented with the method or function predict(),
uses the state transition matrix F to predict the state in the next
time period (epoch). The state is stored as a gaussian (x, P), where
x is the state (column) vector, and P is its covariance. Covariance
matrix P specifies the process covariance. In Bayesian terms, this
prediction is called the *prior*, which you can think of colloquially
as the estimate prior to incorporating the measurement.

The update step, implemented with the method or function `update()`,
incorporates the measurement z with covariance R, into the state
estimate (x, P). The class stores the system uncertainty in S,
the innovation (residual between prediction and measurement in
measurement space) in y, and the Kalman gain in k. The procedural
form returns these variables to you. In Bayesian terms this computes
the *posterior* - the estimate after the information from the
measurement is incorporated.

Whether you use the OO form or procedural form is up to you. If
matrices such as H, R, and F are changing each epoch, you'll probably
opt to use the procedural form. If they are unchanging, the OO
form is perhaps easier to use since you won't need to keep track
of these matrices. This is especially useful if you are implementing
banks of filters or comparing various KF designs for performance;
a trivial coding bug could lead to using the wrong sets of matrices.

This module also offers an implementation of the RTS smoother, and
other helper functions, such as log likelihood computations.

The Saver class allows you to easily save the state of the
KalmanFilter class after every update

This module expects NumPy arrays for all values that expect
arrays, although in a few cases, particularly method parameters,
it will accept types that convert to NumPy arrays, such as lists
of lists. These exceptions are documented in the method or function.

Examples
--------
The following example constructs a constant velocity kinematic
filter, filters noisy data, and plots the results. It also demonstrates
using the Saver class to save the state of the filter at each epoch.

.. code-block:: Python

    import matplotlib.pyplot as plt
    import numpy as np
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise, Saver

    r_std, q_std = 2., 0.003
    cv = KalmanFilter(dim_x=2, dim_z=1)
    cv.x = np.array([[0., 1.]]).T # position, velocity
    cv.F = np.array([[1, dt],[ [0, 1]])
    cv.R = np.array([[r_std^^2]])
    f.H = np.array([[1., 0.]])
    f.P = np.diag([.1^^2, .03^^2)
    f.Q = Q_discrete_white_noise(2, dt, q_std**2)

    saver = Saver(cv)
    for z in range(100):
        cv.predict()
        # The z input must be a (*,1) numpy ndarray. In this case, it is (1,1).
        cv.update(np.array([[z + randn() * r_std]]).T)
        saver.save() # save the filter's state

    saver.to_array()
    plt.plot(saver.x[:, 0])

    # plot all of the priors
    plt.plot(saver.x_prior[:, 0])

    # plot mahalanobis distance
    plt.figure()
    plt.plot(saver.mahalanobis)

This code implements the same filter using the procedural form

    x = np.array([[0., 1.]]).T # position, velocity
    F = np.array([[1, dt],[ [0, 1]])
    R = np.array([[r_std^^2]])
    H = np.array([[1., 0.]])
    P = np.diag([.1^^2, .03^^2)
    Q = Q_discrete_white_noise(2, dt, q_std**2)

    for z in range(100):
        x, P = predict(x, P, F=F, Q=Q)
        x, P = update(x, P, z=np.array([[z + randn() * r_std]]).T, R=R, H=H)
        xs.append(x[0, 0])
    plt.plot(xs)

The mathematical equations for linear Kalman Filter are:

.. math::

    x_k^- = Fx_{k-1} + Bu_k + Q \\tag{1}
    P_k^- = F P_{k-1} F' + Q \\tag{2}
    e_k = z_k - H x_k^-
    S_k = H P H' + R
    K_k = P H' S^{-1}
    x_k = x_k^- + K_k e_k
    P_k = (I - K_k H) P_k^- (I - K_k H)' + K_k R K_k'


For more examples see the test subdirectory, or refer to the
book cited below. In it I both teach Kalman filtering from basic
principles, and teach the use of this library in great detail.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2014-2018 Roger R Labbe Jr.
"""

from __future__ import absolute_import, division
from filterpy.common.helpers import Saver

from typing import Callable, Iterable, Optional
from copy import deepcopy
import sys
import numpy as np
from filterpy.stats import logpdf
from filterpy.common import pretty_str, reshape_z, check_input


class KalmanFilter(object):
    r"""Implements a Kalman filter. You are responsible for setting the
    various state variables to reasonable values; the defaults  will
    not give you a functional filter.

    For now the best documentation is my free book Kalman and Bayesian
    Filters in Python [2]_. The test files in this directory also give you a
    basic idea of use, albeit without much description.

    In brief, you will first construct this object, specifying the size of
    the state vector with dim_x and the size of the measurement vector that
    you will be using with dim_z. These are mostly used to perform size checks
    when you assign values to the various matrices. For example, if you
    specified dim_z=2 and then try to assign a 3x3 matrix to R (the
    measurement noise matrix you will get an assert exception because R
    should be 2x2. (If for whatever reason you need to alter the size of
    things midstream just use the underscore version of the matrices to
    assign directly: your_filter._R = a_3x3_matrix.)

    After construction the filter will have default matrices created for you,
    but you must specify the values for each. It’s usually easiest to just
    overwrite them rather than assign to each element yourself. This will be
    clearer in the example below. All are of type numpy.array.


    Examples
    --------

    Here is a filter that tracks position and velocity using a sensor that only
    reads position.

    First construct the object with the required dimensionality. Here the state
    (`dim_x`) has 2 coefficients (position and velocity), and the measurement
    (`dim_z`) has one. In FilterPy `x` is the state, `z` is the measurement.

    .. code::

        from filterpy.kalman import KalmanFilter
        f = KalmanFilter (dim_x=2, dim_z=1)


    Assign the initial value for the state (position and velocity). You can do this
    with a two dimensional array like so:

        .. code::

            f.x = np.array([[2.],    # position
                            [0.]])   # velocity

    Define the state transition matrix:

        .. code::

            f.F = np.array([[1.,1.],
                            [0.,1.]])

    Define the measurement function. Here we need to convert a position-velocity
    vector into just a position vector, so we use:

        .. code::

        f.H = np.array([[1., 0.]])

    Define the state's covariance matrix P.

    .. code::

        f.P = np.array([[1000.,    0.],
                        [   0., 1000.] ])

    Now assign the measurement noise. Here the dimension is 1x1

    .. code::

        f.R = np.array([[5.]])

    Note that this must be a 2 dimensional array.

    Finally, I will assign the process noise. Here I will take advantage of
    another FilterPy library function:

    .. code::

        from filterpy.common import Q_discrete_white_noise
        f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)


    Now just perform the standard predict/update loop:

    .. code::

        while some_condition_is_true:
            z = get_sensor_reading()
            f.predict()
            f.update(z)

            do_something_with_estimate (f.x)


    **Procedural Form**

    This module also contains stand alone functions to perform Kalman filtering.
    Use these if you are not a fan of objects.

    **Example**

    .. code::

        while True:
            z, R = read_sensor()
            x, P = predict(x, P, F, Q)
            x, P = update(x, P, z, R, H)

    See my book Kalman and Bayesian Filters in Python [2]_.


    You will have to set the following attributes after constructing this
    object for the filter to perform properly. Please note that there are
    various checks in place to ensure that you have made everything the
    'correct' size. However, it is possible to provide incorrectly sized
    arrays such that the linear algebra can not perform an operation.
    It can also fail silently - you can end up with matrices of a size that
    allows the linear algebra to work, but are the wrong shape for the problem
    you are trying to solve.

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

    dim_u : int (optional)
        size of the control input, if it is being used.
        Default value of 0 indicates it is not used.

    compute_log_likelihood : bool (default = True)
        Computes log likelihood by default, but this can be a slow
        computation, so if you never use it you can turn this computation
        off.

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
        are for convenience; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : numpy.array(dim_z, 1)
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        Measurement noise covariance matrix. Also known as the
        observation covariance.

    Q : numpy.array(dim_x, dim_x)
        Process noise covariance matrix. Also known as the transition
        covariance.

    F : numpy.array(dim_x, dim_x)
        State Transition matrix. Also known as `A` in some formulation.

    H : numpy.array(dim_z, dim_x)
        Measurement function. Also known as the observation matrix, or as `C`.

    y : numpy.array
        Residual of the update step. Read only.

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    S :  numpy.array
        System uncertainty (P projected to measurement space). Read only.

    SI :  numpy.array
        Inverse system uncertainty. Read only.

    log_likelihood : float
        log-likelihood of the last measurement. Read only.

    likelihood : float
        likelihood of last measurement. Read only.

        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.

    mahalanobis : float
        mahalanobis distance of the innovation. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

        This is only used to invert self.S. If you know it is diagonal, you
        might choose to set it to filterpy.common.inv_diagonal, which is
        several times faster than numpy.linalg.inv for diagonal matrices.

    alpha : float
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.

    References
    ----------

    .. [1] Dan Simon. "Optimal State Estimation." John Wiley & Sons.
       p. 208-212. (2006)

    .. [2] Roger Labbe. "Kalman and Bayesian Filters in Python"
       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    """

    def __init__(self, dim_x: int, dim_z: int, dim_u: int = 1):
        if dim_x < 1:
            raise ValueError("dim_x must be 1 or greater")
        if dim_z < 1:
            raise ValueError("dim_z must be 1 or greater")
        if dim_u < 1:
            raise ValueError("dim_u must be 1 or greater")

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # All matrices are 2D. Further, all possible inputs are now properties for
        # the class and go through enforced checks when someone attempts to modify
        # them.
        self.__x = np.zeros((dim_x, 1))  # state
        self.__P = np.eye(dim_x)  # uncertainty covariance
        self.__Q = np.eye(dim_x)  # process uncertainty
        self.__B = None  # control transition matrix
        self.__F = np.eye(dim_x)  # state transition matrix
        self.__H = np.zeros((dim_z, dim_x))  # measurement function
        self.__R = np.eye(dim_z)  # measurement uncertainty
        self.__M = np.zeros((dim_x, dim_z))  # process-measurement cross correlation
        self.__z = np.zeros((self.dim_z, 1))  # Start with a zero measurement space. This could be a valid measurement.

        self._alpha_sq = 1.0  # fading memory control

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.__y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = np.log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv

    @property
    def x(self):  # pylint: disable=missing-function-docstring
        return self.__x

    @x.setter
    def x(self, vec: np.ndarray):
        exp_shape = (self.dim_x, 1)
        if check_input(vec, exp_shape, "x"):
            self.__x = vec

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

    def predict(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array, default 0
            Optional control vector.

        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """
        if u is not None:
            exp_shape = (self.dim_u, 1)
            assert check_input(u, exp_shape, "u")

        if B is None:
            B = self.B
        else:
            exp_shape = (self.dim_x, self.dim_u)
            assert check_input(B, exp_shape, "B")

        if F is None:
            F = self.F
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(F, exp_shape, "F")

        if Q is None:
            Q = self.Q
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(Q, exp_shape, "Q")

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = F @ self.x + B @ u
        else:
            self.x = F @ self.x

        # P = ⍺^2 * FPF' + Q
        self.P = self._alpha_sq * F @ self.P @ F.T + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z: np.ndarray, R: np.ndarray = None, H: np.ndarray = None):
        """
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

            If you pass in a value of H, z must be a column vector the
            of the correct size.

        R : (dim_z, dim_z) np.ndarray, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : (dim_z, dim_x) np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            # No innovation is available. Thus, we ignore the update step and return our priors.
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        else:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")

        if R is None:
            R = self.R
        else:
            exp_shape = (self.dim_z, self.dim_z)
            assert check_input(R, exp_shape, "R")

        if H is None:
            H = self.H
        else:
            exp_shape = (self.dim_z, self.dim_x)
            assert check_input(H, exp_shape, "H")

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - H @ self.x

        # common subexpression for speed
        PHT = self.P @ H.T

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = H @ PHT + R
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = PHT @ self.inv(self.S)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K @ self.y

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_steadystate(self, u: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None):
        """
        Predict state (prior) using the Kalman filter state propagation
        equations. Only x is updated, P is left unchanged. See
        update_steadstate() for a longer explanation of when to use this
        method.

        Parameters
        ----------

        u : np.array
            Optional control vector. If non-zero, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.
        """

        if u is None:
            u = np.zeros((self.dim_u, 1))
        exp_shape = (self.dim_u, 1)
        assert check_input(u, exp_shape, "u")
        if B is None:
            B = self.B
        else:
            exp_shape = (self.dim_x, self.dim_u)
            assert check_input(B, exp_shape, "B")

        # x = Fx + Bu
        if B is not None:
            self.x = self.F @ self.x + B @ u
        else:
            self.x = self.F @ self.x

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update_steadystate(self, z: np.ndarray):
        """
        Add a new measurement (z) to the Kalman filter without recomputing
        the Kalman gain K, the state covariance P, or the system
        uncertainty S.

        You can use this for LTI systems since the Kalman gain and covariance
        converge to a fixed value. Precompute these and assign them explicitly,
        or run the Kalman filter using the normal predict()/update(0 cycle
        until they converge.

        The main advantage of this call is speed. We do significantly less
        computation, notably avoiding a costly matrix inversion.

        Use in conjunction with predict_steadystate(), otherwise P will grow
        without bound.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.


        Examples
        --------
        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> # let filter converge on representative data, then save k and P
        >>> for i in range(100):
        >>>     cv.predict()
        >>>     cv.update([i, i, i])
        >>> saved_k = np.copy(cv.K)
        >>> saved_P = np.copy(cv.P)

        later on:

        >>> cv = kinematic_kf(dim=3, order=2) # 3D const velocity filter
        >>> cv.K = np.copy(saved_K)
        >>> cv.P = np.copy(saved_P)
        >>> for i in range(100):
        >>>     cv.predict_steadystate()
        >>>     cv.update_steadystate([i, i, i])
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        else:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")

        # z = reshape_z(z, self.dim_z, self.x.ndim)

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - self.H @ self.x

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K @ self.y

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    def update_correlated(self, z: np.ndarray, R: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None):
        """Add a new measurement (z) to the Kalman filter assuming that
        process noise and measurement noise are correlated as defined in
        the `self.M` matrix.

        A partial derivation can be found in [1]

        If z is None, nothing is changed.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array,  or None
            Optionally provide H to override the measurement function for this
            one call, otherwise  self.H will be used.

        References
        ----------

        .. [1] Bulut, Y. (2011). Applied Kalman filter theory (Doctoral dissertation, Northeastern University).
               http://people.duke.edu/~hpgavin/SystemID/References/Balut-KalmanFilter-PhD-NEU-2011.pdf
        """

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        else:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")

        if R is None:
            R = self.R
        else:
            exp_shape = (self.dim_z, self.dim_z)
            assert check_input(R, exp_shape, "R")

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        else:
            exp_shape = (self.dim_z, self.dim_x)
            assert check_input(H, exp_shape, "H")

        # handle special case: if z is in form [[z]] but x is not a column
        # vector dimensions will not match
        # if self.x.ndim == 1 and np.shape(z) == (1, 1):
        #     z = z[0]

        # if np.shape(z) == ():  # is it scalar, e.g. z=3 or z=np.array(3)
        #     z = np.asarray([z])

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - H @ self.x

        # common subexpression for speed
        PHT = self.P @ H.T

        # project system uncertainty into measurement space
        self.S = H @ PHT + H @ self.M + self.M.T @ H.T + R
        self.SI = self.inv(self.S)

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = (PHT + self.M) @ self.SI

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + self.K @ self.y
        self.P = self.P - self.K @ (H @ self.P + self.M.T)

        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def batch_filter(
        self,
        zs: Iterable[np.ndarray],
        Fs: Optional[Iterable[np.ndarray]] = None,
        Qs: Optional[Iterable[np.ndarray]] = None,
        Hs: Optional[Iterable[np.ndarray]] = None,
        Rs: Optional[Iterable[np.ndarray]] = None,
        Bs: Optional[Iterable[np.ndarray]] = None,
        us: Optional[Iterable[np.ndarray]] = None,
        update_first: bool = False,
        saver: Optional[Saver] = None,
    ):
        """Batch processes a sequences of measurements.

         Parameters
         ----------

         zs : list-like
             list of measurements at each time step `self.dt`. Missing
             measurements must be represented by `None`.

         Fs : None, list-like, default=None
             optional value or list of values to use for the state transition
             matrix F.

             If Fs is None then self.F is used for all epochs.

             Otherwise it must contain a list-like list of F's, one for
             each epoch.  This allows you to have varying F per epoch.

         Qs : None, np.array or list-like, default=None
             optional value or list of values to use for the process error
             covariance Q.

             If Qs is None then self.Q is used for all epochs.

             Otherwise it must contain a list-like list of Q's, one for
             each epoch.  This allows you to have varying Q per epoch.

         Hs : None, np.array or list-like, default=None
             optional list of values to use for the measurement matrix H.

             If Hs is None then self.H is used for all epochs.

             If Hs contains a single matrix, then it is used as H for all
             epochs.

             Otherwise it must contain a list-like list of H's, one for
             each epoch.  This allows you to have varying H per epoch.

         Rs : None, np.array or list-like, default=None
             optional list of values to use for the measurement error
             covariance R.

             If Rs is None then self.R is used for all epochs.

             Otherwise it must contain a list-like list of R's, one for
             each epoch.  This allows you to have varying R per epoch.

         Bs : None, np.array or list-like, default=None
             optional list of values to use for the control transition matrix B.

             If Bs is None then self.B is used for all epochs.

             Otherwise it must contain a list-like list of B's, one for
             each epoch.  This allows you to have varying B per epoch.

         us : None, np.array or list-like, default=None
             optional list of values to use for the control input vector;

             If us is None then None is used for all epochs (equivalent to 0,
             or no control input).

             Otherwise it must contain a list-like list of u's, one for
             each epoch.

        update_first : bool, optional, default=False
             controls whether the order of operations is update followed by
             predict, or predict followed by update. Default is predict->update.

         saver : filterpy.common.Saver, optional
             filterpy.common.Saver object. If provided, saver.save() will be
             called after every epoch

         Returns
         -------

         means : np.array((n,dim_x,1))
             array of the state for each time step after the update. Each entry
             is an np.array. In other words `means[k,:]` is the state at step
             `k`.

         covariance : np.array((n,dim_x,dim_x))
             array of the covariances for each time step after the update.
             In other words `covariance[k,:,:]` is the covariance at step `k`.

         means_predictions : np.array((n,dim_x,1))
             array of the state for each time step after the predictions. Each
             entry is an np.array. In other words `means[k,:]` is the state at
             step `k`.

         covariance_predictions : np.array((n,dim_x,dim_x))
             array of the covariances for each time step after the prediction.
             In other words `covariance[k,:,:]` is the covariance at step `k`.

         Examples
         --------

         .. code-block:: Python

             # this example demonstrates tracking a measurement where the time
             # between measurement varies, as stored in dts. This requires
             # that F be recomputed for each epoch. The output is then smoothed
             # with an RTS smoother.

             zs = [t + random.randn()*4 for t in range (40)]
             Fs = [np.array([[1., dt], [0, 1]] for dt in dts]

             (mu, cov, _, _) = kf.batch_filter(zs, Fs=Fs)
             (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs)
        """

        # pylint: disable=too-many-statements
        n = len(zs)
        if isinstance(zs, Iterable):
            exp_shape = (self.dim_z, 1)
            assert all([check_input(z, exp_shape, "z") for z in zs])
            assert len(zs) == n, f"Length of zs is not the same as zs. Expected {n}, got {len(zs)}."
        else:
            raise TypeError("zs must be an iterable.")

        if Fs is None:
            Fs = [self.F] * n
        else:
            if isinstance(Fs, Iterable):
                exp_shape = (self.dim_x, self.dim_x)
                assert all([check_input(F, exp_shape, "F") for F in Fs])
                assert len(Fs) == n, f"Length of Fs is not the same as zs. Expected {n}, got {len(Fs)}."
            else:
                raise TypeError("Fs must be an iterable.")

        if Qs is None:
            Qs = [self.Q] * n
        else:
            if isinstance(Qs, Iterable):
                exp_shape = (self.dim_x, self.dim_x)
                assert all([check_input(Q, exp_shape, "Q") for Q in Qs])
                assert len(Qs) == n, f"Length of Qs is not the same as zs. Expected {n}, got {len(Qs)}."
            else:
                raise TypeError("Qs must be an iterable.")

        if Hs is None:
            Hs = [self.H] * n
        else:
            if isinstance(Hs, Iterable):
                exp_shape = (self.dim_z, self.dim_x)
                assert all([check_input(H, exp_shape, "H") for H in Hs])
                assert len(Hs) == n, f"Length of Hs is not the same as zs. Expected {n}, got {len(Hs)}."
            else:
                raise TypeError("Hs must be an iterable.")

        if Rs is None:
            Rs = [self.R] * n
        else:
            if isinstance(Rs, Iterable):
                exp_shape = (self.dim_z, self.dim_z)
                assert all([check_input(R, exp_shape, "R") for R in Rs])
                assert len(Rs) == n, f"Length of Rs is not the same as zs. Expected {n}, got {len(Rs)}."
            else:
                raise TypeError("Rs must be an iterable.")

        if Bs is None:
            Bs = [self.B] * n
        else:
            if isinstance(Bs, Iterable):
                exp_shape = (self.dim_x, self.dim_u)
                assert all([check_input(B, exp_shape, "B") for B in Bs])
                assert len(Bs) == n, f"Length of Bs is not the same as zs. Expected {n}, got {len(Bs)}."
            else:
                raise TypeError("Bs must be an iterable.")

        if us is None:
            us = [np.array([[0]])] * n
        else:
            if isinstance(us, Iterable):
                exp_shape = (self.dim_u, 1)
                assert all([check_input(u, exp_shape, "u") for u in us])
                assert len(us) == n, f"Length of us is not the same as zs. Expected {n}, got {len(us)}."
            else:
                raise TypeError("us must be an iterable.")

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = np.zeros((n, self.dim_x))
            means_p = np.zeros((n, self.dim_x))
        else:
            means = np.zeros((n, self.dim_x, 1))
            means_p = np.zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = np.zeros((n, self.dim_x, self.dim_x))
        covariances_p = np.zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances, means_p, covariances_p)

    def rts_smoother(
        self,
        Xs: Iterable[np.ndarray],
        Ps: Iterable[np.ndarray],
        Fs: Optional[Iterable[np.ndarray]] = None,
        Qs: Optional[Iterable[np.ndarray]] = None,
        inv: Callable = np.linalg.inv,
    ):
        """
        Runs the Rauch-Tung-Striebel Kalman smoother on a set of
        means and covariances computed by a Kalman filter. The usual input
        would come from the output of `KalmanFilter.batch_filter()`.

        Parameters
        ----------

        Xs : numpy.array
           array of the means (state variable x) of the output of a Kalman
           filter.

        Ps : numpy.array
            array of the covariances of the output of a kalman filter.

        Fs : list-like collection of numpy.array, optional
            State transition matrix of the Kalman filter at each time step.
            Optional, if not provided the filter's self.F will be used

        Qs : list-like collection of numpy.array, optional
            Process noise of the Kalman filter at each time step. Optional,
            if not provided the filter's self.Q will be used

        inv : function, default numpy.linalg.inv
            If you prefer another inverse function, such as the Moore-Penrose
            pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv


        Returns
        -------

        x : numpy.ndarray
           smoothed means

        P : numpy.ndarray
           smoothed state covariances

        K : numpy.ndarray
            smoother gain at each step

        Pp : numpy.ndarray
           Predicted state covariances

        Examples
        --------

        .. code-block:: Python

            zs = [t + random.randn()*4 for t in range (40)]

            (mu, cov, _, _) = kalman.batch_filter(zs)
            (x, P, K, Pp) = rts_smoother(mu, cov, kf.F, kf.Q)

        """

        n = len(Xs)

        if isinstance(Xs, Iterable):
            exp_shape = (self.dim_x, 1)
            assert all([check_input(X, exp_shape, "x") for X in Xs])
            assert len(Xs) == n, f"Length of Xs is not the same as Xs. Expected {n}, got {len(Xs)}."
        else:
            raise TypeError("Xs must be an iterable.")

        if isinstance(Ps, Iterable):
            exp_shape = (self.dim_x, self.dim_x)
            assert all([check_input(P, exp_shape, "P") for P in Ps])
            assert len(Ps) == n, f"Length of Ps is not the same as Xs. Expected {n}, got {len(Xs)}."
        else:
            raise TypeError("Ps must be an iterable.")

        if Fs is None:
            Fs = [self.F] * n
        else:
            if isinstance(Fs, Iterable):
                exp_shape = (self.dim_x, self.dim_x)
                assert all([check_input(F, exp_shape, "F") for F in Fs])
                assert len(Fs) == n, f"Length of Fs is not the same as Xs. Expected {n}, got {len(Fs)}."
            else:
                raise TypeError("Fs must be an iterable.")

        if Qs is None:
            Qs = [self.Q] * n
        else:
            if isinstance(Qs, Iterable):
                exp_shape = (self.dim_x, self.dim_x)
                assert all([check_input(Q, exp_shape, "Q") for Q in Qs])
                assert len(Qs) == n, f"Length of Qs is not the same as Xs. Expected {n}, got {len(Qs)}."
            else:
                raise TypeError("Qs must be an iterable.")

        # smoother gain
        K = np.zeros((n, self.dim_x, self.dim_x))

        x, P, Pp = Xs.copy(), Ps.copy(), Ps.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = Fs[k + 1] @ P[k] @ Fs[k + 1].T + Qs[k + 1]
            K[k] = P[k] @ Fs[k + 1].T @ inv(Pp[k])
            x[k] += K[k] @ (x[k + 1] - Fs[k + 1] @ x[k])
            P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T

        return (x, P, K, Pp)

    def get_prediction(
        self,
        u: Optional[np.ndarray] = None,
        B: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations and returns it without modifying the object.

        Parameters
        ----------

        u : np.array, default 0
            Optional control vector.

        B : np.array(dim_x, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the prediction.
        """
        if u is not None:
            exp_shape = (self.dim_u, 1)
            assert check_input(u, exp_shape, "u")
        if B is None:
            B = self.B
        else:
            exp_shape = (self.dim_x, self.dim_u)
            assert check_input(B, exp_shape, "B")
        if F is None:
            F = self.F
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(F, exp_shape, "F")
        if Q is None:
            Q = self.Q
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(Q, exp_shape, "Q")

        # x = Fx + Bu
        if B is not None and u is not None:
            x = F @ self.x + B @ u
        else:
            x = F @ self.x

        # P = FPF' + Q
        P = self._alpha_sq * F @ self.P @ F.T + Q

        return x, P

    def get_update(self, z: Optional[np.ndarray] = None):
        """
        Computes the new estimate based on measurement `z` and returns it
        without altering the state of the filter.

        Parameters
        ----------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        Returns
        -------

        (x, P) : tuple
            State vector and covariance array of the update.
        """

        if z is None:
            return self.x, self.P
        else:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")

        R = self.R
        H = self.H
        P = self.P
        x = self.x

        # error (residual) between measurement and prediction
        y = z - H @ x

        # common subexpression for speed
        PHT = P @ H.T

        # project system uncertainty into measurement space
        S = H @ PHT + R

        # map system uncertainty into kalman gain
        K = PHT @ self.inv(S)

        # predict new x with residual scaled by the kalman gain
        x = x + K @ y

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        return x, P

    def residual_of(self, z: np.ndarray):
        """
        Returns the residual for the given measurement (z). Does not alter
        the state of the filter.
        """
        exp_shape = (self.dim_z, 1)
        assert check_input(z, exp_shape, "z")
        return z - self.H @ self.x_prior

    def measurement_of_state(self, x: np.ndarray):
        """
        Helper function that converts a state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman state vector

        Returns
        -------

        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.
        """
        exp_shape = (self.dim_x, 1)
        assert check_input(x, exp_shape, "x")
        return self.H @ x

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = np.exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood

    @property
    def mahalanobis(self):
        """ "
        Mahalanobis distance of measurement. E.g. 3 means measurement
        was 3 standard deviations away from the predicted value.

        Returns
        -------
        mahalanobis : float
        """
        if self._mahalanobis is None:
            self._mahalanobis = np.sqrt(float(self.y.T @ self.SI @ self.y))
        return self._mahalanobis

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq ** 0.5

    def log_likelihood_of(self, z: np.ndarray):
        """
        log likelihood of the measurement `z`. This should only be called
        after a call to update(). Calling after predict() will yield an
        incorrect result."""

        if z is None:
            return np.log(sys.float_info.min)
        else:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")
        return logpdf(z, self.H @ self.x, self.S)

    @alpha.setter
    def alpha(self, value: float):
        if not np.isscalar(value) or value < 1:
            raise ValueError("alpha must be a float greater than 1")

        self._alpha_sq = value ** 2

    def __repr__(self):
        return "\n".join(
            [
                "KalmanFilter object",
                pretty_str("dim_x", self.dim_x),
                pretty_str("dim_z", self.dim_z),
                pretty_str("dim_u", self.dim_u),
                pretty_str("x", self.x),
                pretty_str("P", self.P),
                pretty_str("x_prior", self.x_prior),
                pretty_str("P_prior", self.P_prior),
                pretty_str("x_post", self.x_post),
                pretty_str("P_post", self.P_post),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("SI", self.SI),
                pretty_str("M", self.M),
                pretty_str("B", self.B),
                pretty_str("z", self.z),
                pretty_str("log-likelihood", self.log_likelihood),
                pretty_str("likelihood", self.likelihood),
                pretty_str("mahalanobis", self.mahalanobis),
                pretty_str("alpha", self.alpha),
                pretty_str("inv", self.inv),
            ]
        )

    def test_matrix_dimensions(
        self,
        z: Optional[np.ndarray] = None,
        H: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        F: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
    ):
        """
        Performs a series of asserts to check that the size of everything
        is what it should be. This can help you debug problems in your design.

        If you pass in H, R, F, Q those will be used instead of this object's
        value for those matrices.
        """

        if H is None:
            H = self.H
        else:
            exp_shape = (self.dim_z, self.dim_x)
            assert check_input(H, exp_shape, "H")
        if R is None:
            R = self.R
        else:
            exp_shape = (self.dim_z, self.dim_z)
            assert check_input(R, exp_shape, "R")
        if F is None:
            F = self.F
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(F, exp_shape, "F")
        if Q is None:
            Q = self.Q
        else:
            exp_shape = (self.dim_x, self.dim_x)
            assert check_input(Q, exp_shape, "Q")

        x = self.x
        P = self.P

        exp_shape = (self.dim_x, 1)
        assert check_input(x, exp_shape, "x")
        exp_shape = (self.dim_x, self.dim_x)
        assert check_input(P, exp_shape, "P")
        exp_shape = (self.dim_x, self.dim_x)
        assert check_input(Q, exp_shape, "Q")
        exp_shape = (self.dim_x, self.dim_x)
        assert check_input(F, exp_shape, "F")
        exp_shape = (self.dim_z, self.dim_x)
        assert check_input(H, exp_shape, "H")
        exp_shape = (self.dim_z, self.dim_z)
        assert check_input(R, exp_shape, "R")
        if z is not None:
            exp_shape = (self.dim_z, 1)
            assert check_input(z, exp_shape, "z")


def update(
    x: np.ndarray, P: np.ndarray, z: np.ndarray, R: np.ndarray, H: Optional[np.ndarray] = None, return_all: bool = False
):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    update(1, 2, 1, 1, 1)  # univariate
    update(x, P, 1



    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector

    P : numpy.array(dim_x, dim_x), or float
        Covariance matrix

    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

    R : numpy.array(dim_z, dim_z), or float
        Measurement noise matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    return_all : bool, default False
        If true, y, K, S, and log_likelihood are returned, otherwise
        only x and P are returned.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    P : numpy.array
        Posterior covariance matrix

    y : numpy.array or scalar
        Residua. Difference between measurement and state in measurement space

    K : numpy.array
        Kalman gain

    S : numpy.array
        System uncertainty in measurement space

    log_likelihood : float
        log likelihood of the measurement
    """

    dim_x = x.shape[0]
    exp_shape = (dim_x, 1)
    assert check_input(x, exp_shape, "x")
    exp_shape = (dim_x, dim_x)
    assert check_input(P, exp_shape, "P")
    if z is None:
        if return_all:
            return x, P, None, None, None, None
        return x, P
    else:
        dim_z = z.shape[0]
        exp_shape = (dim_z, 1)
        assert check_input(z, exp_shape, "z")
        exp_shape = (dim_z, dim_z)
        assert check_input(R, exp_shape, "R")
        if H is None:
            H = np.eye(dim_x)
        exp_shape = (dim_z, dim_x)
        assert check_input(H, exp_shape, "H")

    Hx = H @ x

    # error (residual) between measurement and prediction
    y = z - Hx

    # project system uncertainty into measurement space
    S = H @ P @ H.T + R

    # map system uncertainty into kalman gain
    K = P @ H.T @ np.linalg.inv(S)

    # predict new x with residual scaled by the kalman gain
    x = x + K @ y

    # P = (I-KH)P(I-KH)' + KRK'
    KH = K @ H

    I_KH = np.eye(KH.shape[0]) - KH

    P = I_KH @ P @ I_KH.T + K @ R @ K.T

    if return_all:
        # compute log likelihood
        log_likelihood = logpdf(z, H @ x, S)
        return x, P, y, K, S, log_likelihood
    return x, P


def update_steadystate(x: np.ndarray, z: np.ndarray, K: np.ndarray, H: Optional[np.ndarray] = None):
    """
    Add a new measurement (z) to the Kalman filter. If z is None, nothing
    is changed.


    Parameters
    ----------

    x : numpy.array(dim_x, 1), or float
        State estimate vector


    z : (dim_z, 1): array_like
        measurement for this update. z can be a scalar if dim_z is 1,
        otherwise it must be convertible to a column vector.

    K : numpy.array, or float
        Kalman gain matrix

    H : numpy.array(dim_x, dim_x), or float, optional
        Measurement function. If not provided, a value of 1 is assumed.

    Returns
    -------

    x : numpy.array
        Posterior state estimate vector

    Examples
    --------

    This can handle either the multidimensional or unidimensional case. If
    all parameters are floats instead of arrays the filter will still work,
    and return floats for x, P as the result.

    >>> update_steadystate(1, 2, 1)  # univariate
    >>> update_steadystate(x, P, z, H)
    """

    dim_x = x.shape[0]
    exp_shape = (dim_x, 1)
    assert check_input(x, exp_shape, "x")
    exp_shape = (dim_x, dim_x)
    assert check_input(K, exp_shape, "K")

    if z is None:
        return x
    else:
        dim_z = z.shape[0]
        exp_shape = (dim_z, 1)
        assert check_input(z, exp_shape, "z")
        if H is None:
            H = np.array([[1]])

    Hx = H @ x

    # error (residual) between measurement and prediction
    y = z - Hx

    # estimate new x with residual scaled by the kalman gain
    return x + K @ y


def predict(
    x: np.ndarray,
    P: np.ndarray,
    F: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    alpha: float = 1.0,
):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    Q : numpy.array, Optional
        Process noise matrix


    u : numpy.array, Optional, default 0.
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default 0.
        Control transition matrix.

    alpha : float, Optional, default=1.0
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon

    Returns
    -------

    x : numpy.array
        Prior state estimate vector

    P : numpy.array
        Prior covariance matrix
    """

    dim_x = x.shape[0]
    exp_shape = (dim_x, 1)
    assert check_input(x, exp_shape, "x")
    exp_shape = (dim_x, dim_x)
    assert check_input(P, exp_shape, "P")
    if F is None:
        F = np.eye(dim_x, dim_x)
    exp_shape = (dim_x, dim_x)
    assert check_input(F, exp_shape, "F")
    if Q is None:
        Q = np.zeros((dim_x, dim_x))
    exp_shape = (dim_x, dim_x)
    assert check_input(Q, exp_shape, "Q")
    if u is None:
        u = np.array([[0]])
    dim_u = u.shape[0]
    exp_shape = (dim_u, 1)
    assert check_input(u, exp_shape, "u")
    if B is None:
        B = np.zeros((dim_x, dim_u))
    exp_shape = (dim_x, dim_u)
    assert check_input(B, exp_shape, "B")

    x = F @ x + B @ u
    P = (alpha * alpha) * F @ P @ F.T + Q

    return x, P


def predict_steadystate(
    x: np.ndarray, F: np.ndarray, u: Optional[np.ndarray] = np.array([[0]]), B: Optional[np.ndarray] = np.array([[1]])
):
    """
    Predict next state (prior) using the Kalman filter state propagation
    equations. This steady state form only computes x, assuming that the
    covariance is constant.

    Parameters
    ----------

    x : numpy.array
        State estimate vector

    P : numpy.array
        Covariance matrix

    F : numpy.array()
        State Transition matrix

    u : numpy.array, Optional, default np.array([[0]]).
        Control vector. If non-zero, it is multiplied by B
        to create the control input into the system.

    B : numpy.array, optional, default np.array([[1]]).
        Control transition matrix.

    Returns
    -------

    x : numpy.array
        Prior state estimate vector
    """

    dim_x = x.shape[0]
    exp_shape = (dim_x, 1)
    assert check_input(x, exp_shape, "x")
    exp_shape = (dim_x, dim_x)
    assert check_input(F, exp_shape, "F")
    dim_u = u.shape[0]
    exp_shape = (dim_u, 1)
    assert check_input(u, exp_shape, "u")
    exp_shape = (dim_x, dim_u)
    assert check_input(B, exp_shape, "B")

    x = F @ x + B @ u

    return x


def batch_filter(
    x: np.ndarray,
    P: np.ndarray,
    zs: Iterable[np.ndarray],
    Fs: Iterable[np.ndarray],
    Qs: Iterable[np.ndarray],
    Hs: Iterable[np.ndarray],
    Rs: Iterable[np.ndarray],
    Bs: Optional[Iterable[np.ndarray]] = None,
    us: Optional[Iterable[np.ndarray]] = None,
    update_first: bool = False,
    saver: Optional[Saver] = None,
):
    """
    Batch processes a sequences of measurements.

    Parameters
    ----------

    x: np.array
        Initial state of the system.

    P: np.array
        Initial state covariance of the system.

    zs : list-like
        list of measurements at each time step.

    Fs : list-like
        list of values to use for the state transition matrix matrix.

    Qs : list-like
        list of values to use for the process error
        covariance.

    Hs : list-like
        list of values to use for the measurement matrix.

    Rs : list-like
        list of values to use for the measurement error
        covariance.

    Bs : list-like, optional
        list of values to use for the control transition matrix;
        a value of None in any position will cause the filter
        to use `self.B` for that time step.

    us : list-like, optional
        list of values to use for the control input vector;
        a value of None in any position will cause the filter to use
        0 for that time step.

    update_first : bool, optional
        controls whether the order of operations is update followed by
        predict, or predict followed by update. Default is predict->update.

        saver : filterpy.common.Saver, optional
            filterpy.common.Saver object. If provided, saver.save() will be
            called after every epoch

    Returns
    -------

    means : np.array((n,dim_x,1))
        array of the state for each time step after the update. Each entry
        is an np.array. In other words `means[k,:]` is the state at step
        `k`.

    covariance : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the update.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    means_predictions : np.array((n,dim_x,1))
        array of the state for each time step after the predictions. Each
        entry is an np.array. In other words `means[k,:]` is the state at
        step `k`.

    covariance_predictions : np.array((n,dim_x,dim_x))
        array of the covariances for each time step after the prediction.
        In other words `covariance[k,:,:]` is the covariance at step `k`.

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]
        Fs = [kf.F for t in range (40)]
        Hs = [kf.H for t in range (40)]

        (mu, cov, _, _) = kf.batch_filter(zs, Rs=R_list, Fs=Fs, Hs=Hs, Qs=None,
                                          Bs=None, us=None, update_first=False)
        (xs, Ps, Ks, Pps) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=None)

    """

    n = len(zs)
    dim_x = x.shape[0]
    exp_shape = (dim_x, 1)
    assert check_input(x, exp_shape, "x")
    dim_z = zs[0].shape[0]
    if isinstance(zs, Iterable):
        exp_shape = (dim_z, 1)
        assert all([check_input(z, exp_shape, "z") for z in zs])
        assert len(zs) == n, f"Length of zs is not the same as zs. Expected {n}, got {len(zs)}."
    else:
        raise TypeError("zs must be an iterable.")

    if isinstance(Fs, Iterable):
        exp_shape = (dim_x, dim_x)
        assert all([check_input(F, exp_shape, "F") for F in Fs])
        assert len(Fs) == n, f"Length of Fs is not the same as zs. Expected {n}, got {len(Fs)}."
    else:
        raise TypeError("Fs must be an iterable.")

    if isinstance(Qs, Iterable):
        exp_shape = (dim_x, dim_x)
        assert all([check_input(Q, exp_shape, "Q") for Q in Qs])
        assert len(Qs) == n, f"Length of Qs is not the same as zs. Expected {n}, got {len(Qs)}."
    else:
        raise TypeError("Qs must be an iterable.")

    if isinstance(Hs, Iterable):
        exp_shape = (dim_z, dim_x)
        assert all([check_input(H, exp_shape, "H") for H in Hs])
        assert len(Hs) == n, f"Length of Hs is not the same as zs. Expected {n}, got {len(Hs)}."
    else:
        raise TypeError("Hs must be an iterable.")

    if isinstance(Rs, Iterable):
        exp_shape = (dim_z, dim_z)
        assert all([check_input(R, exp_shape, "R") for R in Rs])
        assert len(Rs) == n, f"Length of Rs is not the same as zs. Expected {n}, got {len(Rs)}."
    else:
        raise TypeError("Rs must be an iterable.")

    if us is not None:
        dim_u = us[0].shape[0]
        if isinstance(us, Iterable):
            exp_shape = (dim_u, 1)
            assert all([check_input(u, exp_shape, "u") for u in us])
            assert len(us) == n, f"Length of us is not the same as zs. Expected {n}, got {len(us)}."
        else:
            raise TypeError("us must be an iterable.")

    if Bs is not None:
        if isinstance(Bs, Iterable):
            exp_shape = (dim_x, dim_u)
            assert all([check_input(B, exp_shape, "B") for B in Bs])
            assert len(Bs) == n, f"Length of Bs is not the same as zs. Expected {n}, got {len(Bs)}."
        else:
            raise TypeError("Bs must be an iterable.")

    # mean estimates from Kalman Filter
    means = np.zeros((n, dim_x, 1))
    means_p = np.zeros((n, dim_x, 1))

    # state covariances from Kalman Filter
    covariances = np.zeros((n, dim_x, dim_x))
    covariances_p = np.zeros((n, dim_x, dim_x))

    if us is None:
        dim_u = 1
        us = [np.zeros((dim_u, 1))] * n
        Bs = [np.zeros((dim_x, dim_u))] * n

    if update_first:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = update(x, P, z, R=R, H=H)  # pylint disable=unbalanced-tuple-unpacking
            means[i, :] = x
            covariances[i, :, :] = P

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P
            if saver is not None:
                saver.save()
    else:
        for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

            x, P = predict(x, P, u=u, B=B, F=F, Q=Q)
            means_p[i, :] = x
            covariances_p[i, :, :] = P

            x, P = update(x, P, z, R=R, H=H)  # pylint disable=unbalanced-tuple-unpacking
            means[i, :] = x
            covariances[i, :, :] = P
            if saver is not None:
                saver.save()

    return (means, covariances, means_p, covariances_p)


def rts_smoother(Xs: Iterable[np.ndarray], Ps: Iterable[np.ndarray], Fs: Iterable[np.ndarray], Qs: Iterable[np.ndarray]):
    """
    Runs the Rauch-Tung-Striebel Kalman smoother on a set of
    means and covariances computed by a Kalman filter. The usual input
    would come from the output of `KalmanFilter.batch_filter()`.

    Parameters
    ----------

    Xs : numpy.array
       array of the means (state variable x) of the output of a Kalman
       filter.

    Ps : numpy.array
        array of the covariances of the output of a kalman filter.

    Fs : list-like collection of numpy.array
        State transition matrix of the Kalman filter at each time step.

    Qs : list-like collection of numpy.array, optional
        Process noise of the Kalman filter at each time step.

    Returns
    -------

    x : numpy.ndarray
       smoothed means

    P : numpy.ndarray
       smoothed state covariances

    K : numpy.ndarray
        smoother gain at each step

    pP : numpy.ndarray
       predicted state covariances

    Examples
    --------

    .. code-block:: Python

        zs = [t + random.randn()*4 for t in range (40)]

        (mu, cov, _, _) = kalman.batch_filter(zs)
        (x, P, K, pP) = rts_smoother(mu, cov, kf.F, kf.Q)
    """

    n = len(Xs)

    if isinstance(Xs, Iterable):
        dim_x = Xs[0].shape[0]
        exp_shape = (dim_x, 1)
        assert all([check_input(X, exp_shape, "x") for X in Xs])
        assert len(Xs) == n, f"Length of Xs is not the same as Xs. Expected {n}, got {len(Xs)}."
    else:
        raise TypeError("Xs must be an iterable.")

    if isinstance(Ps, Iterable):
        exp_shape = (dim_x, dim_x)
        assert all([check_input(P, exp_shape, "P") for P in Ps])
        assert len(Ps) == n, f"Length of Ps is not the same as Xs. Expected {n}, got {len(Xs)}."
    else:
        raise TypeError("Ps must be an iterable.")

    if isinstance(Fs, Iterable):
        exp_shape = (dim_x, dim_x)
        assert all([check_input(F, exp_shape, "F") for F in Fs])
        assert len(Fs) == n, f"Length of Fs is not the same as Xs. Expected {n}, got {len(Fs)}."
    else:
        raise TypeError("Fs must be an iterable.")

    if isinstance(Qs, Iterable):
        exp_shape = (dim_x, dim_x)
        assert all([check_input(Q, exp_shape, "Q") for Q in Qs])
        assert len(Qs) == n, f"Length of Qs is not the same as Xs. Expected {n}, got {len(Qs)}."
    else:
        raise TypeError("Qs must be an iterable.")

    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, pP = Xs.copy(), Ps.copy(), Ps.copy()

    for k in range(n - 2, -1, -1):
        pP[k] = Fs[k] @ P[k] @ Fs[k].T + Qs[k]
        K[k] = P[k] @ Fs[k].T @ np.linalg.inv(pP[k])
        x[k] += K[k] @ (x[k + 1] - Fs[k] @ x[k])
        P[k] += K[k] @ (P[k + 1] - pP[k]) @ K[k].T

    return (x, P, K, pP)
