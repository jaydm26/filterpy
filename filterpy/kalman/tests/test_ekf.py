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
import matplotlib.pyplot as plt
import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Saver
from filterpy.examples import RadarSim
from pytest import approx
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis

DO_PLOT = False


def test_ekf():
    def H_of(x):
        """compute Jacobian of H matrix for state x"""

        horiz_dist = x[0]
        altitude = x[2]

        denom = np.sqrt(horiz_dist ** 2 + altitude ** 2)
        return np.array([horiz_dist / denom, np.array([0.0]), altitude / denom]).T

    def hx(x):
        """takes a state variable and returns the measurement that would
        correspond to that state.
        """

        return np.sqrt(x[0] ** 2 + x[2] ** 2)

    dt = 0.05
    proccess_error = 0.05

    rk = ExtendedKalmanFilter(dim_x=3, dim_z=1)

    rk.F = np.eye(3) + np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]) * dt

    def fx(x, dt):
        return rk.F @ x

    rk.x = np.array([[-10.0, 90.0, 1100.0]]).T
    rk.R *= 10
    rk.Q = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.001

    rk.P *= 50

    rs = []
    xs = []
    radar = RadarSim(dt)
    ps = []
    pos = []

    s = Saver(rk)
    for i in range(int(20 / dt)):
        z = radar.get_range(proccess_error)
        pos.append(radar.pos)

        rk.update(np.asarray([[z]]), H_of, hx, R=np.array([hx(rk.x) * proccess_error]))
        ps.append(rk.P)
        rk.predict()

        xs.append(rk.x)
        rs.append(z)
        s.save()

        # test mahalanobis
        a = np.zeros(rk.y.shape)
        maha = scipy_mahalanobis(a, rk.y, rk.SI)
        assert rk.mahalanobis == approx(maha)

    s.to_array()

    xs = np.asarray(xs)
    ps = np.asarray(ps)
    rs = np.asarray(rs)

    p_pos = ps[:, 0, 0]
    p_vel = ps[:, 1, 1]
    p_alt = ps[:, 2, 2]
    pos = np.asarray(pos)

    if DO_PLOT:
        plt.subplot(311)
        plt.plot(xs[:, 0])
        plt.ylabel("position")

        plt.subplot(312)
        plt.plot(xs[:, 1])
        plt.ylabel("velocity")

        plt.subplot(313)
        # plt.plot(xs[:,2])
        # plt.ylabel('altitude')

        plt.plot(p_pos)
        plt.plot(-p_pos)
        plt.plot(xs[:, 0] - pos)


if __name__ == "__main__":
    test_ekf()
