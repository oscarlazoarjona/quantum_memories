# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example of usage of the ORCA memory with the
default settings.
"""

# from time import time
from quantum_memories import orca
from quantum_memories.misc import set_parameters_ladder

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
# import sys

default_params = set_parameters_ladder()
default_params["alpha_rw"] = 0.0
default_params["USE_HG_CTRL"] = False
default_params["USE_HG_SIG"] = True
default_params["nw"] = 0
default_params["ns"] = 0
default_params["nr"] = 0
default_params["energy_pulse2"] = 25E-12
default_params["D"] = 1.15 * default_params["L"]


if __name__ == '__main__':
    name = "test"
    # t0 = time()
    print ("Mode order 0")
    print default_params["Nv"]

    t_sample, Z, Om1, rho31 = orca.solve(default_params, plots=False,
                                         name=name, integrate_velocities=True)

    N = len(t_sample)
    M = len(Z)
    Gri = np.zeros((M, N), complex)
    Gri += rho31[-1, :].reshape((M, 1)).dot(Om1[:, 0].reshape((1, N)))

    for ii in range(4):
        print ("Mode order %i" % (ii + 1))
        default_params["ns"] = ii + 1
        aux = orca.solve(default_params, plots=False,
                         name=name, integrate_velocities=True)
        t_sample, Z, Om1, rho31 = aux
        Gri += rho31[-1, :].reshape((M, 1)).dot(Om1[:, 0].reshape((1, N)))

    Gri /= np.sqrt((abs(Gri)**2).sum())
    U, D, V = svd(Gri)
    K = 1.0 / (D**4).sum()
    print ("Effective mode number: %.3f" % K)
    T, S = np.meshgrid(t_sample, Z)
    plt.figure()
    plt.contourf(T, S, abs(Gri)**2, 256)
    plt.tight_layout()
    # plt.show()
    # sys.exit()

    plt.figure()
    plt.subplot(211)
    plt.plot(Z, np.abs(U[:, 0]))
    plt.plot(Z, np.abs(U[:, 1]))
    plt.plot(Z, np.abs(U[:, 2]))
    plt.plot(Z, np.abs(U[:, 3]))
    plt.subplot(212)
    plt.plot(t_sample, np.abs(V[0, :]))
    plt.plot(t_sample, np.abs(V[1, :]))
    plt.plot(t_sample, np.abs(V[2, :]))
    plt.plot(t_sample, np.abs(V[3, :]))
    plt.tight_layout()
    # plt.show()

    plt.figure()
    plt.subplot(121)
    plt.bar(np.arange(5), D[:5])
    plt.subplot(122)
    plt.bar(np.arange(5), (D[:5])**2)
    plt.tight_layout()
    # plt.show()
    # tsolve = time()-t0
    # t0 = time()
    # eff_in, eff_out, eff = efficiencies(t, Om1, default_params,
    #                                     plots=True, name=name)
    # teff = time()-t0

    # print "Including plotting times:"
    # print "The solve function took", tsolve, "s."
    # print "The efficiencies function took", teff, "s."
    # print "The efficiencies were:", eff_in, eff_out, eff

    # t0 = time()
    # t, Z, vZ, rho, Om1 = orca.solve(default_params, plots=False, name=name)
    # tsolve = time()-t0
    # t0 = time()
    # eff_in, eff_out, eff = efficiencies(t, Om1, default_params,
    #                                     plots=False, name=name)
    # teff = time()-t0

    # print "Not including plotting times:"
    # print "The solve function took", tsolve, "s."
    # print "The efficiencies function took", teff, "s."
    # print "The efficiencies were:", eff_in, eff_out, eff
