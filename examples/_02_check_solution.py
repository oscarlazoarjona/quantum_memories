# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2020 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This script checks the solution from _01_high_efficiency.py against the
equations.
"""
from __future__ import print_function
from pickle import load
import numpy as np
# from matplotlib import pyplot as plt
from quantum_memories import (time_bandwith_product, set_parameters_ladder,
                              calculate_pulse_energy, check_fdm)

from scipy.constants import c

# We establish base parameters.
folder = "__02__check_solution/"
name = ""
plots = False; calculate = False; calculate_greens = False
plots = True
calculate = True
calculate_greens = False

# Set the memory parameters.
if True:
    # The cell to control pulse ratio:
    l = 1.6
    # The cell to control pulse ratio at half light-speed.
    lp = 2*l
    tauw = 300e-12
    D = lp*tauw*c/2
    ########################################################################
    # The bandwidth of the control pulse is chosen so that it is a
    # Fourier-limited square pulse with intensity FWHM in time tauw.
    sigma_power2 = time_bandwith_product("oo")/tauw

    # The bandwidth of the signal pulse is chosen so that it is a
    # Fourier-limited Gaussian pulse with intensity FWHM in time tauw.
    sigma_power1 = time_bandwith_product(1)/tauw
    ########################################################################

    params = {"verbose": 1,
              "nshg": 0, "USE_HG_SIG": True,
              "sigma_power1": sigma_power1,
              "sigma_power2": sigma_power2,
              "Temperature": 273.15 + 115,
              "L": D/1.05,
              "w1": 131e-6,
              "w2": 131e-6,
              "delta1": 9.0*1e9*2*np.pi,
              "USE_SQUARE_CTRL": True, "nwsquare": "oo", "nrsquare": "oo",
              "element": "Rb", "isotope": 87}
    params = set_parameters_ladder(params)

    ########################################################################
    # We use the analytic theory to calculate the optimal pulse energy,
    # (that would allow unit efficiency for a narrowband signal and an
    # infinite cell.)
    Ecrit = calculate_pulse_energy(params)
    # We set the pulse energy to that value.
    params["energy_pulse2"] = Ecrit
    ########################################################################
# We load the saved solutions.
if True:
    sol = load(open("__01__high_efficiency/solution_write.pickle", "r"))
    tau, Z, Bw, Sw = sol

    sol = load(open("__01__high_efficiency/solution_read.pickle", "r"))
    tau, Z, Br, Sr = sol

    print("For the write process")
    check_fdm(params, Bw, Sw, tau, Z, folder=folder, name="_write", plots=True)

    print()
    print("For the read process")
    check_fdm(params, Br, Sr, tau, Z, folder=folder, name="_read", plots=True)
