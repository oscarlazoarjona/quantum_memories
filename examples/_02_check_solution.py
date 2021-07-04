# -*- coding: utf-8 -*-
# Compatible with Python 3.8
# Copyright (C) 2020-2021 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This script checks the solution from _01_high_efficiency.py against the
equations.
"""
from __future__ import print_function
from pickle import load
# import numpy as np
# from matplotlib import pyplot as plt
# from quantum_memories.misc import time_bandwith_product
from quantum_memories.orca import check_fdm

# from scipy.constants import c

# We establish base parameters.
folder = "__02__check_solution/"
name = ""
plots = False; calculate = False; calculate_greens = False
plots = True
calculate = True
calculate_greens = False

########################################################################
# We load the saved solutions.
if True:
    params = load(open("__01__high_efficiency/params.pickle", "rb"))

    sol = load(open("__01__high_efficiency/solution_write.pickle", "rb"))
    tau, Z, Bw, Sw = sol

    sol = load(open("__01__high_efficiency/solution_read.pickle", "rb"))
    tau, Z, Br, Sr = sol

    print("For the write process:")
    check_fdm(params, Bw, Sw, tau, Z, folder=folder, name="_write", plots=True)

    print()
    print("For the read process:")
    check_fdm(params, Br, Sr, tau, Z, folder=folder, name="_read", plots=True)
