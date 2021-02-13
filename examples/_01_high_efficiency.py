# -*- coding: utf-8 -*-
# Compatible with Python 3.8
# Copyright (C) 2020-2021 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This script calculates the numerical storage and retrieval of the
optimal input signal that the analytic theory suggests using feasible
parameters and getting high efficiency.
"""
from __future__ import print_function
from pickle import dump
import numpy as np
from matplotlib import pyplot as plt
from quantum_memories import (time_bandwith_product, build_mesh_fdm,
                              sketch_frame_transform)
from quantum_memories.orca import (set_parameters_ladder,
                                   calculate_pulse_energy, print_params,
                                   calculate_xi0, calculate_F,
                                   calculate_optimal_input_xi, num_integral,
                                   calculate_optimal_input_Z,
                                   calculate_optimal_input_tau,
                                   solve)
from scipy.constants import c

# We establish base parameters.
folder = "__01__high_efficiency/"
name = ""
plots = False; calculate = False
plots = True
calculate = True

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
    print_params(params)
    print("")
    params, Z, tau, tau1, tau2, tau3 = build_mesh_fdm(params)
    sketch_frame_transform(params, folder=folder, draw_readout=False)
# Calculate the optimal signal.
if True:
    ########################################################################
    # In xi-space.
    xi0 = calculate_xi0(params)
    Deltaxi = 2/c/tauw
    xi = np.linspace(xi0-4*Deltaxi/2, xi0+4*Deltaxi/2, 1001)

    Gammaxi = calculate_F(params, xi)
    xi, S0xi = calculate_optimal_input_xi(params, xi)
    Sfxi = Gammaxi*S0xi

    # The initial normalization of the xi signal (instead of integrating).
    N0 = c/2
    Nf = num_integral(np.abs(Sfxi)**2, xi)
    eta_ana = Nf/N0

    print("Critical energy: {:.2f} nJ".format(Ecrit*1e9))
    print("Analytic-theory efficiency: {:.4f}".format(eta_ana))

    ########################################################################
    # In Z-space
    Z__, S0Z = calculate_optimal_input_Z(params)
    ########################################################################
    # In tau-space
    tau, S0tau = calculate_optimal_input_tau(params)

# # We calculate the write-in process.
if calculate:
    # NOTE: set analytic_storage to 2 for faster calculation.
    kwargs = {"plots": True, "folder": folder, "name": "write", "verbose": 0,
              "analytic_storage": 1, "S0t": S0tau}
    tau, Z, Bw, Sw = solve(params, **kwargs)
# We calculate the read-out process.
if calculate:
    B0_stored = Bw[-1, :]
    # NOTE: set analytic_storage to 2 for faster calculation.
    kwargs = {"plots": True, "folder": folder, "name": "read", "verbose": 0,
              "analytic_storage": 1, "B0z": B0_stored}
    tau, Z, Br, Sr = solve(params, **kwargs)
# We calculate the Beam-splitter picture transmissivities and reflectivities.
if calculate:
    NS = num_integral(np.abs(Sw[:, 0])**2, tau)
    NST = num_integral(np.abs(Sw[:, -1])**2, tau)

    Nt1 = tau1.shape[0]
    S0Z_num = Sw[Nt1-1, :]

    TS = NST/NS
    RS = 1 - TS

    NB = num_integral(np.abs(Br[0, :])**2, tau)
    NBT = num_integral(np.abs(Br[-1, :])**2, tau)
    TB = NBT/NB
    RB = 1 - TB

    Nf = num_integral(np.abs(Sr[:, -1])**2, tau)
    eta_num = Nf/NS

    print("Numerical efficiency      : {:.4f}".format(eta_num))
    print("")
    print("Beam-splitter picture transmissivities and reflectivities:")
    print("TB: {:.4f}, RS: {:.4f}".format(TB, RS))
    print("RB: {:.4f}, TS: {:.4f}".format(RB, TS))

    assert np.round(eta_ana, 4) == 0.7953
    assert np.round(eta_num, 4) == 0.7902

    assert np.round(TB, 4) == 0.0922
    assert np.round(RS, 4) == 0.8817
    assert np.round(RB, 4) == 0.9078
    assert np.round(TS, 4) == 0.1183
# Save the results.
if calculate:
    dump([tau, Z, Bw, Sw], open(folder+"solution_write.pickle", "wb"))
    dump([tau, Z, Br, Sr], open(folder+"solution_read.pickle", "wb"))
if calculate and plots:
    fs = 15
    ######################################################################
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(tau*1e9, np.abs(Sw[:, 0])**2*1e-9, "b-", label="Input")
    ax1.plot(tau*1e9, np.abs(Sw[:, -1])**2*1e-9, "r-", label="Leaked")
    ax1.plot(tau*1e9, np.abs(Sr[:, -1])**2*1e-9, "g-", label="Output")

    angle1 = np.unwrap(np.angle(Sw[:, 0]))/2/np.pi
    angle2 = np.unwrap(np.angle(Sw[:, -1]))/2/np.pi
    angle3 = np.unwrap(np.angle(Sr[:, -1]))/2/np.pi
    ax2.plot(tau*1e9, angle1, "b:")
    ax2.plot(tau*1e9, angle2, "r:")
    ax2.plot(tau*1e9, angle3, "g:")

    ax1.set_xlabel(r"$\tau \ [ns]$", fontsize=fs)
    ax1.set_ylabel(r"$|S|^2$  [1/ns]", fontsize=fs)
    ax2.set_ylabel(r"Phase  [revolutions]", fontsize=fs)
    ax1.legend(loc=6)
    plt.savefig(folder+"high_efficiency.png", bbox_inches="tight")
    plt.close()
    ######################################################################

    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax2 = ax.twinx()

    ax.plot(xi, np.abs(Gammaxi), "k-", label=r"$F(\xi)$")
    ax2.plot(xi, np.abs(S0xi), "b-", label=r"$S(\tau=0, \xi)$")
    ax2.plot(xi, np.abs(Sfxi), "g-", label=r"$S(\tau=\tau_f, \xi)$")

    ax.grid(True)
    ax.legend(fontsize=fs, loc=2)
    ax2.legend(fontsize=fs, loc=1)
    ax.set_xlabel(r"$\xi \ [1/m]$", fontsize=fs)
    ax.set_ylabel(r"$F(\xi)$", fontsize=fs)
    ax2.set_ylabel(r"$S(\xi) \ [m]$", fontsize=fs)

    plt.savefig(folder+"Gamma_xi.png", bbox_inches="tight")
    plt.close()
