# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This example calculates the inhomogeneous dephasing."""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
from time import time

from quantum_memories.misc import set_parameters_ladder, efficiencies
from quantum_memories.orca import solve


def model_theoretical(t, amp, sigma):
    """Return points on a gaussian with the given amplitude and dephasing."""
    return amp*np.exp(-(t/sigma)**2)


def efficiencies_delay(i, Nv, explicit_decoherence=None):
    r"""Get the efficiencies with t0r = t0w + (3.5+i) ns."""
    name = "ORCA"+str(i)
    # We set custom parameters (different from those in settings.py)
    t0w = default_params["t0w"]
    params = set_parameters_ladder({"Nv": Nv, "Nt": 51000, "T": 16e-9,
                                    "t0r": t0w+3.5e-9+i*1e-9,
                                    "verbose": 0})
    t0w = params["t0w"]
    # t0r = params["t0r"]
    # print "......................................."
    # print str(i)+"th readout, t0r - t0w =", (t0r - t0w)*1e9, "ns"

    # We call the solver from orca_solver.py
    t, Z, vZ, rho1, Om1 = solve(params, plots=False, name=name)
    # We calculate the efficiencies.
    eff_in, eff_out, eff = efficiencies(t, Om1, params,
                                        plots=True, name=name)

    if explicit_decoherence is not None:
        eff_out = eff_out*explicit_decoherence
        eff = eff_in*eff_out
    # print "eff_in, eff_out, eff =", eff_in, eff_out, eff
    return eff_in, eff_out, eff


###############################################################################
# The number of processors available for parallel computing.
Nprocs = cpu_count()
# We set the default parameters taking them from settings.py.
default_params = set_parameters_ladder()

if __name__ == '__main__':

    ###########################################################################
    # We test various readout times.
    print "Testing readout times..."
    t0 = time()
    Nv = 9
    Ndelay = 8
    tdelay = np.array([3.5e-9 + i*1e-9 for i in range(Ndelay)])

    t0r_list = []
    eff_in_list = []; eff_out_list = []; eff_list = []

    eff_in_list = np.zeros(Ndelay)
    eff_out_list = np.zeros(Ndelay)
    eff_list = np.zeros(Ndelay)

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    procs = [pool.apply_async(efficiencies_delay, [i, Nv])
             for i in range(Ndelay)]

    # We get the results.
    aux = [procs[i].get() for i in range(Ndelay)]
    pool.close()
    pool.join()

    # We save the results with more convenient names.
    for i in range(Ndelay):
        eff_in_list[i], eff_out_list[i], eff_list[i] = aux[i]

    ####################################
    # We plot the total efficiencies.
    plt.title(r"$\mathrm{Dephasing}$", fontsize=20)
    plt.plot(tdelay*1e9, eff_list, "k+",
             label=r"$\eta_{\mathrm{model}}$", ms=10)

    # We fit a gaussian.
    amp_the, sig_the = curve_fit(model_theoretical, tdelay, eff_list,
                                 p0=[1.0, 5.4e-9])[0]

    print "Nv, 1/e time, calculation time:", Nv, sig_the*1e9, "ns",
    print (time() - t0)/60.0, "min"

    # We make a plot of the fitted gaussian.
    tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
    eff_exp = model_theoretical(tdelay_cont, amp_the, 5.4e-9)
    eff_the = model_theoretical(tdelay_cont, amp_the, sig_the)

    plt.plot(tdelay_cont*1e9, eff_exp, "k-",
             label=r"$\eta_{\mathrm{experiment}}$")
    plt.plot(tdelay_cont*1e9, eff_the, "k:",
             label=r"$\eta_{\mathrm{fit}}$")

    plt.xlabel(r"$t_{0\mathrm{r}}-t_{0\mathrm{w}} \ \mathrm{(ns)}$",
               fontsize=20)
    plt.ylabel(r"$\eta$", fontsize=20)
    plt.ylim([0, None])

    plt.legend(fontsize=15)
    plt.savefig("doppler_dephasing.png", bbox_inches="tight")
    plt.savefig("doppler_dephasing.pdf", bbox_inches="tight")
    plt.close("all")

    ###########################################################################
    # We now repeat exactly the same thing using different number of velocity
    # classes.
    Nvmax = 13
    Nv = [2*i + 1 for i in range((Nvmax-1)/2+1)]
    # print Nv
    sigma = np.zeros(len(Nv))
    print
    print "Calculating the best number of velocity classes to use..."
    for jj in range(len(Nv)):
        t0 = time()
        Nvi = Nv[jj]
        Ndelay = 8
        tdelay = np.array([3.5e-9 + i*1e-9 for i in range(Ndelay)])

        t0r_list = []
        eff_in_list = []; eff_out_list = []; eff_list = []

        eff_in_list = np.zeros(Ndelay)
        eff_out_list = np.zeros(Ndelay)
        eff_list = np.zeros(Ndelay)

        # We create the parallel processes.
        pool = Pool(processes=Nprocs)
        # We calculate the efficiencies in parallel.
        procs = [pool.apply_async(efficiencies_delay, [i, Nvi])
                 for i in range(Ndelay)]

        # We get the results.
        aux = [procs[i].get() for i in range(Ndelay)]
        pool.close()
        pool.join()

        # We save the results with more convenient names.
        for i in range(Ndelay):
            eff_in_list[i], eff_out_list[i], eff_list[i] = aux[i]

        ####################################
        # We plot the total efficiencies.
        plt.title(r"$\mathrm{Dephasing}$", fontsize=20)
        plt.plot(tdelay*1e9, eff_list, "k+", label=r"$\eta_{\mathrm{model}}$",
                 ms=10)

        # We fit a gaussian.
        amp_the, sig_the = curve_fit(model_theoretical, tdelay, eff_list,
                                     p0=[1.0, 5.4e-9])[0]

        sigma[jj] = abs(sig_the)
        print "Nv, 1/e time, calculation time:", Nvi, abs(sig_the)*1e9, "ns",
        print (time() - t0)/60.0, "min"

        # We make a plot of the fitted gaussian.
        tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
        eff_exp = model_theoretical(tdelay_cont, amp_the, 5.4e-9)
        eff_the = model_theoretical(tdelay_cont, amp_the, sig_the)

        plt.plot(tdelay_cont*1e9, eff_exp, "k-",
                 label=r"$\eta_{\mathrm{experiment}}$")
        plt.plot(tdelay_cont*1e9, eff_the, "k:",
                 label=r"$\eta_{\mathrm{fit}}$")

        plt.xlabel(r"$t_{0\mathrm{r}}-t_{0\mathrm{w}} \ \mathrm{(ns)}$",
                   fontsize=20)
        plt.ylabel(r"$\eta$", fontsize=20)
        plt.ylim([0, None])

        plt.legend(fontsize=15)
        plt.savefig("doppler_dephasing"+str(Nvi)+".png", bbox_inches="tight")
        plt.savefig("doppler_dephasing"+str(Nvi)+".pdf", bbox_inches="tight")
        plt.close("all")

    # We plot the dephasing as a function of the number of velocity groups
    # being considered.

    # Nv = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # sigma = [3.08691878e-08, 3.05867013e-08, 1.28434666e-08, 1.14965758e-08,
    #          1.14866015e-08, 1.14866867e-08, 1.14867979e-08, 1.14869120e-08,
    #          1.14870206e-08, 1.14871203e-08, 1.14872106e-08]

    sigma = np.array(sigma)
    i = 0
    # print Nv
    # print sigma

    fig, ax = plt.subplots()
    ax.plot(Nv[i:], sigma[i:]*1e9, "r+")
    ax.set_xticks(Nv[i:])

    ax.set_xlabel(r"$N_v$", fontsize=20)
    ax.set_ylabel(r"$1/e \ \mathrm{time \ (ns)}$", fontsize=20)
    plt.savefig("doppler_dephasing_velocities.png", bbox_inches="tight")
    plt.savefig("doppler_dephasing_velocities.pdf", bbox_inches="tight")
    plt.close("all")

    print
    print "So 9 velocity classes seem like a good compromise."
