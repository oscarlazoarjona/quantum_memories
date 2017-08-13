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
from scipy.optimize import minimize

from quantum_memories import hyperfine_orca, orca
from quantum_memories.misc import set_parameters_ladder, efficiencies


def model_theoretical(t, amp, sigma):
    """Return points on a gaussian with the given amplitude and dephasing."""
    return amp*np.exp(-(t/sigma)**2)


def efficiencies_delay(i, Nv, errors, hyperfine=True):
    r"""Get the efficiencies with t0r = t0w + (3.5+i) ns."""
    name = "ORCA"+str(i)
    # We set custom parameters (different from those in settings.py)
    t0w = default_params["t0w"]
    params = set_parameters_ladder({"Nv": Nv, "Nt": 51000, "T": 16e-9,
                                    "t0r": t0w+3.5e-9+i*1e-9,
                                    "verbose": 0, "t_cutoff": 3.5e-9})

    t0w = params["t0w"]
    params["r1"] = params["r1"]*errors[0]
    params["r2"] = params["r2"]*errors[1]

    if hyperfine:
        solve = hyperfine_orca.solve
    else:
        solve = orca.solve
    t, Z, vZ, rho1, Om1 = solve(params, plots=False, name=name)
    # We calculate the efficiencies.
    aux = efficiencies(t, Om1, params, plots=True, name=name)
    eff_in, eff_out, eff = aux

    return eff_in, eff_out, eff


def Chi2(xx, hyperfine=True, efficiencies=False):
    r"""Calculate deviation."""
    t0 = time()
    Nv = 9
    Ndelay = 8
    tdelay = np.array([3.5e-9 + i*1e-9 for i in range(Ndelay)])

    eff_in_list = []; eff_out_list = []; eff_list = []

    eff_in_list = np.zeros(Ndelay)
    eff_out_list = np.zeros(Ndelay)
    eff_list = np.zeros(Ndelay)

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    procs = [pool.apply_async(efficiencies_delay, [i, Nv, xx, hyperfine])
             for i in range(Ndelay)]

    # We get the results.
    aux = [procs[i].get() for i in range(Ndelay)]
    pool.close()
    pool.join()

    # We save the results with more convenient names.
    for i in range(Ndelay):
        eff_in_list[i], eff_out_list[i], eff_list[i] = aux[i]
    eff_list = np.array(eff_list)

    ####################################
    # We fit a gaussian.
    amp_the, sig_the = curve_fit(model_theoretical, tdelay, eff_list,
                                 p0=[1.0, 5.4e-9])[0]
    ####################################
    # We calculate the normalized efficiency assuming a 5.4 ns lifetime.
    eff_gauss = model_theoretical(tdelay, 1.0, 5.4e-9)
    # We calculate the error.
    error = sum([(eff_gauss[ii]-eff_list[ii]/amp_the)**2
                 for ii in range(len(tdelay))])

    t_run = (time() - t0)/60.0
    print "xx, error, Nv, 1/e time, run time:",
    print xx, error, Nv, sig_the*1e9, "ns", t_run, "min"
    # print eff_list/amp_the
    ####################################
    # We make a plot of the fitted gaussian.
    tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
    eff_exp = model_theoretical(tdelay_cont, 1.0, 5.4e-9)
    eff_the = model_theoretical(tdelay_cont, 1.0, sig_the)

    ####################################
    # We plot the normalized efficiencies.
    plt.title(r"$\mathrm{Dephasing}$", fontsize=20)
    plt.plot(tdelay*1e9, eff_list/amp_the, "k+",
             label=r"$\eta_{\mathrm{model}}$", ms=10)

    plt.plot(tdelay_cont*1e9, eff_exp, "k-",
             label=r"$\eta_{\mathrm{experiment}}$")
    plt.plot(tdelay_cont*1e9, eff_the, "k:",
             label=r"$\eta_{\mathrm{fit}}$")

    plt.xlabel(r"$t_{0\mathrm{r}}-t_{0\mathrm{w}} \ \mathrm{(ns)}$",
               fontsize=20)
    plt.ylabel(r"$\eta_N$", fontsize=20)
    plt.ylim([0, None])

    plt.legend(fontsize=15)
    plt.savefig("doppler_dephasing.png", bbox_inches="tight")
    plt.savefig("doppler_dephasing.pdf", bbox_inches="tight")
    plt.close("all")

    if efficiencies:
        return error, eff_list

    return error


def chi2_hyperfine(xx):
    r"""Return the error using the hyperfine model."""
    return Chi2(xx, True)


def chi2_simple(xx):
    r"""Return the error using the simple model."""
    return Chi2(xx, False)


###############################################################################
# The number of processors available for parallel computing.
Nprocs = cpu_count()
# We set the default parameters taking them from settings.py.
default_params = set_parameters_ladder()

optimize = True
optimize = False
if __name__ == '__main__':

    ###########################################################################
    # We test various readout times.
    x0 = [1.0, 1.15]
    x0 = [0.91803913, 1.20252447]
    if optimize:
        print "Optimizing..."
        result = minimize(chi2_hyperfine, x0, method="Nelder-Mead",
                          options={"maxfev": 100})
        print result
        x0 = result.x
    else:
        print "Calculating error for the fitted parameters..."
        err_hyp, eta_the_hyp = Chi2(x0, True, True)
        print "For the hyperfine solver the error is:", err_hyp
        print eta_the_hyp

        err_sim, eta_the_sim = Chi2([1.0, 1.0], False, True)
        print "For the simple solver the error is:", err_sim
        print eta_the_sim

    Ndelay = 8
    tdelay = np.array([3.5e-9 + i*1e-9 for i in range(Ndelay)])
    # eta = np.array([0.696173, 0.56164038, 0.42610179, 0.30242463,
    #                 0.19931623, 0.12067496, 0.06607928, 0.03202731])

    # The experimental efficiencies.
    eta_exp = np.array([17.96301, 16.80307, 15.87454, 14.34508,
                        14.86127, 13.96623, 12.36829, 10.50846,
                        10.77325, 4.20513, 3.87795, 2.96679,
                        2.10454, 2.01409, 1.49769, 1.14375, 0.95395])/100

    tdelay_exp = np.array([2.2680, 2.7540, 2.9160, 3.2400, 3.2400,
                           3.7260, 4.0500, 4.2120, 4.8600, 6.1560,
                           6.8040, 7.4520, 8.1000, 8.1000, 8.5860,
                           9.0720, 9.7200])*1e-9

    eta_exp = np.array([17.96301, 16.80307, 15.92110, 14.36767,
                        14.93705, 13.98694, 12.38937, 10.54274,
                        10.78436, 4.21545, 3.88266, 2.97023,
                        2.10454, 2.01409, 1.49935, 1.14693, 0.95395])/100

    tdelay_exp = np.array([2.2680, 2.7540, 3.0780, 3.4020, 3.5640,
                           3.8880, 4.2120, 4.5360, 5.0220, 6.4800,
                           6.9660, 7.6140, 8.1000, 8.1000, 8.7480,
                           9.2340, 9.7200])*1e-9
    ################################
    # We fit a gauussian to the experimental efficiencies.
    amp_exp, sig_exp = curve_fit(model_theoretical, tdelay_exp, eta_exp,
                                 p0=[1.0, 5.4e-9])[0]
    eta_exp_norm = eta_exp/amp_exp
    # We make a continuous plot.
    tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
    eta_exp_cont = model_theoretical(tdelay_cont, 1.0, sig_exp)
    ################################
    # We fit a gauussian to the hyperfine theoretical efficiencies.
    amp_the_hyp, sig_the_hyp = curve_fit(model_theoretical, tdelay,
                                         eta_the_hyp,
                                         p0=[1.0, 5.4e-9])[0]
    eta_the_hyp_norm = eta_the_hyp/amp_the_hyp
    # We make a continuous plot.
    eta_the_hyp_cont = model_theoretical(tdelay_cont, 1.0, sig_the_hyp)
    ################################
    # We fit a gauussian to the simple theoretical efficiencies.
    amp_the_sim, sig_the_sim = curve_fit(model_theoretical, tdelay,
                                         eta_the_sim,
                                         p0=[1.0, 5.4e-9])[0]
    eta_the_sim_norm = eta_the_sim/amp_the_sim
    # We make a continuous plot.
    eta_the_sim_cont = model_theoretical(tdelay_cont, 1.0, sig_the_sim)
    print "sig_exp, sig_the_hyp, sig_the_sim:",
    print sig_exp*1e9, sig_the_hyp*1e9, sig_the_sim*1e9, "ns"
    ################################
    # We plot everything.
    plt.title(r"$\mathrm{Doppler \ and \ Hyperfine \ Dephasing}$",
              fontsize=15)
    plt.plot(tdelay_cont*1e9, eta_exp_cont, "g-", linewidth=0.7)
    plt.plot(tdelay_cont*1e9, eta_the_hyp_cont, "b-", linewidth=0.7)
    plt.plot(tdelay_cont*1e9, eta_the_sim_cont, "r-", linewidth=0.7)

    plt.scatter(tdelay_exp*1e9, eta_exp_norm, marker="d", s=20,
                facecolors='none', edgecolors='green',
                label="Experiment")
    plt.plot(tdelay*1e9, eta_the_hyp_norm, "+b", label="Hyperfine Theory")
    plt.plot(tdelay*1e9, eta_the_sim_norm, "+r", label="Simple Theory")

    plt.xlim([0, None])
    plt.xlabel(r"$\mathrm{Readout \ time \ (ns)}$", fontsize=15)
    plt.ylabel(r"$\eta_N$", fontsize=15)
    plt.legend()

    plt.savefig("doppler_dephasing.png", bbox_inches="tight")
    plt.close("all")

    ###########################################################################
    # We now repeat exactly the same thing using different number of velocity
    # classes.
    # Nvmax = 13
    # Nv = [2*i + 1 for i in range((Nvmax-1)/2+1)]
    # # print Nv
    # sigma = np.zeros(len(Nv))
    # print
    # print "Calculating the best number of velocity classes to use..."
    # for jj in range(len(Nv)):
    #     t0 = time()
    #     Nvi = Nv[jj]
    #     Ndelay = 8
    #     tdelay = np.array([3.5e-9 + i*1e-9 for i in range(Ndelay)])
    #
    #     t0r_list = []
    #     eff_in_list = []; eff_out_list = []; eff_list = []
    #
    #     eff_in_list = np.zeros(Ndelay)
    #     eff_out_list = np.zeros(Ndelay)
    #     eff_list = np.zeros(Ndelay)
    #
    #     # We create the parallel processes.
    #     pool = Pool(processes=Nprocs)
    #     # We calculate the efficiencies in parallel.
    #     procs = [pool.apply_async(efficiencies_delay, [i, Nvi])
    #              for i in range(Ndelay)]
    #
    #     # We get the results.
    #     aux = [procs[i].get() for i in range(Ndelay)]
    #     pool.close()
    #     pool.join()
    #
    #     # We save the results with more convenient names.
    #     for i in range(Ndelay):
    #         eff_in_list[i], eff_out_list[i], eff_list[i] = aux[i]
    #
    #     ####################################
    #     # We plot the total efficiencies.
    #     plt.title(r"$\mathrm{Dephasing}$", fontsize=20)
    #     plt.plot(tdelay*1e9, eff_list, "k+",
    #              label=r"$\eta_{\mathrm{model}}$",
    #              ms=10)
    #
    #     # We fit a gaussian.
    #     amp_the, sig_the = curve_fit(model_theoretical, tdelay, eff_list,
    #                                  p0=[1.0, 5.4e-9])[0]
    #
    #     sigma[jj] = abs(sig_the)
    #     print "Nv, 1/e time, calculation time:", Nvi, abs(sig_the)*1e9, "ns",
    #     print (time() - t0)/60.0, "min"
    #
    #     # We make a plot of the fitted gaussian.
    #     tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
    #     eff_exp = model_theoretical(tdelay_cont, amp_the, 5.4e-9)
    #     eff_the = model_theoretical(tdelay_cont, amp_the, sig_the)
    #
    #     plt.plot(tdelay_cont*1e9, eff_exp, "k-",
    #              label=r"$\eta_{\mathrm{experiment}}$")
    #     plt.plot(tdelay_cont*1e9, eff_the, "k:",
    #              label=r"$\eta_{\mathrm{fit}}$")
    #
    #     plt.xlabel(r"$t_{0\mathrm{r}}-t_{0\mathrm{w}} \ \mathrm{(ns)}$",
    #                fontsize=20)
    #     plt.ylabel(r"$\eta$", fontsize=20)
    #     plt.ylim([0, None])
    #
    #     plt.legend(fontsize=15)
    #     plt.savefig("doppler_dephasing"+str(Nvi)+".png", bbox_inches="tight")
    #     plt.savefig("doppler_dephasing"+str(Nvi)+".pdf", bbox_inches="tight")
    #     plt.close("all")
    #
    # # We plot the dephasing as a function of the number of velocity groups
    # # being considered.
    #
    # # Nv = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    # # sigma = [3.08691878e-08, 3.05867013e-08, 1.28434666e-08,
    # #          1.14965758e-08, 1.14866015e-08, 1.14866867e-08,
    # #          1.14867979e-08, 1.14869120e-08, 1.14870206e-08,
    # #          1.14871203e-08, 1.14872106e-08]
    #
    # sigma = np.array(sigma)
    # i = 0
    # # print Nv
    # # print sigma
    #
    # fig, ax = plt.subplots()
    # ax.plot(Nv[i:], sigma[i:]*1e9, "r+")
    # ax.set_xticks(Nv[i:])
    #
    # ax.set_xlabel(r"$N_v$", fontsize=20)
    # ax.set_ylabel(r"$1/e \ \mathrm{time \ (ns)}$", fontsize=20)
    # plt.savefig("doppler_dephasing_velocities.png", bbox_inches="tight")
    # plt.savefig("doppler_dephasing_velocities.pdf", bbox_inches="tight")
    # plt.close("all")
    #
    # print
    # print "So 9 velocity classes seem like a good compromise."
