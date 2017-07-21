# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This example fits parameters to measured efficiencies.

We have a series of efficiencies measured for different control pulse
energies. We fit the electric dipole matrix elements and the position where
the signal and control pulses meet.
"""

from matplotlib import pyplot as plt
from scipy.constants import physical_constants, c
import numpy as np
from multiprocessing import Pool, cpu_count
from scipy.optimize import minimize
from time import time

from quantum_memories.misc import set_parameters_ladder, Measurement
from quantum_memories.orca import efficiencies_r1r2t0w
from quantum_memories.orca import efficiencies_t0wenergies
# from quantum_memories.orca import optimize_signal
# from quantum_memories.settings_ladder import optimize


def chi2(xx):
    r"""Get the sum of squared residues for the fit of the efficiencies.

    This is the objective function of the minimization process.
    """
    t0 = time()
    eff_in = np.zeros(len(energies))
    eff_out = np.zeros(len(energies))
    eff = np.zeros(len(energies))

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    procs = [pool.apply_async(efficiencies_r1r2t0w,
                              [energies[i], xx, explicit_decoherence,
                               str(i)])
             for i in range(len(energies))]

    # We get the results.
    aux = [procs[i].get() for i in range(len(energies))]
    pool.close()
    pool.join()

    # We save the results with more convenient names.
    for i in range(len(energies)):
        eff_in[i], eff_out[i], eff[i] = aux[i]

    # We calculate the sum of squared residues.
    chi22 = 0.0
    chi22 += sum([(eff_meas[i]-eff[i])**2 for i in range(len(energies))])
    chi22 += sum([(eff_out_meas[i]-eff_out[i])**2
                 for i in range(len(energies))])
    chi22 += sum([(eff_in_meas[i]-eff_in[i])**2
                  for i in range(len(energies))])

    if optimize:
        print xx, chi22, time() - t0

    #########################################################################
    # We calculate the optimized efficiencies.
    # opt_eff = []
    # act_eff = []
    # print "Calculating optimized efficiencies..."
    # energies2 = np.linspace(10e-12, 2500e-12, 20)
    # energies2 = energies
    # energies2 = np.linspace(10e-12, 1000e-12, 40)
    # for i in range(len(energies2)):
    #
    #     params = efficiencies_r1r2t0w(energies2[i], xx, explicit_decoherence,
    #                                   str(i), return_params=True)
    #     print
    #     print "Data point", i, energies2[i]*1e12
    #     params["USE_HG_SIG"] = True
    #
    #     # params["L"] = 0.01
    #     # params["D"] = 1.05*params["L"]
    #     # params["Nt"] = 180000
    #     # params["sampling_rate"] = 50*8
    #
    #     params["t0w"] = params["t0s"]
    #     t0s_new = 2e-9
    #     corr = t0s_new-params["t0s"]
    #     params["t0s"] = t0s_new
    #     params["t0w"] = params["t0s"]
    #     params["t0r"] = params["t0w"]+3.5e-9
    #
    #     corr2 = 1e-9
    #     params["t_cutoff"] = 2.7e-9
    #     params["t0w"] = params["t0s"] - 5.373288999998793e-11
    #     params["t0r"] = params["t0w"] + 2.26e-9 + corr2
    #     params["delta1"] = -7.06e9*2*np.pi
    #     params["energy_pulse1"] = energies2[i]
    #     params["alpha_rw"] = np.sqrt(954.89075375/48.8)
    #     params["Temperature"] = 273.15 + 119.9
    #
    #     Llong = 0.14
    #     corr3 = Llong - params["L"]
    #     params["L"] = Llong
    #     params["D"] = 1.05*params["L"]
    #     params["t0w"] = params["t0w"] + corr3/c
    #     params["t0r"] = params["t0r"] + corr3/c
    #     params["t_cutoff"] = 4.5e-9
    #
    #     # params["t_cutoff"] = params["t_cutoff"]+corr+corr2*0.5
    #     params["explicit_decoherence"] = 1.0
    #     aux = optimize_signal(params, i, plots=True, check=True)
    #     opt_in, opt_out, opt_eta, act_eta = aux
    #     opt_eff += [opt_eta]
    #     act_eff += [act_eta]
    # chi22 = 1.0
    # plt.plot(energies2*1e12, opt_eff, "g-")
    # plt.plot(energies2*1e12, act_eff, "g--")

    # We plot the measured efficiencies.
    plt.errorbar(energies*1e12, eff_in_meas, yerr=error_in,
                 fmt="ro", ms=3, capsize=2, label=r"$\eta_{\mathrm{in}}$")
    plt.errorbar(energies*1e12, eff_out_meas, yerr=error_out,
                 fmt="bo", ms=3, capsize=2, label=r"$\eta_{\mathrm{out}}$")
    plt.errorbar(energies*1e12, eff_meas, yerr=error_tot,
                 fmt="ko", ms=3, capsize=2, label=r"$\eta_{\mathrm{tot}}$")

    # We plot the calculated efficiencies.
    plt.plot(energies*1e12, eff_in, "r-")
    plt.plot(energies*1e12, eff_out, "b-")
    plt.plot(energies*1e12, eff, "k-")

    plt.ylim([-0.02, None])
    plt.xlabel(r"$E_c \ \mathrm{(pJ)}$", fontsize=20)
    plt.ylabel(r"$\mathrm{Efficiency}$", fontsize=20)
    plt.legend(fontsize=15, loc=2)

    plt.savefig("efficiencies.png", bbox_inches="tight")
    plt.close("all")

    return chi22


# We set the default parameters, taken from settings.py.
default_params = set_parameters_ladder(fitted_couplings=False)
optimize = True
optimize = False
if __name__ == '__main__':

    Nprocs = cpu_count()

    a0 = physical_constants["Bohr radius"][0]

    # These are the control powers in Watts.
    powers = [0.0919800, 0.0700700, 0.0495500, 0.0306500, 0.0104100,
              0.0050300, 0.0020800, 0.0062800, 0.0208700, 0.0407100,
              0.0590500, 0.0791000]
    # The measured in-efficiencies.
    eff_in_meas = [0.720255, 0.724511, 0.675580, 0.530461, 0.211418,
                   0.067589, 0.034490, 0.114708, 0.370442, 0.562352,
                   0.722136, 0.754702]
    eff_in_meas = [0.720255, 0.724511, 0.675580, 0.530461, 0.211418,
                   0.067589, 0.034490, 0.114708, 0.370442, 0.562352,
                   0.722136, 0.754702]

    # The measured total efficiencies.
    eff_meas = [0.0999324, 0.1341076, 0.1653211, 0.1491960, 0.0489706,
                0.0152489, 0.0025864, 0.0217840, 0.1143661, 0.1662100,
                0.1680261, 0.1234118]

    # We sort things properly:
    aux = map(None, powers, eff_meas, eff_in_meas)
    aux = sorted(aux)

    powers = np.array([it[0] for it in aux])
    eff_meas = np.array([it[1] for it in aux])
    eff_in_meas = np.array([it[2] for it in aux])
    eff_out_meas = eff_meas/eff_in_meas

    # We calculate the error bars for the readout efficiency.
    error_in = []
    error_out = []
    error_tot = []
    for i in range(len(eff_meas)):
        dati = Measurement(eff_meas[i], eff_in_meas[i]*0.01)
        dat_ini = Measurement(eff_in_meas[i], eff_meas[i]*0.01)
        dat_outi = dati/dat_ini

        error_in += [eff_in_meas[i]*0.01]
        error_out += [dat_outi.sigma]
        error_tot += [eff_meas[i]*0.01]

    rep_rate = 160e6
    energies = powers/rep_rate*9/10.0

    # We set the default parameters, taken from settings.py.
    explicit_decoherence = np.exp(-(3.5/5.4)**2)

    # The default errors for r1, r2, and t0w. The first two are the fractions
    # of the initial guesses for r1, r2 that are to be used, and the third is
    # the correction to t0w in ns.
    # These are the fits for the experiments with the coherent pulses.
    x14 = [0.23765377, 0.78910769, -0.32736357]
    # x14 = [0.23695544, 0.7606034, -0.35714997]
    # x14 = [0.25088347, 0.73943283, -0.43628541]
    # x14 = [0.23382219, 0.81674232, -0.41974531]
    x14 = [0.23380502, 0.81678002, -0.41974288]
    x14 = [0.23543177, 0.81360687, -0.420853]  # A nice one! 0.0201248140575
    x14 = [0.2556521, 0.72474758, -0.43663623]
    # Couplings that reproduce the sing-photon experiment.
    x15 = [0.2556521, 0.63474758, -0.43663623]

    if optimize:
        print "Optimizing..."
        result = minimize(chi2, x14, method="Nelder-Mead",
                          options={"maxfev": 100})
        print result
        x14 = result.x
    else:
        print "Calculating error for the fitted parameters..."
        err = chi2(x14)
        print "The error is", err

    tss = x14[2]*1e-9
    print "For the coherent pulses experiment:"
    print "The fitted values are:"
    print "r1:", x14[0]*default_params["r1"]/a0, "Bohr radii"
    print "r2:", x14[1]*default_params["r2"]/a0, "Bohr radii"
    print "t0w - t0s:",
    print tss*1e9, "ns"
    print "overlap at:", tss/2.0*c/2.0*1e2,
    print "cm from the center of the cell."

    #########################################################################
    # We create a continuous plot.
    print
    print "Calculating a continuous plot..."
    energies_cont = np.linspace(0.0, 550e-12, 150)
    eff_in = np.zeros(len(energies_cont))
    eff_out = np.zeros(len(energies_cont))
    eff = np.zeros(len(energies_cont))

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    procs = [pool.apply_async(efficiencies_r1r2t0w,
                              [energies_cont[i], x14, explicit_decoherence])
             for i in range(len(energies_cont))]

    # We get the results.
    aux = [procs[i].get() for i in range(len(energies_cont))]
    pool.close()
    pool.join()

    # We save the results with more convenient names.
    for i in range(len(energies_cont)):
        eff_in[i], eff_out[i], eff[i] = aux[i]

    # We plot the measured efficiencies.
    plt.errorbar(energies*1e12, eff_in_meas, yerr=error_in,
                 fmt="ro", ms=3, capsize=2, label=r"$\eta_{\mathrm{in}}$")
    plt.errorbar(energies*1e12, eff_out_meas, yerr=error_out,
                 fmt="bo", ms=3, capsize=2, label=r"$\eta_{\mathrm{out}}$")
    plt.errorbar(energies*1e12, eff_meas, yerr=error_tot,
                 fmt="ko", ms=3, capsize=2, label=r"$\eta_{\mathrm{tot}}$")

    # We plot the calculated efficiencies.
    plt.plot(energies_cont*1e12, eff_in, "r-")
    plt.plot(energies_cont*1e12, eff_out, "b-")
    plt.plot(energies_cont*1e12, eff, "k-")

    plt.ylim([-0.02, None])
    plt.xlim([energies_cont[0]*1e12, energies_cont[-1]*1e12])

    plt.xlabel(r"$E_c \ \mathrm{(pJ)}$", fontsize=20)
    plt.ylabel(r"$\mathrm{Efficiency}$", fontsize=20)
    plt.legend(fontsize=15, loc=2)

    plt.savefig("control_energies.png", bbox_inches="tight")
    plt.savefig("control_energies.pdf", bbox_inches="tight")
    plt.close("all")

    ##########################################################################
    # We use the fitted values to the efficiencies at
    # different meeting points
    print
    print "Calculating the efficiencies for the single-photon experiment..."
    ewrite = 210e-12  # Joules
    eread = 970e-12  # Joules

    smin = -0.8; smax = -smin  # -0.19
    sleft = -0.30; sright = -0.27
    Ns = 8*4
    s = np.linspace(smin, smax, Ns)

    eta_in = np.zeros(Ns)
    eta_out = np.zeros(Ns)
    eta_tot = np.zeros(Ns)

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    explicit_decoherence = np.exp(-(3.5/5.4)**2)
    explicit_decoherence = np.exp(-(4.5/5.4)**2)
    procs = [pool.apply_async(efficiencies_t0wenergies,
                              [[s[i], ewrite, eread], explicit_decoherence])
             for i in range(Ns)]

    # We get the results.
    aux = [procs[i].get() for i in range(Ns)]
    pool.close()
    pool.join()

    for i in range(Ns): eta_in[i], eta_out[i], eta_tot[i] = aux[i]

    #############################################
    # Let us plot this.
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.plot(s, eta_in, "r:",
             label=r"$\mathrm{Theory} \ \eta_{\mathrm{in }}$")
    ax1.plot(s, eta_out, "b:",
             label=r"$\mathrm{Theory} \ \eta_{\mathrm{out}}$")
    ax1.plot(s, eta_tot, "k:",
             label=r"$\mathrm{Theory} \ \eta_{\mathrm{tot}}$")
    ax1.legend(fontsize=15, loc=1)

    ax1.plot([sleft, sleft], [0, 0.8], color="green", alpha=0.5)
    ax1.plot([sright, sright], [0, 0.8], color="green", alpha=0.5)

    eta_in_ind_phot = 0.685
    eta_tot_ind_phot = 0.15
    eta_out_ind_phot = eta_tot_ind_phot/eta_in_ind_phot

    error_in = 0.019
    error_tot = 0.019
    error_out = (Measurement(eta_tot_ind_phot, error_tot) /
                 Measurement(eta_in_ind_phot, error_in)).sigma

    ax1.fill_between(s, eta_in_ind_phot-error_in, eta_in_ind_phot+error_in,
                     facecolor='red', alpha=0.5)
    ax1.fill_between(s, eta_out_ind_phot-error_out, eta_out_ind_phot+error_out,
                     facecolor='blue', alpha=0.5)
    ax1.fill_between(s, eta_tot_ind_phot-error_tot, eta_tot_ind_phot+error_tot,
                     facecolor='black', alpha=0.5)

    ax1.plot(s, [eta_in_ind_phot]*Ns, "r")
    ax1.plot(s, [eta_out_ind_phot]*Ns, "b")
    ax1.plot(s, [eta_tot_ind_phot]*Ns, "k")

    ax1.set_xlabel(r"$t_{\mathrm{w0}} -t_{\mathrm{s0}} \ \mathrm{(ns)}$",
                   fontsize=20)
    ax1.set_ylabel(r"$\mathrm{Efficiency}$", fontsize=20)
    ax1.set_xlim([smin, smax])
    ax1.set_ylim([0, None])

    ax2 = ax1.twiny()

    def tick_function(X):
        """Return the tick values in space."""
        V = c*X*1e-9/2.0*100

        return [str(z) for z in V]

    x2_ticks = [-12.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0]
    x2_ticks = range(-12, 13, 2)
    new_tick_locations = np.array([i/c/1e-9*2/100 for i in x2_ticks])
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"$\mathrm{Overlap \ point \ (cm)}$", fontsize=20)

    plt.savefig("delay_dependence.png", bbox_inches="tight")
    plt.savefig("delay_dependence.pdf", bbox_inches="tight")
    plt.close("all")
    ##############################################

    print
    print "For the single-photon experiment:"
    print "The fitted values are:"
    print "r1:", x14[0]*default_params["r1"]/a0, "Bohr radii"
    print "r2:", x14[1]*default_params["r2"]/a0, "Bohr radii"
    print "t0w - t0s:", (sleft+sright)/2.0, "ns"
    print "overlap at:", (sleft+sright)*1e-9/2.0/2.0*c/2.0*1e2,
    print "cm from the center of the cell."
    print "overlap at:", tss/2.0*c/2.0*1e2
