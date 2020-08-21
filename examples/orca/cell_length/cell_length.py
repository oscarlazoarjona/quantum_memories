# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example with the default settings."""

from time import time
import numpy as np
from scipy.constants import c, hbar, epsilon_0, physical_constants
from scipy.optimize import minimize
# from matplotlib import pyplot as plt

from quantum_memories.misc import set_parameters_ladder, efficiencies
from quantum_memories import orca


e_charge = physical_constants["elementary charge"][0]


def efficiency_optimize(tmeet, tdelay,
                        energy_write_error, energy_read_error,
                        delta1_error, Temperature_error, Lerror):
    r"""Return the selected efficiencies."""
    t0 = time()
    name = "optimal_efficiency"
    # We get the default values.
    r1 = default_params["r1"]
    r2 = default_params["r2"]
    t0s = default_params["t0s"]
    t0w = default_params["t0w"]
    t0r = default_params["t0r"]
    energy_pulse2 = default_params["energy_pulse2"]
    delta1 = default_params["delta1"]
    w1 = default_params["w1"]
    Temperature = default_params["Temperature"]
    L = default_params["L"]

    # We calculate the new parameters
    t0w = t0s + tmeet*1e-9
    t0r = t0w + tdelay*1e-9
    alpha_rw = np.sqrt(energy_read_error/energy_write_error)
    energy_pulse2 = energy_pulse2*energy_write_error
    # energy_pulse_read = energy_pulse2*energy_read_error/energy_write_error
    delta1 = delta1*delta1_error
    Temperature = Temperature - 273.15
    Temperature = Temperature*Temperature_error
    Temperature = Temperature + 273.15
    L = L*Lerror

    # print "energies:", energy_pulse2, energy_pulse_read
    Nt = 25500
    Nt = 35500
    params = {"r1": r1, "r2": r2, "t_cutoff": t_cutoff,
              "t0w": t0w, "t0r": t0r, "energy_pulse2": energy_pulse2,
              "alpha_rw": alpha_rw, "delta1": delta1,
              "Temperature": Temperature, "L": L,
              "Nv": 1, "Nt": Nt, "verbose": 0}
    params = set_parameters_ladder(params)
    omega_laser1 = params["omega_laser1"]

    ###########################################################################
    t, Z, vZ, rho, Om1 = orca.solve(params, plots=plotresult, name=name)

    # plt.plot(t*1e9, np.angle(rho[:, 1, :, 25]))
    # plt.savefig("phases.png", bbox_inches="tight")
    # plt.close("all")
    # We calculate the explicit_decoherence
    explicit_decoherence = np.exp(-(tdelay/5.4)**2)

    aux = efficiencies(t, Om1, params,
                       plots=True, name=name,
                       explicit_decoherence=explicit_decoherence)
    eff_in, eff_out, eff = aux
    ###########################################################################
    # We get the level photons per nanosecond at the right end of the cell.
    for i in range(len(t)):
        if t[i] > params["t_cutoff"]:
            i_cutoff = i
            break
    const1 = np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1
    phots = np.real(Om1[:, -1]*Om1[:, -1].conjugate())*const1*1e-9
    ###########################################################################
    # phots_in = np.real(Om1[:, 0]*Om1[:, 0].conjugate())*const1*1e-9
    # trace = file("trace.csv", "w")
    # trace.write(str(list(t*1e9))[1:-1])
    # trace.write("\n")
    # trace.write(str(list(phots))[1:-1])
    # trace.write("\n")
    # trace.write(str(list(phots_in))[1:-1])
    # trace.close()

    ###########################################################################

    teff = time()-t0
    # We penalize for various desired lab-conditions!
    if phots[i_cutoff] > 0.01:
        return -1.0, -1.0, -1.0, teff
    elif delta1_error < 0.5:
        return -2.0, -2.0, -2.0, teff
    # elif energy_pulse2 >= 1500e-12:
    #     return -3.0, -3.0, -3.0, teff
    # elif energy_pulse2*alpha_rw**2 >= 1500e-12:
    #     return -4.0, -4.0, -4.0, teff
    elif Temperature-273.15 >= 120.0:
        return -5.0, -5.0, -5.0, teff
    else:
        return eff_in, eff_out, eff, teff


def inefficiency_in(x):
    r"""Return the read-in efficiency."""
    tmeet, tdelay, energy_write_error, energy_read_error = x[:4]
    delta1_error, Temperature_error = x[4:]
    aux = efficiency_optimize(tmeet, tdelay,
                              energy_write_error, energy_read_error,
                              delta1_error, Temperature_error)

    eff_in, eff_out, eff, teff = aux
    print x, eff_in, eff, teff
    return 1-eff_in


def inefficiency_tot(x):
    r"""Return the efficiency."""
    tmeet, tdelay, energy_write_error, energy_read_error = x[:4]
    delta1_error, Temperature_error, Lerror = x[4:]
    aux = efficiency_optimize(tmeet, tdelay,
                              energy_write_error, energy_read_error,
                              delta1_error, Temperature_error, Lerror)

    eff_in, eff_out, eff, teff = aux
    print_params(x)
    print x, eff_in, eff, teff
    return 1-eff


def print_params(x):
    r"""Get parameters."""
    print "Meeting point from the center of the cell:", x[0]*c/2*1e-7, "cm"
    print "Readout time:                             ", x[1], "ns"
    print "Write control pulse energy:               ", x[2] * \
        default_params["energy_pulse2"]*1e12, "pJ"
    print "Read control pulse energy:                ", x[3] * \
        default_params["energy_pulse2"]*1e12, "pJ"
    print "Detuning:                                 ", x[4] * \
        default_params["delta1"]/2/np.pi*1e-9, "GHz"

    Temperature = default_params["Temperature"] - 273.15
    Temperature = Temperature*x[5]
    print "Temperature:                              ", Temperature,
    print "Celsius degrees"
    print "Cell length:                              ", x[6] * \
        default_params["L"]*100, "cm."


default_params = set_parameters_ladder()
# t_cutoff = 3.0e-9
optimize = False
# optimize = True
if __name__ == '__main__':

    # This set of parameters has the highest total efficiency I've seen, but
    # it works for zero detuning (it is not ORCA).
    # read-in efficiency: 0.9998
    # total efficiency:   0.884
    x0 = [-1.96926424e-01, 1.47885844e+00, 2.24690956e+01, 1.71542686e+02,
          -9.11451876e-04, 1.00451703e+00, 1.0]
    t_cutoff = 1.9e-9

    # This set of parameters has the -3 GHz detuning, but very high control
    # pulse energies.
    # read-in efficiency: 0.9972
    # total efficiency:   0.8258
    x0 = [-2.14949065e-01, 1.99730372e+00, 7.34677360e+01, 5.73368265e+02,
          5.00000179e-01, 1.26338969e+00, 1.0]

    x0 = [-1.73145733e-01, 1.85864299e+00, 6.14331996e+01, 9.23529599e+02,
          5.00130045e-01, 1.28751780e+00, 8.91922071e-01]

    t_cutoff = 2.4e-9

    # This set of parameters is both ORCA and feasible.
    # read-in efficiency: 0.8225
    # total efficiency:   0.5105
    x0 = [0.05373289, 2.26084362, 1.95388604, 38.19563015,
          1.17706132, 1.33333333, 1.0]
    t_cutoff = 2.7e-9

    # print_params(x0)
    if optimize:
        print "Optimizing..."
        plotresult = False
        res = minimize(inefficiency_tot, x0,
                       method="Nelder-Mead", options={"maxfev": 1000})
        print res
        x0 = res.x

    print "Calculating inefficiency..."
    plotresult = True
    print inefficiency_tot(x0)
