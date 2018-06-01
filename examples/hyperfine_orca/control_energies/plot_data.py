# -*- coding: utf-8 -*-
# Copyright (C) 2017 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This script plots the measured efficiencies for the coherent pulse
experiment, and the fitted curves using the orca.solve Maxwell-Bloch
integrator.
"""
from matplotlib import pyplot as plt
import numpy as np
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


experimental_data = np.load("experimental_data.npz")
fitted_data = np.load("fitted_data.npz")

energies = experimental_data["energies"]
eff_in_meas = experimental_data["eff_in_meas"]
eff_out_meas = experimental_data["eff_out_meas"]
eff_meas = experimental_data["eff_meas"]
error_out = experimental_data["error_out"]
error_in = experimental_data["error_in"]
error_tot = experimental_data["error_tot"]

energies_cont = fitted_data["energies_cont"]
eff_in = fitted_data["eff_in"]
eff_out = fitted_data["eff_out"]
eff = fitted_data["eff"]

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
plt.show()
