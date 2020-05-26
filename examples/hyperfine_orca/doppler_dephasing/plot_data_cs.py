# -*- coding: utf-8 -*-
# Copyright (C) 2017 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""This is a template."""
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt

from scipy.constants import physical_constants
from tabulate import tabulate
from fast import State, Transition, Integer
from fast import all_atoms
from math import pi, sqrt


def gaussian_formula(t, amp, sigma):
    """Return points on a gaussian with the given amplitude and dephasing."""
    return amp*np.exp(-(t/sigma)**2)


def simple_formula(t, amp, gamma, sigma, Delta):
    """Return points on the modeled simple efficiency."""
    return amp*np.exp(-gamma*t - (Delta*sigma*t)**2)


def hyperfine_formula(t, amp, gamma, sigma, Delta, omega87, omega97,
                      A, B, C, phib, phic):
    """Return points on the modeled hyperfine efficiency."""
    eta = amp*np.exp(-gamma*t - (Delta*sigma*t)**2)
    eta = eta*abs(A +
                  B*np.exp(1j*omega87*t+1j*phib) +
                  C*np.exp(1j*omega97*t+1j*phic))**2
    eta = eta/abs(A+B*np.exp(1j*phib)+C*np.exp(1j*phic))**2
    return eta


def get_model(amp, gamma, sigma, Delta, omega87, omega97, fit_gamma=None):
    r"""Get a model to fit."""
    def f(t, A, B, C, phib, phic):
        return hyperfine_formula(t, amp, gamma, sigma, Delta,
                                 omega87, omega97, A, B, C, phib, phic)

    def g(t, gamma):
        return hyperfine_formula(t, amp, gamma, sigma, Delta,
                                 omega87, omega97, A, B, C, phib, phic)

    if fit_gamma is not None:
        A, B, C, phib, phic = fit_gamma
        return g
    else:
        return f


res = pd.read_csv("eta_the_cs.csv")
tdelay = np.array(res["tdelay_the"])
eta_sim_the = np.array(res["eta_sim_the"])
eta_hyp_the = np.array(res["eta_hyp_the"])

res = pd.read_csv("eta_exp_cs.csv")
tdelay_exp = np.array(res["tdelay_exp"])
eta_exp = np.array(res["etan_exp"])

###############################################################################
# We calculate the numbers for the lifetime formulas.

Rb85 = all_atoms[0]
Cs133 = all_atoms[2]

mRb = Rb85.mass
mCs = Cs133.mass

kB = physical_constants["Boltzmann constant"][0]

e1rb85 = State("Rb", 85, 5, 0, 1/Integer(2))
e2rb85 = State("Rb", 85, 5, 1, 3/Integer(2))
e3rb85 = State("Rb", 85, 5, 2, 5/Integer(2))

e1cs133 = State("Cs", 133, 6, 0, 1/Integer(2))
e2cs133 = State("Cs", 133, 6, 1, 3/Integer(2))
e3cs133 = State("Cs", 133, 6, 2, 5/Integer(2))

t1rb85 = Transition(e2rb85, e1rb85)
t2rb85 = Transition(e3rb85, e2rb85)
k1rb85 = 2*pi/t1rb85.wavelength
k2rb85 = 2*pi/t2rb85.wavelength
kmrb85 = abs(k2rb85-k1rb85)

t1cs133 = Transition(e2cs133, e1cs133)
t2cs133 = Transition(e3cs133, e2cs133)
k1cs133 = 2*pi/t1cs133.wavelength
k2cs133 = 2*pi/t2cs133.wavelength
kmcs133 = abs(k2cs133-k1cs133)

# We calculate the Doppler and spontaneous decay lifetimes
T = 273.15+90
sigmarb = sqrt(kB*T/mRb)
sigmacs = sqrt(kB*T/mCs)
gamma32cs = t2cs133.einsteinA
gamma32rb = t2rb85.einsteinA

taucs = (-gamma32cs+sqrt(4*(kmcs133*sigmacs)**2 +
         gamma32cs**2))/2/(kmcs133*sigmacs)**2

taurb = (-gamma32rb+sqrt(4*(kmrb85*sigmarb)**2 +
         gamma32rb**2))/2/(kmrb85*sigmarb)**2

print "A few properties of our alkalis:"
table = [["85Rb", "133Cs"],
         ["m", mRb, mCs],
         ["Deltak", kmrb85, kmcs133],
         ["sigma_v", sigmarb, sigmacs],
         ["gamma_32", gamma32rb/2/pi*1e-6, gamma32cs/2/pi*1e-6],
         ["tau", taurb*1e9, taucs*1e9]]

print tabulate(table, headers="firstrow")
###############################################################################
# We make continous plots for the formulas
tdelay_cont = np.linspace(0, tdelay[-1]*1.05, 500)
eta_simple_cont = simple_formula(tdelay_cont, 1.0, gamma32cs,
                                 sigmacs, kmcs133)
omega87 = 2*pi*27e6
omega97 = omega87 + 2*pi*23e6

# We fit the hyperfine formula to the hyperfine model.

f = get_model(1.0, gamma32cs, sigmacs, kmcs133, omega87, omega97)
p0 = [1.0/3, 1.0/3, 1.0/3, 0.0, 0.0]
res = curve_fit(f, tdelay, eta_hyp_the, p0=p0)
p0 = res[0]
print p0
eta_hyperfine_cont = f(tdelay_cont, *p0)

# We fit the hyperfine formula to the experiment.

g = get_model(1.0, gamma32cs, sigmacs, kmcs133, omega87, omega97, p0)
p0 = list(p0)+[2*pi*3.2e6*0.73]
p0 = [2*pi*3.2e6*0.73]
# p0 = [1.0/3, 1.0/3, 1.0/3, 0.0, 0.0]
res = curve_fit(g, tdelay_exp, eta_exp, p0=p0)
p0 = res[0]
print p0[-1]/2/pi*1e-6, gamma32cs/2/pi*1e-6
eta_exp_cont = g(tdelay_cont, *p0)

# We fit gaussians to everything.
amp_exp, sig_exp = curve_fit(gaussian_formula, tdelay_exp, eta_exp,
                             p0=[1.0, 5.4e-9])[0]
eta_exp_cont_gau = gaussian_formula(tdelay_cont, 1.0, sig_exp)

amp_exp, sig_sim = curve_fit(gaussian_formula, tdelay, eta_sim_the,
                             p0=[1.0, 5.4e-9])[0]
eta_sim_cont_gau = gaussian_formula(tdelay_cont, 1.0, sig_sim)

amp_exp, sig_hyp = curve_fit(gaussian_formula, tdelay, eta_hyp_the,
                             p0=[1.0, 5.4e-9])[0]
eta_hyp_cont_gau = gaussian_formula(tdelay_cont, 1.0, sig_hyp)

###############################################################################
# We find the 1/e lifetime for the hyperfine theory:
tau_hfs = 0
for i in range(len(eta_hyperfine_cont)):
    if eta_hyperfine_cont[i] < np.exp(-1.0):
        tau_hfs = tdelay_cont[i]
        break
print "The 1/e lifetime using the hyperfine theory is", tau_hfs*1e9, "ns."

tau_hfs = 0
for i in range(len(eta_hyperfine_cont)):
    if eta_exp_cont[i] < np.exp(-1.0):
        tau_hfs = tdelay_cont[i]
        break
print "The 1/e lifetime for the experiment is", tau_hfs*1e9, "ns."

###############################################################################
# We make a plot using only gaussian fits:
plt.title(r"$^{133}\mathrm{Cs \ Memory \ Lifetime}$", fontsize=15)
plt.plot(tdelay*1e9, eta_sim_the, "r+", label=r"$\mathrm{Simple Theory}$")
plt.plot(tdelay*1e9, eta_hyp_the, "b+", label=r"$\mathrm{Hyperfine Theory}$")

plt.scatter(tdelay_exp*1e9, eta_exp, marker="d", s=20,
            facecolors='none', edgecolors='green',
            label=r"$\mathrm{Experiment}$")
plt.plot(tdelay_cont*1e9, eta_exp_cont_gau, "g-", linewidth=0.75,
         label=r"$\mathrm{Gaussian \ fit}$")
plt.plot(tdelay_cont*1e9, eta_sim_cont_gau, "r-", linewidth=0.75,
         label=r"$\mathrm{Gaussian \ fit}$")
plt.plot(tdelay_cont*1e9, eta_hyp_cont_gau, "b-", linewidth=0.75,
         label=r"$\mathrm{Gaussian \ fit}$")

plt.xlim([0, 11])
plt.xlabel(r"$\tau \ \mathrm{[ns]}$", fontsize=15)
plt.ylabel(r"$\eta_\mathrm{N}$", fontsize=15)
plt.legend(loc=3, fontsize=12)

plt.savefig("doppler_dephasing_cs1.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_cs1.pdf", bbox_inches="tight")
plt.close("all")

###############################################################################
# We make a plot including the ab-initio formulas:
plt.title(r"$^{133}\mathrm{Cs \ Memory \ Lifetime}$", fontsize=15)
# plt.text(10, 0.9, "a", fontsize=15)
plt.plot(tdelay*1e9, eta_sim_the, "r+", label=r"$\mathrm{Simple \ Theory}$")
plt.plot(tdelay_cont*1e9, eta_simple_cont, "r-", linewidth=0.75,
         label=r"$\eta_{\mathrm{fs}}$")

plt.plot(tdelay*1e9, eta_hyp_the, "b+",
         label=r"$\mathrm{Hyperfine \ Theory}$")
plt.plot(tdelay_cont*1e9, eta_hyperfine_cont, "b-", linewidth=0.75,
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$ ")

plt.scatter(tdelay_exp*1e9, eta_exp, marker="d", s=20,
            facecolors='none', edgecolors='green',
            label=r"$\mathrm{Experiment}$")
plt.plot(tdelay_cont*1e9, eta_exp_cont, "g-", linewidth=0.75,
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$ ")

plt.xlim([0, 11])
plt.xlabel(r"$\tau \ \mathrm{[ns]}$", fontsize=15)
plt.ylabel(r"$\eta_\mathrm{N}$", fontsize=15)
plt.legend(loc=3, fontsize=12)

plt.savefig("doppler_dephasing_cs2.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_cs2.pdf", bbox_inches="tight")
plt.close("all")

###############################################################################
# We make a plot including the ab-initio formulas for the paper:
plt.title(r"$^{133}\mathrm{Cs \ Memory \ Lifetime}$", fontsize=15)
# plt.text(10, 0.9, "a", fontsize=15)
# plt.plot(tdelay*1e9, eta_simple, "r+", label=r"$\mathrm{Simple \ Theory}$")
plt.plot(tdelay_cont*1e9, eta_simple_cont, "g-",
         label=r"$\mathrm{Theory \ (with \ pumping)}$")

# plt.plot(tdelay*1e9, eta_hyperfine, "b+",
#          label=r"$\mathrm{Hyperfine \ Theory}$")
plt.plot(tdelay_cont*1e9, eta_hyperfine_cont, "b-",
         label=r"$\mathrm{Theory \ (without \ pumping)}$", zorder=2)

plt.scatter(tdelay_exp*1e9, eta_exp, marker="d", s=25,
            facecolors='none', edgecolors=(1, 0.5, 0),
            label=r"$\mathrm{Experiment}$", zorder=3)
plt.plot(tdelay_cont*1e9, eta_exp_cont, "-", color=(1, 0.5, 0))

plt.xlim([0, 11])
plt.xlabel(r"$\tau \ \mathrm{[ns]}$", fontsize=15)
plt.ylabel(r"$\eta_\mathrm{N}$", fontsize=15)
plt.legend(loc=3, fontsize=12)

plt.savefig("doppler_dephasing_cs3.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_cs3.pdf", bbox_inches="tight")
plt.close("all")

###############################################################################
# We save all the data
curves = np.asarray([tdelay_cont,
                     eta_simple_cont, eta_hyperfine_cont, eta_exp_cont,
                     eta_sim_cont_gau, eta_hyp_cont_gau, eta_exp_cont_gau]).T
curves = pd.DataFrame(curves)
curves.to_csv("eta_cont_cs.csv",
              header=["tdelay_cont",
                      "eta_simple_cont", "eta_hyperfine_cont",
                      "eta_exp_cont",
                      "eta_sim_cont_gau", "eta_hyp_cont_gau",
                      "eta_exp_cont_gau"])
