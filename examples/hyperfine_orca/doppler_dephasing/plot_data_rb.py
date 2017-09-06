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


def hyperfine_formula(t, amp, gamma, sigma, Delta, omega87, omega97, omega107,
                      A, B, C, D, phib, phic, phid):
    """Return points on the modeled hyperfine efficiency."""
    eta = amp*np.exp(-gamma*t - (Delta*sigma*t)**2)
    eta = eta*abs(A +
                  B*np.exp(1j*omega87*t+1j*phib) +
                  C*np.exp(1j*omega97*t+1j*phic) +
                  D*np.exp(1j*omega107*t+1j*phid))**2
    # eta = eta/abs(A+B*np.exp(1j*phib)+C*np.exp(1j*phic))**2
    return eta


def get_model(gamma, sigma, Delta, omega87, omega97, omega107, fit_gamma=None):
    r"""Get a model to fit."""
    def f(t, amp, A, B, C, D, phib, phic, phid):
        return hyperfine_formula(t, amp, gamma, sigma, Delta,
                                 omega87, omega97, omega107,
                                 A, B, C, D, phib, phic, phid)

    def g(t, gamma):
        return hyperfine_formula(t, amp, gamma, sigma, Delta,
                                 omega87, omega97, A, B, C, phib, phic)

    if fit_gamma is not None:
        A, B, C, phib, phic = fit_gamma
        return g
    else:
        return f


tdelay_exp = [5.1788101408450705e-09, 9.9592504225352128e-09,
              1.5337245633802818e-08, 2.011768591549296e-08,
              2.529649605633803e-08, 3.0276121126760566e-08,
              3.5255746478873246e-08, 4.0434557746478886e-08,
              4.52149971830986e-08, 5.0393808450704239e-08,
              5.517424788732396e-08, 6.015387323943664e-08,
              6.5315278873239461e-08, 7.0511492957746506e-08,
              7.529193239436622e-08, 8.0072374647887346e-08,
              8.5649554929577499e-08, 9.0629180281690165e-08,
              9.5210433802816913e-08, 1.0019005915492959e-07,
              1.0516968450704226e-07, 1.1054767887323945e-07,
              1.1532812112676057e-07, 1.2050692957746481e-07,
              1.2508818591549298e-07, 1.3026699718309861e-07,
              1.3564499154929577e-07, 1.404254309859155e-07,
              1.4580342535211266e-07, 1.5018549577464785e-07,
              1.5516512112676056e-07, 1.6034393239436617e-07,
              1.6532355774647887e-07, 1.7010399718309857e-07,
              1.7528280845070421e-07, 1.8046161690140844e-07,
              1.8544124225352112e-07, 1.904208676056338e-07,
              1.9520130985915493e-07, 2.0018093521126761e-07]

eta_exp = [0.20500400458756521, 0.06670728759622839,
           0.030760132618571991, 0.053227104479607261,
           0.10964417892068903, 0.16855758121498934,
           0.20500400458756521, 0.16056932219972492,
           0.045738113859262283, 0.021773329752779624,
           0.055224170998595869, 0.089673534912872471,
           0.10989380958780416, 0.10614931427763159,
           0.074196285277262727, 0.03675132511484798,
           0.011788032461283361, 0.023770375089700164,
           0.045238845464342668, 0.049232978502319565,
           0.038748384573147235, 0.022771852421239813,
           0.0092916551832402938, 0.0037997805067081221,
           0.0042990418409383779, 0.0097909165174705493,
           0.012287293795513616, 0.010290248458592754,
           0.0047983031751684729, -0.0006936421082558077,
           -0.0006936421082558077, -0.0016921647767161587,
           -0.0006936421082558077, 0.00030488056020454339,
           0.00030488056020454339, -0.00019438077402571241,
           -0.00069364210825596827, -0.00069364210825596827,
           -0.00019438077402571241, -0.00019438077402571241]

eta_hyp = [0.06233438, 0.00794021, 0.00257542, 0.00398528, 0.00833282,
           0.03126975, 0.05285888, 0.05447093, 0.02845603, 0.00445009,
           0.01632845, 0.02317063, 0.01021712, 0.02350214, 0.0380736,
           0.02224003, 0.00251957, 0.00718363, 0.01727268, 0.02114651,
           0.01280552, 0.00557187, 0.00713151, 0.00351195, 0.0022527,
           0.01314799, 0.0169971, 0.00835812, 0.00223538, 0.00139977,
           0.0004076, 0.00016485, 0.0018742, 0.00532722, 0.00699998,
           0.00332763, 0.00023941, 0.00037594, 0.00037573, 0.00030398]

tdelay_exp = np.array(tdelay_exp)
eta_exp = np.array(eta_exp)
eta_hyp = np.array(eta_hyp)*eta_exp[0]/eta_hyp[0]
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
tdelay_cont = np.linspace(0, tdelay_exp[-1]*1.05, 500)
eta_simple_cont = simple_formula(tdelay_cont, 1.0, gamma32rb,
                                 sigmarb, kmrb85)*0.22
omega87 = 2*pi*28.8254e6
omega97 = omega87 + 2*pi*22.954e6
omega107 = omega97 + 2*pi*15.9384e6

# We fit the hyperfine formula to the experimental data.
f = get_model(gamma32rb, sigmarb, kmrb85, omega87, omega97, omega107)
p0 = [1.0, 1.0/3, 1.0/3, 1.0/3, 0.0, 0.0, 0.0, 0.0]
res = curve_fit(f, tdelay_exp, eta_exp, p0=p0)
p0 = res[0]
amp, A, B, C, D, phib, phic, phid = p0
amp01 = amp*abs(A + B*np.exp(1j*phib) + C*np.exp(1j*phic)+D*np.exp(1j*phid))**2
eta_exp_cont = f(tdelay_cont, *p0)/amp01

# We fit the hyperfine formula to the hyperfine model.
res = curve_fit(f, tdelay_exp, eta_hyp, p0=p0)
p0 = res[0]
amp, A, B, C, D, phib, phic, phid = p0
print A, B, C, D
amp02 = amp*abs(A + B*np.exp(1j*phib) + C*np.exp(1j*phic)+D*np.exp(1j*phid))**2
eta_hyp_cont = f(tdelay_cont, *p0)/amp02

# We make a plot including the ab-initio formulas:
###############################################################################
plt.title(r"$^{87}\mathrm{Rb \ Memory \ Lifetime}$", fontsize=15)
plt.plot(tdelay_exp*1e9, eta_hyp/amp02, "rx",
         label=r"$\mathrm{Hyperfine \ Theory}$")

plt.plot(tdelay_cont*1e9, eta_hyp_cont, "r-",
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$")

plt.plot(tdelay_cont*1e9, eta_simple_cont/eta_simple_cont[0], "g-",
         label=r"$\mathrm{Simple \ Theory}$")

plt.xlim([0, tdelay_cont[-1]*1e9])
plt.ylim([-0.02, 1.02])
plt.xlabel(r"$\mathrm{Readout \ time \ (ns)}$", fontsize=15)
plt.ylabel(r"$\eta_N$", fontsize=15)
plt.legend(loc=1, fontsize=12)

plt.savefig("doppler_dephasing_rb1.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_rb1.pdf", bbox_inches="tight")

#############################################
plt.plot(tdelay_exp*1e9, eta_exp/amp01, "bx",
         label=r"$\mathrm{Experiment}$")
plt.plot(tdelay_cont*1e9, eta_exp_cont, "b-",
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$")
plt.legend(loc=1, fontsize=12)

plt.savefig("doppler_dephasing_rb2.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_rb2.pdf", bbox_inches="tight")

plt.close("all")

# We save all the data.
###############################################################################
continous = np.asarray([tdelay_cont, eta_hyp_cont,
                        eta_simple_cont/eta_simple_cont[0]]).T
continous = pd.DataFrame(continous)
continous.to_csv("eta_continous_rb.csv",
                 header=["tdelay_cont", "etan_hyp", "etan_sim"])

maxwell_bloch = np.asarray([tdelay_exp, eta_hyp/amp02]).T
maxwell_bloch = pd.DataFrame(maxwell_bloch)
maxwell_bloch.to_csv("eta_maxwell_bloch_rb.csv",
                     header=["tdelay_exp", "etan_hyp"])
