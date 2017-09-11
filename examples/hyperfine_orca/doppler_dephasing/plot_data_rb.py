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

tdelay_hyp = np.linspace(5e-9, 200e-9, 40*5)
eta_hyp = [6.49292859e-02, 5.07382837e-02, 3.72369196e-02, 2.53027370e-02,
           1.55529475e-02, 8.30006291e-03, 3.55003301e-03, 1.04036579e-03,
           3.10074579e-04, 7.89166056e-04, 1.89333699e-03, 3.10977792e-03,
           4.06238220e-03, 4.54870105e-03, 4.54593917e-03, 4.18832154e-03,
           3.72243106e-03, 3.44999435e-03, 3.66867785e-03, 4.62069832e-03,
           6.45670720e-03, 9.21901639e-03, 1.28443964e-02, 1.71832101e-02,
           2.20290340e-02, 2.71516926e-02, 3.23268102e-02, 3.73565539e-02,
           4.20787257e-02, 4.63642798e-02, 5.01060320e-02, 5.32033427e-02,
           5.55483620e-02, 5.70190292e-02, 5.74823940e-02, 5.68093679e-02,
           5.48992035e-02, 5.17094189e-02, 4.72850647e-02, 4.17805480e-02,
           3.54679462e-02, 2.87276684e-02, 2.20202379e-02, 1.58413587e-02,
           1.06656516e-02, 6.88691070e-03, 4.76401366e-03, 4.38138522e-03,
           5.63111059e-03, 8.22079292e-03, 1.17073629e-02, 1.55530074e-02,
           1.91958079e-02, 2.21251704e-02, 2.39511342e-02, 2.44573284e-02,
           2.36296432e-02, 2.16562276e-02, 1.88986884e-02, 1.58386209e-02,
           1.30073039e-02, 1.09087957e-02, 9.94758243e-03, 1.03710851e-02,
           1.22350238e-02, 1.53961088e-02, 1.95324368e-02, 2.41879746e-02,
           2.88341175e-02, 3.29391083e-02, 3.60353805e-02, 3.77756391e-02,
           3.79705464e-02, 3.66039907e-02, 3.38252781e-02, 2.99210201e-02,
           2.52720957e-02, 2.03028565e-02, 1.54301327e-02, 1.10189900e-02,
           7.35039491e-03, 4.60370389e-03, 2.85424203e-03, 2.08401199e-03,
           2.20173021e-03, 3.06754052e-03, 4.51771176e-03, 6.38538753e-03,
           8.51492118e-03, 1.07689175e-02, 1.30287872e-02, 1.51908811e-02,
           1.71609059e-02, 1.88494323e-02, 2.01706591e-02, 2.10454367e-02,
           2.14086698e-02, 2.12194420e-02, 2.04712494e-02, 1.92008297e-02,
           1.74913992e-02, 1.54699917e-02, 1.32963217e-02, 1.11477068e-02,
           9.19750740e-03, 7.59242269e-03, 6.43395894e-03, 5.76328135e-03,
           5.55555984e-03, 5.72353213e-03, 6.13091440e-03, 6.61324900e-03,
           7.00367668e-03, 7.15935278e-03, 6.98495574e-03, 6.44883567e-03,
           5.58976788e-03, 4.51253655e-03, 3.37400509e-03, 2.36019286e-03,
           1.65921138e-03, 1.43186721e-03, 1.78838297e-03, 2.76969271e-03,
           4.33970213e-03, 6.38838984e-03, 8.74345561e-03, 1.11922786e-02,
           1.35081129e-02, 1.54779625e-02, 1.69265503e-02, 1.77354298e-02,
           1.78554245e-02, 1.73068547e-02, 1.61731057e-02, 1.45864330e-02,
           1.27081045e-02, 1.07073932e-02, 8.74127105e-03, 6.94005568e-03,
           5.39375909e-03, 4.14869024e-03, 3.21185551e-03, 2.55423908e-03,
           2.12288703e-03, 1.85408537e-03, 1.68291960e-03, 1.55274907e-03,
           1.42288155e-03, 1.26964567e-03, 1.08549464e-03, 8.77514561e-04,
           6.60852869e-04, 4.54475130e-04, 2.77375984e-04, 1.44878833e-04,
           6.81641395e-05, 5.43469430e-05, 1.07840697e-04, 2.32490793e-04,
           4.32669142e-04, 7.14327428e-04, 1.08354963e-03, 1.54523025e-03,
           2.10027580e-03, 2.74191594e-03, 3.45408943e-03, 4.20891200e-03,
           4.96746908e-03, 5.68289904e-03, 6.30260816e-03, 6.77549352e-03,
           7.05760016e-03, 7.11694071e-03, 6.93964817e-03, 6.53236487e-03,
           5.92167971e-03, 5.15285733e-03, 4.28457926e-03, 3.38223460e-03,
           2.51099788e-03, 1.72888114e-03, 1.08178247e-03, 5.93276280e-04,
           2.72028526e-04, 1.06638665e-04, 7.04134086e-05, 1.26402824e-04,
           2.33284988e-04, 3.51424036e-04, 4.48180762e-04, 5.01998112e-04,
           5.09333495e-04, 4.63761824e-04, 3.84893007e-04, 2.94669021e-04,
           2.18240891e-04, 1.79421482e-04, 1.96753482e-04, 2.80866472e-04]

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
res = curve_fit(f, tdelay_hyp, eta_hyp, p0=p0)
p0 = res[0]
amp, A, B, C, D, phib, phic, phid = p0
print A, B, C, D
amp02 = amp*abs(A + B*np.exp(1j*phib) + C*np.exp(1j*phic)+D*np.exp(1j*phid))**2
eta_hyp_cont = f(tdelay_cont, *p0)/amp02

###############################################################################
# We find the 1/e lifetime with pumping.
tau_hfs = 0
eta_pump = eta_simple_cont/eta_simple_cont[0]

for i in range(len(eta_pump)):
    if eta_pump[i] < np.exp(-1.0):
        tau_fs = tdelay_cont[i]
        break
print "The 1/e lifetime with pumping is", tau_fs*1e9, "ns."
#######################################
# We find the 1/e lifetime without pumping.
tau_hfs = 0
eta_no_pump = eta_hyp_cont

for i in range(len(eta_pump)):
    if eta_no_pump[i] < np.exp(-1.0):
        tau_fs = tdelay_cont[i]
        break
print "The 1/e lifetime without pumping is", tau_fs*1e9, "ns."

# We make a plot including the ab-initio formulas:
###############################################################################
plt.title(r"$^{87}\mathrm{Rb \ Memory \ Lifetime}$", fontsize=15)
plt.plot(tdelay_hyp*1e9, eta_hyp/amp02, "rx-",
         label=r"$\mathrm{Hyperfine \ Theory}$", ms=5)

plt.plot(tdelay_cont*1e9, eta_hyp_cont, "b-",
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$")

plt.plot(tdelay_cont*1e9, eta_simple_cont/eta_simple_cont[0], "g-",
         label=r"$\mathrm{Simple \ Theory}$")

plt.xlim([0, tdelay_cont[-1]*1e9])
plt.ylim([0, 1.02])
plt.xlabel(r"$\tau \ \mathrm{[ns]}$", fontsize=15)
plt.ylabel(r"$\eta_\mathrm{N}$", fontsize=15)
plt.legend(loc=1, fontsize=12)

plt.savefig("doppler_dephasing_rb1.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_rb1.pdf", bbox_inches="tight")

#############################################
plt.plot(tdelay_exp*1e9, eta_exp/amp01, "x", color=(1, 0.5, 0),
         label=r"$\mathrm{Experiment}$", ms=5)
plt.plot(tdelay_cont*1e9, eta_exp_cont, "-", color=(1, 0.5, 0),
         label=r"$\eta_{\mathrm{hfs}} \ \mathrm{fit}$")
plt.legend(loc=1, fontsize=12)

plt.savefig("doppler_dephasing_rb2.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_rb2.pdf", bbox_inches="tight")

plt.close("all")
#############################################
plt.title(r"$^{87}\mathrm{Rb \ Memory \ Lifetime}$", fontsize=15)

plt.plot(tdelay_cont*1e9, eta_hyp_cont, "b-",
         label=r"$\mathrm{Theory \ (without \ pumping)}$")

plt.plot(tdelay_cont*1e9, eta_simple_cont/eta_simple_cont[0], "g-",
         label=r"$\mathrm{Theory \ (with \ pumping)}$")


plt.xlim([0, tdelay_cont[-1]*1e9])
plt.ylim([0, 1.02])
plt.xlabel(r"$\tau \ \mathrm{[ns]}$", fontsize=15)
plt.ylabel(r"$\eta_\mathrm{N}$", fontsize=15)
plt.legend(loc=1, fontsize=12)

plt.savefig("doppler_dephasing_rb3.png", bbox_inches="tight")
plt.savefig("doppler_dephasing_rb3.pdf", bbox_inches="tight")
# We save all the data.
###############################################################################
continous = np.asarray([tdelay_cont, eta_hyp_cont,
                        eta_simple_cont/eta_simple_cont[0]]).T
continous = pd.DataFrame(continous)
continous.to_csv("eta_cont_rb.csv",
                 header=["tdelay_cont", "etan_hyp", "etan_sim"])

maxwell_bloch = np.asarray([tdelay_hyp, eta_hyp/amp02]).T
maxwell_bloch = pd.DataFrame(maxwell_bloch)
maxwell_bloch.to_csv("eta_the_rb.csv",
                     header=["tdelay_exp", "etan_hyp"])
