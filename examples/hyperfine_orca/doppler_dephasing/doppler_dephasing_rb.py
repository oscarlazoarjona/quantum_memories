# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This example calculates the inhomogeneous dephasing."""

import numpy as np
from matplotlib import pyplot as plt
# from scipy.optimize import curve_fit
from multiprocessing import Pool, cpu_count
# from time import time
from scipy.optimize import minimize

from quantum_memories import hyperfine_orca, orca
from quantum_memories.misc import set_parameters_ladder, efficiencies
import pandas as pd
from math import pi


def model_theoretical(t, amp, sigma):
    """Return points on a gaussian with the given amplitude and dephasing."""
    return amp*np.exp(-(t/sigma)**2)


def efficiencies_delay(i, Nv, errors, tdelayi, Ti, hyperfine=True):
    r"""Get the efficiencies with t0r = t0w + (3.5+i) ns."""
    name = "ORCA_rb"+str(i)
    # We set custom parameters (different from those in settings.py)
    t0w = default_params["t0w"]
    print t0w+tdelayi, Ti
    # We use the magic detuning
    delta1 = 2*pi*1055.3893691431863e9
    params = set_parameters_ladder({"Nv": Nv, "Nt": 4*51000, "T": Ti,
                                    "t0r": t0w+tdelayi,
                                    "verbose": 0, "t_cutoff": 3.5e-9,
                                    "delta1": delta1,
                                    "element": "Rb", "isotope": 87})

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
    print "finished", params["t0r"], eff

    return eff_in, eff_out, eff


def Chi2(xx, tdelay_the, hyperfine=True, efficiencies=False):
    r"""Calculate deviation."""
    Nv = 9
    tdelay = tdelay_the[:Ndelay]
    T_list = tdelay+5e-9

    eff_in_list = []; eff_out_list = []; eff_list = []

    eff_in_list = np.zeros(Ndelay)
    eff_out_list = np.zeros(Ndelay)
    eff_list = np.zeros(Ndelay)

    ####################################
    for i in range(Ndelay):
        eff_list[i] = 0.15-i*0.01
    eff_list = np.array(eff_list)

    # We create the parallel processes.
    pool = Pool(processes=Nprocs)
    # We calculate the efficiencies in parallel.
    procs = [pool.apply_async(efficiencies_delay,
                              [i, Nv, xx, tdelay[i], T_list[i], hyperfine])
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
    error = 1.0

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

    # The experimental efficiencies.
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

    tdelay_exp = np.array(tdelay_exp)
    eta_exp = np.array(eta_exp)

    tdelay_the = np.linspace(5e-9, 400e-9, 20*5+1)

    ###########################################################################
    # We test various readout times.
    Ndelay = 8
    Ndelay = len(tdelay_the)
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
        err_hyp, eta_the_hyp = Chi2(x0, tdelay_the, True, True)
        print "eta_the_hyp:", eta_the_hyp
        # err_hyp = 0.1; eta_the_hyp = 0.15 - np.linspace(0, 0.1, 8)

    eta_the_hyp_norm = eta_the_hyp*eta_exp[0]/eta_the_hyp[0]

    ################################
    # We plot everything.
    plt.title(r"$\mathrm{Doppler \ and \ Hyperfine \ Dephasing \ for \ Rb}$",
              fontsize=15)

    plt.plot(tdelay_exp*1e9, eta_exp, "o-b", label="Experiment")
    plt.plot(tdelay_the[:Ndelay]*1e9, eta_the_hyp_norm,
             "+-r", label="Hyperfine Theory")

    plt.xlim([0, None])
    plt.xlabel(r"$\mathrm{Readout \ time \ (ns)}$", fontsize=15)
    plt.ylabel(r"$\eta_N$", fontsize=15)
    plt.legend()

    plt.savefig("doppler_dephasing_rb.png", bbox_inches="tight")
    plt.savefig("doppler_dephasing_rb.pdf", bbox_inches="tight")
    plt.close("all")

    # We save the data.
    models = np.asarray([tdelay_the[:Ndelay], eta_the_hyp]).T
    models = pd.DataFrame(models)
    experimental_data = np.asarray([tdelay_exp, eta_exp]).T
    experimental_data = pd.DataFrame(experimental_data)

    models.to_csv("eta_the_rb.csv",
                  header=["tdelay", "etan_hyperfine"])

    experimental_data.to_csv("eta_exp_rb.csv",
                             header=["tdelay_exp", "etan_exp"])

# Calculating error for the fitted parameters...
# 6.18012452835e-09 1e-08
# 1.01301245283e-08 1.395e-08
# 1.40801245283e-08 1.79e-08
# 1.80301245283e-08 2.185e-08
# 2.19801245283e-08 2.58e-08
# 2.59301245283e-08 2.975e-08
# 2.98801245283e-08 3.37e-08
# 3.38301245283e-08 3.765e-08
# finished 6.18012452835e-09 7.54021595563e-09
# 3.77801245283e-08 4.16e-08
# finished 1.01301245283e-08 1.75486287278e-09
# 4.17301245283e-08 4.555e-08
# finished 1.40801245283e-08 3.35072338981e-11
# 4.56801245283e-08 4.95e-08
# finished 1.80301245283e-08 5.04822815423e-10
# 4.96301245283e-08 5.345e-08
# finished 2.19801245283e-08 5.27900776787e-10
# 5.35801245283e-08 5.74e-08
# finished 2.59301245283e-08 9.45328694549e-10
# 5.75301245283e-08 6.135e-08
# finished 2.98801245283e-08 2.86320681714e-09
# 6.14801245283e-08 6.53e-08
# finished 3.38301245283e-08 5.42175370976e-09
# 6.54301245283e-08 6.925e-08
# finished 3.77801245283e-08 7.31592227263e-09
# 6.93801245283e-08 7.32e-08
# finished 4.17301245283e-08 7.24202480804e-09
# 7.33301245283e-08 7.715e-08
# finished 4.56801245283e-08 4.45092363239e-09
# 7.72801245283e-08 8.11e-08
# finished 4.96301245283e-08 1.14686668064e-09
# 8.12301245283e-08 8.505e-08
# finished 5.35801245283e-08 8.96155226532e-10
# 8.51801245283e-08 8.9e-08
# finished 5.75301245283e-08 3.04293372292e-09
# 8.91301245283e-08 9.295e-08
# finished 6.14801245283e-08 3.49136302675e-09
# 9.30801245283e-08 9.69e-08
# finished 6.54301245283e-08 1.94586900202e-09
# 9.70301245283e-08 1.0085e-07
# finished 6.93801245283e-08 2.32597584903e-09
# 1.00980124528e-07 1.048e-07
# finished 7.33301245283e-08 5.19235497373e-09
# 1.04930124528e-07 1.0875e-07
# finished 7.72801245283e-08 6.32930496629e-09
# 1.08880124528e-07 1.127e-07
# finished 8.12301245283e-08 3.87263278417e-09
# 1.12830124528e-07 1.1665e-07
# finished 8.51801245283e-08 9.42483423007e-10
# 1.16780124528e-07 1.206e-07
# finished 8.91301245283e-08 4.74793093965e-10
# 1.20730124528e-07 1.2455e-07
# finished 9.30801245283e-08 2.13692885159e-09
# 1.24680124528e-07 1.285e-07
# finished 9.70301245283e-08 4.22698283398e-09
# 1.28630124528e-07 1.3245e-07
# /home/oscar/anaconda2/lib/python2.7/site-packages/scipy/integrate/_ode.py:1035: UserWarning: dopri5: larger nmax is needed
#   self.messages.get(idid, 'Unexpected idid=%s' % idid))
# finished 1.28630124528e-07 2.30381390731e-18
# 1.32580124528e-07 1.364e-07
# finished 1.32580124528e-07 2.75842239166e-19
# 1.36530124528e-07 1.4035e-07
# finished 1.36530124528e-07 0.0
# 1.40480124528e-07 1.443e-07
# finished 1.40480124528e-07 0.0
# 1.44430124528e-07 1.4825e-07
# finished 1.44430124528e-07 0.0
# 1.48380124528e-07 1.522e-07
# finished 1.48380124528e-07 0.0
# 1.52330124528e-07 1.5615e-07
# finished 1.52330124528e-07 0.0
# 1.56280124528e-07 1.601e-07
# finished 1.56280124528e-07 0.0
# 1.60230124528e-07 1.6405e-07
# finished 1.00980124528e-07 5.11768006601e-09
# 1.64180124528e-07 1.68e-07
# finished 1.60230124528e-07 0.0
# 1.68130124528e-07 1.7195e-07
# /home/oscar/anaconda2/lib/python2.7/site-packages/scipy/integrate/_ode.py:1035: UserWarning: dopri5: larger nmax is needed
#   self.messages.get(idid, 'Unexpected idid=%s' % idid))
# finished 1.64180124528e-07 0.0
# 1.72080124528e-07 1.759e-07
# finished 1.68130124528e-07 0.0
# 1.76030124528e-07 1.7985e-07
# finished 1.72080124528e-07 0.0
# 1.79980124528e-07 1.838e-07
# finished 1.76030124528e-07 0.0
# 1.83930124528e-07 1.8775e-07
# finished 1.79980124528e-07 0.0
# 1.87880124528e-07 1.917e-07
# finished 1.83930124528e-07 0.0
# 1.91830124528e-07 1.9565e-07
# finished 1.87880124528e-07 0.0
# 1.95780124528e-07 1.996e-07
# finished 1.91830124528e-07 0.0
# 1.99730124528e-07 2.0355e-07
# finished 1.95780124528e-07 0.0
# 2.03680124528e-07 2.075e-07
# finished 1.99730124528e-07 0.0
# 2.07630124528e-07 2.1145e-07
# finished 2.03680124528e-07 0.0
# 2.11580124528e-07 2.154e-07
# finished 2.07630124528e-07 0.0
# 2.15530124528e-07 2.1935e-07
# finished 2.11580124528e-07 0.0
# 2.19480124528e-07 2.233e-07
# finished 2.15530124528e-07 0.0
# 2.23430124528e-07 2.2725e-07
# finished 1.04930124528e-07 3.92855547752e-09
# 2.27380124528e-07 2.312e-07
# finished 2.19480124528e-07 0.0
# 2.31330124528e-07 2.3515e-07
# finished 2.23430124528e-07 0.0
# 2.35280124528e-07 2.391e-07
# /home/oscar/anaconda2/lib/python2.7/site-packages/scipy/integrate/_ode.py:1035: UserWarning: dopri5: larger nmax is needed
#   self.messages.get(idid, 'Unexpected idid=%s' % idid))
# finished 2.27380124528e-07 0.0
# 2.39230124528e-07 2.4305e-07
# finished 2.31330124528e-07 0.0
# 2.43180124528e-07 2.47e-07
# finished 2.35280124528e-07 0.0
# 2.47130124528e-07 2.5095e-07
# finished 2.39230124528e-07 0.0
# 2.51080124528e-07 2.549e-07
# finished 2.43180124528e-07 0.0
# 2.55030124528e-07 2.5885e-07
# finished 2.47130124528e-07 0.0
# 2.58980124528e-07 2.628e-07
# finished 2.51080124528e-07 0.0
# 2.62930124528e-07 2.6675e-07
# finished 2.55030124528e-07 0.0
# 2.66880124528e-07 2.707e-07
# finished 2.58980124528e-07 0.0
# 2.70830124528e-07 2.7465e-07
# finished 2.62930124528e-07 0.0
# 2.74780124528e-07 2.786e-07
# finished 2.66880124528e-07 0.0
# 2.78730124528e-07 2.8255e-07
# finished 2.74780124528e-07 0.0
# 2.82680124528e-07 2.865e-07
# finished 2.78730124528e-07 0.0
# 2.86630124528e-07 2.9045e-07
# finished 2.82680124528e-07 0.0
# 2.90580124528e-07 2.944e-07
# finished 2.86630124528e-07 0.0
# 2.94530124528e-07 2.9835e-07
# finished 2.90580124528e-07 0.0
# 2.98480124528e-07 3.023e-07
# finished 2.94530124528e-07 0.0
# 3.02430124528e-07 3.0625e-07
# finished 2.98480124528e-07 0.0
# 3.06380124528e-07 3.102e-07
# finished 3.06380124528e-07 0.0
# 3.10330124528e-07 3.1415e-07
# finished 3.02430124528e-07 0.0
# 3.14280124528e-07 3.181e-07
# finished 3.14280124528e-07 0.0
# 3.18230124528e-07 3.2205e-07
# finished 3.10330124528e-07 0.0
# 3.22180124528e-07 3.26e-07
# finished 3.18230124528e-07 0.0
# 3.26130124528e-07 3.2995e-07
# finished 3.22180124528e-07 0.0
# 3.30080124528e-07 3.339e-07
# finished 3.26130124528e-07 0.0
# 3.34030124528e-07 3.3785e-07
# finished 3.30080124528e-07 0.0
# 3.37980124528e-07 3.418e-07
# finished 3.34030124528e-07 0.0
# 3.41930124528e-07 3.4575e-07
# finished 3.37980124528e-07 0.0
# 3.45880124528e-07 3.497e-07
# finished 3.45880124528e-07 0.0
# 3.49830124528e-07 3.5365e-07
# finished 3.41930124528e-07 0.0
# 3.53780124528e-07 3.576e-07
# finished 3.49830124528e-07 0.0
# 3.57730124528e-07 3.6155e-07
# finished 3.53780124528e-07 0.0
# 3.61680124528e-07 3.655e-07
# finished 2.70830124528e-07 0.0
# 3.65630124528e-07 3.6945e-07
# finished 3.57730124528e-07 0.0
# 3.69580124528e-07 3.734e-07
# finished 3.61680124528e-07 0.0
# 3.73530124528e-07 3.7735e-07
# finished 3.65630124528e-07 0.0
# 3.77480124528e-07 3.813e-07
# finished 3.69580124528e-07 0.0
# 3.81430124528e-07 3.8525e-07
# finished 3.73530124528e-07 0.0
# 3.85380124528e-07 3.892e-07
# finished 3.77480124528e-07 0.0
# 3.89330124528e-07 3.9315e-07
# finished 3.81430124528e-07 0.0
# 3.93280124528e-07 3.971e-07
# finished 3.85380124528e-07 0.0
# 3.97230124528e-07 4.0105e-07
# finished 3.93280124528e-07 0.0
# 4.01180124528e-07 4.05e-07
# finished 3.89330124528e-07 0.0
# finished 3.97230124528e-07 0.0
# finished 4.01180124528e-07 0.0
# finished 1.08880124528e-07 2.01050117311e-09
# finished 1.12830124528e-07 1.6007320183e-09
# finished 1.16780124528e-07 2.1231843993e-09
# finished 1.20730124528e-07 1.374573966e-09
# finished 1.24680124528e-07 3.8179851344e-10
# eta_the_hyp: [  7.54021596e-09   1.75486287e-09   3.35072339e-11   5.04822815e-10
#    5.27900777e-10   9.45328695e-10   2.86320682e-09   5.42175371e-09
#    7.31592227e-09   7.24202481e-09   4.45092363e-09   1.14686668e-09
#    8.96155227e-10   3.04293372e-09   3.49136303e-09   1.94586900e-09
#    2.32597585e-09   5.19235497e-09   6.32930497e-09   3.87263278e-09
#    9.42483423e-10   4.74793094e-10   2.13692885e-09   4.22698283e-09
#    5.11768007e-09   3.92855548e-09   2.01050117e-09   1.60073202e-09
#    2.12318440e-09   1.37457397e-09   3.81798513e-10   2.30381391e-18
#    2.75842239e-19   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#    0.00000000e+00]
