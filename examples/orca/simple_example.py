# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example of usage of the ORCA memory with the
default settings.

>>> params = set_parameters_ladder()
>>> print params["e_charge"]
1.6021766208e-19
>>> print params["hbar"]
1.05457180014e-34
>>> print params["c"]
299792458.0
>>> print params["epsilon_0"]
8.85418781762e-12
>>> print params["kB"]
1.38064852e-23
>>> print params["Omega"]
1.0
>>> print params["distance_unit"]
1.0
>>> print params["element"]
Cs
>>> print params["isotope"]
133
>>> print params["Nt"]
25500
>>> print params["Nz"]
50
>>> print params["Nv"]
9
>>> print params["Nrho"]
2
>>> print params["T"]
8e-09
>>> print params["L"]
0.072
>>> print params["sampling_rate"]
50
>>> print params["keep_data"]
sample
>>> print params["Temperature"]
363.15
>>> print params["Nsigma"]
4
>>> print params["delta1"]
-37699111843.1
>>> print params["sigma_power1"]
807222536.902
>>> print params["sigma_power2"]
883494520.871
>>> print params["w1"]
0.00028
>>> print params["w2"]
0.00032
>>> print params["t0s"]
1.18012452835e-09
>>> print params["t0w"]
1.18012452835e-09
>>> print params["t0r"]
4.68012452835e-09
>>> print params["alpha_rw"]
1.0
>>> print params["t_cutoff"]
3e-09
>>> print params["verbose"]
1

We get the depenent paramters:
>>> print params["mass"]
2.2069469161e-25
>>> print params["gamma21"]
32886191.8978
>>> print params["gamma32"]
14878582.8074
>>> print params["omega21"]
2.20993141261e+15
>>> print params["omega32"]
2.05306420003e+15

"""

from time import time

from quantum_memories import orca
from quantum_memories.misc import set_parameters_ladder, efficiencies
import numpy as np

cus_params = {"Temperature": 273.15+90,
              "alpha_rw": np.sqrt(20.0), "Nz": 50}
params = set_parameters_ladder(cus_params)

if __name__ == '__main__':
    name = "test"

    # Benchmark with plotting.
    t0 = time()
    t, Z, vZ, rho, Om1 = orca.solve(params, plots=True, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, Om1, params,
                                        plots=True, name=name)
    teff = time()-t0
    nfun = 0
    print "Including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff

    # Benchmark without plotting.
    t0 = time()
    t, Z, vZ, rho, Om1 = orca.solve(params, plots=False, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, Om1, params,
                                        plots=False, name=name)
    teff = time()-t0
    nfun = 0
    print
    print "Not including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff

# if __name__ == "__main__":
#     import doctest
#     print doctest.testmod(verbose=False)
