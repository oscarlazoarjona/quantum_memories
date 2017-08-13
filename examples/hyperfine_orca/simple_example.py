# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example of usage of the ORCA memory with the
default settings.
"""

from time import time

from quantum_memories import hyperfine_orca
from quantum_memories.misc import set_parameters_ladder, efficiencies
import numpy as np

# [6.34, 8.77, 14.50, 19.37]
# [6.50, 9.85, 14.58, 19.92]

cus_params = {"Temperature": 273.15+90,
              "alpha_rw": np.sqrt(20.0), "Nz": 50}
params = set_parameters_ladder(cus_params)

if __name__ == '__main__':
    name = "test"

    # Benchmark with plotting.
    t0 = time()
    # hyperfine_orca.solve(params, plots=True, name=name)
    t, Z, vZ, rho, E01 = hyperfine_orca.solve(params, plots=True, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, E01, params,
                                        plots=True, name=name, rabi=False)
    teff = time()-t0
    nfun = 0
    print "Including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff

    # Benchmark without plotting.
    t0 = time()
    t, Z, vZ, rho, Om1 = hyperfine_orca.solve(params, plots=False, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, Om1, params,
                                        plots=False, name=name, rabi=False)
    teff = time()-t0
    nfun = 0
    print
    print "Including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff
