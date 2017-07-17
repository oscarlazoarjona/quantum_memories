# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example of usage of the ORCA memory with the
default settings.
"""

from time import time

from quantum_memories import orca
from quantum_memories.misc import set_parameters_ladder, efficiencies
import numpy as np


default_params = set_parameters_ladder({"Temperature": 273.15+90,
                                        "alpha_rw": np.sqrt(20.0)})
if __name__ == '__main__':
    name = "test"

    # Benchmark with plotting.
    t0 = time()
    t, Z, vZ, rho, Om1 = orca.solve(default_params, plots=True, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, Om1, default_params,
                                        plots=True, name=name)
    teff = time()-t0
    nfun = 0
    print "Including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff

    # Benchmark without plotting.
    t0 = time()
    t, Z, vZ, rho, Om1 = orca.solve(default_params, plots=False, name=name)
    tsolve = time()-t0
    t0 = time()
    eff_in, eff_out, eff = efficiencies(t, Om1, default_params,
                                        plots=False, name=name)
    teff = time()-t0

    print
    print "Not including plotting times:"
    print "The solve function took", tsolve, "s."
    print "The efficiencies function took", teff, "s."
    print "The efficiencies were:", eff_in, eff_out, eff
