# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************
r"""This is a simple example of usage of the ORCA memory with the
default settings.
"""

# from time import time
from quantum_memories import orca
from quantum_memories.misc import set_parameters_ladder

params = set_parameters_ladder()
params["alpha_rw"] = 1.0
params["USE_HG_CTRL"] = False
params["nw"] = 0
params["ns"] = 0
params["nr"] = 0
params["Nv"] = 1
params["energy_pulse2"] = 25E-12
params["D"] = 1.15 * params["L"]
params["verbose"] = 0

if __name__ == '__main__':
    name = "test"
    aux = orca.greens(params, plots=True, verbose=True)
    Gri, t_sample, t_out, phi, eta, psi = aux
