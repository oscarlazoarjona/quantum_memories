# -*- coding: utf-8 -*-
# Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona
#                      2017 Benjamin Brecht
# mailto: oscar.lazoarjona@physics.ox.ac.uk
# mailto: benjamin.brecht@physics.ox.ac.uk
r"""This is a repository for solvers for the Maxwell-Bloch equations of \
various quantum memories, and calculations using them.
"""

__version__ = "1.1"

from quantum_memories.misc import (time_bandwith_product, build_mesh_fdm,
                                   rayleigh_range, rel_error, glo_error)

from quantum_memories.orca import set_parameters_ladder
#
from quantum_memories.graphical import sketch_frame_transform, plot_solution
