# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************

r"""This file establishes all the parameters needed."""

# We import fundamental constants:
from scipy.constants import c, hbar, epsilon_0
from scipy.constants import physical_constants
from math import sqrt, log, pi
from misc import vapour_number_density

# These flags control whether my software FAST [1] is used to calculate
# the parameters of the atom or the Bloch equations.
rewrite = True; rewrite = False
calculate_atom = False  # ; calculate_atom = True
calculate_bloch = False  # ; calculate_bloch=True
make_smoother = True  # ; make_smoother=False
change_rep_rate = True  # ; change_rep_rate=False
change_read_power = True  # ; change_read_power=False
ignore_lower_f = False; ignore_lower_f = True
run_long = False; run_long = True

optimize = True; optimize = False
verbose = 1

# We choose the units we want.
units = "SI"  # ; units="fancy"
if verbose >= 2: print "We are using "+units+" units!"

a0 = physical_constants["Bohr radius"][0]
e_charge = physical_constants["elementary charge"][0]
kB = physical_constants["Boltzmann constant"][0]

# The memory protocol to be used.
red_detuned = True
magic = False

# The extent of the simulation given by the number of dynamic variables
# Nrho, the number of time steps Nt, and the number of z points Nz.
Nrho = 2
Nt = 25500; Nz = 50

# The number of velocity groups to consider (better an odd number)
Nv = 9
# The number of standard deviations to consider on either side of the velocity
# distribution.
Nsigma = 4

# The data for the time discretization.
# The total time of the simulation (in s).
T = 8e-9
# T = 16e-9
# The time step.
dt = T/(Nt-1)

# The data for the spacial discretization.
# Cell length (in m).
L = 0.072

# Spatial extent of the simulation (in m).
D = 1.05 * L
optical_depth = 0.05e5
# The simulation will be done spanning -D/2 <= z <= D/2
zL = -0.5 * D  # left boundary of the simulation
zR = +0.5 * D  # right boundary of the simulation

######################
# The temperature of the cell.
Temperature = 90.0 + 273.15

# We should be able to choose whether to keep all of data, to just keep a
# sample at a certain rate, or to keep only the current-time data.

keep_data = "all"
keep_data = "sample"
# The sampling rate for the output. If sampling_rate=2 every second time step
# will be saved in memory and returned. If Nt is a multiple of sampling_rate
# then the length of the output should be Nt/sampling_rate.
sampling_rate = 50


################################################
# The characteristics of the beams:

# The waists of the beams (in m):
w1 = 280e-6
w2 = 320e-6

# The full widths at half maximum of the gaussian envelope of the powers
# spectra (in Hz).
sigma_power1 = 1.0e9
sigma_power2 = 1.0e9

sigma_power1 = 0.807222536902e9
sigma_power1 = 1.0e9
sigma_power2 = 0.883494520871e9

# We calculate the duration of the pulses from the standard deviations
tau1 = 2/pi * sqrt(log(2.0))/sigma_power1
tau2 = 2/pi * sqrt(log(2.0))/sigma_power2

tau1 = 2*sqrt(2)*log(2)/pi / sigma_power1
tau2 = 2*sqrt(2)*log(2)/pi / sigma_power2

# The time of arrival of the beams
t0s = 1.1801245283489222e-09
t0w = t0s
t0r = t0w + 3.5e-9
alpha_rw = 1.0

t_cutoff = t0r+D/2/c+tau1
t_cutoff = 3.0e-9

######################
# The detuning of the (in rad/s):
if red_detuned:
    delta1 = -2*pi*6e9
else:
    delta1 = 2*pi*6e9
# The detuning of the control field (in rad/s):
delta2 = delta1
# This is the two-photon transition condition.

######################
# We choose an atom:
element = "Rb"; isotope = 85; n_atom = 5
element = "Rb"; isotope = 87; n_atom = 5
element = "Cs"; isotope = 133; n_atom = 6
n_atomic0 = vapour_number_density(Temperature, element)
# We calculate (or impose) the properties of the atom:
if calculate_atom:
    from fast import State, Transition, make_list_of_states
    from fast import calculate_boundaries, Integer, calculate_matrices
    from fast import fancy_r_plot, fancy_matrix_plot, Atom
    from matplotlib import pyplot

    atom = Atom(element, isotope)

    g = State(element, isotope, n_atom, 0, 1/Integer(2))
    e = State(element, isotope, n_atom, 1, 3/Integer(2))
    fine_states = [g, e]

    hyperfine_states = make_list_of_states(fine_states, "hyperfine", verbose=0)
    magnetic_states = make_list_of_states(fine_states, "magnetic", verbose=0)

    aux = calculate_boundaries(fine_states, magnetic_states)
    fine_bounds, hyperfine_bounds = aux

    g1 = hyperfine_states[0]
    g2 = hyperfine_states[1]

    print fine_bounds, hyperfine_bounds

    g1_index = hyperfine_bounds[0][0]
    e1_index = fine_bounds[1][0]

    g2_index = hyperfine_bounds[1][-1]-1
    e2_index = fine_bounds[1][-1]-1

    if verbose >= 1:
        print
        print "Calculating atomic properties ..."
        print "We are choosing the couplings of"
        print magnetic_states[g1_index], magnetic_states[e1_index]
        print "For the g1 <-> e transition, and the couplings of"
        print magnetic_states[g2_index], magnetic_states[e2_index]
        print "For the g2 <-> e transition."

    # We calculate the matrices for the given states.
    Omega = 1.0  # We choose the calculated frequencies to be in radians.
    omega, gamma, r = calculate_matrices(magnetic_states, Omega)

    # We plot these matrices.
    path = ''; name = element+str(isotope)
    fig = pyplot.figure(); ax = fig.add_subplot(111)
    fancy_matrix_plot(ax, omega, magnetic_states, path, name+'_omega.png',
                      take_abs=True, colorbar=True)
    fig = pyplot.figure(); ax = fig.add_subplot(111)
    fancy_matrix_plot(ax, gamma, magnetic_states, path, name+'_gamma.png',
                      take_abs=True, colorbar=True)
    fig = pyplot.figure(); ax = fig.add_subplot(111)
    fancy_r_plot(r, magnetic_states, path, name+'_r.png',
                 complex_matrix=True)
    pyplot.close("all")

    # We get the parameters for the simplified scheme.
    # The couplings.
    r31 = r[2][e1_index][g1_index]
    r32 = r[2][e2_index][g2_index]

    # The FAST function calculate_matrices always returns r in
    # Bohr radii, so we convert. By contrast, it returns omega
    # and gamma in units scaled by Omega. If Omega=1e6 this means
    # 10^6 rad/s. So we do not have to rescale omega or gamma.
    r31 = r31*a0
    r32 = r32*a0

    # The decay frequencies.
    if red_detuned:
        gamma31 = gamma[e2_index][g2_index]
        gamma32 = 0
    else:
        gamma31 = 0
        gamma32 = gamma[e2_index][g2_index]

    omega31 = Transition(e, g1).omega
    omega32 = Transition(e, g2).omega
    omega21 = Transition(g2, g1).omega

    print gamma31, gamma32
    print omega31, omega32, omega21
    print r31, r32
    print atom.mass
else:
    if (element, isotope) == ("Rb", 85):
        if red_detuned:
            gamma31, gamma32 = (38107518.888, 0.0)
        else:
            gamma31, gamma32 = (0.0, 38107518.888)
        omega31, omega32 = (2.41420163422e+15, 2.41418256014e+15)
        omega21 = 19074074839.2
        r31, r32 = (1.73263595681e-10, 2.23682340192e-10)
        mass = 1.40999341816e-25

    elif (element, isotope) == ("Rb", 87):
        if red_detuned:
            gamma31, gamma32 = (38107518.888, 0.0)
        else:
            gamma31, gamma32 = (0.0, 38107518.888)
        omega31, omega32 = (2.41421734741e+15, 2.41417440383e+15)
        omega21 = 42943577360.1
        r31, r32 = (1.29143059323e-10, 2.23682340192e-10)
        mass = 1.44316087206e-25

    elif (element, isotope) == ("Cs", 133):
        if red_detuned:
            gamma31, gamma32 = (32886191.8978, 0.0)
        else:
            gamma31, gamma32 = (0.0, 32886191.8978)
        omega31, omega32 = (2.20999035592e+15, 2.20993259691e+15)
        omega21 = 57759008871.6
        r31, r32 = (2.00516655729e-10, 2.37254506627e-10)
        mass = 2.2069469161e-25

# The frequencies of the optical fields.
omega_laser1 = delta1 + omega31
omega_laser2 = delta2 + omega32

# A decoherence frequency
gammaB = 2*pi*15e6

# ######################
# The energies of the photons.
energy_phot1 = hbar*omega_laser1
energy_phot2 = hbar*omega_laser2

# The energies of the pulses.
if (magic and red_detuned) or (not magic and not red_detuned):
    energy_pulse31 = 25e-12      # Joules.
    energy_pulse32 = 1*energy_phot2  # Joules.
else:
    energy_pulse31 = 1*energy_phot1  # Joules.
    energy_pulse32 = 25e-12      # Joules.

################################################

# The fancy units should be picked so that the factors multiplied in
# each of the terms of the equations are of similar magnitude.

# Ideally, the various terms should also be of similar magnitude, but
# changing the units will not change the relative importance of terms.
# Otherwise physics would change depending on the units!
# However, it should be possible to choose units such that the largest
# terms should be close to 1.

if units == "SI":
    Omega = 1.0  # The frequency unit in Hz.
    distance_unit = 1.0  # The distance unit in m.
elif units == "fancy":
    # The frequency scale for frequency in Hz.
    Omega = 1e9
    # Omega=1.0

    # The distance unit in m.
    distance_unit = 1.0

    # An interesting option would be to make
    # distance_unit=c*Omega
    # This way the beams will propagate in time-space diagrams at 45 degree
    # angles.

    # To use fancy units we need to rescale fundamental constants.
    # [ hbar ] = J s
    #          = kg * m^2 * s / s^2
    #          = kg * m^2 * Hz
    hbar = hbar/distance_unit**2 / Omega  # fancy units.

    # [ epsilon_0 ] = A^2 * s^4 / ( kg * m^3 )
    #               = C^2 * s^2 / ( kg * m^3 )
    #               = C^2  / ( Hz^2 kg * m^3 )
    epsilon_0 = epsilon_0 * Omega**2 * distance_unit**3

    # [ c ] = m / s
    c = c/distance_unit/Omega

    # [ kB ] = J K^-1
    #        = (kg m^2/s^2) K^-1
    #        = (kg m^2 Hz^2) K^-1
    kB = kB / distance_unit**2 / Omega**2

    # Rescale time:
    T = T*Omega
    dt = dt*Omega

    # We must also rescale the cell:
    L = L/distance_unit
    D = D/distance_unit
    zL = zL/distance_unit
    zR = zR/distance_unit

    # We must also rescale the characteristics of the pulses.
    w1 = w1/distance_unit
    w2 = w2/distance_unit

    sigma_power1 = sigma_power1/Omega
    sigma_power2 = sigma_power2/Omega

    tau1 = tau1*Omega
    tau2 = tau2*Omega

    t0s = t0s*Omega
    t0w = t0w*Omega
    t0r = t0r*Omega

    t_cutoff = t_cutoff*Omega

    gamma31 = gamma31/Omega
    gamma32 = gamma32/Omega

    omega21 = omega21/Omega
    omega31 = omega31/Omega
    omega32 = omega32/Omega

    delta1 = delta1/Omega
    delta2 = delta2/Omega

    omega_laser1 = omega_laser1/Omega
    omega_laser2 = omega_laser2/Omega

    gammaB = gammaB/Omega

    r31 = r31/distance_unit
    r32 = r32/distance_unit

    # J = kg * m^2 / s^2
    energy_phot1 = energy_phot1 / distance_unit**2 / Omega**2
    energy_phot2 = energy_phot2 / distance_unit**2 / Omega**2

    energy_pulse1 = energy_pulse1 / distance_unit**2 / Omega**2
    energy_pulse2 = energy_pulse2 / distance_unit**2 / Omega**2


# [1] https://github.com/oscarlazoarjona/fast
