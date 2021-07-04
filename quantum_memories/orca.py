# -*- coding: utf-8 -*-
# Compatible with Python 3.8
# Copyright (C) 2020-2021 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Orca related routines."""
from time import time
import warnings
import numpy as np
from scipy.constants import physical_constants, c, hbar, epsilon_0, mu_0
from scipy.sparse import spdiags
from scipy.sparse import eye as sp_eye

from matplotlib import pyplot as plt
from sympy import oo

from quantum_memories.misc import (time_bandwith_product,
                                   vapour_number_density, rayleigh_range,
                                   ffftfreq, iffftfft, interpolator, sinc,
                                   hermite_gauss, num_integral, build_Z_mesh,
                                   build_t_mesh, build_mesh_fdm, harmonic,
                                   rel_error, glo_error, get_range)

from quantum_memories.fdm import (derivative_operator,
                                  fdm_derivative_operators, bfmt, bfmtf,
                                  set_block,
                                  solve_fdm)
from quantum_memories.graphical import plot_solution


def set_parameters_ladder(custom_parameters=None, fitted_couplings=True,
                          calculate_atom=False):
    r"""Set the parameters for a ladder memory.

    Only completely independent parameters are taken from settings.py.
    The rest are derived from them.
    """
    #########################################################################
    # We set the default values of independent parameters
    if True:
        ignore_lower_f = False; ignore_lower_f = True
        verbose = 1

        a0 = physical_constants["Bohr radius"][0]
        e_charge = physical_constants["elementary charge"][0]
        kB = physical_constants["Boltzmann constant"][0]

        # The number of time steps Nt, and the number of z points Nz.
        Nt = 1020
        Nz = 50

        # The number of velocity groups to consider (better an odd number)
        Nv = 1
        # The number of standard deviations to consider on either side
        # of the velocity distribution.
        Nsigma = 4

        # The data for the time discretization.
        # The total time of the simulation (in s).
        T = 8e-9
        # T = 16e-9
        # The time step.
        # dt = T/(Nt-1)

        # The data for the spacial discretization.
        # Cell length (in m).
        L = 0.072

        ######################
        # The temperature of the cell.
        Temperature = 90.0 + 273.15

        ################################################
        # The characteristics of the beams:

        # The waists of the beams (in m):
        w1 = 280e-6
        w2 = 320e-6

        # The full widths at half maximum of the gaussian envelope of
        # the powers spectra (in Hz).
        sigma_power1 = 1.0e9
        sigma_power2 = 1.0e9

        sigma_power1 = 0.807222536902e9
        sigma_power2 = 0.883494520871e9

        # This corresponds to 300 ps.
        sigma_power1 = 1.47090400101768e9
        sigma_power2 = 1.47090400101768e9

        # The time of arrival of the beams
        t0s = 1.1801245283489222e-09
        t0w = t0s
        t0r = t0w + 3.5e-9
        wr_ratio = 1.0

        # t_cutoff = t0r+D/2/c+tau1
        t_cutoff = 3.0e-9

        ######################
        # The detuning of the signal field (in Hz):
        delta1 = -2*np.pi*9e9
        # The detuning of the control field (in Hz):
        # This is the two-photon transition condition.
        delta2 = -delta1
        # We choose an atom:
        element = "Cs"; isotope = 133; n_atom = 6

        # Control pulse energy.
        energy_pulse2 = 50e-12  # Joules.

        # The default flags.
        USE_HG_CTRL = False
        USE_HG_SIG = False
        USE_SQUARE_SIG = False
        USE_SQUARE_CTRL = False

        nshg = 0; nwhg = 0; nrhg = 0
        nssquare = 1; nwsquare = 1; nrsquare = 1
    ################################################
    # We set the default values of the independent parameters.
    pms = {"e_charge": e_charge,
           "hbar": hbar,
           "c": c,
           "epsilon_0": epsilon_0,
           "kB": kB,
           "element": element,
           "isotope": isotope,
           "Nt": Nt,
           "Nz": Nz,
           "Nv": Nv,
           "T": T,
           "L": L,
           "Temperature": Temperature,
           "Nsigma": Nsigma,
           "delta1": delta1,
           "sigma_power1": sigma_power1,
           "sigma_power2": sigma_power2,
           "w1": w1,
           "w2": w2,
           "t0s": t0s,
           "t0w": t0w,
           "t0r": t0r,
           "energy_pulse2": energy_pulse2,
           "wr_ratio": wr_ratio,
           "t_cutoff": t_cutoff,
           "element": element,
           "isotope": isotope,
           "verbose": verbose,
           "USE_HG_SIG": USE_HG_SIG,
           "USE_HG_CTRL": USE_HG_CTRL,
           "USE_SQUARE_SIG": USE_SQUARE_SIG,
           "USE_SQUARE_CTRL": USE_SQUARE_CTRL,
           "nshg": nshg, "nwhg": nwhg, "nrhg": nrhg,
           "nssquare": nssquare, "nwsquare": nwsquare, "nrsquare": nrsquare,
           "ntauw": 1.0, "N": 101,
           "pumping": 0.0,
           "with_focusing": False,
           "rep_rate": 80e6}
    # NOTE: if an independent parameter is added here, it must also
    # be added in the next block of code to update it.

    #########################################################################
    # We replace independent parameters by custom ones if given.
    if True:
        if custom_parameters is None:
            custom_parameters = {}

        pm_names_ind = pms.keys()
        pm_names_dep = ["mass", "gamma21", "gamma32", "omega21", "omega32",
                        "omega_laser1", "omega_laser2", "delta2", "r1", "r2",
                        "taus", "tauw", "taur", "energy_pulse1"]

        for i in custom_parameters:
            if (i not in pm_names_ind) and (i not in pm_names_dep):
                raise ValueError(str(i)+" is not a valid parameter name.")

        # We replace "oo" by oo.
        aux = ["nssquare", "nwsquare", "nrsquare"]
        for key in aux:
            if key in custom_parameters and custom_parameters[key] == "oo":
                custom_parameters[key] = oo

        # Quick code generation for the folliwing block.
        # for name in pm_names_ind:
        #     line1 = 'if "{}" in custom_parameters.keys():'
        #     print(line1.format(name))
        #     line2 = '    pms["{}"] = custom_parameters["{}"]'
        #     print(line2.format(name, name))
        #     line3 = '    {} = custom_parameters["{}"]'
        #     print(line3.format(name, name))
        if True:
            if "e_charge" in custom_parameters.keys():
                pms["e_charge"] = custom_parameters["e_charge"]
                e_charge = custom_parameters["e_charge"]
            # if "hbar" in custom_parameters.keys():
            #     pms["hbar"] = custom_parameters["hbar"]
            #     hbar = custom_parameters["hbar"]
            # if "c" in custom_parameters.keys():
            #     pms["c"] = custom_parameters["c"]
            #     c = custom_parameters["c"]
            # if "epsilon_0" in custom_parameters.keys():
            #     pms["epsilon_0"] = custom_parameters["epsilon_0"]
            #     epsilon_0 = custom_parameters["epsilon_0"]
            if "kB" in custom_parameters.keys():
                pms["kB"] = custom_parameters["kB"]
                kB = custom_parameters["kB"]
            if "element" in custom_parameters.keys():
                pms["element"] = custom_parameters["element"]
                element = custom_parameters["element"]
            if "isotope" in custom_parameters.keys():
                pms["isotope"] = custom_parameters["isotope"]
                isotope = custom_parameters["isotope"]
            if "Nt" in custom_parameters.keys():
                pms["Nt"] = custom_parameters["Nt"]
                Nt = custom_parameters["Nt"]
            if "Nz" in custom_parameters.keys():
                pms["Nz"] = custom_parameters["Nz"]
                Nz = custom_parameters["Nz"]
            if "Nv" in custom_parameters.keys():
                pms["Nv"] = custom_parameters["Nv"]
                Nv = custom_parameters["Nv"]
            if "T" in custom_parameters.keys():
                pms["T"] = custom_parameters["T"]
                T = custom_parameters["T"]
            if "L" in custom_parameters.keys():
                pms["L"] = custom_parameters["L"]
                L = custom_parameters["L"]
            if "Temperature" in custom_parameters.keys():
                pms["Temperature"] = custom_parameters["Temperature"]
                Temperature = custom_parameters["Temperature"]
            if "Nsigma" in custom_parameters.keys():
                pms["Nsigma"] = custom_parameters["Nsigma"]
                Nsigma = custom_parameters["Nsigma"]
            if "delta1" in custom_parameters.keys():
                pms["delta1"] = custom_parameters["delta1"]
                delta1 = custom_parameters["delta1"]
            if "sigma_power1" in custom_parameters.keys():
                pms["sigma_power1"] = custom_parameters["sigma_power1"]
                sigma_power1 = custom_parameters["sigma_power1"]
            if "sigma_power2" in custom_parameters.keys():
                pms["sigma_power2"] = custom_parameters["sigma_power2"]
                sigma_power2 = custom_parameters["sigma_power2"]
            if "w1" in custom_parameters.keys():
                pms["w1"] = custom_parameters["w1"]
                w1 = custom_parameters["w1"]
            if "w2" in custom_parameters.keys():
                pms["w2"] = custom_parameters["w2"]
                w2 = custom_parameters["w2"]
            if "t0s" in custom_parameters.keys():
                pms["t0s"] = custom_parameters["t0s"]
                t0s = custom_parameters["t0s"]
            if "t0w" in custom_parameters.keys():
                pms["t0w"] = custom_parameters["t0w"]
                t0w = custom_parameters["t0w"]
            if "t0r" in custom_parameters.keys():
                pms["t0r"] = custom_parameters["t0r"]
                t0r = custom_parameters["t0r"]
            if "energy_pulse2" in custom_parameters.keys():
                pms["energy_pulse2"] = custom_parameters["energy_pulse2"]
                energy_pulse2 = custom_parameters["energy_pulse2"]
            if "wr_ratio" in custom_parameters.keys():
                pms["wr_ratio"] = custom_parameters["wr_ratio"]
                wr_ratio = custom_parameters["wr_ratio"]
            if "t_cutoff" in custom_parameters.keys():
                pms["t_cutoff"] = custom_parameters["t_cutoff"]
                t_cutoff = custom_parameters["t_cutoff"]
            if "element" in custom_parameters.keys():
                pms["element"] = custom_parameters["element"]
                element = custom_parameters["element"]
            if "isotope" in custom_parameters.keys():
                pms["isotope"] = custom_parameters["isotope"]
                isotope = custom_parameters["isotope"]
            if "verbose" in custom_parameters.keys():
                pms["verbose"] = custom_parameters["verbose"]
                verbose = custom_parameters["verbose"]
            if "USE_HG_SIG" in custom_parameters.keys():
                pms["USE_HG_SIG"] = custom_parameters["USE_HG_SIG"]
                USE_HG_SIG = custom_parameters["USE_HG_SIG"]
            if "USE_HG_CTRL" in custom_parameters.keys():
                pms["USE_HG_CTRL"] = custom_parameters["USE_HG_CTRL"]
                USE_HG_CTRL = custom_parameters["USE_HG_CTRL"]
            if "USE_SQUARE_SIG" in custom_parameters.keys():
                pms["USE_SQUARE_SIG"] = custom_parameters["USE_SQUARE_SIG"]
                USE_SQUARE_SIG = custom_parameters["USE_SQUARE_SIG"]
            if "USE_SQUARE_CTRL" in custom_parameters.keys():
                pms["USE_SQUARE_CTRL"] = custom_parameters["USE_SQUARE_CTRL"]
                USE_SQUARE_CTRL = custom_parameters["USE_SQUARE_CTRL"]
            if "nshg" in custom_parameters.keys():
                pms["nshg"] = custom_parameters["nshg"]
                nshg = custom_parameters["nshg"]
            if "nwhg" in custom_parameters.keys():
                pms["nwhg"] = custom_parameters["nwhg"]
                nwhg = custom_parameters["nwhg"]
            if "nrhg" in custom_parameters.keys():
                pms["nrhg"] = custom_parameters["nrhg"]
                nrhg = custom_parameters["nrhg"]
            if "nssquare" in custom_parameters.keys():
                pms["nssquare"] = custom_parameters["nssquare"]
                nssquare = custom_parameters["nssquare"]
            if "nwsquare" in custom_parameters.keys():
                pms["nwsquare"] = custom_parameters["nwsquare"]
                nwsquare = custom_parameters["nwsquare"]
            if "nrsquare" in custom_parameters.keys():
                pms["nrsquare"] = custom_parameters["nrsquare"]
                nrsquare = custom_parameters["nrsquare"]
            if "N" in custom_parameters.keys():
                pms["N"] = custom_parameters["N"]
                nrsquare = custom_parameters["N"]
            if "ntauw" in custom_parameters.keys():
                pms["ntauw"] = custom_parameters["ntauw"]
                nrsquare = custom_parameters["ntauw"]
            if "pumping" in custom_parameters.keys():
                pms["pumping"] = custom_parameters["pumping"]
            if "with_focusing" in custom_parameters.keys():
                pms["with_focusing"] = custom_parameters["with_focusing"]
            if "rep_rate" in custom_parameters.keys():
                pms["rep_rate"] = custom_parameters["rep_rate"]
    #########################################################################

    if calculate_atom:
        from fast import State, Transition, make_list_of_states, Atom
        from fast import calculate_boundaries, Integer
        from fast import calculate_matrices
        # from fast import fancy_r_plot, fancy_matrix_plot
        from fast import vapour_number_density
        # from matplotlib import pyplot

        atom = Atom(element, isotope)
        mass = atom.mass
        n_atom = atom.ground_state_n
        n_atomic0 = vapour_number_density(Temperature, element)

        g = State(element, isotope, n_atom, 0, 1/Integer(2))
        e = State(element, isotope, n_atom, 1, 3/Integer(2))
        l = State(element, isotope, n_atom, 2, 5/Integer(2))
        fine_states = [g, e, l]

        magnetic_states = make_list_of_states(fine_states,
                                              "magnetic", verbose=0)

        bounds = calculate_boundaries(fine_states, magnetic_states)

        g_index = bounds[0][0][1]-1
        e_index = bounds[0][1][1]-1
        l_index = bounds[1][6][1]-1

        g = magnetic_states[g_index]
        e = magnetic_states[e_index]
        l = magnetic_states[l_index]

        if verbose >= 1:
            print
            print("Calculating atomic properties ...")
            print("We are choosing the couplings of")
            print(magnetic_states[g_index], magnetic_states[e_index],)
            print(magnetic_states[l_index])
            print("as a basis to estimate the values of gamma_ij, r^l.")

        # We calculate the matrices for the given states.
        omega, gamma, r = calculate_matrices(magnetic_states, 1.0)

        # We plot these matrices.
        # path = ''; name = element+str(isotope)
        # fig = pyplot.figure(); ax = fig.add_subplot(111)
        # fancy_matrix_plot(ax, omega, magnetic_states, path,
        #                   name+'_omega.png',
        #                   take_abs=True, colorbar=True)
        # fig = pyplot.figure(); ax = fig.add_subplot(111)
        # fancy_matrix_plot(ax, gamma, magnetic_states, path,
        #                   name+'_gamma.png',
        #                   take_abs=True, colorbar=True)
        # fig = pyplot.figure(); ax = fig.add_subplot(111)
        # fancy_r_plot(r, magnetic_states, path, name+'_r.png',
        #              complex_matrix=True)
        # pyplot.close("all")

        # We get the parameters for the simplified scheme.
        # The couplings.
        r1 = r[2][e_index][g_index]
        r2 = r[2][l_index][e_index]

        # The FAST function calculate_matrices always returns r in
        # Bohr radii, so we convert. By contrast, it returns omega
        # and gamma in units scaled by Omega. If Omega=1e6 this means
        # 10^6 rad/s. So we do not have to rescale omega or gamma.
        r1 = r1*a0
        r2 = r2*a0

        # The decay frequencies.
        gamma21 = gamma[e_index][g_index]
        gamma32 = gamma[l_index][e_index]
        # print gamma21, gamma32

        # We determine which fraction of the population is in the lower
        # and upper ground states. The populations will be approximately
        # those of a thermal state. At room temperature the populations
        # of all Zeeman states will be approximately equal.
        fs = State(element, isotope, n_atom, 0, 1/Integer(2)).fperm
        # lower_fraction = (2*fs[0]+1)/(2*fs[0]+1.0 + 2*fs[1]+1.0)
        upper_fraction = (2*fs[1]+1)/(2*fs[0]+1.0 + 2*fs[1]+1.0)

        if ignore_lower_f:
            g_index = bounds[0][0][1]-1
            e_index = bounds[1][3][1]-1

            g = magnetic_states[g_index]
            e = magnetic_states[e_index]
            n_atomic0 = upper_fraction*n_atomic0

        else:
            g_index = bounds[0][0][1]-1
            e_index = bounds[0][1][1]-1
            l_index = bounds[1][6][1]-1

            g = magnetic_states[g_index]
            e = magnetic_states[e_index]
            l = magnetic_states[l_index]

        omega21 = Transition(e, g).omega
        omega32 = Transition(l, e).omega
        # print omega21, omega32
        # print r1, r2
        # print n_atomic0
        # print atom.mass
    else:
        if (element, isotope) == ("Rb", 85):

            gamma21, gamma32 = (38107518.888, 3102649.47106)
            if ignore_lower_f:
                omega21, omega32 = (2.4141820325e+15, 2.42745336743e+15)
            else:
                omega21, omega32 = (2.41418319096e+15, 2.42745220897e+15)
            r1, r2 = (2.23682340192e-10, 5.48219440757e-11)
            mass = 1.40999341816e-25
            if ignore_lower_f:
                n_atomic0 = 1.8145590576e+18
            else:
                n_atomic0 = 3.11067267018e+18

        elif (element, isotope) == ("Rb", 87):
            gamma21, gamma32 = (38107518.888, 3102649.47106)
            if ignore_lower_f:
                omega21, omega32 = (2.41417295963e+15, 2.42745419204e+15)
            else:
                omega21, omega32 = (2.41417562114e+15, 2.42745153053e+15)
            r1, r2 = (2.23682340192e-10, 5.48219440757e-11)
            r1, r2 = (1.58167299508e-10, 4.47619298768e-11)
            mass = 1.44316087206e-25
            if ignore_lower_f:
                n_atomic0 = 1.94417041886e+18
            else:
                n_atomic0 = 3.11067267018e+18

        elif (element, isotope) == ("Cs", 133):
            gamma21, gamma32 = (32886191.8978, 14878582.8074)
            if ignore_lower_f:
                omega21, omega32 = (2.20993141261e+15, 2.05306420003e+15)
            else:
                omega21, omega32 = (2.20993425498e+15, 2.05306135765e+15)
            r1, r2 = (2.37254506627e-10, 1.54344650829e-10)
            r1, r2 = (1.67764270425e-10, 1.26021879628e-10)
            mass = 2.2069469161e-25
            if ignore_lower_f:
                n_atomic0 = 4.72335166533e+18
            else:
                n_atomic0 = 8.39706962725e+18

    # We calculate dependent parameters
    if True:
        # The frequencies of the optical fields.
        omega_laser1 = delta1 + omega21
        omega_laser2 = delta2 + omega32

        ######################
        # The energies of the photons.
        energy_phot1 = hbar*omega_laser1
        # The energies of the pulses.
        energy_pulse1 = 1*energy_phot1  # Joules.

        delta1 = pms["delta1"]
        delta2 = -delta1
        omega_laser1 = delta1 + omega21
        omega_laser2 = delta2 + omega32

        if USE_SQUARE_CTRL:
            tauw = time_bandwith_product(nwsquare)/sigma_power2
            taur = time_bandwith_product(nrsquare)/sigma_power2
        else:
            tauw = time_bandwith_product(1) / sigma_power2
            taur = time_bandwith_product(1) / sigma_power2

        if USE_SQUARE_SIG:
            taus = time_bandwith_product(nssquare)/sigma_power1
        else:
            taus = time_bandwith_product(1) / sigma_power1

        # We make a few checks
        if pms["Nv"] == 2:
            raise ValueError("Nv = 2 is a very bad choice.")

        pms.update({"mass": mass,
                    "gamma21": gamma21,
                    "gamma32": gamma32,
                    "omega21": omega21,
                    "omega32": omega32,
                    "omega_laser1": omega_laser1,
                    "omega_laser2": omega_laser2,
                    "delta2": delta2,
                    "r1": r1,
                    "r2": r2,
                    "energy_pulse1": energy_pulse1,
                    "energy_pulse2": energy_pulse2,
                    "taus": taus,
                    "tauw": tauw,
                    "taur": taur})

        cond1 = "r1" not in custom_parameters
        cond2 = "r2" not in custom_parameters
        if fitted_couplings and cond1 and cond2:
            pms.update({"r1": pms["r1"]*0.2556521})
            pms.update({"r2": pms["r2"]*0.72474758})

    # We force any custom dependent parameters.
    for name in pm_names_dep:
        if name in custom_parameters:
            if pms["verbose"] >= 1:
                mes = "WARNING: parameter " + name
                mes += " may be inconsistent with independent parameters."
                print(mes)
            pms.update({name: custom_parameters[name]})

    return pms


def print_params(params):
    r"""Print parameters."""
    # Nt = params["Nt"]
    # Nz = params["Nz"]
    # T = params["T"]
    L = params["L"]
    # hbar = params["hbar"]
    # epsilon_0 = params["epsilon_0"]
    # e_charge = params["e_charge"]
    # c = params["c"]
    # r2 = params["r2"]
    t0s = params["t0s"]
    t0w = params["t0w"]
    t0r = params["t0r"]
    sigma_power1 = params["sigma_power1"]
    sigma_power2 = params["sigma_power2"]
    taus = params["taus"]
    tauw = params["tauw"]
    Omega = calculate_Xi(params)
    w1 = params["w1"]
    w2 = params["w2"]
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    energy_pulse2 = params["energy_pulse2"]
    rep_rate = params["rep_rate"]
    Temperature = params["Temperature"]
    pumping = params["pumping"]

    n = vapour_number_density(params)
    kappa = calculate_kappa(params)
    ZRs, ZRc = rayleigh_range(params)
    Ecrit = calculate_pulse_energy(params)

    # print("Grid size: %i x %i = %i points" % (Nt, Nz, Nt*Nz))
    # print("Spacetime size: %2.3f ns, %2.3f cm" % (T*1e9, D*100))
    print("Atom: {}{}".format(params["element"], params["isotope"]))
    print("delta1: %2.3f GHz" % (delta1/2/np.pi*1e-9))
    print("delta2: %2.3f GHz" % (delta2/2/np.pi*1e-9))
    print("Rabi frequency: %2.3f GHz" % (Omega/2/np.pi*1e-9))

    aux = (sigma_power1*1e-9, sigma_power2*1e-9)
    print("Signal & Control bandwidth: %2.3f GHz, %2.3f GHz" % aux)
    aux = (taus*1e9, tauw*1e9)
    print("Signal & Control duration: %2.3f ns, %2.3f ns" % aux)
    aux = (w1*1e6, w2*1e6)
    print("Signal & Control waists: %2.3f um, %2.3f um" % aux)
    aux = (2*ZRs*100, 2*ZRc*100)
    print("Signal & Control double Rayleigh range: %2.3f cm, %2.3f cm" % aux)

    print("Control pulse energy : {:10.3f} nJ".format(energy_pulse2*1e9))
    print("Critical pulse energy: {:10.3f} nJ".format(Ecrit*1e9))
    print("Average control power: {:10.3f} W".format(energy_pulse2*rep_rate))
    print("Critical average control power: {:10.3f} W".format(Ecrit*rep_rate))
    aux = [t0s*1e9, t0w*1e9, t0r*1e9]
    print("t0s: {:2.3f} ns, t0w: {:2.3f} ns, t0r: {:2.3f} ns".format(*aux))
    print("L: {:2.3f} cm".format(L*100))
    print("Temperature: {:6.2f} °C".format(Temperature-273.15))
    print("n: {:.2e} m^-3 ".format(n))
    print("kappa: {:.2e} sqrt((m s)^-1)".format(kappa))
    print("Pumping: {}".format(pumping))


def calculate_Gamma21(params):
    r"""Get the complex detuning."""
    gamma21 = params["gamma21"]
    delta1 = params["delta1"]
    return gamma21/2 - 1j*delta1


def calculate_Gamma32(params):
    r"""Get the complex detuning."""
    gamma32 = params["gamma32"]
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    return gamma32/2 - 1j*(delta1+delta2)


def calculate_Xi(params):
    r"""Calculate the effective (time averaged) Rabi frequency."""
    energy_pulse2 = params["energy_pulse2"]
    hbar = params["hbar"]
    e_charge = params["e_charge"]
    r2 = params["r2"]
    w2 = params["w2"]
    m = params["nwsquare"]
    sigma_power2 = params["sigma_power2"]

    tbp = time_bandwith_product(m)
    T_Xi = tbp/sigma_power2

    Xi = 4 * e_charge**2*r2**2 * energy_pulse2 * c * mu_0
    Xi = Xi/(hbar**2*w2**2*np.pi*T_Xi)
    Xi = np.sqrt(np.float64(Xi))

    return Xi


def calculate_Xitz(params, Xit, tau2, Z):
    r"""Calculate the Rabi frequency as a function of tau and z."""
    Xi0 = calculate_Xi(params)
    w2 = params["w2"]
    tauw = params["tauw"]
    with_focusing = params["with_focusing"]
    Nt2 = len(tau2)
    Nz = len(Z)
    if with_focusing:
        zRS, zRXi = rayleigh_range(params)
        wz = w2*np.sqrt(1 + (Z/zRXi)**2)
        wz = np.outer(np.ones(Nt2), wz)
    else:
        wz = w2*np.ones((Nt2, Nz))

    if Xit == "square":
        Xi = Xi0*np.ones((Nt2, Nz))
    else:
        Xi = Xi0*np.sqrt(tauw)*np.outer(Xit(tau2), np.ones(Nz))

    return Xi*w2/wz


def calculate_power(params, Xi):
    r"""Calculate the power of the given Rabi frequency."""
    hbar = params["hbar"]
    e_charge = params["e_charge"]
    r2 = params["r2"]
    w2 = params["w2"]

    wz = w2

    dim = len(np.array(Xi).shape)
    if dim == 0:
        wz = w2
    elif dim == 1:
        Z = build_Z_mesh(params)
        if Xi.shape == Z.shape:
            # We assume that Omega is given as a function of z.
            zRS, zRXi = rayleigh_range(params)
            wz = w2*np.sqrt(1 + (Z/zRXi)**2)
        else:
            wz = w2
    elif dim == 2:
        Nt = Xi.shape[0]
        zRS, zRXi = rayleigh_range(params)
        wz = w2*np.sqrt(1 + (Z/zRXi)**2)
        wz = np.outer(np.ones(Nt), wz)

    return np.pi*(hbar*wz*np.abs(Xi)/e_charge/r2)**2 / 4/c/mu_0


def calculate_kappa(params):
    r"""Calculate the kappa parameter."""
    # We calculate the number density assuming Cs 133
    omega_laser1 = params["omega_laser1"]
    element = params["element"]
    isotope = params["isotope"]
    r1 = params["r1"]
    e_charge = params["e_charge"]
    hbar = params["hbar"]
    epsilon_0 = params["epsilon_0"]
    pumping = params["pumping"]

    n_atomic0 = vapour_number_density(params)
    if pumping != 1.0 or pumping:
        if element == "Cs":
            fground = [3, 4]
        elif element == "Rb":
            if isotope == 85:
                fground = [2, 3]
            else:
                fground = [1, 2]

        upper = 2*fground[1]+1
        lower = 2*fground[0]+1
        tot = upper + lower
        frac = upper/tot + pumping*lower/tot
        n_atomic0 = frac*n_atomic0

    return e_charge*r1*np.sqrt(n_atomic0*omega_laser1/(hbar*epsilon_0))


def calculate_OmegaBS(params):
    r"""Calculate the memory Rabi frequency."""
    delta1 = params["delta1"]
    Xi = calculate_Xi(params)
    Gamma21 = calculate_Gamma21(params)
    kappa = calculate_kappa(params)

    return Xi*delta1*kappa/2/np.abs(Gamma21)**2


def calculate_delta_stark(params):
    r"""Calculate the Stark shift."""
    delta1 = params["delta1"]
    Xi = calculate_Xi(params)
    Gamma21 = calculate_Gamma21(params)

    return -delta1*np.abs(Xi)**2/4/np.abs(Gamma21)**2


def calculate_delta_disp(params):
    r"""Calculate the dispersion shift."""
    delta1 = params["delta1"]
    Gamma21 = calculate_Gamma21(params)
    kappa = calculate_kappa(params)

    return -delta1*np.abs(kappa)**2/4/np.abs(Gamma21)**2


def calculate_Delta(params):
    r"""Calculate the two-photon detuning."""
    delta1 = params["delta1"]
    delta2 = params["delta2"]

    return delta1 + delta2


def calculate_xi0(params):
    r"""Return xi0, the position of the peak for Gamma(xi)."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)

    delta_stark = calculate_delta_stark(params)
    delta_disp = calculate_delta_disp(params)
    Delta = calculate_Delta(params)

    return -(Delta + delta_disp + delta_stark)/np.pi/c


def calculate_xi0p(params):
    r"""Return xi0p, the imaginary counterpart to xi0."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)

    OmegaBS = calculate_OmegaBS(params)

    return np.abs(OmegaBS)/np.pi/c


def calculate_phi0(params):
    r"""Return phi0."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    tauw = params["tauw"]
    kappa = calculate_kappa(params)
    Omega = calculate_Omega(params)

    phi0 = c*kappa**2 - 2*delta1**2 - 2*delta1*delta2 + 2*np.abs(Omega)**2
    phi0 = tauw*(phi0)/(2*delta1)
    return phi0

#
# def calculate_z0(params):
#     r"""Return phi0."""
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#
#     tauw = params["tauw"]
#     return tauw*c/2
#
#

#
#

#
#
# def calculate_Ctilde(params):
#     r"""Calculate the coupling Ctilde."""
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     tauw = params["tauw"]
#     Omega = calculate_Omega(params)
#     kappa = calculate_kappa(params)
#     Gamma21 = calculate_Gamma21(params)
#     Ctilde = tauw*kappa*Omega*np.sqrt(c/2)/np.sqrt(-Gamma21**2)
#     return Ctilde
#
#
# def calculate_beta(params, xi=None):
#     r"""Return the beta function."""
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if xi is None:
#         Z = build_Z_mesh(params)
#         xi = ffftfreq(Z)
#     Gamma21 = calculate_Gamma21(params)
#     Gamma32 = calculate_Gamma32(params)
#     Omega = calculate_Omega(params)
#     kappa = calculate_kappa(params)
#
#     beta = np.zeros(xi.shape[0], complex)
#     beta += Omega*np.conj(Omega)/(2*np.abs(Omega)**2)
#     beta += -Gamma21*Gamma32/(2*np.abs(Omega)**2)
#     beta += c*kappa**2/(8*np.abs(Omega)**2)
#     beta += Gamma21**2*Gamma32**2/(2*c*kappa**2*np.abs(Omega)**2)
#     beta += Omega**2*np.conj(Omega)**2/(2*c*kappa**2*np.abs(Omega)**2)
#     beta += 1j*np.pi*Gamma21*c*xi/(2*np.abs(Omega)**2)
#     beta += -np.pi**2*Gamma21**2*c*xi**2/(2*kappa**2*np.abs(Omega)**2)
#     beta += Gamma21*Gamma32*Omega*np.conj(Omega)/(c*kappa**2*np.abs(Omega)**2)
#     beta += -1j*np.pi*Gamma21**2*Gamma32*xi/(kappa**2*np.abs(Omega)**2)
#     aux = -1j*np.pi*Gamma21*Omega*xi*np.conj(Omega)
#     beta += aux/(kappa**2*np.abs(Omega)**2)
#     beta = np.sqrt(beta)
#
#     return beta
#
#
# def calculate_F(params, xi=None):
#     r"""Return the beta function."""
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if xi is None:
#         Z = build_Z_mesh(params)
#         xi = ffftfreq(Z)
#
#     z0 = calculate_z0(params)
#     phi0 = calculate_phi0(params)
#
#     Ctilde = calculate_Ctilde(params)
#     beta = calculate_beta(params, xi)
#
#     Fxi = -Ctilde**2*np.exp(-1j*(phi0 + 2*np.pi*z0*xi))*sinc(Ctilde*beta)**2
#     return Fxi
#
#
# def calculate_optimal_delta2(params):
#     r"""Calculate the detuning of the control field to obtain two-photon
#     resonance and also compensate for Stark shifting.
#     """
#     delta1 = params["delta1"]
#     delta2 = params["delta2"]
#     Omega = calculate_Omega(params)
#     kappa = calculate_kappa(params)
#     delta2 = - delta1 + (np.abs(Omega)**2 - c/2*kappa**2)/delta1
#
#     return delta2
#
#
# def calculate_optimal_input_xi(params, xi=None, force_xi0=False,
#                                with_critical_energy=True):
#     r"""Calculate the optimal `xi`-space input for the given parameters.
#
#     Note that this returns a Gaussian pulse of time duration params["taus"]
#     """
#     params_ = params.copy()
#     if not params_["USE_SQUARE_CTRL"] or str(params_["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if xi is None:
#         Z = build_Z_mesh(params_)
#         xi = ffftfreq(Z)
#
#     if with_critical_energy:
#         energy_pulse2 = calculate_pulse_energy(params_)
#         params_["energy_pulse2"] = energy_pulse2
#     xi0 = calculate_xi0(params_)
#
#     taus = params_["taus"]
#     tauw = params_["tauw"]
#     DeltanuS_num = time_bandwith_product(1)/taus
#     DeltaxiS_num = DeltanuS_num*2/c
#     sigma_xi = DeltaxiS_num/(2*np.sqrt(np.log(2)))
#
#     # We make sure that the oscillations in the signal are not too fast.
#     Nu0 = np.abs(c*xi0)
#     if (taus*Nu0 > 5.0 or tauw*Nu0 > 5.0) and not force_xi0:
#         mes = "The optimal signal has a linear phase that is too fast "
#         mes += "for the grid to represent accurately. "
#         mes += "Using a flat phase instead."
#         mes += " Set force_xi0=True to override this (but don't Fourier"
#         mes += " transform into input for a z-space problem, please). "
#         mes += "The best thing is to set "
#         mes += "`params[delta2] = calculate_optimal_delta2(params)`"
#         warnings.warn(mes)
#         # warnings.filterwarnings('ignore', mes)
#         xi0 = 0.0
#
#     Zoff = params_["tauw"]/2*(c/2)
#     Sin = hermite_gauss(0, xi-xi0, sigma_xi)
#     Sin = Sin*np.exp(2*np.pi*1j*Zoff*xi)
#     # We normalize so that the integral of the signal mod square over tau
#     # is 1.
#     Sin = Sin*np.sqrt(c/2)
#     return xi, Sin
#
#
# def calculate_optimal_input_Z(params, Z=None, with_critical_energy=True):
#     r"""Calculate the optimal `Z`-space input for the given parameters.
#
#     Note this returns a Gaussian pulse of time duration params["taus"]
#     """
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if Z is None:
#         band = True
#         Zp = build_Z_mesh(params)
#     else:
#         band = False
#
#     # We get a reasonable xi and Z mesh.
#     xi0 = calculate_xi0(params)
#     Deltaxi = 2/c/params["tauw"]
#
#     a1 = xi0+20*Deltaxi/2
#     a2 = xi0-20*Deltaxi/2
#     aa = np.amax(np.abs(np.array([a1, a2])))
#     xi = np.linspace(-aa, aa, 1001)
#     kwargs = {"with_critical_energy": with_critical_energy}
#     xi, S0xi = calculate_optimal_input_xi(params, xi, **kwargs)
#
#     # We Fourier transform it.
#     Z = ffftfreq(xi)
#     S0Z = iffftfft(S0xi, xi)
#
#     taus = params["taus"]
#     tauw = params["tauw"]
#     Nu0 = np.abs(c*xi0)
#     if taus*Nu0 > 5.0 or tauw*Nu0 > 5.0:
#         mes = "The optimal signal has a linear phase that is too fast "
#         mes += "for the grid to represent accurately. "
#         mes += "Using a flat phase instead."
#         warnings.warn(mes)
#         warnings.filterwarnings('ignore', mes)
#         xi0 = 0.0
#
#         Z = np.linspace(-0.25, 0.25, 1001)
#
#         DeltanuS_num = time_bandwith_product(1)/taus
#         DeltaxiS_num = DeltanuS_num*2/c
#         sigma_xi = DeltaxiS_num/(2*np.sqrt(np.log(2)))
#         Zoff = tauw/2*(c/2)
#
#         Sin = hermite_gauss(0, xi-xi0, sigma_xi)
#         Sin = Sin*np.exp(2*np.pi*1j*Zoff*xi)*np.sqrt(c/2)
#
#         # S0Z = hermite_gauss(0, Z+Zoff, 1/np.pi**2*np.sqrt(2)/sigma_xi)
#         S0Z = hermite_gauss(0, Z+Zoff, 1/2.0/np.pi/sigma_xi)
#         S0Z = S0Z*np.sqrt(c/2)
#         # S0Z = S0Z*np.exp(2*np.pi*1j*Zoff*xi)*np.sqrt(c/2)
#
#     if not band:
#         S0Z_interp = interpolator(Z, S0Z, kind="cubic")
#         S0Z = S0Z_interp(Zp)
#         return Zp, S0Z
#     else:
#         return Z, S0Z
#
#
# def calculate_optimal_input_tau(params, tau=None, with_critical_energy=True):
#     r"""Calculate the optimal `tau`-space input for the given parameters.
#
#     Note this returns a Gaussian pulse of time duration params["taus"]
#     """
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if tau is None:
#         tau = build_t_mesh(params)
#
#     kappa = calculate_kappa(params)
#     Gamma21 = calculate_Gamma21(params)
#
#     kwargs = {"with_critical_energy": with_critical_energy}
#     Z, S0Z = calculate_optimal_input_Z(params, **kwargs)
#     S0Z_interp = interpolator(Z, S0Z, kind="cubic")
#
#     L = params["L"]
#     tau0 = params["t0w"] - params["tauw"]/2
#     S0tau = S0Z_interp(-L/2 - c*(tau-tau0)/2)
#
#     S0tau = S0tau*np.exp(-c*kappa**2*(tau-tau0)/(2*Gamma21))
#     S0tau = S0tau/np.sqrt(num_integral(np.abs(S0tau)**2, tau))
#
#     return tau, S0tau
#
#
# def approximate_optimal_input(params, tau=None, Z=None, mode="hg0", shift=0.0):
#     r"""Get optimal input."""
#     c = params["c"]
#     xi0 = calculate_xi0(params)
#     phi0 = calculate_phi0(params)
#     Zoff = params["tauw"]/2*(c/2)
#     kappa0 = calculate_kappa(params)
#     Gamma21 = calculate_Gamma21(params)
#     L = params["L"]
#
#     taus = params["taus"]
#     DeltanuS_num = time_bandwith_product(1)/taus
#     DeltaxiS_num = DeltanuS_num*2/c
#     sig_xi = DeltaxiS_num/(2*np.sqrt(np.log(2)))
#     sig_z = 1/2/np.pi/sig_xi
#
#     t0 = params["t0w"] - params["tauw"]/2
#     t0 = - params["tauw"]/2 + shift*L/c
#     # t0 = 0.0
#
#     if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
#         mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
#         raise ValueError(mes)
#     if tau is None:
#         tau = build_t_mesh(params)
#     if Z is None:
#         Z = build_Z_mesh(params)
#
#     tau0 = -c*t0/2 + L/2 - Zoff
#     # tau0 = -c*t0/2 + L/2 #- Zoff
#     #########################################################
#     S0t = np.exp(2*1j*np.pi*xi0*(-c*tau/2-tau0))
#     S0t *= np.exp(1j*phi0)
#     # S0t *= hermite_gauss(0, -c*tau/2 - tau0, sig_t)
#     if mode[:2] == "hg":
#         nn = int(mode[-1])
#         S0t *= hermite_gauss(nn, -c*tau/2 - tau0, sig_z)
#     elif mode[:2] == "ha":
#         nn = int(mode[-1])
#         S0t *= harmonic(nn, -c*tau/2 - tau0, taus*c)
#     else:
#         raise ValueError
#     S0t *= np.exp(-c*kappa0**2*(tau-t0)/(2*Gamma21))*np.sqrt(c/2)
#
#     #########################################################
#     tau0 = -c*t0/2 - Z - Zoff
#     tau_ini = tau[0]
#     S0z = np.exp(2*1j*np.pi*xi0*(-c*tau_ini/2-tau0))
#     S0z *= np.exp(1j*phi0)
#     if mode[:2] == "hg":
#         nn = int(mode[-1])
#         S0z *= hermite_gauss(0, -c*tau_ini/2 - tau0, sig_z)
#     elif mode[:2] == "ha":
#         nn = int(mode[-1])
#         S0z *= harmonic(1, -c*tau_ini/2 - tau0, taus*c)
#     S0z *= np.exp(-c*kappa0**2*(tau_ini-t0)/(2*Gamma21))*np.sqrt(c/2)
#
#     return S0t, S0z, tau, Z
#
#


def calculate_pulse_energy(params, order=0):
    r"""Calculate the necessary pulse energy for unit efficiency."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)

    tauw = params["tauw"]
    delta1 = params["delta1"]
    hbar = params["hbar"]
    # epsilon_0 = params["epsilon_0"]
    w2 = params["w2"]
    e_charge = params["e_charge"]
    r2 = params["r2"]

    kappa = calculate_kappa(params)

    En = np.pi**3*(delta1*hbar*w2)**2
    return En/(tauw*mu_0*c*(r2*e_charge*kappa)**2)

#
#
# def calculate_efficiencies(tau, Z, Bw, Sw, Br, Sr, verbose=0):
#     r"""Calculate the memory efficiencies for a given write-read
#     process.
#
#     These are the total memory efficiency, TB, RS, RB, TS.
#     """
#     L = Z[-1] - Z[0]
#     tau_iniS = tau[0]
#     tau_iniQ = tau_iniS - L*2/c
#     tauQ0 = (tau_iniS-tau_iniQ)/(Z[0]-Z[-1])*(Z-Z[0]) + tau_iniS
#
#     tau_finS = tau[-1]
#     tau_finQ = tau_finS + L*2/c
#     tauQf = (tau_finS-tau_finQ)/(Z[-1]-Z[0])*(Z-Z[-1]) + tau_finS
#
#     # The initial photon number.
#     NS = num_integral(np.abs(Sw[:, 0])**2, tau)
#     NS += num_integral(np.abs(Sw[0, :])**2, tauQ0)
#     # The transmitted photon number.
#     NST = num_integral(np.abs(Sw[:, -1])**2, tau)
#     NST += num_integral(np.abs(Sw[-1, :])**2, tauQf)
#     # The retrieved photon number.
#     Nf = num_integral(np.abs(Sr[:, -1])**2, tau)
#     Nf += num_integral(np.abs(Sr[-1, :])**2, tauQf)
#     # The initial spin-wave number.
#     NB = num_integral(np.abs(Br[0, :])**2, tau)
#     # The transmitted. spin-wave number.
#     NBT = num_integral(np.abs(Br[-1, :])**2, tau)
#
#     # Nt1 = tau1.shape[0]
#     # S0Z_num = Sw[Nt1-1, :]
#
#     TS = NST/NS
#     RS = 1 - TS
#
#     TB = NBT/NB
#     RB = 1 - TB
#
#     eta_num = Nf/NS
#
#     if verbose > 0:
#         print("Numerical efficiency      : {:.4f}".format(eta_num))
#         print("")
#         print("Beam-splitter picture transmissivities and reflectivities:")
#         print("TB: {:.4f}, RS: {:.4f}".format(TB, RS))
#         print("RB: {:.4f}, TS: {:.4f}".format(RB, TS))
#
#     return eta_num, TB, RS, RB, TS
#
#
# #############################################################################
# # Finite difference ORCA routines.
#
#
# def eqs_fdm(params, tau, Z, Omegat="square", case=0, adiabatic=True,
#             pt=4, pz=4, plots=False, folder="", sparse=False):
#     r"""Calculate the matrix form of equations `Wp X = 0`."""
#     if not adiabatic:
#         nv = 3
#     else:
#         nv = 2
#     # We unpack parameters.
#     if True:
#         with_focusing = params["with_focusing"]
#         Nt = tau.shape[0]
#         Nz = Z.shape[0]
#         Gamma21 = calculate_Gamma21(params)
#         Gamma32 = calculate_Gamma32(params)
#         kappa = calculate_kappa(params)
#         Omega0 = calculate_Omega(params)
#
#         tauw = params["tauw"]
#
#         nX = nv*Nt*Nz
#         Ntz = Nt*Nz
#     # We build the derivative matrices.
#     if True:
#         args = [tau, Z]
#         kwargs = {"pt": pt, "pz": pz, "plots": plots, "folder": folder,
#                   "sparse": sparse}
#         DT, DZ = fdm_derivative_operators(*args, **kwargs)
#
#     # We calculate Omega(tau, z) as an array, and then as a flattened,
#     # diagonal matrix.
#     w0Xi = params["w2"]
#     zRS, zRXi = rayleigh_range(params)
#     wXi = w0Xi*np.sqrt(1 + (Z/zRXi)**2)
#
#     if with_focusing:
#         Omegaz = Omega0*w0Xi/wXi
#     else:
#         Omegaz = Omega0*np.ones(Nz)
#
#     if Omegat == "square":
#         Omegatauz = np.outer(np.ones(Nt), Omegaz).flatten()
#     elif with_focusing:
#         Omegatauz = Omega0*np.sqrt(tauw)
#         Omegatauz *= np.outer(Omegat(tau), w0Xi/wXi).flatten()
#     else:
#         Omegatauz = Omega0*np.sqrt(tauw)
#         Omegatauz *= np.outer(Omegat(tau), np.ones(Nz)).flatten()
#
#     if sparse:
#         eye = sp_eye(Ntz, format=bfmt)
#         A = bfmtf((nX, nX), dtype=np.complex128)
#         Omega = spdiags(Omegatauz, [0], Ntz, Ntz, format=bfmt)
#     else:
#         eye = np.eye(Ntz)
#         A = np.zeros((nX, nX), complex)
#         Omega = np.diag(Omegatauz)
#
#     # We build the A matrix.
#     if True:
#         # Empty space.
#         if case == 0 and adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma32*eye)
#             A = set_block(A, 1, 1, c/2*DZ)
#         # Storage phase.
#         elif case == 1 and adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma32*eye)
#             A = set_block(A, 1, 1, c*kappa**2/2/Gamma21*eye)
#             A = set_block(A, 1, 1, c/2*DZ)
#         # Memory write/read phase.
#         elif case == 2 and adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma32*eye)
#             A = set_block(A, 1, 1, c*kappa**2/2/Gamma21*eye)
#             A = set_block(A, 1, 1, c/2*DZ)
#
#             aux1 = np.abs(Omega)**2/Gamma21
#             aux2 = kappa*Omega/Gamma21
#             aux3 = c*kappa*np.conjugate(Omega)/2/Gamma21
#
#             A = set_block(A, 0, 0, aux1)
#             A = set_block(A, 0, 1, aux2)
#             A = set_block(A, 1, 0, aux3)
#
#         elif case == 0 and not adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#             A = set_block(A, 2, 2, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma21*eye)
#             A = set_block(A, 1, 1, Gamma32*eye)
#             A = set_block(A, 2, 2, c/2*DZ)
#
#         elif case == 1 and not adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#             A = set_block(A, 2, 2, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma21*eye)
#             A = set_block(A, 1, 1, Gamma32*eye)
#             A = set_block(A, 2, 2, c/2*DZ)
#             A = set_block(A, 0, 2, 1j*kappa*eye)
#             A = set_block(A, 2, 0, 1j*kappa*c/2*eye)
#         elif case == 2 and not adiabatic:
#             # We set the time derivatives.
#             A = set_block(A, 0, 0, DT)
#             A = set_block(A, 1, 1, DT)
#             A = set_block(A, 2, 2, DT)
#
#             # We set the right-hand side terms.
#             A = set_block(A, 0, 0, Gamma21*eye)
#             A = set_block(A, 1, 1, Gamma32*eye)
#             A = set_block(A, 2, 2, c/2*DZ)
#             A = set_block(A, 0, 2, 1j*kappa*eye)
#             A = set_block(A, 2, 0, 1j*kappa*c/2*eye)
#
#             A = set_block(A, 0, 1, 1j*np.conjugate(Omega))
#             A = set_block(A, 1, 0, 1j*Omega)
#
#     if plots:
#         ################################################################
#         # Plotting Wp.
#         plt.figure(figsize=(15, 15))
#         plt.title("$A'$")
#         plt.imshow(np.log(np.abs(A)))
#         plt.savefig(folder+"A.png", bbox_inches="tight")
#         plt.close("all")
#
#     return A
#
#
# def solve_fdm_block(params, S0t, S0z, B0z, tau, Z, P0z=None, Omegat="square",
#                     case=0, method=0, pt=4, pz=4,
#                     plots=False, folder="", name="", verbose=0):
#     r"""We solve using the finite difference method for given
#     boundary conditions, and with time and space precisions `pt` and `pz`.
#
#     INPUT:
#
#     -  ``params`` - dict, the problem's parameters.
#
#     -  ``S0t`` - array, the S(Z=-L/2, t) boundary condition.
#
#     -  ``S0z`` - array, the S(Z, t=0) boundary condition.
#
#     -  ``B0z`` - array, the B(Z, t=0) boundary condition.
#
#     -  ``tau`` - array, the time axis.
#
#     -  ``Z`` - array, the space axis.
#
#     -  ``P0z`` - array, the P(Z, t=0) boundary condition (default None).
#
#     -  ``Omegat`` - function, a function that returns the temporal mode of the
#                     Rabi frequency at time tau (default "square").
#
#     -  ``case`` - int, the dynamics to solve for: 0 for free space, 1 for
#                   propagation through vapour, 2 for propagation through vapour
#                   and non-zero control field.
#
#     -  ``method`` - int, the fdm method to use: 0 to solve the full space, 1
#                   to solve by time step slices.
#
#     -  ``pt`` - int, the precision order for the numerical time derivate. Must
#                 be even.
#
#     -  ``pz`` - int, the precision order for the numerical space derivate. Must
#                 be even.
#
#     -  ``plots`` - bool, whether to make plots.
#
#     -  ``folder`` - str, the directory to save plots in.
#
#     -  ``verbose`` - int, a vague measure much of messages to print.
#
#     OUTPUT:
#
#     A solution to the equations for the given case and boundary conditions.
#
#     """
#     t00_tot = time()
#     # We unpack parameters.
#     if True:
#         Nt = params["Nt"]
#         Nz = params["Nz"]
#         # Nt_prop = pt + 1
#
#         # The number of functions.
#         nv = 2
#     # We make pre-calculations.
#     if True:
#         if P0z is not None:
#             P = np.zeros((Nt, Nz), complex)
#             P[0, :] = P0z
#         B = np.zeros((Nt, Nz), complex)
#         S = np.zeros((Nt, Nz), complex)
#         B[0, :] = B0z
#         S[0, :] = S0z
#     if method == 0:
#         # We solve the full block.
#         sparse = True
#         aux1 = [params, tau, Z]
#         aux2 = {"Omegat": Omegat, "pt": pt, "pz": pz, "case": case,
#                 "folder": folder, "plots": False, "sparse": sparse,
#                 "adiabatic": P0z is None}
#         # print(Omegat)
#         t00 = time()
#         A = eqs_fdm(*aux1, **aux2)
#         if verbose > 0: print("FDM Eqs time  : {:.3f} s".format(time()-t00))
#         #############
#         # New method.
#         # We get the input indices.
#         t00 = time()
#         auxi = np.arange(nv*Nt*Nz).reshape((nv, Nt, Nz))
#         indsB0 = auxi[0, 0, :].tolist()
#         indsQ0 = np.flip(auxi[1, 0, :]).tolist()
#         indsS0 = auxi[1, :, 0].tolist()
#         inds0_ = indsB0 + indsQ0 + indsS0
#
#         indsBf = auxi[0, -1, :].tolist()
#         indsSf = auxi[1, :, -1].tolist()
#         indsQf = np.flip(auxi[1, -1, :]).tolist()
#         indsf_ = indsBf + indsSf + indsQf
#         indsf_ = auxi.flatten().tolist()
#
#         # We build the input vector.
#         input = np.zeros((len(inds0_), 1), complex)
#         input[:Nz, 0] = B0z
#         input[Nz:2*Nz, 0] = np.flip(S0z)
#         input[2*Nz:, 0] = S0t
#
#         Y = solve_fdm(A, inds0_, indsf_, input=input)
#         B, S = np.reshape(Y, (nv, Nt, Nz))
#         if verbose > 0: print("FDM Sol time  : {:.3f} s".format(time()-t00))
#         #############
#
#     # Report running time.
#     if verbose > 0:
#         runtime_tot = time() - t00_tot
#         aux = [runtime_tot, Nt, Nz, Nt*Nz]
#         mes = "FDM block time: {:.3f} s for a grid of {} x {} = {} points."
#         print(mes.format(*aux))
#     # Plotting.
#     if plots:
#         plt.figure(figsize=(15, 8))
#         plt.subplot(1, 2, 1)
#         plt.title("$B$ numeric")
#         cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(B), shading="auto")
#         plt.colorbar(cs)
#         plt.ylabel(r"$\tau$ (ns)")
#         plt.xlabel("$Z$ (cm)")
#
#         plt.subplot(1, 2, 2)
#         plt.title("$S$ numeric")
#         cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(S), shading="auto")
#         plt.colorbar(cs)
#         plt.ylabel(r"$\tau$ (ns)")
#         plt.xlabel("$Z$ (cm)")
#         aux = folder+"solution_numeric"+name+".png"
#         plt.savefig(aux, bbox_inches="tight")
#         plt.close("all")
#
#     if P0z is not None:
#         return P, B, S
#     else:
#         return B, S
#
#
# def solve(params, S0t=None, S0z=None, B0z=None, P0z=None, Omegat="square",
#           method=0, pt=4, pz=4,
#           folder="", name="", plots=False, verbose=0,
#           seed=None, analytic_storage=True, return_modes=False):
#     r"""We solve using the finite difference method for given
#     boundary conditions, and with time and space precisions `pt` and `pz`.
#     """
#     adiabatic = P0z is None
#     t00f = time()
#     # We unpack parameters.
#     if True:
#         aux = build_mesh_fdm(params)
#         params, Z, tau, tau1, tau2, tau3 = aux
#         Nt = params["Nt"]
#         Nz = params["Nz"]
#         kappa = calculate_kappa(params)
#         Gamma21 = calculate_Gamma21(params)
#         Gamma32 = calculate_Gamma32(params)
#         # Omega = calculate_Omega(params)
#         taus = params["taus"]
#         t0s = params["t0s"]
#         L = Z[-1] - Z[0]
#
#         Nt1 = tau1.shape[0]
#         Nt2 = tau2.shape[0]
#         Nt3 = tau3.shape[0]
#
#         # We initialize the solution.
#         if not adiabatic:
#             P = np.zeros((Nt, Nz), complex)
#         B = np.zeros((Nt, Nz), complex)
#         S = np.zeros((Nt, Nz), complex)
#     # We solve in the initial region.
#     if True:
#         if verbose > 0: t000f = time()
#         B_exact1 = np.zeros((Nt1, Nz))
#         S_exact1 = np.zeros((Nt1, Nz))
#         ttau1 = np.outer(tau1, np.ones(Nz))
#         ZZ1 = np.outer(np.ones(Nt1), Z)
#
#         nshg = params["nshg"]
#         if seed == "S":
#             sigs = taus/(2*np.sqrt(2*np.log(2)))*np.sqrt(2)
#             S_exact1 = hermite_gauss(nshg, -t0s + ttau1 - 2*ZZ1/c, sigs)
#             S_exact1 = S_exact1*np.exp(-(ZZ1+L/2)*kappa**2/Gamma21)
#             S[:Nt1, :] = S_exact1
#         elif seed == "B":
#             nshg = nshg + 1
#             B_exact1 = harmonic(nshg, ZZ1, L)
#             B[:Nt1, :] = B_exact1
#         elif S0t is not None or S0z is not None or B0z is not None:
#             if S0t is not None:
#                 S0t_interp = interpolator(tau, S0t, kind="cubic")
#                 S_exact1 = S0t_interp(ttau1 - 2*(ZZ1+L/2)/c)
#                 S_exact1 = S_exact1*np.exp(-(ZZ1+L/2)*kappa**2/Gamma21)
#                 S[:Nt1, 1:] += S_exact1[:, 1:]
#                 S[:, 0] = S0t
#             if S0z is not None:
#                 t00 = params["t0w"] - params["tauw"]/2
#                 S0z_interp = interpolator(Z, S0z, kind="cubic")
#                 S_exact2 = S0z_interp(ZZ1-c/2*(ttau1-tau1[0]))
#                 S_exact2 *= np.exp(-c*kappa**2*(ttau1-t00)/(2*Gamma21))
#
#                 S[:Nt1, 1:] += S_exact2[:, 1:]
#                 S[0, :] = S0z
#
#             if B0z is not None:
#                 B_exact1 = np.outer(np.ones(Nt1), B0z)
#                 B[:Nt1, :] = B_exact1
#         else:
#             mes = "Either of S0t, B0z, or seed must be given as arguments"
#             raise ValueError(mes)
#         if verbose > 0: print("region 1 time : {:.3f} s".format(time()-t000f))
#     # We obtain the input modes for the memory region.
#     if True:
#         if S0t is None and B0z is None:
#             if seed == "P":
#                 # We seed with a harmonic mode.
#                 raise NotImplementedError
#             elif seed == "B":
#                 # We seed with a harmonic mode.
#                 B02z = harmonic(nshg, Z, L)
#                 S02z = np.zeros(Nz, complex)
#                 S02t = np.zeros(Nt2, complex)
#             elif seed == "S":
#                 # We seed with a Hermite-Gauss mode.
#                 # HG modes propagate as:
#                 #  S(tau, Z) = HG_n(t-t0s - 2*Z/c, sigma)
#                 #            x exp(-(Z+D/2)*kappa**2/Gamma21)
#                 #
#                 # We calculate the gaussian standard deviation.
#                 B02z = np.zeros(Nz, complex)
#                 S02z = hermite_gauss(nshg, tau2[0] - t0s - 2*Z/c, sigs)
#                 S02z = S02z*np.exp(-(Z+L/2)*kappa**2/Gamma21)
#                 S02t = hermite_gauss(nshg, tau2 - t0s + L/c, sigs)
#             else:
#                 raise ValueError
#         else:
#             if S0t is not None:
#                 B02z = B_exact1[Nt1-1, :]
#                 S02z = S_exact1[Nt1-1, :]
#                 S02t = S0t[Nt1-1:Nt1+Nt2-1]
#             if B0z is not None:
#                 B02z = B0z
#                 S02z = np.zeros(Nz, complex)
#                 S02t = np.zeros(Nt2, complex)
#     # We solve in the memory region using the FDM.
#     if True:
#         if verbose > 0: t000f = time()
#         params_memory = params.copy()
#         params_memory["Nt"] = Nt2
#
#         aux1 = [params_memory, S02t, S02z, B02z, tau2, Z]
#         aux2 = {"Omegat": Omegat, "method": method, "pt": pt, "pz": pz,
#                 "folder": folder, "plots": False,
#                 "verbose": verbose-1, "P0z": P0z, "case": 2}
#         if adiabatic:
#             B2, S2 = solve_fdm_block(*aux1, **aux2)
#             B[Nt1-1:Nt1+Nt2-1] = B2
#             S[Nt1-1:Nt1+Nt2-1] = S2
#         else:
#             P2, B2, S2 = solve_fdm_block(*aux1, **aux2)
#             P[Nt1-1:Nt1+Nt2-1] = P2
#             B[Nt1-1:Nt1+Nt2-1] = B2
#             S[Nt1-1:Nt1+Nt2-1] = S2
#
#         if verbose > 0: print("region 2 time : {:.3f} s".format(time()-t000f))
#     # We solve in the storage region.
#     if True:
#         if verbose > 0: t000f = time()
#         B03z = B[Nt1+Nt2-2, :]
#         S03z = S[Nt1+Nt2-2, :]
#         if seed == "S":
#             S03t = hermite_gauss(nshg, tau3 - t0s + L/c, sigs)
#         elif S0t is not None:
#             S03t = S0t[Nt1+Nt2-2:]
#         else:
#             S03t = np.zeros(Nt3, complex)
#
#         params_storage = params.copy()
#         params_storage["Nt"] = Nt3
#         aux1 = [params_storage, S03t, S03z, B03z, tau3, Z]
#         aux2 = {"pt": pt, "pz": pz, "folder": folder, "plots": False,
#                 "verbose": 1, "P0z": P0z, "case": 1}
#
#         # We calculate analyticaly.
#         if adiabatic:
#             if analytic_storage > 0:
#                 t03 = tau3[0]
#                 ttau3 = np.outer(tau3, np.ones(Nz))
#                 ZZ3 = np.outer(np.ones(Nt3), Z)
#
#                 B[Nt1+Nt2-2:] = B03z*np.exp(-Gamma32*(ttau3 - t03))
#
#                 # The region determined by S03z
#
#                 S03z_reg = np.where(ttau3 <= t03 + (2*ZZ3+L)/c, 1.0, 0.0)
#                 # The region determined by S03t
#                 S03t_reg = 1 - S03z_reg
#                 S03z_f = interpolator(Z, S03z, kind="cubic")
#                 S03t_f = interpolator(tau3, S03t, kind="cubic")
#
#             if analytic_storage == 1:
#                 S03z_reg = S03z_reg*S03z_f(ZZ3 - (ttau3-t03)*c/2)
#                 S03z_reg = S03z_reg*np.exp(-c*kappa**2/2/Gamma21*(ttau3-t03))
#
#                 S03t_reg = S03t_reg*S03t_f(ttau3 - (2*ZZ3+L)/c)
#                 S03t_reg = S03t_reg*np.exp(-kappa**2/Gamma21*(ZZ3+L/2))
#                 S3 = S03z_reg + S03t_reg
#                 S[Nt1+Nt2-2:] = S3
#             elif analytic_storage == 2:
#                 aux1 = S03z_f(L/2 - (tau3-t03)*c/2)
#                 aux1 = aux1*np.exp(-c*kappa**2/2/Gamma21*(tau3-t03))
#
#                 aux2 = S03t_f(tau3 - (2*L/2+L)/c)
#                 aux2 = aux2*np.exp(-kappa**2/Gamma21*(L/2+L/2))
#
#                 Sf3t = S03z_reg[:, -1]*aux1 + S03t_reg[:, -1]*aux2
#                 S[Nt1+Nt2-2:, -1] = Sf3t
#
#                 tff = tau3[-1]
#                 aux3 = S03z_f(Z - (tff-t03)*c/2)
#                 aux3 = aux3*np.exp(-c*kappa**2/2/Gamma21*(tff-t03))
#
#                 aux4 = S03t_f(tff - (2*Z+L)/c)
#                 aux4 = aux4*np.exp(-kappa**2/Gamma21*(Z+L/2))
#
#                 Sf3z = S03z_reg[-1, :]*aux3 + S03t_reg[-1, :]*aux4
#                 S[-1, :] = Sf3z
#                 S[Nt1+Nt2-2:, 0] = S03t
#
#             elif analytic_storage == 0:
#                 B3, S3 = solve_fdm_block(*aux1, **aux2)
#                 B[Nt1+Nt2-2:] = B3
#                 S[Nt1+Nt2-2:] = S3
#             else:
#                 raise ValueError
#         else:
#             P3, B3, S3 = solve_fdm_block(*aux1, **aux2)
#             P[Nt1+Nt2-2:] = P3
#             B[Nt1+Nt2-2:] = B3
#             S[Nt1+Nt2-2:] = S3
#
#         if verbose > 0: print("region 3 time : {:.3f} s".format(time()-t000f))
#     if verbose > 0:
#         print("Full exec time: {:.3f} s".format(time()-t00f))
#     # Plotting.
#     if plots:
#         fs = 15
#         if verbose > 0: print("Plotting...")
#
#         fig, ax1 = plt.subplots()
#         ax2 = ax1.twinx()
#         ax1.plot(tau*1e9, np.abs(S[:, 0])**2*1e-9, "b-")
#         ax1.plot(tau*1e9, np.abs(S[:, -1])**2*1e-9, "g-")
#
#         angle1 = np.unwrap(np.angle(S[:, -1]))/2/np.pi
#         ax2.plot(tau*1e9, angle1, "g:")
#
#         ax1.set_xlabel(r"$\tau \ [ns]$", fontsize=fs)
#         ax1.set_ylabel(r"Signal  [1/ns]", fontsize=fs)
#         ax2.set_ylabel(r"Phase  [revolutions]", fontsize=fs)
#         plt.savefig(folder+"Sft_"+name+".png", bbox_inches="tight")
#         plt.close()
#
#         plt.figure(figsize=(15, 9))
#         plt.subplot(1, 2, 1)
#         plt.title("$B$ FDM")
#         cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(B)**2, shading="auto")
#         plt.colorbar(cs)
#         plt.ylabel(r"$\tau$ (ns)")
#         plt.xlabel("$Z$ (cm)")
#
#         plt.subplot(1, 2, 2)
#         plt.title("$S$ FDM")
#         cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(S)**2*1e-9, shading="auto")
#         plt.colorbar(cs)
#         plt.ylabel(r"$\tau$ (ns)")
#         plt.xlabel("$Z$ (cm)")
#
#         plt.savefig(folder+"solution_fdm_"+name+".png", bbox_inches="tight")
#         plt.close("all")
#
#     if adiabatic:
#         if return_modes:
#             B0 = B02z
#             S0 = S[:, 0]
#             B1 = B03z
#             S1 = S[:, -1]
#             return tau, Z, B0, S0, B1, S1
#         else:
#             return tau, Z, B, S
#     else:
#         return tau, Z, P, B, S
#
#
# def check_block_fdm(params, B, S, tau, Z, case=0, P=None,
#                     pt=4, pz=4, folder="", plots=False, verbose=1):
#     r"""Check the equations in an FDM block."""
#     # We build the derivative operators.
#     Nt = tau.shape[0]
#     Nz = Z.shape[0]
#     Gamma32 = calculate_Gamma32(params)
#     Gamma21 = calculate_Gamma21(params)
#     Omega = calculate_Omega(params)
#     kappa = calculate_kappa(params)
#     Dt = derivative_operator(tau, p=pt)
#     Dz = derivative_operator(Z, p=pt)
#
#     adiabatic = P is None
#
#     # Empty space.
#     if case == 0 and adiabatic:
#         # We get the time derivatives.
#         DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
#         DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T
#
#         DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
#         rhsB = -Gamma32*B
#         rhsS = -c/2*DzS
#     # Storage phase.
#     elif case == 1 and adiabatic:
#         # We get the time derivatives.
#         DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
#         DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T
#
#         DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
#         rhsB = -Gamma32*B
#         rhsS = -c/2*DzS - c*kappa**2/2/Gamma21*S
#     # Memory write/read phase.
#     elif case == 2 and adiabatic:
#         # We get the time derivatives.
#         DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
#         DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T
#
#         DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
#         rhsB = -Gamma32*B
#         rhsS = -c/2*DzS - c*kappa**2/2/Gamma21*S
#
#         rhsB += -np.abs(Omega)**2/Gamma21*B
#         rhsB += -kappa*Omega/Gamma21*S
#         rhsS += -c*kappa*np.conjugate(Omega)/2/Gamma21*B
#
#     else:
#         raise ValueError
#
#     if True:
#         # We put zeros into the boundaries.
#         ig = pt/2 + 1
#         ig = pt + 1
#         ig = 1
#
#         DtB[:ig, :] = 0
#         DtS[:ig, :] = 0
#         DtS[:, :ig] = 0
#
#         rhsB[:ig, :] = 0
#         rhsS[:ig, :] = 0
#         rhsS[:, :ig] = 0
#
#         # We put zeros in all the boundaries to neglect border effects.
#         DtB[-ig:, :] = 0
#         DtS[-ig:, :] = 0
#         DtB[:, :ig] = 0
#         DtB[:, -ig:] = 0
#         DtS[:, -ig:] = 0
#
#         rhsB[-ig:, :] = 0
#         rhsS[-ig:, :] = 0
#         rhsB[:, :ig] = 0
#         rhsB[:, -ig:] = 0
#         rhsS[:, -ig:] = 0
#
#     if True:
#         Brerr = rel_error(DtB, rhsB)
#         Srerr = rel_error(DtS, rhsS)
#
#         Bgerr = glo_error(DtB, rhsB)
#         Sgerr = glo_error(DtS, rhsS)
#
#         i1, j1 = np.unravel_index(Srerr.argmax(), Srerr.shape)
#         i2, j2 = np.unravel_index(Sgerr.argmax(), Sgerr.shape)
#
#         with warnings.catch_warnings():
#             mes = r'divide by zero encountered in log10'
#             warnings.filterwarnings('ignore', mes)
#
#             aux1 = list(np.log10(get_range(Brerr)))
#             aux1 += [np.log10(np.mean(Brerr))]
#             aux1 += list(np.log10(get_range(Srerr)))
#             aux1 += [np.log10(np.abs(np.mean(Srerr)))]
#
#             aux2 = list(np.log10(get_range(Bgerr)))
#             aux2 += [np.log10(np.mean(Bgerr))]
#             aux2 += list(np.log10(get_range(Sgerr)))
#             aux2 += [np.log10(np.mean(Sgerr))]
#
#         aux1[1], aux1[2] = aux1[2], aux1[1]
#         aux1[-1], aux1[-2] = aux1[-2], aux1[-1]
#         aux2[1], aux2[2] = aux2[2], aux2[1]
#         aux2[-1], aux2[-2] = aux2[-2], aux2[-1]
#
#         if verbose > 0:
#             print("Left and right hand sides comparison:")
#             print("        Bmin   Bave   Bmax   Smin   Save   Smax")
#             mes = "{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}"
#             print("rerr: "+mes.format(*aux1))
#             print("gerr: "+mes.format(*aux2))
#     if plots:
#         args = [tau, Z, Brerr, Srerr, folder, "check_01_eqs_rerr"]
#         kwargs = {"log": True, "ii": i1, "jj": j1}
#         plot_solution(*args, **kwargs)
#
#         args = [tau, Z, Bgerr, Sgerr, folder, "check_02_eqs_gerr"]
#         kwargs = {"log": True, "ii": i2, "jj": j2}
#         plot_solution(*args, **kwargs)
#
#     return aux1, aux2, Brerr, Srerr, Bgerr, Sgerr
#
#
# def check_fdm(params, B, S, tau, Z, P=None,
#               pt=4, pz=4, folder="", name="check", plots=False, verbose=1):
#     r"""Check the equations in an FDM block."""
#     params, Z, tau, tau1, tau2, tau3 = build_mesh_fdm(params)
#     N1 = len(tau1)
#     N2 = len(tau2)
#     # N3 = len(tau3)
#
#     # S1 = S[:N1]
#     S2 = S[N1-1:N1-1+N2]
#     # S3 = S[N1-1+N2-1:N1-1+N2-1+N3]
#
#     # B1 = B[:N1]
#     B2 = B[N1-1:N1-1+N2]
#     # B3 = B[N1-1+N2-1:N1-1+N2-1+N3]
#
#     Brerr = np.zeros(B.shape)
#     Srerr = np.zeros(B.shape)
#     Bgerr = np.zeros(B.shape)
#     Sgerr = np.zeros(B.shape)
#
#     print("the log_10 of relative and global errors (for B and S):")
#     ####################################################################
#     kwargs = {"case": 2, "folder": folder, "plots": False}
#     aux = check_block_fdm(params, B2, S2, tau2, Z, **kwargs)
#     checks2_rerr, checks2_gerr, B2rerr, S2rerr, B2gerr, S2gerr = aux
#
#     Brerr[N1-1:N1-1+N2] = B2rerr
#     Srerr[N1-1:N1-1+N2] = S2rerr
#     Bgerr[N1-1:N1-1+N2] = B2gerr
#     Sgerr[N1-1:N1-1+N2] = S2gerr
#     ####################################################################
#     if plots:
#         plot_solution(tau, Z, Brerr, Srerr, folder, "rerr"+name, log=True)
#         plot_solution(tau, Z, Bgerr, Sgerr, folder, "gerr"+name, log=True)
