# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2020 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Orca related routines."""
import warnings
import numpy as np
from scipy.constants import physical_constants, c, hbar, epsilon_0, mu_0
from sympy import oo

from misc import (time_bandwith_product, vapour_number_density, rayleigh_range,
                  ffftfreq, iffftfft, interpolator, sinc, hermite_gauss,
                  num_integral, build_Z_mesh, build_t_mesh)


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
           "ntauw": 1.0, "N": 101}
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
    # D = L*1.05
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
    Omega = calculate_Omega(params)
    w1 = params["w1"]
    w2 = params["w2"]
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    energy_pulse2 = params["energy_pulse2"]

    Temperature = params["Temperature"]
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
    aux = [t0s*1e9, t0w*1e9, t0r*1e9]
    print("t0s: {:2.3f} ns, t0w: {:2.3f} ns, t0r: {:2.3f} ns".format(*aux))
    print("L: {:2.3f} cm".format(L*100))
    print("Temperature: {:6.2f} °C".format(Temperature-273.15))
    print("n: {:.2e} m^-3 ".format(n))
    print("kappa: {:.2e} sqrt((m s)^-1)".format(kappa))


def calculate_Omega(params):
    r"""Calculate the effective (time averaged) Rabi frequency."""
    energy_pulse2 = params["energy_pulse2"]
    hbar = params["hbar"]
    e_charge = params["e_charge"]
    r2 = params["r2"]
    w2 = params["w2"]
    m = params["nwsquare"]
    sigma_power2 = params["sigma_power2"]

    tbp = time_bandwith_product(m)
    tau_Omega = tbp/sigma_power2

    Omega = energy_pulse2*e_charge**2*r2**2
    Omega = Omega/(np.pi*c*epsilon_0*hbar**2*w2**2*tau_Omega)
    Omega = np.float64(Omega)
    Omega = np.sqrt(Omega)

    return Omega


def calculate_xi0(params):
    r"""Return xi0, the position of the peak for F(xi)."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    kappa = calculate_kappa(params)
    Omega = calculate_Omega(params)

    xi0 = -(c*kappa**2 + 2*delta1**2 + 2*delta1*delta2 - 2*np.abs(Omega)**2)
    xi0 = xi0/(2*np.pi*c*delta1)
    return xi0


def calculate_kappa(params, pumped=True):
    r"""Calculate the kappa parameter."""
    # We calculate the number density assuming Cs 133
    omega_laser1 = params["omega_laser1"]
    element = params["element"]
    isotope = params["isotope"]
    r1 = params["r1"]
    e_charge = params["e_charge"]
    hbar = params["hbar"]
    epsilon_0 = params["epsilon_0"]

    n_atomic0 = vapour_number_density(params)
    if not pumped:
        if element == "Cs":
            fground = [3, 4]
        elif element == "Rb":
            if isotope == 85:
                fground = [2, 3]
            else:
                fground = [1, 2]
        upper_fraction = (2*fground[1]+1)/(2*fground[0]+1.0 + 2*fground[1]+1.0)
        n_atomic0 = upper_fraction*n_atomic0

    return e_charge*np.sqrt(n_atomic0*omega_laser1/(c*hbar*epsilon_0))*r1


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


def calculate_Ctilde(params):
    r"""Calculate the coupling Ctilde."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    tauw = params["tauw"]
    Omega = calculate_Omega(params)
    kappa = calculate_kappa(params)
    Gamma21 = calculate_Gamma21(params)
    Ctilde = tauw*kappa*Omega*np.sqrt(c/2)/np.sqrt(-Gamma21**2)
    return Ctilde


def calculate_beta(params, xi=None):
    r"""Return the beta function."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    if xi is None:
        Z = build_Z_mesh(params)
        xi = ffftfreq(Z)
    Gamma21 = calculate_Gamma21(params)
    Gamma32 = calculate_Gamma32(params)
    Omega = calculate_Omega(params)
    kappa = calculate_kappa(params)

    beta = np.zeros(xi.shape[0], complex)
    beta += Omega*np.conj(Omega)/(2*np.abs(Omega)**2)
    beta += -Gamma21*Gamma32/(2*np.abs(Omega)**2)
    beta += c*kappa**2/(8*np.abs(Omega)**2)
    beta += Gamma21**2*Gamma32**2/(2*c*kappa**2*np.abs(Omega)**2)
    beta += Omega**2*np.conj(Omega)**2/(2*c*kappa**2*np.abs(Omega)**2)
    beta += 1j*np.pi*Gamma21*c*xi/(2*np.abs(Omega)**2)
    beta += -np.pi**2*Gamma21**2*c*xi**2/(2*kappa**2*np.abs(Omega)**2)
    beta += Gamma21*Gamma32*Omega*np.conj(Omega)/(c*kappa**2*np.abs(Omega)**2)
    beta += -1j*np.pi*Gamma21**2*Gamma32*xi/(kappa**2*np.abs(Omega)**2)
    aux = -1j*np.pi*Gamma21*Omega*xi*np.conj(Omega)
    beta += aux/(kappa**2*np.abs(Omega)**2)
    beta = np.sqrt(beta)

    return beta


def calculate_F(params, xi=None):
    r"""Return the beta function."""
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    if xi is None:
        Z = build_Z_mesh(params)
        xi = ffftfreq(Z)

    tauw = params["tauw"]
    delta1 = params["delta1"]
    delta2 = params["delta2"]
    kappa = calculate_kappa(params)
    Omega = calculate_Omega(params)

    z0 = tauw*c/2
    phi0 = c*kappa**2 - 2*delta1**2 - 2*delta1*delta2 + 2*np.abs(Omega)**2
    phi0 = tauw*(phi0)/(2*delta1)

    Ctilde = calculate_Ctilde(params)
    beta = calculate_beta(params, xi)

    Fxi = -Ctilde**2*np.exp(-1j*(phi0 + 2*np.pi*z0*xi))*sinc(Ctilde*beta)**2
    return Fxi


def calculate_optimal_input_xi(params, xi=None):
    r"""Calculate the optimal `xi`-space input for the given parameters.

    Note that this returns a Gaussian pulse of time duration params["taus"]
    """
    params_ = params.copy()
    if not params_["USE_SQUARE_CTRL"] or str(params_["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    if xi is None:
        Z = build_Z_mesh(params_)
        xi = ffftfreq(Z)

    energy_pulse2 = calculate_pulse_energy(params_)
    params_["energy_pulse2"] = energy_pulse2
    xi0 = calculate_xi0(params_)

    taus = params_["taus"]
    tauw = params_["tauw"]
    DeltanuS_num = time_bandwith_product(1)/taus
    DeltaxiS_num = DeltanuS_num*2/c
    sigma_xi = DeltaxiS_num/(2*np.sqrt(np.log(2)))

    # We make sure that the oscillations in the signal are not too fast.
    T0 = np.abs(1/c/xi0)
    if taus/T0 > 5.0 or tauw/T0 > 5.0:
        mes = "The optimal signal has a linear phase that is too fast "
        mes += "for the grid to represent accurately. "
        mes += "Using a flat phase instead."
        warnings.warn(mes)
        warnings.filterwarnings('ignore', mes)
        xi0 = 0.0

    Zoff = params_["tauw"]/2*(c/2)
    Sin = hermite_gauss(0, xi-xi0, sigma_xi)
    Sin = Sin*np.exp(2*np.pi*1j*Zoff*xi)

    # We normalize so that the integral of the signal mod square over tau
    # is 1.
    Sin = Sin*np.sqrt(c/2)
    return xi, Sin


def calculate_optimal_input_Z(params, Z=None):
    r"""Calculate the optimal `Z`-space input for the given parameters.

    Note this returns a Gaussian pulse of time duration params["taus"]
    """
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    if Z is None:
        band = True
        Zp = build_Z_mesh(params)
    else:
        band = False

    # We get a reasonable xi and Z mesh.
    xi0 = calculate_xi0(params)
    Deltaxi = 2/c/params["tauw"]

    a1 = xi0+20*Deltaxi/2
    a2 = xi0-20*Deltaxi/2
    aa = np.amax(np.abs(np.array([a1, a2])))
    xi = np.linspace(-aa, aa, 1001)
    xi, S0xi = calculate_optimal_input_xi(params, xi)

    # We Fourier transform it.
    Z = ffftfreq(xi)
    S0Z = iffftfft(S0xi, xi)

    taus = params["taus"]
    tauw = params["tauw"]
    T0 = np.abs(1/c/xi0)
    if taus/T0 > 5.0 or tauw/T0 > 5.0:
        mes = "The optimal signal has a linear phase that is too fast "
        mes += "for the grid to represent accurately. "
        mes += "Using a flat phase instead."
        warnings.warn(mes)
        warnings.filterwarnings('ignore', mes)
        xi0 = 0.0

        Z = np.linspace(-0.25, 0.25, 1001)

        DeltanuS_num = time_bandwith_product(1)/taus
        DeltaxiS_num = DeltanuS_num*2/c
        sigma_xi = DeltaxiS_num/(2*np.sqrt(np.log(2)))
        Zoff = tauw/2*(c/2)

        Sin = hermite_gauss(0, xi-xi0, sigma_xi)
        Sin = Sin*np.exp(2*np.pi*1j*Zoff*xi)*np.sqrt(c/2)

        # S0Z = hermite_gauss(0, Z+Zoff, 1/np.pi**2*np.sqrt(2)/sigma_xi)
        S0Z = hermite_gauss(0, Z+Zoff, 1/2.0/np.pi/sigma_xi)
        S0Z = S0Z*np.sqrt(c/2)
        # S0Z = S0Z*np.exp(2*np.pi*1j*Zoff*xi)*np.sqrt(c/2)

    if not band:
        S0Z_interp = interpolator(Z, S0Z, kind="cubic")
        S0Z = S0Z_interp(Zp)
        return Zp, S0Z
    else:
        return Z, S0Z


def calculate_optimal_input_tau(params, tau=None):
    r"""Calculate the optimal `tau`-space input for the given parameters.

    Note this returns a Gaussian pulse of time duration params["taus"]
    """
    if not params["USE_SQUARE_CTRL"] or str(params["nwsquare"]) != "oo":
        mes = 'USE_SQUARE_CTRL must be True, and "nwsquare" must be "oo".'
        raise ValueError(mes)
    if tau is None:
        tau = build_t_mesh(params)

    kappa = calculate_kappa(params)
    Gamma21 = calculate_Gamma21(params)

    Z, S0Z = calculate_optimal_input_Z(params)
    S0Z_interp = interpolator(Z, S0Z, kind="cubic")

    D = params["L"]*1.05
    tau0 = params["t0w"] - params["tauw"]/2

    S0tau = S0Z_interp(-D/2 - c*(tau-tau0)/2)
    S0tau = S0tau*np.exp(-c*kappa**2*(tau-tau0)/(2*Gamma21))
    S0tau = S0tau/np.sqrt(num_integral(np.abs(S0tau)**2, tau))

    return tau, S0tau


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

    aux = np.pi**3*delta1**2*hbar**2*w2**2
    aux = aux/(2*tauw*r2**2*c**2*e_charge**2*kappa**2*mu_0)

    return aux

#############################################################################
# Finite difference ORCA routines.
