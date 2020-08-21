# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2020 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Python module initializer."""
import warnings
from time import time
import numpy as np
from numpy import sinc as normalized_sinc
from scipy.constants import physical_constants, c, hbar, epsilon_0, mu_0
from scipy.constants import k as k_B
from scipy.sparse import linalg, csr_matrix, spmatrix, bmat, spdiags
from scipy.sparse import kron as sp_kron
from scipy.sparse import eye as sp_eye
from scipy.interpolate import interp1d
from scipy.special import hermite
from scipy.misc import factorial
from sympy import log, pi, oo, zeros, Matrix
from sympy import factorial as sym_fact
from math import factorial as num_fact
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
##############################################################################
# Basic routines.


def rel_error(a, b):
    r"""Get the relative error between two quantities."""
    scalar = not hasattr(a, "__getitem__")
    if scalar:
        a = np.abs(a)
        b = np.abs(b)
        if a == 0.0 and b == 0.0:
            return 0.0
        if a > b:
            return 1 - b/a
        else:
            return 1 - a/b

    shape = [2] + list(a.shape)
    aux = np.zeros(shape)
    aux[0, :] = np.abs(a)
    aux[1, :] = np.abs(b)

    small = np.amin(aux, axis=0)
    large = np.amax(aux, axis=0)

    # Replace zeros with one, to avoid zero-division errors.
    small[small == 0] = 1
    large[large == 0] = 1
    err = 1-small/large
    if scalar:
        return err[0]
    return err


def glo_error(a, b):
    r"""Get the "global" relative error between two quantities."""
    scale = np.amax([np.amax(np.abs(a)), np.amax(np.abs(b))])
    if scale == 0.0:
        return np.zeros(a.shape)
    return np.abs(a-b)/scale


def interpolator(xp, fp, kind="linear"):
    r"""Return an interpolating function that extrapolates to zero."""
    F = interp1d(xp, fp, kind)

    def f(x):
        if isinstance(x, np.ndarray):
            return np.array([f(xi) for xi in x])
        if xp[0] <= x <= xp[-1]:
            return F(x)
        else:
            return 0.0

    return f


def ffftfreq(t):
    r"""Calculate the angular frequency axis for a given time axis."""
    dt = t[1]-t[0]
    nu = np.fft.fftshift(np.fft.fftfreq(t.size, dt))
    return nu


def ffftfft(f, t):
    r"""Calculate the Fourier transform."""
    dt = t[1]-t[0]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(f)))*dt


def iffftfft(f, nu):
    r"""Calculate the inverse Fourier transform."""
    Deltanu = nu[-1]-nu[0]
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f)))*Deltanu


##############################################################################
# Mode shape routines.


def time_bandwith_product(m=1, symbolic=False):
    r"""Return an approximate of the time-bandwidth product for a generalized
    pulse.
    """
    if symbolic and m == 1:
        return 2*log(2)/pi

    if m == 1:
        return 2*np.log(2)/np.pi
    elif str(m) == "oo":
        return 0.885892941378901
    else:
        a, b, B, p = (0.84611760622587673, 0.44076249541699231,
                      0.87501561821518636, 0.64292796298081856)

        return (B-b)*(1-np.exp(-a*(m-1)**p))+b


def hermite_gauss(n, x, sigma):
    """Generate normalized Hermite-Gauss mode."""
    X = x / sigma
    result = hermite(n)(X) * np.exp(-X**2 / 2)
    result /= np.sqrt(factorial(n) * np.sqrt(np.pi) * 2**n * sigma)
    return result


def harmonic(n, x, L):
    r"""Generate a normalized harmonic mode."""
    omega = np.pi/L
    h = np.sin(n*omega*(x + L/2))/np.sqrt(L/2)
    h = h*np.where(np.abs(x) < L/2, 1.0, 0.0)
    return h


def sinc(x):
    u"""The non-normalized sinc.

              ⎧   1        for x = 0
    sinc(x) = ⎨
              ⎩ sin(x)/x   otherwise

    """
    return normalized_sinc(x/np.pi)


def num_integral(f, dt):
    """We integrate using the trapezium rule."""
    if hasattr(dt, "__getitem__"):
        dt = dt[1]-dt[0]
    F = sum(f[1:-1])
    F += (f[1] + f[-1])*0.5
    return np.real(F*dt)


##############################################################################
# Memory routines.


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


def build_t_mesh(params, uniform=True, return_bounds=False):
    r"""Build a variable density mesh for the time axis.

    We ensure that within three fwhm there are a tenth of the
    points, or at least 200.
    """
    Nfwhms = 2.0
    Nleast = 100
    fra = 0.01
    Nt = params["Nt"]
    T = params["T"]
    t0w = params["t0w"]
    t0r = params["t0r"]
    tauw = params["tauw"]
    taur = params["taur"]

    if uniform:
        return np.linspace(-T/2, T/2, Nt)

    # We determine how many points go in each control field region.
    Nw = int(fra*Nt); Nr = int(fra*Nt)
    if Nw < Nleast: Nw = Nleast
    if Nr < Nleast: Nr = Nleast

    # The density outside these regions should be uniform.
    Trem = T - Nfwhms*tauw - Nfwhms*taur
    Nrem = Nt - Nw - Nr

    t01 = 0.0; tf1 = t0w-Nfwhms*tauw/2
    t02 = t0w+Nfwhms*tauw/2; tf2 = t0r-Nfwhms*taur/2
    t03 = t0r+Nfwhms*taur/2; tf3 = T

    T1 = tf1 - t01
    T2 = tf2 - t02
    T3 = tf3 - t03

    N1 = int(Nrem*T1/Trem)
    N2 = int(Nrem*T2/Trem)
    N3 = int(Nrem*T3/Trem)
    # We must make sure that these numbers all add up to Nt
    Nt_wrong = N1-1 + Nw + N2-2 + Nr + N3-1
    # print N1, Nw, N2, Nr, N3
    # print Nt, Nt_wrong
    # We correct this error:
    N2 = N2 + Nt - Nt_wrong
    Nt_wrong = N1-1 + Nw + N2-2 + Nr + N3-1

    tw = np.linspace(t0w - Nfwhms*tauw/2, t0w + Nfwhms*tauw/2, Nw)
    tr = np.linspace(t0r - Nfwhms*taur/2, t0r + Nfwhms*taur/2, Nr)
    t1 = np.linspace(t01, tf1, N1)
    t2 = np.linspace(t02, tf2, N2)
    t3 = np.linspace(t03, tf3, N3)

    t = np.zeros(Nt)
    t[:N1-1] = t1[:-1]

    a1 = 0; b1 = N1-1
    aw = b1; bw = aw+Nw
    a2 = bw; b2 = a2+N2-2
    ar = b2; br = ar+Nr
    a3 = br; b3 = a3+N3-1

    t[a1:b1] = t1[:-1]
    t[aw:bw] = tw
    t[a2:b2] = t2[1:-1]
    t[ar:br] = tr
    t[a3:b3] = t3[1:]

    # print t[a1:b1].shape, t[aw:bw].shape, t[a2:b2].shape, t[ar:br].shape,
    # print t[a3:b3].shape

    if return_bounds:
        return (a1, aw, a2, ar, a3)
    return t


def build_Z_mesh(params, uniform=True, on_cell_edge=False):
    r"""Return a Z mesh for a given cell length and number of points."""
    L = params["L"]
    Nz = params["Nz"]
    if on_cell_edge:
        D = L*1.0
    else:
        D = L*1.05
    Z = np.linspace(-D/2, D/2, Nz)
    return Z


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


def vapour_pressure(params):
    r"""Return the vapour pressure of rubidium or cesium in Pascals.

    This function receives as input the temperature in Kelvins and the
    name of the element.

    >>> print vapour_pressure(25.0 + 273.15,"Rb")
    5.31769896107e-05
    >>> print vapour_pressure(39.3 + 273.15,"Rb")
    0.000244249795696
    >>> print vapour_pressure(90.0 + 273.15,"Rb")
    0.0155963687128
    >>> print vapour_pressure(25.0 + 273.15,"Cs")
    0.000201461144963
    >>> print vapour_pressure(28.5 + 273.15,"Cs")
    0.000297898928349
    >>> print vapour_pressure(90.0 + 273.15,"Cs")
    0.0421014384667

    The element must be in the database.

    >>> print vapour_pressure(90.0 + 273.15,"Ca")
    Traceback (most recent call last):
    ...
    ValueError: Ca is not an element in the database for this function.

    References:
    [1] Daniel A. Steck, "Cesium D Line Data," available online at
        http://steck.us/alkalidata (revision 2.1.4, 23 December 2010).
    [2] Daniel A. Steck, "Rubidium 85 D Line Data," available online at
        http://steck.us/alkalidata (revision 2.1.5, 19 September 2012).
    [3] Daniel A. Steck, "Rubidium 87 D Line Data," available online at
        http://steck.us/alkalidata (revision 2.1.5, 19 September 2012).

    """
    Temperature = params["Temperature"]
    element = params["element"]
    if element == "Rb":
        Tmelt = 39.30+273.15  # K.
        if Temperature < Tmelt:
            P = 10**(2.881+4.857-4215.0/Temperature)  # Torr.
        else:
            P = 10**(2.881+4.312-4040.0/Temperature)  # Torr.
    elif element == "Cs":
        Tmelt = 28.5 + 273.15  # K.
        if Temperature < Tmelt:
            P = 10**(2.881+4.711-3999.0/Temperature)  # Torr.
        else:
            P = 10**(2.881+4.165-3830.0/Temperature)  # Torr.
    else:
        s = str(element)
        s += " is not an element in the database for this function."
        raise ValueError(s)

    P = P * 101325.0/760.0  # Pascals.
    return P


def vapour_number_density(params):
    r"""Return the number of atoms in a rubidium or cesium vapour in m^-3.

    It receives as input the temperature in Kelvins and the
    name of the element.

    >>> print vapour_number_density(90.0 + 273.15,"Cs")
    8.39706962725e+18

    """
    Temperature = params["Temperature"]
    return vapour_pressure(params)/k_B/Temperature


def rayleigh_range(params):
    r"""Return the Rayleigh range for signal and control."""
    ws = params["w1"]
    wc = params["w1"]
    lams = c/(params["omega21"]/2/np.pi)
    lamc = c/(params["omega32"]/2/np.pi)

    return np.pi*ws**2/lams, np.pi*wc**2/lamc


##############################################################################
# Finite difference methods.


def build_mesh_fdm(params, verbose=0):
    r"""Build mesh for the FDM in the control field region.

    We choose a mesh such that the region where the cell overlaps with the
    control field has approximately `N**2` points, the time and space
    steps satisfy approximately dz/dtau = c/2, and length in time of the mesh
    is duration `params["T"]*ntau`.

    """
    tauw = params["tauw"]
    ntauw = params["ntauw"]
    N = params["N"]
    Z = build_Z_mesh(params)
    D = Z[-1] - Z[0]
    # We calculate NtOmega and Nz such that we have approximately
    # NtOmega
    NtOmega = int(round(N*np.sqrt(c*ntauw*tauw/2/D)))
    Nz = int(round(N*np.sqrt(2*D/c/ntauw/tauw)))

    dt = ntauw*tauw/(NtOmega-1)
    # We calculate a t0w that is approximately at 3/2*ntauw*tauw + 2*D/c
    Nt1 = int(round((ntauw*tauw + 2*D/c)/dt))+1

    t01 = 0.0; tf1 = t01 + (Nt1-1)*dt
    t02 = tf1; tf2 = t02 + (NtOmega-1)*dt
    t03 = tf2; tf3 = t03 + (Nt1-1)*dt
    t0w = (tf2 + t02)/2

    t01 -= t0w; tf1 -= t0w
    t02 -= t0w; tf2 -= t0w
    t03 -= t0w; tf3 -= t0w
    t0w = 0.0

    tau1 = np.linspace(t01, tf1, Nt1)
    tau2 = np.linspace(t02, tf2, NtOmega)
    tau3 = np.linspace(t03, tf3, Nt1)

    Nt = 2*Nt1 + NtOmega - 2
    T = tf3-t01
    tau = np.linspace(t01, tf3, Nt)
    Z = build_Z_mesh(params)

    params_new = params.copy()
    params_new["Nt"] = Nt
    params_new["Nz"] = Nz
    params_new["T"] = T
    params_new["t0w"] = t0w
    params_new["t0s"] = t0w
    Z = build_Z_mesh(params_new)

    if verbose > 0:
        Nt1 = tau1.shape[0]
        Nt2 = tau2.shape[0]
        Nt3 = tau3.shape[0]
        T1 = tau1[-1] - tau1[0]
        T2 = tau2[-1] - tau2[0]
        T3 = tau3[-1] - tau3[0]
        T1_tar = ntauw*tauw + 2*D/c
        # dt1 = tau1[1] - tau1[0]
        # dt2 = tau2[1] - tau2[0]
        # dt3 = tau3[1] - tau3[0]

        aux1 = Nt1+Nt2+Nt3-2
        total_size = aux1*Nz
        aux2 = [Nt1, Nt2, Nt3, Nz, aux1, Nz, total_size]
        mes = "Grid size: ({} + {} + {}) x {} = {} x {} = {} points"
        print(mes.format(*aux2))

        dz = Z[1]-Z[0]
        dt = tau[1]-tau[0]

        ratio1 = float(NtOmega)/float(Nz)
        ratio2 = float(Nz)/float(NtOmega)

        mes = "The control field region has {} x {} = {} =? {} points"
        print(mes.format(Nt2, Nz, Nt2*Nz, N**2))
        mes = "The W matrix would be (5 x 2 x Nz)^2 = {} x {} = {} points"
        NW = 2*5*Nz
        print(mes.format(NW, NW, NW**2))
        mes = "The ratio of steps is (dz/dt)/(c/2) = {:.3f}"
        print(mes.format(dz/dt/(c/2)))
        aux = [T1/tauw, T2/tauw, T3/tauw]
        mes = "T1/tauw, T3/tauw, T3/tauw : {:.3f}, {:.3f}, {:.3f}"
        print(mes.format(*aux))
        mes = "T1/T1_tar, T3/T1_tar: {:.3f}, {:.3f}"
        print(mes.format(T1/T1_tar, T3/T1_tar))

        # aux = [dt1*1e9, dt2*1e9, dt3*1e9]
        # print("dt1, dt2, dt3 : {} {} {}".format(*aux))

        if total_size > 1.3e6:
            mes = "The mesh size is larger than 1.3 million, the computer"
            mes += " might crash!"
            warnings.warn(mes)
        if ratio1 > 15:
            mes = "There are too many t-points in the control region: {}"
            mes = mes.format(NtOmega)
            warnings.warn(mes)
        if ratio2 > 15:
            mes = "There are too many Z-points in the control region, "
            mes = "the computer might crash!"
            warnings.warn(mes)
        if Nz > 500:
            mes = "There are too many Z-points! I'll prevent this from "
            mes += "crashing."
            raise ValueError(mes)
    return params_new, Z, tau, tau1, tau2, tau3


def set_row(W, ii):
    r"""Set a given row to zero and its diagonal element to -1."""
    indr = np.where(W.row == ii)[0]
    indc = np.where(W.col == ii)[0]

    for ind in indr:
        W.data[ind] = 0.0

    indrc = list(set(indr).intersection(set(indc)))
    if len(indrc) == 0:
        W.data = np.append(W.data, -1.0)
        W.row = np.append(W.row, ii)
        W.col = np.append(W.col, ii)
    elif len(indrc) == 1:
        W.data[indrc[0]] = -1.0
    else:
        raise ValueError(str(indrc))

    return W


def set_col(W, ii):
    r"""Set a given column to zero and its diagonal element to 1."""
    indr = np.where(W.row == ii)[0]
    indc = np.where(W.col == ii)[0]

    for ind in indc:
        W.data[ind] = 0.0

    indrc = list(set(indr).intersection(set(indc)))
    if len(indrc) == 0:
        W.data = np.append(W.data, 1.0)
        W.row = np.append(W.row, ii)
        W.col = np.append(W.col, ii)
    elif len(indrc) == 1:
        W.data[indrc[0]] = 1.0
    else:
        raise ValueError(str(indrc))

    return W


def transform_system(Wp, xb, tau, boundary_indices=[0],
                     symbolic=False, sparse=False, verbose=0):
    r"""Transform a system of equations of the form $W' x = 0$ into $W x = b$
    by imposing the values of $x_i$ for $i\in$ `boundary indices`.
    """
    if verbose > 0: t00 = time()
    if hasattr(Wp, "__call__"):
        # We have time-dependent equations.
        W = Wp(tau)
    else:
        # We have time-independent equations.
        W = Wp.copy()
    N = W.shape[0]
    if symbolic:
        zero_row = zeros(1, N)
        zero_col = zeros(N, 1)
    else:
        zero_row = 0.0
        zero_col = 0.0
    if verbose > 0: print("555 time: {}".format(time()-t00))

    if verbose > 0: t00 = time()
    if sparse:
        W = W.tocoo()
    if verbose > 0: print("666 time: {}".format(time()-t00))

    if verbose > 0: t00 = time()
    for i in boundary_indices:
        if sparse:
            W = set_row(W, i)
        else:
            W[i, :] = zero_row
            W[i, i] = -1
    if verbose > 0: print("777 time: {}".format(time()-t00))
    # raise ValueError

    if verbose > 0: t00 = time()
    if symbolic:
        b = -W*xb
    elif sparse:
        b = -W.dot(xb)
    else:
        b = -np.dot(W, xb)
    if verbose > 0: print("888 time: {}".format(time()-t00))

    if verbose > 0: t00 = time()
    for i in boundary_indices:
        if sparse:
            W = set_col(W, i)
        else:
            W[:, i] = zero_col
            W[i, i] = 1
    if verbose > 0: print("999 time: {}".format(time()-t00))

    if verbose > 0: t00 = time()
    if sparse:
        W = W.tocsr()
    if verbose > 0: print("111 time: {}".format(time()-t00))

    return W, b


def impose_boundary(params, Wp, tau, S0t, S0z, B0z, P0z=None, sparse=False):
    r"""Impose boudary conditions."""
    # We unpack parameters.
    if P0z is not None:
        nv = 3
    else:
        nv = 2
    # We unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]
        nX = nv*Nt*Nz
    # We build Xb.
    if True:
        Xb_ = np.zeros((nv, Nt, Nz), complex)
        # An auxiliary array to find the boundary condition indices.
        aux = np.zeros((nv, Nt, Nz), int)
        zero_bound = np.zeros(Nt)

        adiabatic = P0z is None
        if not adiabatic:
            ex = 1
            # We set up P(tau=0, Z)
            Xb_[0, 0, :] = P0z
            aux[0, 0, :] = 1
            # We set up P(tau, Z=-D/2)
            Xb_[0, :, 0] = zero_bound
            aux[0, :, 0] = 1
            # We set up P(tau, Z=+D/2)
            Xb_[0, :, -1] = zero_bound
            aux[0, :, -1] = 1
        else:
            ex = 0

        # We set up B(tau=0, Z)
        Xb_[0+ex, 0, :] = B0z
        aux[0+ex, 0, :] = 1
        # We set up S(tau=0, Z)
        Xb_[1+ex, 0, :] = S0z
        aux[1+ex, 0, :] = 2

        # We set up B(tau, Z=-D/2)
        Xb_[0+ex, :, 0] = zero_bound
        aux[0+ex, :, 0] = 3
        # We set up B(tau, Z=+D/2)
        Xb_[0+ex, :, -1] = zero_bound
        aux[0+ex, :, -1] = 4
        # We set up S(tau, Z=-D/2)
        Xb_[1+ex, :, 0] = S0t
        aux[1+ex, :, 0] = 5

        # We flatten Xb_.
        Xb = np.reshape(Xb_, nX)
        # We find the boundary_indices.
        aux = np.reshape(aux, nX)
        boundary_indices = [ind for ind, auxi in enumerate(aux) if auxi != 0]
    # We transform the system.
    if True:
        W, b = transform_system(Wp, Xb, tau, sparse=sparse,
                                boundary_indices=boundary_indices)
        return W, b, Xb_


def solve_fdm_subblock(params, Wp, S0t, S0z, B0z, tau, Z,
                       P0z=None, return_block=False, folder="",
                       plots=False):
    r"""We solve using the finite difference method on a block for given
    boundary conditions, and with time and space precisions `pt` and `pz`.
    """
    if P0z is not None:
        nv = 3
    else:
        nv = 2
    # We unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]
        nX = nv*Nt*Nz
    # We transform the system.
    if True:
        W, b, Xb_ = impose_boundary(params, Wp, tau, S0t, S0z, B0z)
        Ws = csr_matrix(W)
        bs = csr_matrix(np.reshape(b, (nX, 1)))
    # We solve the transformed system.
    if True:
        Xsol = linalg.spsolve(Ws, bs)
        Xsol_ = np.reshape(Xsol, (nv, Nt, Nz))
    if plots:
        ################################################################
        # Plotting W and B.
        plt.figure(figsize=(15, 7))
        plt.subplot(1, 2, 1)
        plt.title("$W$")
        plt.imshow(np.log(np.abs(W)))

        plt.subplot(1, 2, 2)
        plt.title("$B$")
        plt.plot(np.abs(b))
        plt.savefig(folder+"W-b.png", bbox_inches="tight")
        plt.close("all")

        ################################################################
        # Plotting B and S.
        plt.figure(figsize=(19, 9))

        plt.subplot(4, 1, 1)
        plt.title("$B$ boundary")
        plt.imshow(np.abs(Xb_[0, :, :]))

        plt.subplot(4, 1, 2)
        plt.title("$B$ solution")
        plt.imshow(np.abs(Xsol_[0, :, :]))

        plt.subplot(4, 1, 3)
        plt.title("$S$ boundary")
        plt.imshow(np.abs(Xb_[1, :, :]))

        plt.subplot(4, 1, 4)
        plt.title("$S$ solution")
        plt.imshow(np.abs(Xsol_[1, :, :]))
        plt.savefig(folder+"BS.png", bbox_inches="tight")
        plt.close("all")
    # We unpack the solution and return it.
    if return_block:
        Bsol = Xsol_[0]
        Ssol = Xsol_[1]
    else:
        if P0z is not None:
            Psol = Xsol_[0, -1, :]
            Bsol = Xsol_[1, -1, :]
            Ssol = Xsol_[2, -1, :]
            return Psol, Bsol, Ssol
        else:
            Bsol = Xsol_[0, -1, :]
            Ssol = Xsol_[1, -1, :]
            return Bsol, Ssol


def solve_fdm_block(params, S0t, S0z, B0z, tau, Z, P0z=None, Omegat="square",
                    case=0, method=0, pt=4, pz=4,
                    plots=False, folder="", name="", verbose=0):
    r"""We solve using the finite difference method for given
    boundary conditions, and with time and space precisions `pt` and `pz`.

    INPUT:

    -  ``params`` - dict, the problem's parameters.

    -  ``S0t`` - array, the S(Z=-L/2, t) boundary condition.

    -  ``S0z`` - array, the S(Z, t=0) boundary condition.

    -  ``B0z`` - array, the B(Z, t=0) boundary condition.

    -  ``tau`` - array, the time axis.

    -  ``Z`` - array, the space axis.

    -  ``P0z`` - array, the P(Z, t=0) boundary condition (default None).

    -  ``Omegat`` - function, a function that returns the temporal mode of the
                    Rabi frequency at time tau (default "square").

    -  ``case`` - int, the dynamics to solve for: 0 for free space, 1 for
                  propagation through vapour, 2 for propagation through vapour
                  and non-zero control field.

    -  ``method`` - int, the fdm method to use: 0 to solve the full space, 1
                  to solve by time step slices.

    -  ``pt`` - int, the precision order for the numerical time derivate. Must
                be even.

    -  ``pz`` - int, the precision order for the numerical space derivate. Must
                be even.

    -  ``plots`` - bool, whether to make plots.

    -  ``folder`` - str, the directory to save plots in.

    -  ``verbose`` - int, a vague measure much of messages to print.

    OUTPUT:

    A solution to the equations for the given case and boundary conditions.

    """
    t00_tot = time()
    # We unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]
        Nt_prop = pt + 1

        # The number of functions.
        nv = 2

    # We make pre-calculations.
    if True:
        if P0z is not None:
            P = np.zeros((Nt, Nz), complex)
            P[0, :] = P0z
        B = np.zeros((Nt, Nz), complex)
        S = np.zeros((Nt, Nz), complex)
        B[0, :] = B0z
        S[0, :] = S0z
    if method == 0:
        # We solve the full block.
        sparse = True
        aux1 = [params, tau, Z]
        aux2 = {"Omegat": Omegat, "pt": pt, "pz": pz, "case": case,
                "folder": folder, "plots": False, "sparse": sparse,
                "adiabatic": P0z is None}
        # print(Omegat)
        t00 = time()
        Wp = eqs_fdm(*aux1, **aux2)
        if verbose > 0: print("FDM Eqs time  : {:.3f} s".format(time()-t00))
        # print(type(Wp))
        args = [params, Wp, tau, S0t, S0z, B0z]
        t00 = time()
        W, b, Xb_ = impose_boundary(*args, sparse=sparse)
        if verbose > 0: print("FDM Bdr time  : {:.3f} s".format(time()-t00))

        t00 = time()
        if sparse:
            X = linalg.spsolve(W, b)
            mes = "FDM Sol time  : {:.3f} s"
            if verbose > 0: print(mes.format(time()-t00))
        else:
            X = np.linalg.solve(W, b)
            mes = "FDM Sol time  : {:.3f} s"
            if verbose > 0: print(mes.format(time()-t00))
        B, S = np.reshape(X, (nv, Nt, Nz))

    elif method == 1:
        # We solve by time step slices.
        ##################################################################
        S0t_interp = interpolator(tau, S0t, kind="cubic")
        params_prop = params.copy()
        params_prop["Nt"] = Nt_prop
        dtau = tau[1] - tau[0]
        params_prop["T"] = dtau
        tau_slice = build_t_mesh(params_prop, uniform=True)

        t00_eqs = time()
        ##################################################################
        # The equations are here!
        aux1 = [params_prop, tau_slice, Z]
        aux2 = {"Omegat": Omegat, "pt": pt, "pz": pz, "case": case,
                "folder": folder, "plots": False,
                "adiabatic": P0z is None}
        Wp = eqs_fdm(*aux1, **aux2)

        # We solve the system.
        ##################################################################
        if verbose > 0:
            runtime_eqs = time()-t00_eqs
            print("Eqs time: {} s.".format(runtime_eqs))
        ##################################################################
        # The for loop for propagation is this.
        tt = 0
        for tt in range(Nt-1):
            if verbose > 1: print("t_{} = {}".format(tt, tau[tt]))
            # We specify the block to solve.
            tau_prop = tau[tt] + tau_slice
            B0z_prop = B[tt, :]
            S0z_prop = S[tt, :]
            S0t_prop = S0t_interp(tau_prop)

            aux1 = [params_prop, Wp, S0t_prop, S0z_prop, B0z_prop,
                    tau_prop, Z]
            aux2 = {"folder": folder, "plots": False, "P0z": P0z}
            if P0z is None:
                Bslice, Sslice = solve_fdm_subblock(*aux1, **aux2)
                B[tt+1, :] = Bslice
                S[tt+1, :] = Sslice
            else:
                Pslice, Bslice, Sslice = solve_fdm_subblock(*aux1, **aux2)
                P[tt+1, :] = Pslice
                B[tt+1, :] = Bslice
                S[tt+1, :] = Sslice

    # Report running time.
    if verbose > 0:
        runtime_tot = time() - t00_tot
        aux = [runtime_tot, Nt, Nz, Nt*Nz]
        mes = "FDM block time: {:.3f} s for a grid of {} x {} = {} points."
        print(mes.format(*aux))
    # Plotting.
    if plots:
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        plt.title("$B$ numeric")
        cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(B))
        plt.colorbar(cs)
        plt.ylabel(r"$\tau$ (ns)")
        plt.xlabel("$Z$ (cm)")

        plt.subplot(1, 2, 2)
        plt.title("$S$ numeric")
        cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(S))
        plt.colorbar(cs)
        plt.ylabel(r"$\tau$ (ns)")
        plt.xlabel("$Z$ (cm)")
        aux = folder+"solution_numeric"+name+".png"
        plt.savefig(aux, bbox_inches="tight")
        plt.close("all")

    if P0z is not None:
        return P, B, S
    else:
        return B, S


def solve_fdm(params, S0t=None, S0z=None, B0z=None, P0z=None, Omegat="square",
              method=0, pt=4, pz=4,
              folder="", name="", plots=False, verbose=0,
              seed=None, analytic_storage=True, return_modes=False):
    r"""We solve using the finite difference method for given
    boundary conditions, and with time and space precisions `pt` and `pz`.
    """
    adiabatic = P0z is None
    t00f = time()
    # We unpack parameters.
    if True:
        aux = build_mesh_fdm(params)
        params, Z, tau, tau1, tau2, tau3 = aux
        Nt = params["Nt"]
        Nz = params["Nz"]
        kappa = calculate_kappa(params)
        Gamma21 = calculate_Gamma21(params)
        Gamma32 = calculate_Gamma32(params)
        Omega = calculate_Omega(params)
        taus = params["taus"]
        t0s = params["t0s"]
        D = Z[-1] - Z[0]

        Nt1 = tau1.shape[0]
        Nt2 = tau2.shape[0]
        Nt3 = tau3.shape[0]

        # We initialize the solution.
        if not adiabatic:
            P = np.zeros((Nt, Nz), complex)
        B = np.zeros((Nt, Nz), complex)
        S = np.zeros((Nt, Nz), complex)

    # We solve in the initial region.
    if True:
        if verbose > 0: t000f = time()
        B_exact1 = np.zeros((Nt1, Nz))
        S_exact1 = np.zeros((Nt1, Nz))
        ttau1 = np.outer(tau1, np.ones(Nz))
        ZZ1 = np.outer(np.ones(Nt1), Z)

        nshg = params["nshg"]
        if seed == "S":
            sigs = taus/(2*np.sqrt(2*np.log(2)))*np.sqrt(2)
            S_exact1 = hermite_gauss(nshg, -t0s + ttau1 - 2*ZZ1/c, sigs)
            S_exact1 = S_exact1*np.exp(-(ZZ1+D/2)*kappa**2/Gamma21)
            S[:Nt1, :] = S_exact1
        elif seed == "B":
            nshg = nshg + 1
            B_exact1 = harmonic(nshg, ZZ1, D)
            B[:Nt1, :] = B_exact1
        elif S0t is not None or B0z is not None:
            if S0t is not None:
                S0t_interp = interpolator(tau, S0t, kind="cubic")
                S_exact1 = S0t_interp(ttau1 - 2*(ZZ1+D/2)/c)
                S_exact1 = S_exact1*np.exp(-(ZZ1+D/2)*kappa**2/Gamma21)
                S[:Nt1, :] = S_exact1
            if B0z is not None:
                B_exact1 = np.outer(np.ones(Nt1), B0z)
                B[:Nt1, :] = B_exact1
        else:
            mes = "Either of S0t, B0z, or seed must be given as arguments"
            raise ValueError(mes)
        if verbose > 0: print("region 1 time : {:.3f} s".format(time()-t000f))
    # We obtain the input modes for the memory.
    if True:
        if S0t is None and B0z is None:
            if seed == "P":
                # We seed with a harmonic mode.
                raise NotImplementedError
            elif seed == "B":
                # We seed with a harmonic mode.
                B02z = harmonic(nshg, Z, D)
                S02z = np.zeros(Nz, complex)
                S02t = np.zeros(Nt2, complex)
            elif seed == "S":
                # We seed with a Hermite-Gauss mode.
                # HG modes propagate as:
                #  S(tau, Z) = HG_n(t-t0s - 2*Z/c, sigma)
                #            x exp(-(Z+D/2)*kappa**2/Gamma21)
                #
                # We calculate the gaussian standard deviation.
                B02z = np.zeros(Nz, complex)
                S02z = hermite_gauss(nshg, tau2[0] - t0s - 2*Z/c, sigs)
                S02z = S02z*np.exp(-(Z+D/2)*kappa**2/Gamma21)
                S02t = hermite_gauss(nshg, tau2 - t0s + D/c, sigs)
            else:
                raise ValueError
        else:
            if S0t is not None:
                B02z = B_exact1[Nt1-1, :]
                S02z = S_exact1[Nt1-1, :]
                S02t = S0t[Nt1-1:Nt1+Nt2-1]
            if B0z is not None:
                B02z = B0z
                S02z = np.zeros(Nz, complex)
                S02t = np.zeros(Nt2, complex)

    # We solve in the memory region using the FDM.
    if True:
        if verbose > 0: t000f = time()
        params_memory = params.copy()
        params_memory["Nt"] = Nt2
        aux1 = [params_memory, S02t, S02z, B02z, tau2, Z]
        aux2 = {"Omegat": Omegat, "method": method, "pt": pt, "pz": pz,
                "folder": folder, "plots": False,
                "verbose": verbose-1, "P0z": P0z, "case": 2}
        if adiabatic:
            B2, S2 = solve_fdm_block(*aux1, **aux2)
            B[Nt1-1:Nt1+Nt2-1] = B2
            S[Nt1-1:Nt1+Nt2-1] = S2
        else:
            P2, B2, S2 = solve_fdm_block(*aux1, **aux2)
            P[Nt1-1:Nt1+Nt2-1] = P2
            B[Nt1-1:Nt1+Nt2-1] = B2
            S[Nt1-1:Nt1+Nt2-1] = S2
        if verbose > 0: print("region 2 time : {:.3f} s".format(time()-t000f))
    # We solve in the storage region.
    if True:
        if verbose > 0: t000f = time()
        B03z = B[Nt1+Nt2-2, :]
        S03z = S[Nt1+Nt2-2, :]
        if seed == "S":
            S03t = hermite_gauss(nshg, tau3 - t0s + D/c, sigs)
        else:
            S03t = np.zeros(Nt3, complex)

        params_storage = params.copy()
        params_storage["Nt"] = Nt3
        aux1 = [params_storage, S03t, S03z, B03z, tau3, Z]
        aux2 = {"pt": pt, "pz": pz, "folder": folder, "plots": False,
                "verbose": 1, "P0z": P0z, "case": 1}

        # We calculate analyticaly.
        if adiabatic:
            if analytic_storage > 0:
                t03 = tau3[0]
                ttau3 = np.outer(tau3, np.ones(Nz))
                ZZ3 = np.outer(np.ones(Nt3), Z)

                arg = -np.abs(Omega)**2/Gamma21 - Gamma32
                B[Nt1+Nt2-2:] = B03z*np.exp(arg*(ttau3 - t03))

                # The region determined by S03z

                S03z_reg = np.where(ttau3 <= t03 + (2*ZZ3+D)/c, 1.0, 0.0)
                # The region determined by S03t
                S03t_reg = 1 - S03z_reg

                S03z_f = interpolator(Z, S03z, kind="cubic")
                S03t_f = interpolator(tau3, S03t, kind="cubic")

            if analytic_storage == 1:
                S03z_reg = S03z_reg*S03z_f(ZZ3 - (ttau3-t03)*c/2)
                S03z_reg = S03z_reg*np.exp(-c*kappa**2/2/Gamma21*(ttau3-t03))

                S03t_reg = S03t_reg*S03t_f(ttau3 - (2*ZZ3+D)/c)
                S03t_reg = S03t_reg*np.exp(-kappa**2/Gamma21*(ZZ3+D/2))

                S[Nt1+Nt2-2:] = S03z_reg + S03t_reg
            elif analytic_storage == 2:
                aux1 = S03z_f(D/2 - (tau3-t03)*c/2)
                aux1 = aux1*np.exp(-c*kappa**2/2/Gamma21*(tau3-t03))

                aux2 = S03t_f(tau3 - (2*D/2+D)/c)
                aux2 = aux2*np.exp(-kappa**2/Gamma21*(D/2+D/2))

                Sf3t = S03z_reg[:, -1]*aux1 + S03t_reg[:, -1]*aux2
                S[Nt1+Nt2-2:, -1] = Sf3t

                tff = tau3[-1]
                aux3 = S03z_f(Z - (tff-t03)*c/2)
                aux3 = aux3*np.exp(-c*kappa**2/2/Gamma21*(tff-t03))

                aux4 = S03t_f(tff - (2*Z+D)/c)
                aux4 = aux4*np.exp(-kappa**2/Gamma21*(Z+D/2))

                Sf3z = S03z_reg[-1, :]*aux3 + S03t_reg[-1, :]*aux4
                S[-1, :] = Sf3z
                S[Nt1+Nt2-2:, 0] = S03t

            elif analytic_storage == 0:
                B3, S3 = solve_fdm_block(*aux1, **aux2)
                B[Nt1+Nt2-2:] = B3
                S[Nt1+Nt2-2:] = S3
            else:
                raise ValueError
        else:
            P3, B3, S3 = solve_fdm_block(*aux1, **aux2)
            P[Nt1+Nt2-2:] = P3
            B[Nt1+Nt2-2:] = B3
            S[Nt1+Nt2-2:] = S3
        if verbose > 0: print("region 3 time : {:.3f} s".format(time()-t000f))
    if verbose > 0:
        print("Full exec time: {:.3f} s".format(time()-t00f))
    if plots:
        fs = 15
        if verbose > 0: print("Plotting...")

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(tau*1e9, np.abs(S[:, 0])**2*1e-9, "b-")
        ax1.plot(tau*1e9, np.abs(S[:, -1])**2*1e-9, "g-")

        angle1 = np.unwrap(np.angle(S[:, -1]))/2/np.pi
        ax2.plot(tau*1e9, angle1, "g:")

        ax1.set_xlabel(r"$\tau \ [ns]$", fontsize=fs)
        ax1.set_ylabel(r"Signal  [1/ns]", fontsize=fs)
        ax2.set_ylabel(r"Phase  [revolutions]", fontsize=fs)
        plt.savefig(folder+"Sft_"+name+".png", bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(15, 9))
        plt.subplot(1, 2, 1)
        plt.title("$B$ FDM")
        cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(B)**2)
        plt.colorbar(cs)
        plt.ylabel(r"$\tau$ (ns)")
        plt.xlabel("$Z$ (cm)")

        plt.subplot(1, 2, 2)
        plt.title("$S$ FDM")
        cs = plt.pcolormesh(Z*100, tau*1e9, np.abs(S)**2*1e-9)
        plt.colorbar(cs)
        plt.ylabel(r"$\tau$ (ns)")
        plt.xlabel("$Z$ (cm)")

        plt.savefig(folder+"solution_fdm_"+name+".png", bbox_inches="tight")
        plt.close("all")

    if adiabatic:
        if return_modes:
            B0 = B02z
            S0 = S[:, 0]
            B1 = B03z
            S1 = S[:, -1]
            return tau, Z, B0, S0, B1, S1
        else:
            return tau, Z, B, S
    else:
        return tau, Z, P, B, S


#############################################
# Finite difference equations.


bfmt = "csr"
bfmtf = csr_matrix


def D_coefficients(p, j, xaxis=None, d=1, symbolic=False):
    r"""Calculate finite difference coefficients that approximate the
    derivative of order ``d`` to precision order $p$ on point $j$ of an
    arbitrary grid.

    INPUT:

    -  ``p`` - int, the precission order of the approximation.

    -  ``j`` - int, the point where the approximation is centered.

    -  ``xaxis`` - an array, the grid on which the function is represented.

    -  ``d`` - int, the order of the derivative.

    -  ``symbolic`` - a bool, whether to return symbolic coefficients.

    OUTPUT:

    An array of finite difference coefficients.

    Examples
    ========

    First order derivatives:
    >>> from sympy import pprint
    >>> pprint(D_coefficients(2, 0, symbolic=True))
    [-3/2  2  -1/2]
    >>> pprint(D_coefficients(2, 1, symbolic=True))
    [-1/2  0  1/2]
    >>> pprint(D_coefficients(2, 2, symbolic=True))
    [1/2  -2  3/2]

    Second order derivatives:
    >>> pprint(D_coefficients(2, 0, d=2, symbolic=True))
    [1  -2  1]
    >>> pprint(D_coefficients(3, 0, d=2, symbolic=True))
    [2  -5  4  -1]
    >>> pprint(D_coefficients(3, 1, d=2, symbolic=True))
    [1  -2  1  0]
    >>> pprint(D_coefficients(3, 2, d=2, symbolic=True))
    [0  1  -2  1]
    >>> pprint(D_coefficients(3, 3, d=2, symbolic=True))
    [-1  4  -5  2]

    A non uniform grid:
    >>> x = np.array([1.0, 3.0, 5.0])
    >>> print(D_coefficients(2, 1, xaxis=x))
    [-0.25  0.    0.25]

    """
    def poly_deri(x, a, n):
        if a-n >= 0:
            if symbolic:
                return sym_fact(a)/sym_fact(a-n)*x**(a-n)
            else:
                return num_fact(a)/float(num_fact(a-n))*x**(a-n)
        else:
            return 0.0

    if d > p:
        mes = "Cannot calculate a derivative of order "
        mes += "`d` larger than precision `p`."
        raise ValueError(mes)
    Nt = p+1
    if symbolic:
        arr = Matrix
    else:
        arr = np.array

    if xaxis is None:
        xaxis = arr([i for i in range(Nt)])

    zp = arr([poly_deri(xaxis[j], i, d) for i in range(Nt)])
    eqs = arr([[xaxis[ii]**jj for jj in range(Nt)] for ii in range(Nt)])

    if symbolic:
        coefficients = zp.transpose()*eqs.inv()
    else:
        coefficients = np.dot(zp.transpose(), np.linalg.inv(eqs))
    return coefficients


def derivative_operator(xaxis, p=2, symbolic=False, sparse=False):
    u"""A matrix representation of the differential operator for an arbitrary
    xaxis.

    Multiplying the returned matrix by a discretized function gives the second
    order centered finite difference for all points except the extremes, where
    a forward and backward second order finite difference is used for the
    first and last points respectively.

    Setting higher=True gives a fourth order approximation for the extremes.

    Setting symbolic=True gives a symbolic exact representation of the
    coefficients.

    INPUT:

    -  ``xaxis`` - an array, the grid on which the function is represented.

    -  ``p`` - int, the precission order of the approximation.

    -  ``symbolic`` - a bool, whether to return symbolic coefficients.

    -  ``sparse`` - a bool, whether to return a sparse matrix.

    OUTPUT:

    A 2-d array representation of the differential operator.

    Examples
    ========

    >>> from sympy import pprint
    >>> D = derivative_operator(range(5))
    >>> print(D)
    [[-1.5  2.  -0.5  0.   0. ]
     [-0.5  0.   0.5  0.   0. ]
     [ 0.  -0.5  0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.5]
     [ 0.   0.   0.5 -2.   1.5]]

    >>> D = derivative_operator(range(5), p=4)
    >>> print(D)
    [[-2.08333333  4.         -3.          1.33333333 -0.25      ]
     [-0.25       -0.83333333  1.5        -0.5         0.08333333]
     [ 0.08333333 -0.66666667  0.          0.66666667 -0.08333333]
     [-0.08333333  0.5        -1.5         0.83333333  0.25      ]
     [ 0.25       -1.33333333  3.         -4.          2.08333333]]

    >>> D = derivative_operator(range(5), p=4, symbolic=True)
    >>> pprint(D)
    ⎡-25                           ⎤
    ⎢────    4     -3   4/3   -1/4 ⎥
    ⎢ 12                           ⎥
    ⎢                              ⎥
    ⎢-1/4   -5/6  3/2   -1/2  1/12 ⎥
    ⎢                              ⎥
    ⎢1/12   -2/3   0    2/3   -1/12⎥
    ⎢                              ⎥
    ⎢-1/12  1/2   -3/2  5/6    1/4 ⎥
    ⎢                              ⎥
    ⎢                          25  ⎥
    ⎢ 1/4   -4/3   3     -4    ──  ⎥
    ⎣                          12  ⎦

    >>> D = derivative_operator([1, 2, 4, 6, 7], p=2, symbolic=True)
    >>> pprint(D)
    ⎡-4/3  3/2   -1/6   0     0 ⎤
    ⎢                           ⎥
    ⎢-2/3  1/2   1/6    0     0 ⎥
    ⎢                           ⎥
    ⎢ 0    -1/4   0    1/4    0 ⎥
    ⎢                           ⎥
    ⎢ 0     0    -1/6  -1/2  2/3⎥
    ⎢                           ⎥
    ⎣ 0     0    1/6   -3/2  4/3⎦


    """
    def rel_dif(a, b):
        if a > b:
            return 1-b/a
        else:
            return 1-a/b
    #########################################################################
    if symbolic and sparse:
        mes = "There is no symbolic sparse implementation."
        raise NotImplementedError(mes)

    N = len(xaxis); h = xaxis[1] - xaxis[0]
    if p % 2 != 0:
        raise ValueError("The precission must be even.")
    if N < p+1:
        raise ValueError("N < p+1!")

    if symbolic:
        D = zeros(N)
    else:
        D = np.zeros((N, N))
    #########################################################################
    hlist = [xaxis[i+1] - xaxis[i] for i in range(N-1)]
    err = np.any([rel_dif(hlist[i], h) >= 1e-5 for i in range(N-1)])
    if not err:
        coefficients = [D_coefficients(p, i, symbolic=symbolic)
                        for i in range(p+1)]
        mid = int((p+1)/2)

        # We put in place the middle coefficients.
        for i in range(mid, N-mid):
            a = i-mid; b = a+p+1
            D[i, a:b] = coefficients[mid]

        # We put in place the forward coefficients.
        for i in range(mid):
            D[i, :p+1] = coefficients[i]

        # We put in place the backward coefficients.
        for i in range(N-mid, N):
            D[i, N-p-1:N] = coefficients[p+1-N+i]

        D = D/h
    else:
        # We generate a p + 1 long list for each of the N rows.
        for i in range(N):
            if i < p/2:
                a = 0
                jj = i
            elif i >= N - p/2:
                a = N - p - 1
                jj = (i - (N - p - 1))
            else:
                a = i - p/2
                jj = p/2
            b = a + p + 1
            D[i, a: b] = D_coefficients(p, jj, xaxis=xaxis[a:b],
                                        symbolic=symbolic)
    if sparse:
        return bfmtf(D)
    return D


def fdm_derivative_operators(params, tau, Z, pt=4, pz=4, sparse=False,
                             plots=False, folder=""):
    r"""Calculate the block-matrix representation of the derivatives."""
    # We unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]

    # We build the derivative matrices.
    if True:
        Dt = derivative_operator(tau, p=pt, sparse=sparse)
        Dz = derivative_operator(Z, p=pz, sparse=sparse)
        if sparse:
            DT = sp_kron(Dt, sp_eye(Nz), format=bfmt)
            DZ = sp_kron(sp_eye(Nt), Dz, format=bfmt)
        else:
            DT = np.kron(Dt, np.eye(Nz))
            DZ = np.kron(np.eye(Nt), Dz)

    # Plotting.
    if plots and not sparse:
        print("Plotting...")

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.title("$D_t$")
        plt.imshow(np.abs(Dt))

        plt.subplot(2, 2, 2)
        plt.title("$D_z$")
        plt.imshow(np.abs(Dz))

        plt.subplot(2, 2, 3)
        plt.title("$D_T$")
        plt.imshow(np.abs(DT))

        plt.subplot(2, 2, 4)
        plt.title("$D_Z$")
        plt.imshow(np.abs(DZ))

        plt.savefig(folder+"derivatives.png", bbox_inches="tight")
        plt.close("all")

    return DT, DZ


def set_block(A, i, j, B):
    r"""A quick function to set block matrices."""
    nX = A.shape[0]
    Ntz = B.shape[0]
    nv = nX/Ntz
    if isinstance(A, spmatrix):
        ZERO = bfmtf((Ntz, Ntz))
        block = [[ZERO for jj in range(nv)] for ii in range(nv)]
        block[i][j] = B
        A += bmat(block, format=bfmt)
    else:
        ai = i*Ntz; bi = (i+1)*Ntz
        aj = j*Ntz; bj = (j+1)*Ntz
        A[ai:bi, aj:bj] += B
    return A


def eqs_fdm(params, tau, Z, Omegat="square", case=0, adiabatic=True,
            pt=4, pz=4, plots=False, folder="", sparse=False):
    r"""Calculate the matrix form of equations `Wp X = 0`."""
    if not adiabatic:
        nv = 3
    else:
        nv = 2
    # We unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]
        Gamma21 = calculate_Gamma21(params)
        Gamma32 = calculate_Gamma32(params)
        kappa = calculate_kappa(params)
        Omega = calculate_Omega(params)

        nX = nv*Nt*Nz
        Ntz = Nt*Nz
    # We build the derivative matrices.
    if True:
        args = [params, tau, Z]
        kwargs = {"pt": pt, "pz": pz, "plots": plots, "folder": folder,
                  "sparse": sparse}
        DT, DZ = fdm_derivative_operators(*args, **kwargs)

    if sparse:
        eye = sp_eye(Ntz, format=bfmt)
        Wp = bfmtf((nX, nX), dtype=np.complex128)
    else:
        eye = np.eye(Ntz)
        Wp = np.zeros((nX, nX), complex)

    # We build the Wp matrix.
    if True:
        # Empty space.
        if case == 0 and adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma32*eye)
            Wp = set_block(Wp, 1, 1, c/2*DZ)
        # Storage phase.
        elif case == 1 and adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma32*eye)
            Wp = set_block(Wp, 1, 1, c*kappa**2/2/Gamma21*eye)
            Wp = set_block(Wp, 1, 1, c/2*DZ)
        # Memory write/read phase.
        elif case == 2 and adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma32*eye)
            Wp = set_block(Wp, 1, 1, c*kappa**2/2/Gamma21*eye)
            Wp = set_block(Wp, 1, 1, c/2*DZ)

            if hasattr(Omegat, "__call__"):
                def Wpt(t):
                    if sparse:
                        aux1 = Omegat(t)
                        aux1 = spdiags(aux1, 0, Nt, Nt, format=bfmt)
                        aux2 = sp_eye(Nz, format=bfmt)
                        Omegatm = sp_kron(aux1, aux2, format=bfmt)
                    else:
                        Omegatm = np.kron(np.diag(Omegat(t)), np.eye(Nz))
                    aux1 = np.abs(Omegatm)**2/Gamma21
                    aux2 = kappa*Omegatm/Gamma21
                    aux3 = c*kappa*np.conjugate(Omegatm)/2/Gamma21
                    Wp_ = Wp.copy()
                    Wp_ = set_block(Wp_, 0, 0, aux1)
                    Wp_ = set_block(Wp_, 0, 1, aux2)
                    Wp_ = set_block(Wp_, 1, 0, aux3)
                    return Wp_

            else:
                aux1 = np.abs(Omega)**2/Gamma21
                aux2 = kappa*Omega/Gamma21
                aux3 = c*kappa*np.conjugate(Omega)/2/Gamma21
                Wp = set_block(Wp, 0, 0, aux1*eye)
                Wp = set_block(Wp, 0, 1, aux2*eye)
                Wp = set_block(Wp, 1, 0, aux3*eye)

        elif case == 0 and not adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)
            Wp = set_block(Wp, 2, 2, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma21*eye)
            Wp = set_block(Wp, 1, 1, Gamma32*eye)
            Wp = set_block(Wp, 2, 2, c/2*DZ)

        elif case == 1 and not adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)
            Wp = set_block(Wp, 2, 2, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma21*eye)
            Wp = set_block(Wp, 1, 1, Gamma32*eye)
            Wp = set_block(Wp, 2, 2, c/2*DZ)
            Wp = set_block(Wp, 0, 2, 1j*kappa*eye)
            Wp = set_block(Wp, 2, 0, 1j*kappa*c/2*eye)
        elif case == 2 and not adiabatic:
            # We set the time derivatives.
            Wp = set_block(Wp, 0, 0, DT)
            Wp = set_block(Wp, 1, 1, DT)
            Wp = set_block(Wp, 2, 2, DT)

            # We set the right-hand side terms.
            Wp = set_block(Wp, 0, 0, Gamma21*eye)
            Wp = set_block(Wp, 1, 1, Gamma32*eye)
            Wp = set_block(Wp, 2, 2, c/2*DZ)
            Wp = set_block(Wp, 0, 2, 1j*kappa*eye)
            Wp = set_block(Wp, 2, 0, 1j*kappa*c/2*eye)

            if hasattr(Omegat, "__call__"):
                def Wpt(t):
                    if sparse:
                        aux1 = Omegat(t)
                        aux1 = spdiags(aux1, 0, Nt, Nt, format=bfmt)
                        aux2 = sp_eye(Nz, format=bfmt)
                        Omegatm = sp_kron(aux1, aux2, format=bfmt)
                    else:
                        Omegatm = np.kron(np.diag(Omegat(t)), np.eye(Nz))
                    aux = np.conjugate(Omegatm)
                    Wp_ = Wp.copy()
                    Wp_ = set_block(Wp_, 0, 1, 1j*aux)
                    Wp_ = set_block(Wp_, 1, 0, 1j*Omegatm)
                    return Wp_
            else:
                aux = 1j*np.conjugate(Omega)
                Wp = set_block(Wp, 0, 1, aux*eye)
                Wp = set_block(Wp, 1, 0, 1j*Omega*eye)

    if hasattr(Omegat, "__call__"):
        return Wpt
    else:
        if plots:
            ################################################################
            # Plotting Wp.
            plt.figure(figsize=(15, 15))
            plt.title("$W'$")
            plt.imshow(np.log(np.abs(Wp)))
            plt.savefig(folder+"Wp.png", bbox_inches="tight")
            plt.close("all")

        return Wp


##############################################################################
# Checks.

def check_block_fdm(params, B, S, tau, Z, case=0, P=None,
                    pt=4, pz=4, folder="", plots=False, verbose=1):
    r"""Check the equations in an FDM block."""
    # We build the derivative operators.
    Nt = tau.shape[0]
    Nz = Z.shape[0]
    Gamma32 = calculate_Gamma32(params)
    Gamma21 = calculate_Gamma21(params)
    Omega = calculate_Omega(params)
    kappa = calculate_kappa(params)
    Dt = derivative_operator(tau, p=pt)
    Dz = derivative_operator(Z, p=pt)

    adiabatic = P is None

    # Empty space.
    if case == 0 and adiabatic:
        # We get the time derivatives.
        DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
        DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T

        DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
        rhsB = -Gamma32*B
        rhsS = -c/2*DzS
    # Storage phase.
    elif case == 1 and adiabatic:
        # We get the time derivatives.
        DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
        DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T

        DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
        rhsB = -Gamma32*B
        rhsS = -c/2*DzS - c*kappa**2/2/Gamma21*S
    # Memory write/read phase.
    elif case == 2 and adiabatic:
        # We get the time derivatives.
        DtB = np.array([np.dot(Dt, B[:, jj]) for jj in range(Nz)]).T
        DtS = np.array([np.dot(Dt, S[:, jj]) for jj in range(Nz)]).T

        DzS = np.array([np.dot(Dz, S[ii, :]) for ii in range(Nt)])
        rhsB = -Gamma32*B
        rhsS = -c/2*DzS - c*kappa**2/2/Gamma21*S

        rhsB += -np.abs(Omega)**2/Gamma21*B
        rhsB += -kappa*Omega/Gamma21*S
        rhsS += -c*kappa*np.conjugate(Omega)/2/Gamma21*B

    else:
        raise ValueError

    if True:
        # We put zeros into the boundaries.
        ig = pt/2 + 1
        ig = pt + 1
        ig = 1

        DtB[:ig, :] = 0
        DtS[:ig, :] = 0
        DtS[:, :ig] = 0

        rhsB[:ig, :] = 0
        rhsS[:ig, :] = 0
        rhsS[:, :ig] = 0

        # We put zeros in all the boundaries to neglect border effects.
        DtB[-ig:, :] = 0
        DtS[-ig:, :] = 0
        DtB[:, :ig] = 0
        DtB[:, -ig:] = 0
        DtS[:, -ig:] = 0

        rhsB[-ig:, :] = 0
        rhsS[-ig:, :] = 0
        rhsB[:, :ig] = 0
        rhsB[:, -ig:] = 0
        rhsS[:, -ig:] = 0

    if True:
        Brerr = rel_error(DtB, rhsB)
        Srerr = rel_error(DtS, rhsS)

        Bgerr = glo_error(DtB, rhsB)
        Sgerr = glo_error(DtS, rhsS)

        i1, j1 = np.unravel_index(Srerr.argmax(), Srerr.shape)
        i2, j2 = np.unravel_index(Sgerr.argmax(), Sgerr.shape)

        with warnings.catch_warnings():
            mes = r'divide by zero encountered in log10'
            warnings.filterwarnings('ignore', mes)

            aux1 = list(np.log10(get_range(Brerr)))
            aux1 += [np.log10(np.mean(Brerr))]
            aux1 += list(np.log10(get_range(Srerr)))
            aux1 += [np.log10(np.abs(np.mean(Srerr)))]

            aux2 = list(np.log10(get_range(Bgerr)))
            aux2 += [np.log10(np.mean(Bgerr))]
            aux2 += list(np.log10(get_range(Sgerr)))
            aux2 += [np.log10(np.mean(Sgerr))]

        aux1[1], aux1[2] = aux1[2], aux1[1]
        aux1[-1], aux1[-2] = aux1[-2], aux1[-1]
        aux2[1], aux2[2] = aux2[2], aux2[1]
        aux2[-1], aux2[-2] = aux2[-2], aux2[-1]

        if verbose > 0:
            print("Left and right hand sides comparison:")
            print("        Bmin   Bave   Bmax   Smin   Save   Smax")
            mes = "{:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}"
            print("rerr: "+mes.format(*aux1))
            print("gerr: "+mes.format(*aux2))
    if plots:
        args = [tau, Z, Brerr, Srerr, folder, "check_01_eqs_rerr"]
        kwargs = {"log": True, "ii": i1, "jj": j1}
        plot_solution(*args, **kwargs)

        args = [tau, Z, Bgerr, Sgerr, folder, "check_02_eqs_gerr"]
        kwargs = {"log": True, "ii": i2, "jj": j2}
        plot_solution(*args, **kwargs)

    return aux1, aux2, Brerr, Srerr, Bgerr, Sgerr


def check_fdm(params, B, S, tau, Z, P=None,
              pt=4, pz=4, folder="", name="check", plots=False, verbose=1):
    r"""Check the equations in an FDM block."""
    params, Z, tau, tau1, tau2, tau3 = build_mesh_fdm(params)
    N1 = len(tau1)
    N2 = len(tau2)
    # N3 = len(tau3)

    # S1 = S[:N1]
    S2 = S[N1-1:N1-1+N2]
    # S3 = S[N1-1+N2-1:N1-1+N2-1+N3]

    # B1 = B[:N1]
    B2 = B[N1-1:N1-1+N2]
    # B3 = B[N1-1+N2-1:N1-1+N2-1+N3]

    Brerr = np.zeros(B.shape)
    Srerr = np.zeros(B.shape)
    Bgerr = np.zeros(B.shape)
    Sgerr = np.zeros(B.shape)

    print("the log_10 of relative and global errors (for B and S):")
    ####################################################################
    kwargs = {"case": 2, "folder": folder, "plots": False}
    aux = check_block_fdm(params, B2, S2, tau2, Z, **kwargs)
    checks2_rerr, checks2_gerr, B2rerr, S2rerr, B2gerr, S2gerr = aux

    Brerr[N1-1:N1-1+N2] = B2rerr
    Srerr[N1-1:N1-1+N2] = S2rerr
    Bgerr[N1-1:N1-1+N2] = B2gerr
    Sgerr[N1-1:N1-1+N2] = S2gerr
    ####################################################################
    if plots:
        plot_solution(tau, Z, Brerr, Srerr, folder, "rerr"+name, log=True)
        plot_solution(tau, Z, Bgerr, Sgerr, folder, "gerr"+name, log=True)


#############################################################################
# Graphical routines.


def sketch_frame_transform(params, folder="", name="", draw_readout=False,
                           auxiliaries=False):
    r"""Make a sketech of the frame transform."""
    def transform(x):
        t, z = x
        tau = t + z/c
        zp = z
        return (tau, zp)

    def itransform(x):
        tau, zp = x
        t = tau - zp/c
        z = zp
        return (t, z)

    def plot_curve(x, fmt, **kwargs):
        plt.plot(x[1]*100, x[0]*1e9, fmt, **kwargs)

    # Unpack parameters.
    if True:
        Nt = params["Nt"]
        Nz = params["Nz"]
        # T = params["T"]
        # L = params["L"]
        c = params["c"]
        t0s = params["t0s"]
        t0w = params["t0w"]
        t0r = params["t0r"]
        taus = params["taus"]
        tauw = params["tauw"]
        ntauw = params["ntauw"]
        t = build_t_mesh(params)
        Z = build_Z_mesh(params)
        D = Z[-1] - Z[0]

    # Define axes:
    if True:
        zaxis = (t[0]*np.ones(Nz), Z)
        taxis = (t, 0*np.ones(Nt))
        zaxisp = transform(zaxis)
        taxisp = transform(taxis)
    # Define original curves.
    if True:
        end_axis = (t[-1]*np.ones(Nz), Z)
        xmL2 = (t, -D/2*np.ones(Nt))
        xpL2 = (t, +D/2*np.ones(Nt))

        S1 = (Z/c+t0s-taus/2, Z)
        S2 = (Z/c+t0s+taus/2, Z)
        S3 = (Z[int(Nz/2):]/c+t0r-taus/2, Z[int(Nz/2):])
        S4 = (Z[int(Nz/2):]/c+t0r+taus/2, Z[int(Nz/2):])
        Om1 = (t0w-Z/c-ntauw*tauw/2, Z)
        Om2 = (t0w-Z/c+ntauw*tauw/2, Z)
        Om3 = (t0r-Z/c-ntauw*tauw/2, Z)
        Om4 = (t0r-Z/c+ntauw*tauw/2, Z)
        ###########
        zpaxisp = (t[0]*np.ones(Nz), Z)
        zp_end_axisp = (t[-1]*np.ones(Nz), Z)
        ###########
        aux1 = (t0w - D/c - ntauw*tauw/2 + Z*2/c, Z)
        aux2 = (t0w - D/c - 3*ntauw*tauw/2 + Z*2/c, Z)
        aux3 = (t0w + D/c + ntauw*tauw/2 + Z*2/c, Z)
        aux4 = (t0w + D/c + 3*ntauw*tauw/2 + Z*2/c, Z)

    ######################
    # Transform everything.
    if True:
        end_axisp = transform(end_axis)
        xmL2p = transform(xmL2)
        xpL2p = transform(xpL2)

        S1p = transform(S1)
        S2p = transform(S2)
        S3p = transform(S3)
        S4p = transform(S4)
        Om1p = transform(Om1)
        Om2p = transform(Om2)
        Om3p = transform(Om3)
        Om4p = transform(Om4)
        ###########
        zpaxis = itransform(zpaxisp)
        zp_end_axis = itransform(zp_end_axisp)

    # Plot original frame.
    if True:

        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plot_curve(zaxis, "m--")
        plot_curve(end_axis, "m--")
        plot_curve(zpaxis, "m:")
        plot_curve(zp_end_axis, "m:")
        plot_curve(taxis, "g--")
        plot_curve(xmL2, "c-")
        plot_curve(xpL2, "c-", label="Cell windows")
        plot_curve(S1, "b-", lw=1)
        plot_curve(S2, "b-", lw=1)
        plot_curve(Om1, "r-", lw=1)
        plot_curve(Om2, "r-", lw=1)

        plt.fill_between(Om1[1]*100, Om1[0]*1e9, Om2[0]*1e9, color="r",
                         alpha=0.25, label=r"$\Omega$")
        plt.fill_between(S1[1]*100, S1[0]*1e9, S2[0]*1e9, color="b",
                         alpha=0.25, label=r"$S$")
        if draw_readout:
            plot_curve(S3, "b-")
            plot_curve(S4, "b-")
            plot_curve(Om3, "r-")
            plot_curve(Om4, "r-")

        plt.ylabel(r"$t \ \mathrm{(ns)}$", fontsize=15)
        plt.xlabel(r"$Z \ \mathrm{(cm)}$", fontsize=15)
        plt.legend(loc=2)
    # Plot transformed frame.
    if True:
        plt.subplot(1, 2, 2)
        plot_curve(end_axisp, "m--")
        plot_curve(zaxisp, "m--")
        plot_curve(zpaxisp, "m:")
        plot_curve(zp_end_axisp, "m:")
        plot_curve(taxisp, "g--")
        plot_curve(xmL2p, "c-")
        plot_curve(xpL2p, "c-", label="Cell windows")
        plot_curve(S1p, "b-", lw=1)
        plot_curve(S2p, "b-", lw=1)
        plot_curve(Om1p, "r-", lw=1)
        plot_curve(Om2p, "r-", lw=1)
        if auxiliaries:
            plot_curve(aux1, "k-", lw=1, alpha=0.25)
            plot_curve(aux2, "k-", lw=1, alpha=0.25)
            plot_curve(aux3, "k-", lw=1, alpha=0.25)
            plot_curve(aux4, "k-", lw=1, alpha=0.25)
        plt.fill_between(Om1p[1]*100, Om1p[0]*1e9, Om2p[0]*1e9,
                         color="r", alpha=0.25, label=r"$\Omega$")
        plt.fill_between(S1p[1]*100, S1p[0]*1e9, S2p[0]*1e9,
                         color="b", alpha=0.25, label=r"$S$")
        plt.legend(loc=2)

        if draw_readout:
            plot_curve(S3p, "b-")
            plot_curve(S4p, "b-")
            plot_curve(Om3p, "r-")
            plot_curve(Om4p, "r-")

        plt.ylabel(r"$\tau \ \mathrm{(ns)}$", fontsize=15)
        plt.xlabel(r"$Z' \ \mathrm{(cm)}$", fontsize=15)
        # plt.ylim(None, 1.5)
        plt.savefig(folder+"sketch_"+name+".png", bbox_inches="tight")
        plt.close()


def get_range(fp):
    r"""Get the range of an array."""
    fp = np.abs(fp)
    aux = fp.copy()
    aux[aux == 0] = np.amax(fp)
    vmin = np.amin(aux)

    vmax = np.amax(fp)
    return np.array([vmin, vmax])


def get_lognorm(fp):
    r"""Get a log norm to plot 2d functions."""
    fp = np.abs(fp)
    aux = fp.copy()
    aux[aux == 0] = np.amax(fp)
    vmin = np.amin(aux)

    vmax = np.amax(fp)
    if vmin < 1e-15:
        vmin = 1e-15
    if vmin == vmax:
        vmin = 1e-15
        vmax = 1.0
    if vmax == 0:
        vmax = 1.0

    return LogNorm(vmin=vmin, vmax=vmax)


def plot_solution(tau, Z, B, S, folder, name,
                  log=False, colorbar=True, ii=None, jj=None):
    r"""Plot a solution."""
    plt.figure(figsize=(19, 8))
    plt.subplot(1, 2, 1)
    if log:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(B), norm=get_lognorm(B))
    else:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(B))
    if colorbar: plt.colorbar(cb)
    plt.ylabel(r"$\tau$ (ns)")
    plt.xlabel("$Z$ (cm)")

    plt.subplot(1, 2, 2)
    if log:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(S), norm=get_lognorm(S))
    else:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(S))
    if ii is not None:
        plt.plot(Z[jj]*100, tau[ii]*1e9, "rx")
    if colorbar: plt.colorbar(cb)
    plt.ylabel(r"$\tau$ (ns)")
    plt.xlabel("$Z$ (cm)")
    plt.savefig(folder+name+".png", bbox_inches="tight")
    plt.close("all")
