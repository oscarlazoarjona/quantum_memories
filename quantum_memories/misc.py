# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#                            2017 Benjamin Brecht                       *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************

"""This is a library for simulations of the ORCA memory [1].

References:
    [1] https://arxiv.org/abs/1704.00013
"""


from math import pi, sqrt, log
import numpy as np
from matplotlib import pyplot as plt
from colorsys import hls_to_rgb
from settings_ladder import omega21, omega32
from scipy.constants import k as k_B
from scipy.special import hermite
from scipy.misc import factorial


def hg(n, x, x0, sigma):
    """Generate normalized Hermite-Gauss mode.

    That is,

    .. math::
        int |HG(x)|^2 dx = 1.

    Note that for the purpose of this code, the mode is re-normalised
    such that the 0th order mode (the fundamental Gaussian) has a
    peak height of one. This renormalisation is necessary to conform
    to the definitions in the quantum memories code.
    """
    X = (x - x0) / sigma
    result = hermite(n)(X) * np.exp(-X**2 / 2) /\
        sqrt(factorial(n) * sqrt(pi) * 2**n * sigma)
    # In the next line, the renormalisation happens.
    result *= sqrt(sqrt(pi) * sigma)
    return result


def Omega2_HG(Z, ti, sigma2w, sigma2r, Omega2, t0w, t0r,
              alpha_rw, nw=0, nr=0, c=299792458):
    r"""Calculate the control field distribution.
    This function allows you to choose different energies, widths,
    and temporal modes for write and read pulses, respectively.


    Arguments:
    Z -- position axis (numpy.ndarray)
    ti -- current instant in time
    sigma2w -- spectral intensity FWHM of the write pulse
    sigma2r -- spectral intensity FWHM of the read pulse
    Omega2 -- peak Rabi frequency of the write pulse
    t0w -- temporal offset of the write pulse
    t0r -- temporal offset of the read pulse
    alpha_rw -- scaling between write and read pulse


    Keyword Arguments:
    nw -- temporal mode order of the write pulse (default: 0)
    nr -- temporal mode order of the read pulse (default: 0)
    c -- speed of light (default: 299792458 m/s)


    Return:
    ctrl -- numpy.ndarray containing the complex control field
    """
    tauw = sqrt(log(2)) / (pi * sigma2w)  # width of write pulse
    taur = sqrt(log(2)) / (pi * sigma2r)  # width of read pulse
    # Calculate the write pulse
    ctrl_w = Omega2 * hg(nw, t0w - Z / c, ti, tauw)
    # Calculate the read pulse
    ctrl_r = Omega2 * hg(nr, t0r - Z / c, ti, taur)
    ctrl = ctrl_w + alpha_rw * ctrl_r
    return ctrl


def Omega1_boundary_HG(t, sigma1, Omega1, t0s, D, ns=0, c=299792458):
    r"""Calculate the boundary conditions for the signal field.


    Arguments:
    t -- time axis (numpy.ndarray).
    sigma1 -- spectral intensity FWHM of the signal pulse.
    Omega1 -- peak Rabi frequency of the signal pulse.
    t0s -- temporal offset of the signal pulse.
    D -- spatial extent of the calculation.


    Keyword Arguments:
    ns -- temporal mode order of the signal pulse (default: 0)
    c -- speed of light (default: 299792458 m/s)


    Return:
    sig_bound -- numpy.ndarray containing the complex signal
    """
    tau = sqrt(log(2)) / (pi * sigma1)
    sig_bound = Omega1 * hg(ns, t, t0s - D / 2 / c, tau)
    return sig_bound


def Omega1_initial_HG(Z, sigma1, Omega1, t0s, ns=0, c=299792458):
    r"""Calculate the initial signal field.


    Arguments:
    Z -- space axis (numpy.ndarray)
    sigma1 -- spectral intensity FWHM of the signal pulse
    Omega1 -- peak Rabi frequency of the signal pulse
    t0s -- temporal offset of the signal pulse


    Keyword Arguments:
    ns -- temporal mode order of the signal pulse (default: 0)
    c -- speed of light (default: 299792458 m/s)


    Return:
    sig_init -- numpy.ndarray containing the complex initial signal
    """
    tau = sqrt(log(2)) / (pi * sigma1)
    sig_init = Omega1 * hg(ns, -t0s, Z / c, tau)
    return sig_init


def simple_complex_plot(x, y, f, name, amount="", modsquare=False):
    """Plot the real, imaginary and mod square of a function f."""
    plt.figure(figsize=(18, 6))
    fs = 15

    plt.subplot(1, 3, 1)
    plt.title(r"$ \mathfrak{Re}"+amount+"$", fontsize=fs)
    cs = plt.pcolormesh(x, y, np.real(f))
    plt.xlabel(r"$Z \ \mathrm{(cm)}$", fontsize=fs)
    plt.ylabel(r"$t \ \mathrm{(ns)}$", fontsize=fs)
    plt.colorbar(cs)

    plt.subplot(1, 3, 2)
    plt.title(r"$ \mathfrak{Im}"+amount+"$", fontsize=fs)
    cs = plt.pcolormesh(x, y, np.imag(f))
    plt.xlabel(r"$Z \ \mathrm{(cm)}$", fontsize=fs)
    plt.colorbar(cs)

    plt.subplot(1, 3, 3)
    plt.title(r"$|"+amount+"|^2$", fontsize=fs)
    if modsquare:
        plt.title(r"$|"+amount+"|^2$", fontsize=fs)
        cs = plt.pcolormesh(x, y, np.real(f*f.conjugate()))
    else:
        plt.title(r"$|"+amount+"|$", fontsize=fs)
        cs = plt.pcolormesh(x, y, np.abs(f))
    plt.xlabel(r"$Z \ \mathrm{(cm)}$", fontsize=fs)
    plt.colorbar(cs)

    plt.savefig(name, bbox_inches="tight")
    plt.close("all")


def colorize(z):
    r"""Return an array of rgb tuples to visualize a complex matrix.

    The lightness of each pixel represents the magnitude of the corresponding,
    number, and it's hue the argument.
    """
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi) / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    return c


def cheb(N):
    r"""Generate Chebyshev matrix."""
    if N == 0:
        D = 0.
        x = 1.
    else:
        n = np.arange(0, N + 1)
        x = np.cos(np.pi * n / N).reshape(N + 1, 1)
        # x is a column vector
        c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**n).reshape(N+1, 1)
        # c is a column vector with a 2 as first and last element, not the
        # speed of light! and ones in the middle. The signs are alternating.
        X = np.tile(x, (1, N + 1))
        # X combines N+1 colunm vectors x to a matrix
        dX = X - X.T
        D = np.dot(c, 1. / c.T) / (dX + np.eye(N + 1))
        D -= np.diag(np.sum(D.T, axis=0))
    return D, x.reshape(N+1)
    # return the D matrix and x as a row vector


def cDz(fz, c, cheb_diff_mat):
    r"""Calculate the Z derivative times c."""
    return c*np.dot(fz, cheb_diff_mat)


def set_parameters_ladder(custom_parameters=None, fitted_couplings=True):
    r"""Set the parameters for a ladder memory.

    Only completely independent parameters are taken from settings.py.
    The rest are derived from them.
    """
    if custom_parameters is None:
        custom_parameters = {}
    pm_names = ["e_charge", "hbar", "c", "epsilon_0", "kB",
                "Omega", "distance_unit", "element", "isotope",
                "Nt", "Nz", "Nv", "Nrho", "T", "L", "sampling_rate",
                "keep_data", "mass", "Temperature", "Nsigma",
                "gamma21", "gamma32", "omega21", "omega21", "r1", "r2",
                "delta1", "sigma_power1", "sigma_power2",
                "w1", "w2", "energy_pulse1", "energy_pulse2",
                "t0s", "t0w", "t0r", "alpha_rw", "t_cutoff", "verbose"]
    pms = {}

    for i in custom_parameters:
        if i not in pm_names:
            raise ValueError(str(i)+" is not a valid parameter name.")

    for name in pm_names:
        if name in custom_parameters:
            pms.update({name: custom_parameters[name]})
        else:
            s = "from settings_ladder import "+name
            exec(s)
            s = "pms.update({'"+name+"':"+name+"})"
            exec(s)

    delta1 = pms["delta1"]
    delta2 = -delta1
    omega_laser1 = delta1 + omega21
    omega_laser2 = delta2 + omega32
    pms.update({"omega_laser1": omega_laser1, "omega_laser2": omega_laser2})
    # We make a few checks
    if pms["Nv"] == 2:
        raise ValueError("Nv = 2 is a very bad choice.")

    if pms["Nt"] % pms["sampling_rate"] != 0:
        raise ValueError("Nt must be a multiple of the sampling_rate.")

    cond1 = "r1" not in custom_parameters
    cond2 = "r2" not in custom_parameters
    # if fitted_couplings and cond1 and cond2:
    #     pms.update({"r1": pms["r1"]*0.23543177})
    #     pms.update({"r2": pms["r2"]*0.81360687})
    if fitted_couplings and cond1 and cond2:
        pms.update({"r1": pms["r1"]*0.2556521})
        pms.update({"r2": pms["r2"]*0.72474758})
    # if fitted_couplings and cond1 and cond2:
    #     pms.update({"r1": pms["r1"]*0.2556521})
    #     pms.update({"r2": pms["r2"]*0.63474758})

    pms["ns"] = 1
    pms["nw"] = 1
    pms["nr"] = 1
    pms["USE_HG_CTRL"] = False
    pms["USE_HG_SIG"] = False

    return pms


def set_parameters_lambda(custom_parameters=None, fitted_couplings=True):
    r"""Set the parameters for a lambda memory.

    Only completely independent parameters are taken from settings.py.
    The rest are derived from them.
    """
    if custom_parameters is None:
        custom_parameters = {}
    pm_names = ["magic", "red_detuned",
                "e_charge", "hbar", "c", "epsilon_0", "kB",
                "Omega", "distance_unit", "element", "isotope",
                "Nt", "Nz", "Nv", "Nrho", "T", "L", "sampling_rate",
                "keep_data", "mass", "Temperature", "Nsigma",
                "gamma31", "gamma32", "omega31", "omega31", "omega31",
                "r31", "r32",
                "delta1", "sigma_power1", "sigma_power2",
                "w1", "w2", "energy_pulse31", "energy_pulse32",
                "t0s", "t0w", "t0r", "alpha_rw", "t_cutoff", "verbose"]
    pms = {}

    for i in custom_parameters:
        if i not in pm_names:
            raise ValueError(str(i)+" is not a valid parameter name.")

    for name in pm_names:
        if name in custom_parameters:
            pms.update({name: custom_parameters[name]})
        else:
            s = "from settings_lambda import "+name
            exec(s)
            s = "pms.update({'"+name+"':"+name+"})"
            exec(s)

    delta1 = pms["delta1"]
    delta2 = -delta1
    omega_laser1 = delta1 + omega21
    omega_laser2 = delta2 + omega32
    pms.update({"omega_laser1": omega_laser1, "omega_laser2": omega_laser2})
    # We make a few checks
    if pms["Nv"] == 2:
        raise ValueError("Nv = 2 is a very bad choice.")

    if pms["Nt"] % pms["sampling_rate"] != 0:
        raise ValueError("Nt must be a multiple of the sampling_rate.")

    return pms


def efficiencies(t, Om1, params, plots=False, name="",
                 explicit_decoherence=1.0):
    r"""Calculate the efficiencies for a given solution of the signal."""
    e_charge = params["e_charge"]
    hbar = params["hbar"]
    c = params["c"]
    epsilon_0 = params["epsilon_0"]
    Omega = params["Omega"]
    Nt = len(t)
    r1 = params["r1"]

    omega_laser1 = params["omega_laser1"]
    w1 = params["w1"]
    t_cutoff = params["t_cutoff"]

    # We calculate the number of photons.
    const1 = np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1

    dphotons_ini_dt = const1 * np.real(Om1[:, +0]*Om1[:, +0].conjugate())
    dphotons_out_dt = const1 * np.real(Om1[:, -1]*Om1[:, -1].conjugate())

    dphase_ini = np.unwrap(np.angle(Om1[:, +0]))
    # dphase_out = np.angle(Om1[:, -1])
    # dphase_tra = np.array([dphase_out[i] for i in range(Nt)
    #                        if t[i] < t_cutoff])
    # dphase_ret = np.array([dphase_out[i] for i in range(Nt)
    #                        if t[i] > t_cutoff])

    dt = t[1]-t[0]

    # We separate the output at the cutoff time.
    dphotons_out_dt_tr = [dphotons_out_dt[i] for i in range(Nt)
                          if t[i] < t_cutoff]
    dphotons_out_dt_re = [dphotons_out_dt[i] for i in range(Nt)
                          if t[i] > t_cutoff]

    t_tr = np.array([t[i] for i in range(Nt) if t[i] < t_cutoff])
    t_re = np.array([t[i] for i in range(Nt) if t[i] > t_cutoff])
    dphotons_out_dt_tr = np.array(dphotons_out_dt_tr)
    dphotons_out_dt_re = np.array(dphotons_out_dt_re)*explicit_decoherence

    if plots:
        fig, ax1 = plt.subplots()
        ax1.plot(t*Omega*1e9, dphotons_ini_dt/Omega*1e-9, "g-",
                 label=r"$\mathrm{Signal} \ @ \ z=-D/2$")
        ax1.plot(t_tr*Omega*1e9, dphotons_out_dt_tr/Omega*1e-9, "r-",
                 label=r"$\mathrm{Signal} \ @ \ z=+D/2$")
        ax1.plot(t_re*Omega*1e9, dphotons_out_dt_re/Omega*1e-9, "b-",
                 label=r"$\mathrm{Signal} \ @ \ z=+D/2$")
        ax1.set_xlabel(r"$ t \ (\mathrm{ns})$", fontsize=20)
        ax1.set_ylabel(r"$ \mathrm{photons/ns}$", fontsize=20)
        plt.legend(fontsize=15)

        ax2 = ax1.twinx()
        ax2.plot(t*Omega*1e9, dphase_ini*180/np.pi, "g:")
        # ax2.plot(t_tr*Omega*1e9, dphase_tra*180/np.pi, "r:")
        # ax2.plot(t_re*Omega*1e9, dphase_ret*180/np.pi, "b:")
        ax2.set_ylabel(r"$ \mathrm{Phase \ (degrees)}$", fontsize=20)

        plt.savefig(name+"_inout.png", bbox_inches="tight")
        plt.close("all")

    Ntr = sum([dphotons_out_dt[i] for i in range(Nt)
               if t[i] < t_cutoff])*dt
    Nre = sum([dphotons_out_dt[i] for i in range(Nt)
               if t[i] > t_cutoff])*dt

    # We integrate using the trapezium rule.
    Nin = sum(dphotons_ini_dt[1:-1])
    Nin += (dphotons_ini_dt[1] + dphotons_ini_dt[-1])*0.5
    Nin = Nin*dt

    Ntr = sum(dphotons_out_dt_tr[1:-1])
    Ntr += (dphotons_out_dt_tr[1] + dphotons_out_dt_tr[-1])*0.5
    Ntr = Ntr*dt

    Nre = sum(dphotons_out_dt_re[1:-1])
    Nre += (dphotons_out_dt_re[1] + dphotons_out_dt_re[-1])*0.5
    Nre = Nre*dt

    eff_in = (Nin-Ntr)/Nin
    eff_out = Nre/(Nin-Ntr)
    eff = eff_in*eff_out

    return eff_in, eff_out, eff


def vapour_pressure(Temperature, element):
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


def vapour_number_density(Temperature, element):
    r"""Return the number of atoms in a rubidium or cesium vapour in m^-3.

    It receives as input the temperature in Kelvins and the
    name of the element.

    >>> print vapour_number_density(90.0 + 273.15,"Cs")
    8.39706962725e+18

    """
    return vapour_pressure(Temperature, element)/k_B/Temperature


class Measurement(object):
    r"""A class for error propagation arithmetic."""

    def __init__(self, value, sigma):
        r"""A class for error propagation arithmetic."""
        self.value = float(value)
        self.sigma = sigma

    def __str__(self):
        r"""The string method for Measurement."""
        return '('+str(self.value)+', '+str(self.sigma)+')'

    def __mul__(self, other, cov=0.0):
        r"""Multiplication."""
        # Scalar multiplication
        if isinstance(other, float) or isinstance(other, int):
            return Measurement(other*self.value, abs(other)*self.sigma)
        # Measurement multiplication
        elif isinstance(other, Measurement):
            sigmaf = self.value**2 * other.sigma**2
            sigmaf += other.value**2 * self.sigma**2
            sigmaf += 2*self.value*other.value*cov
            sigmaf = sqrt(sigmaf)
            return Measurement(self.value*other.value, sigmaf)

    def __rmul__(self, other):
        r"""Reverse multiplication."""
        return self.__mul__(other)

    def __add__(self, other, cov=0.0):
        r"""Addition."""
        # Scalar addition
        if isinstance(other, float) or isinstance(other, int):
            return Measurement(other+self.value, self.sigma)
        # Measurement addition
        elif isinstance(other, Measurement):
            sigmaf = self.sigma**2 + other.sigma**2 + 2*cov
            sigmaf = sqrt(sigmaf)
            return Measurement(self.value + other.value, sigmaf)

    def __radd__(self, other):
        r"""Reverse addition."""
        return self.__add__(other)

    def __sub__(self, other, cov=0.0):
        r"""Substraction."""
        # Scalar substraction
        if isinstance(other, float) or isinstance(other, int):
            return Measurement(-other+self.value, self.sigma)
        # Measurement substraction
        elif isinstance(other, Measurement):
            sigmaf = self.sigma**2 + other.sigma**2 - 2*cov
            sigmaf = sqrt(sigmaf)
            return Measurement(self.value - other.value, sigmaf)

    def __rsub__(self, other):
        r"""Reverse substraction."""
        if isinstance(other, float) or isinstance(other, int):
            other = Measurement(other, 0.0)

        return other.__sub__(self)

    def __div__(self, other, cov=0.0):
        r"""Division."""
        # Scalar division.
        if isinstance(other, float) or isinstance(other, int):
            other = Measurement(other, 0.0)
        # Measurement division.
        sigmaf = (self.sigma/self.value)**2
        sigmaf += (other.sigma/other.value)**2 - 2*cov/(self.value*other.value)
        sigmaf = sqrt(sigmaf)
        sigmaf = sqrt((self.value/other.value)**2)*sigmaf

        return Measurement(self.value / other.value, sigmaf)

    def __rdiv__(self, other):
        r"""Reverse division."""
        if isinstance(other, float) or isinstance(other, int):
            other = Measurement(other, 0.0)

        return other.__div__(self)

    def __neg__(self):
        r"""Negative."""
        return Measurement(-self.value, self.sigma)

    def __pow__(self, other, cov=0.0):
        r"""Power."""
        # Scalar power.
        if isinstance(other, float) or isinstance(other, int):
            other = Measurement(other, 0.0)
        # Measurement power.
        sigmaf = (other.value*self.sigma/self.value)**2
        sigmaf += (log(self.value)*other.sigma)**2
        sigmaf += 2*other.value*log(self.value)*cov/self.value
        sigmaf = sqrt(sigmaf)

        return Measurement(self.value ** other.value, sigmaf)

    def __rpow__(self, other):
        r"""Reverse power."""
        if isinstance(other, float) or isinstance(other, int):
            other = Measurement(other, 0.0)

        return other.__pow__(self)
