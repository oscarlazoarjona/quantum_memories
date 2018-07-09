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
from scipy.constants import physical_constants, c, hbar, epsilon_0

import numpy as np
from matplotlib import pyplot as plt
from colorsys import hls_to_rgb
from settings_ladder import omega21, omega32
from scipy.constants import k as k_B
from scipy.special import hermite
from scipy.misc import factorial
from scipy.interpolate import interp1d
from sympy import Matrix, Integer
from sympy import zeros as symb_zeros
from sympy import factorial as symb_factorial
from scipy.special import ai_zeros


def optimal_mesh(n, tau, T, D):
    r"""Get the optimal mesh for a Hermite-Gauss function."""
    bandwidth = hg_bandwidth(n, tau)
    dt = 1/bandwidth/10.0
    dz = c*dt
    Nt = int(T/dt)
    Nz = int(D/dz)
    return Nt, Nz


def last_root(n):
    r"""Get the last zero of the corresponding Hermite polynomial."""
    # i1 = ai_zeros(1)[0][0]/-3**(-1/3.0)
    if n < 0:
        raise ValueError
    if n == 0:
        return None
    a1 = ai_zeros(1)[0][0]/-2**(1/3.0)
    Lam = np.sqrt(2*n+1)
    root = Lam
    root += (-a1)*Lam**(-1/3.0)
    root += (-1/10.0 * a1**2) * Lam**(-5/3.0)
    root += (9/280.0 - 11/350.0 * a1**3) * Lam**(-3.0)
    root += (277/12600.0 * a1 - 823/63000.0 * a1**4) * Lam**(-13.0/3.0)
    return root/np.sqrt(2*np.pi)


def hg_bandwidth(n, tau):
    r"""Get the bandwidth of a Hermite-Gauss function."""
    if n == 0:
        bandwidth = 1.0
    else:
        bandwidth = last_root(n)+1.0
    bandwidth = bandwidth/tau
    return bandwidth


def hg_duration(n, tau):
    r"""Get the pulse duration of a Hermite-Gauss function."""
    return hg_bandwidth(n, tau)*tau**2*np.sqrt(2*np.pi)


def optimal_signal_bandwidth(L, tau2):
    r"""For given tau2 (control field duration) and L we pick the \
    maximum bandwidth.
    """
    #
    bandwidth_c = 1/tau2
    bandwidth_L = c/L
    bandwidth_probe = max([bandwidth_c, bandwidth_L])
    # print "bandwidths L, control, probe:",
    # print bandwidth_L*1e-9, bandwidth_c*1e-9, bandwidth_probe*1e-9
    return bandwidth_probe


def get_coeffs(order, accur, direction="backward"):
    """The coefficients of a discrete derivative.

    INPUT:

    - ``order`` - an integer, the order of the derivative.
    - ``accur`` - an integer, the accurracy of the derivative.
    - ``direction`` - a string indicating the direction of the derivative, \
    either `backward`, `centered`, or `forward`. By default, `backward`.

    OUTPUT:

    - A list of symbolic rational numbers.

    The most common approximation.
    >>> get_coeffs(1, 1)
    [-1, 1]

    A centered, second order derivative.
    >>> get_coeffs(2, 2, "centered")
    [1, -2, 1, 0]

    A high accuracy first derivative.
    >>> get_coeffs(1, 8)
    [1/8, -8/7, 14/3, -56/5, 35/2, -56/3, 14, -8, 761/280]

    A high accuracy fifth derivative.
    >>> get_coeffs(5, 4)
    [35/6, -305/6, 195, -2581/6, 1790/3, -1065/2, 895/3, -575/6, 27/2]

    """
    points = order+accur

    if direction == "backward":
        s = Matrix([Integer(i) for i in range(-points+1, 1)]).transpose()
    elif direction == "forward":
        s = Matrix([Integer(i) for i in range(points)]).transpose()
    elif direction == "centered":
        if accur % 2 != 0:
            s = "accurracy has to be even for centered derivatives."
            raise ValueError(s)
        s = Matrix([Integer(i)
                    for i in range(-points/2+1, points/2+1)]).transpose()
    else:
        s = 'direction options are "backward", "centered", forward""'
        raise ValueError(s)

    S = symb_zeros(points, points)
    for i in range(points):
        S[i, :] = Matrix([s[0, j]**i for j in range(points)]).transpose()

    d = symb_zeros(points, 1)
    d[order] = symb_factorial(order)

    sol = S.inv()*d
    return [i for i in sol]


def calculate_coeff_table(accur_max, direction="backward", numeric=True):
    u"""Calculate a table of coefficients for higher accurracy first order \
    backward derivatives.

    INPUT:

    - ``order`` - an integer, the order of the derivatives.
    - ``accur_max`` - an integer, the maximum accurracy of the derivatives \
    in the table.
    - ``numeric`` - a boolean indicating whether to return symbolic numbers, \
    floats. By default, `True`.

    OUTPUT:

    - Either a list of lists with symbolic numbers or a numpy array.

    >>> from sympy import pprint
    >>> table = calculate_coeff_table(5, numeric=False)
    >>> pprint(table)
    ⎡ 0     0     0     0   -1   1  ⎤
    ⎢                               ⎥
    ⎢ 0     0     0    1/2  -2  3/2 ⎥
    ⎢                               ⎥
    ⎢ 0     0   -1/3   3/2  -3  11/6⎥
    ⎢                               ⎥
    ⎢                            25 ⎥
    ⎢ 0    1/4  -4/3    3   -4   ── ⎥
    ⎢                            12 ⎥
    ⎢                               ⎥
    ⎢                           137 ⎥
    ⎢-1/5  5/4  -10/3   5   -5  ─── ⎥
    ⎣                            60 ⎦

    The equivalent forward table:
    >>> table = calculate_coeff_table(5, direction="forward", numeric=False)
    >>> pprint(table)
    ⎡ -1    1   0     0     0     0 ⎤
    ⎢                               ⎥
    ⎢-3/2   2  -1/2   0     0     0 ⎥
    ⎢                               ⎥
    ⎢-11/6  3  -3/2  1/3    0     0 ⎥
    ⎢                               ⎥
    ⎢-25                            ⎥
    ⎢────   4   -3   4/3   -1/4   0 ⎥
    ⎢ 12                            ⎥
    ⎢                               ⎥
    ⎢-137                           ⎥
    ⎢─────  5   -5   10/3  -5/4  1/5⎥
    ⎣  60                           ⎦

    >>> print calculate_coeff_table(5)
    [[ 0.          0.          0.          0.         -1.          1.        ]
     [ 0.          0.          0.          0.5        -2.          1.5       ]
     [ 0.          0.         -0.33333333  1.5        -3.          1.83333333]
     [ 0.          0.25       -1.33333333  3.         -4.          2.08333333]
     [-0.2         1.25       -3.33333333  5.         -5.          2.28333333]]


    """
    order = 1
    coef_table = [[0 for ii in range(accur_max+1)] for jj in range(accur_max)]
    for i in range(1, accur_max+1):
        tab = get_coeffs(order, i, direction)
        if direction == "backward":
            coef_table[i-1][accur_max+1-(order+i):] = tab
        elif direction == "forward":
            coef_table[i-1][:order+i] = tab
        elif direction == "centered":
            s = "I hate centered tables, go make your own."
            raise NotImplementedError(s)
        else:
            raise ValueError

    coef_table = [coef_table[jj][:accur_max+1] for jj in range(accur_max)]
    if numeric:
        coef_table = np.array([[float(coef_table[ii][jj])
                               for jj in range(accur_max+1)]
                               for ii in range(accur_max)])
    else:
        coef_table = Matrix(coef_table)
    return coef_table


def Dt_order_backward(f, t, coef_table, accur=1):
    """A backward derivative with accurracy `accur`.

    INPUT:

    - ``f`` - a numpy array representing a function.
    - ``t`` - a numpy array representing the funtion's independent variable.
    - ``coef_table`` - a coefficient table calculated with \
    `calculate_coef_table`.
    - ``accur`` - an int indicating the desired accurracy.

    OUTPUT:

    A number representing the derivative of f at the last point `t[-1]`.

    """
    dt = t[1]-t[0]
    max_accur = len(coef_table)
    coefs = coef_table[accur-1][max_accur-accur:]
    return sum([coefs[j]*f[j] for j in range(accur + 1)])/dt


def derivative_bounds(ii, accur_max, Nt, direction="forward"):
    """An auxiliary function to get the bounds necessary to calculate \
    derivatives of the boundary conditions..."""
    if direction == "forward":
        if ii == Nt-1:
            return (Nt-2, Nt)
        elif ii <= Nt-accur_max:
            return (ii, ii+accur_max+1)
        else:
            return (ii, Nt)
    elif direction == "backward":
        if ii <= accur_max:
            return (0, ii+1)
        else:
            return (ii-accur_max, ii+1)


def sketch_cell(params, folder="", name="sketch"):
    r"""Plot a sketch of the cell in space-time showing control and signal \
    fields.
    """
    def Omega2(ti, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw):
        Om2 = Omega2_peak*np.exp(
            -4*log(2.0)*(ti-t0w+Z/c)**2/tau2**2)
        Om2 += Omega2_peak*np.exp(
            -4*log(2.0)*(ti-t0r+Z/c)**2/tau2**2)*alpha_rw
        return Om2
    if True:
        USE_SQUARE_CTRL = params["USE_SQUARE_CTRL"]
        USE_HG_CTRL = params["USE_HG_CTRL"]
        sigma_power2 = params["sigma_power2"]
        t0s = params["t0s"]
        t0w = params["t0w"]
        USE_HG_CTRL = params["USE_HG_CTRL"]
        t0r = params["t0r"]
        alpha_rw = params["alpha_rw"]
        nw = params["nw"]
        nr = params["nr"]
        T = params["T"]
        L = params["L"]
        Nt = params["Nt"]/params["sampling_rate"]
        Nz = params["Nz"]
        tau2 = params["tau2"]
        tau1 = params["tau1"]
        t_cutoff = params["t_cutoff"]

        Omega2_peak = 1.0
        t_ini = np.linspace(0, T, Nt)
        Z = build_Z_mesh(L, Nz)

        if USE_HG_CTRL:
            Om2_mesh = [Omega2_HG(Z, ti, sigma_power2, sigma_power2,
                                  Omega2_peak, t0w, t0r,
                                  alpha_rw, nw=nw, nr=nr) for ti in t_ini]
        elif USE_SQUARE_CTRL:
            Om2_mesh = [Omega2_square(Omega2_peak, Z, ti, tau2, t0w, t0r,
                                      alpha_rw) for ti in t_ini]
            slice_ = Omega2_square(Omega2_peak, Z, t0w, tau2, t0w, t0r,
                                   alpha_rw)
        else:
            Om2_mesh = [Omega2(ti, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw)
                        for ti in t_ini]
            slice_ = Omega2(t0w, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw)

        Om2_mesh = np.array(Om2_mesh)
        Om2_mesh = Om2_mesh**2

    plt.close("all")
    cp = plt.pcolormesh(Z*100, t_ini*1e9, Om2_mesh)
    plt.colorbar(cp)

    input_signal1 = t0s-tau1/2+Z/c
    input_signal2 = t0s+tau1/2+Z/c

    output_signal1 = t0r-tau1/2+Z/c
    output_signal2 = t0r+tau1/2+Z/c
    plt.plot(Z*100, input_signal1*1e9, "b-")
    plt.plot(Z*100, input_signal2*1e9, "b-")
    plt.plot(Z*100, output_signal1*1e9, "b-")
    plt.plot(Z*100, output_signal2*1e9, "b-")

    plt.plot(Z*100, np.ones(len(Z))*t_cutoff*1e9, "g-")

    plt.plot([-L/2*100, -L/2*100], [0, T*1e9], "r-", linewidth=1)
    plt.plot([L/2*100, L/2*100], [0, T*1e9], "r-", linewidth=1)
    plt.xlabel(r"$ Z \ (\mathrm{cm})$", fontsize=20)
    plt.ylabel(r"$ t \ (\mathrm{ns})$", fontsize=20)

    plt.xlim([Z[0]*100, Z[-1]*100])
    plt.ylim(0, T*1e9)
    plt.savefig(folder+"params_Om2_"+name+".png", bbox_inches="tight")
    plt.close("all")

    plt.plot(Z*100, slice_, "b+-")
    plt.savefig(folder+"params_control_"+name+".png", bbox_inches="tight")
    plt.close("all")


def build_Z_mesh(L, Nz):
    r"""Return a Z mesh for a given cell length and number of points."""
    D = L*1.05
    zL = -0.5 * D
    cheb_diff_mat, cheb_mesh = cheb(Nz-1)
    cheb_diff_mat = cheb_diff_mat.T / zL
    Z = zL * cheb_mesh.T
    return Z


def heaviside(x):
    r"""The Heaviside function."""
    return np.where(x <= 0, 0.0, 1.0) + np.where(x == 0, 0.5, 0.0)


def upper_hyperfine_density(element, isotope, Temperature):
    r"""We calculate the atomic density of the upper hyperfine level."""
    if element == "Rb":
        if isotope == 85:
            fground = [2, 3]
        elif isotope == 87:
            fground = [1, 2]
    elif element == "Cs":
        if isotope == 133:
            fground = [3, 4]

    n_atomic0 = vapour_number_density(Temperature, element)
    upper_fraction = (2*fground[1]+1)/(2*fground[0]+1.0 + 2*fground[1]+1.0)
    return upper_fraction*n_atomic0


def cell_atomic_density(element, isotope, Temperature, L, Nz,
                        upper_hyperfine=False):
    r"""Return the atomic density as a function of Z."""
    Z = build_Z_mesh(L, Nz)
    if upper_hyperfine:
        n_atomic0 = upper_hyperfine_density(element, isotope, Temperature)
    else:
        n_atomic0 = vapour_number_density(Temperature, element)
    return n_atomic0*(-heaviside(Z - 0.5 * L) + heaviside(0.5 * L + Z))


def empty_points(n_atomic):
    r"""Return the number of points that are empty in a cell mesh."""
    empty = 0
    for i in range(len(n_atomic)):
        if n_atomic[i] == 0.0:
            empty += 1
        else:
            break
    return empty


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


def num_integral(t, f):
    """We integrate using the trapezium rule."""
    dt = t[1]-t[0]
    F = sum(f[1:-1])
    F += (f[1] + f[-1])*0.5
    return np.real(F*dt)


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
              alpha_rw, nw=0, nr=0):
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
    c = 299792458.0
    tauw = sqrt(log(2)) / (pi * sigma2w)  # width of write pulse
    taur = sqrt(log(2)) / (pi * sigma2r)  # width of read pulse
    # Calculate the write pulse
    ctrl_w = Omega2 * hg(nw, t0w - Z / c, ti, tauw)
    # Calculate the read pulse
    ctrl_r = Omega2 * hg(nr, t0r - Z / c, ti, taur)
    ctrl = ctrl_w + alpha_rw * ctrl_r
    return ctrl


def Omega1_boundary_HG(t, sigma1, Omega1, t0s, D, ns=0):
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


def Omega1_initial_HG(Z, sigma1, Omega1, t0s, ns=0):
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
    c = 299792458.0
    tau = sqrt(log(2)) / (pi * sigma1)
    sig_init = Omega1 * hg(ns, -t0s, Z / c, tau)
    return sig_init


def square(t, tau):
    r"""This is a template."""
    f = np.where(t/tau >= -0.5, 1.0, 0.0)*np.where(t/tau <= 0.5, 1.0, 0.0)
    f = f*sqrt(tau)
    return f


def Omega2_square(Omega2, Z, ti, tau2, t0w, t0r, alpha_rw):
    r"""Calculate the control field Rabi frequency for a specific ti."""
    c = 299792458.0
    pulse = Omega2/np.sqrt(tau2)*square((ti-t0w) + Z/c, tau2)
    pulse += Omega2/np.sqrt(tau2)*square((ti-t0r) + Z/c, tau2)*alpha_rw

    return pulse


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
    # plt.axes().set_aspect("equal")
    plt.title(r"$|"+amount+"|^2$", fontsize=fs)
    if modsquare:
        plt.title(r"$|"+amount+"|^2$", fontsize=fs)
        cs = plt.pcolormesh(x, y, np.real(f*f.conjugate()))
    else:
        plt.title(r"$|"+amount+"|$", fontsize=fs)
        cs = plt.pcolormesh(x, y, np.abs(f))

    # Nz = len(x)
    # for i in range(len(y)):
    #     # print y[0], y[i], y[-1]
    #     plt.plot(x, np.ones(Nz)*y[i], "r+", ms=5)

    # plt.ylim([0.85, 0.89])
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
    #########################################################################
    # We set the default values of independent parameters
    if True:
        # rewrite = True; rewrite = False
        calculate_atom = False  # ; calculate_atom = True
        # calculate_bloch = False  # ; calculate_bloch=True
        # make_smoother = True  # ; make_smoother=False
        # change_rep_rate = True  # ; change_rep_rate=False
        # change_read_power = True  # ; change_read_power=False
        ignore_lower_f = False; ignore_lower_f = True
        # run_long = False; run_long = True

        # optimize = True; optimize = False
        verbose = 1
        # We choose the units we want.
        units = "SI"  # ; units="fancy"
        if verbose >= 2: print "We are using "+units+" units!"

        a0 = physical_constants["Bohr radius"][0]
        e_charge = physical_constants["elementary charge"][0]
        kB = physical_constants["Boltzmann constant"][0]

        # The extent of the simulation given by the number of dynamic variables
        # Nrho, the number of time steps Nt, and the number of z points Nz.
        Nrho = 2
        Nt = 25500
        Nz = 50

        # The number of velocity groups to consider (better an odd number)
        Nv = 9
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

        # optical_depth = 0.05e5
        # The simulation will be done spanning -D/2 <= z <= D/2
        # zL = -0.5 * D  # left boundary of the simulation
        # zR = +0.5 * D  # right boundary of the simulation

        ######################
        # The temperature of the cell.
        Temperature = 90.0 + 273.15

        # We should be able to choose whether to keep all of data, to
        # just keep a sample at a certain rate, or to keep only the
        # current-time data.

        keep_data = "all"
        keep_data = "sample"
        # The sampling rate for the output. If sampling_rate=2 every
        # second time step will be saved in memory and returned. If
        # Nt is a multiple of sampling_rate then the length of the
        # output should be Nt/sampling_rate.
        sampling_rate = 50

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
        # sigma_power1 = 1.0e9
        sigma_power2 = 0.883494520871e9

        # We calculate the duration of the pulses from the standard deviations
        # tau1 = 2/pi * sqrt(log(2.0))/sigma_power1
        # tau2 = 2/pi * sqrt(log(2.0))/sigma_power2

        # tau1 = 2*sqrt(2)*log(2)/pi / sigma_power1
        # tau2 = 2*sqrt(2)*log(2)/pi / sigma_power2

        # The time of arrival of the beams
        t0s = 1.1801245283489222e-09
        t0w = t0s
        t0r = t0w + 3.5e-9
        alpha_rw = 1.0

        # t_cutoff = t0r+D/2/c+tau1
        t_cutoff = 3.0e-9

        ######################
        # The detuning of the signal field (in Hz):
        delta1 = -2*pi*6e9
        # The detuning of the control field (in Hz):
        delta2 = -delta1
        # This is the two-photon transition condition.
        ##################################################################
        # We choose an atom:
        element = "Cs"; isotope = 133; n_atom = 6

        # Control pulse energy.
        energy_pulse2 = 50e-12  # Joules.

    ################################################
    Omega = 1.0  # We choose the frequencies to be in radians/s.
    distance_unit = 1.0

    # The fancy units should be picked so that the factors multiplied in
    # each of the terms of the equations are of similar magnitude.

    # Ideally, the various terms should also be of similar magnitude, but
    # changing the units will not change the relative importance of terms.
    # Otherwise physics would change depending on the units!
    # However, it should be possible to choose units such that the largest
    # terms should be close to 1.

    # We set the default values of the independent parameters.
    pms = {"e_charge": e_charge,
           "hbar": hbar,
           "c": c,
           "epsilon_0": epsilon_0,
           "kB": kB,
           "Omega": Omega,
           "distance_unit": distance_unit,
           "element": element,
           "isotope": isotope,
           "Nt": Nt,
           "Nz": Nz,
           "Nv": Nv,
           "Nrho": Nrho,
           "T": T,
           "L": L,
           "sampling_rate": sampling_rate,
           "keep_data": keep_data,
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
           "alpha_rw": alpha_rw,
           "t_cutoff": t_cutoff,
           "element": element,
           "isotope": isotope,
           "verbose": verbose}

    #########################################################################
    # We replace independent parameters by custom ones if given.
    if True:
        if custom_parameters is None:
            custom_parameters = {}

        pm_names_ind = ["e_charge", "hbar", "c", "epsilon_0", "kB",
                        "Omega", "distance_unit", "element", "isotope", "Nt",
                        "Nz", "Nv", "Nrho", "T", "L", "sampling_rate",
                        "keep_data", "Temperature", "Nsigma", "delta1",
                        "sigma_power1", "sigma_power2", "w1", "w2",
                        "t0s", "t0w", "t0r",
                        "energy_pulse2", "alpha_rw", "t_cutoff",
                        "element", "isotope", "verbose",
                        "USE_HG_CTRL", "USE_HG_SIG", "USE_SB_SIG",
                        "USE_SQUARE_CTRL", "USE_SQUARE_SIG"]

        pm_names_dep = ["mass", "gamma21", "gamma32", "omega21", "omega32",
                        "omega_laser1", "omega_laser2", "delta2", "r1", "r2",
                        "ns", "nw", "nr", "tau1", "tau2", "energy_pulse1"]

        for i in custom_parameters:
            if (i not in pm_names_ind) and (i not in pm_names_dep):
                raise ValueError(str(i)+" is not a valid parameter name.")

        for name in pm_names_ind:
            if name in custom_parameters:
                pms.update({name: custom_parameters[name]})
                if type(custom_parameters[name]) is str:
                    s = name+"= '"+str(custom_parameters[name])+"'"
                else:
                    s = name+"= "+str(custom_parameters[name])
                exec(s)

    #########################################################################
    # We calculate dependent parameters
    if calculate_atom:
        from fast import State, Transition, make_list_of_states
        from fast import calculate_boundaries, Integer
        from fast import calculate_matrices
        from fast import fancy_r_plot, fancy_matrix_plot
        from fast import vapour_number_density
        from matplotlib import pyplot

        # atom = Atom(element, isotope)
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
            print "Calculating atomic properties ..."
            print "We are choosing the couplings of"
            print magnetic_states[g_index], magnetic_states[e_index],
            print magnetic_states[l_index]
            print "as a basis to estimate the values of gamma_ij, r^l."

        # We calculate the matrices for the given states.
        Omega = 1.0  # We choose the frequencies to be in radians/s.
        distance_unit = 1.0
        omega, gamma, r = calculate_matrices(magnetic_states, Omega)

        # We plot these matrices.
        path = ''; name = element+str(isotope)
        fig = pyplot.figure(); ax = fig.add_subplot(111)
        fancy_matrix_plot(ax, omega, magnetic_states, path,
                          name+'_omega.png',
                          take_abs=True, colorbar=True)
        fig = pyplot.figure(); ax = fig.add_subplot(111)
        fancy_matrix_plot(ax, gamma, magnetic_states, path,
                          name+'_gamma.png',
                          take_abs=True, colorbar=True)
        fig = pyplot.figure(); ax = fig.add_subplot(111)
        fancy_r_plot(r, magnetic_states, path, name+'_r.png',
                     complex_matrix=True)
        pyplot.close("all")

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
            mass = 2.2069469161e-25
            if ignore_lower_f:
                n_atomic0 = 4.72335166533e+18
            else:
                n_atomic0 = 8.39706962725e+18

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
    tau1 = 2*sqrt(2)*log(2)/pi / sigma_power1
    tau2 = 2*sqrt(2)*log(2)/pi / sigma_power2

    pms.update({"omega_laser1": omega_laser1, "omega_laser2": omega_laser2})

    # We make a few checks
    if pms["Nv"] == 2:
        raise ValueError("Nv = 2 is a very bad choice.")

    if pms["Nt"] % pms["sampling_rate"] != 0:
        raise ValueError("Nt must be a multiple of the sampling_rate.")

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
                "tau1": tau1,
                "tau2": tau2,
                "ns": 1,
                "nw": 1,
                "nr": 1,
                "USE_HG_CTRL": False,
                "USE_HG_SIG": False,
                "USE_SB_SIG": False})

    if "USE_SQUARE_SIG" not in custom_parameters:
        pms.update({"USE_SQUARE_SIG": False})
    if "USE_SQUARE_CTRL" not in custom_parameters:
        pms.update({"USE_SQUARE_CTRL": False})

    cond1 = "r1" not in custom_parameters
    cond2 = "r2" not in custom_parameters
    if fitted_couplings and cond1 and cond2:
        pms.update({"r1": pms["r1"]*0.2556521})
        pms.update({"r2": pms["r2"]*0.72474758})

    # We force any custom dependent parameters.
    for name in pm_names_dep:
        if name in custom_parameters:
            if pms["verbose"] >= 1:
                print "WARNING: parameter", name,
                print "may be inconsistent with independent parameters."
            pms.update({name: custom_parameters[name]})

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
                 explicit_decoherence=1.0, rabi=True):
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
    if rabi:
        const1 = np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1
    else:
        const1 = np.pi*c*epsilon_0*(w1)**2/16.0/hbar/omega_laser1

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
    Nin += (dphotons_ini_dt[0] + dphotons_ini_dt[-1])*0.5
    Nin = Nin*dt

    Ntr = sum(dphotons_out_dt_tr[1:-1])
    Ntr += (dphotons_out_dt_tr[0] + dphotons_out_dt_tr[-1])*0.5
    Ntr = Ntr*dt

    Nre = sum(dphotons_out_dt_re[1:-1])
    Nre += (dphotons_out_dt_re[0] + dphotons_out_dt_re[-1])*0.5
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
