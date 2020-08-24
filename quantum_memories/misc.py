# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2020 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Miscellaneous routines."""
import numpy as np
import warnings
from scipy.interpolate import interp1d
from numpy import sinc as normalized_sinc
from scipy.special import hermite
from scipy.misc import factorial
from sympy import log, pi
from scipy.constants import k as k_B
from scipy.constants import c


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


def glo_error(a, b, scale=None):
    r"""Get the "global" relative error between two quantities."""
    if scale is None:
        scale = np.amax([np.amax(np.abs(a)), np.amax(np.abs(b))])
    if scale == 0.0:
        return np.zeros(a.shape)
    return np.abs(a-b)/scale


def get_range(fp):
    r"""Get the range of an array."""
    fp = np.abs(fp)
    aux = fp.copy()
    aux[aux == 0] = np.amax(fp)
    vmin = np.amin(aux)

    vmax = np.amax(fp)
    return np.array([vmin, vmax])


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
# Alkali vapour routines.


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
# Finite difference miscellaneous routines.


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
