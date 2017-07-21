# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#                            2017 Benjamin Brecht                       *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************

"""This is a library for simulations of the ORCA memory [1].

The equations under the linear approximation are solved for an arbitrary number
of coupled velocity classes. For a detailed derivation of the equations see
the jupyter notebook included "Ladder memory complete equations.ipynb".

The solver is called, and it returns a solution. The solution can then
be analysed by various other functions.

References:
    [1] https://arxiv.org/abs/1704.00013
"""

import numpy as np
from math import pi, sqrt, log
from matplotlib import pyplot as plt
from matplotlib import rcParams

from misc import cheb, cDz, simple_complex_plot, set_parameters_ladder
from misc import efficiencies, vapour_number_density
from misc import Omega2_HG, Omega1_initial_HG, Omega1_boundary_HG
from scipy.constants import physical_constants
from scipy.linalg import svd
from scipy.integrate import complex_ode as ode
# from scipy.integrate import odeint
# import warnings

a0 = physical_constants["Bohr radius"][0]
# We set matplotlib to use a nice latex font.
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'


def unpack_sol(yy, Nt, Nrho, Nv, Nz):
    r"""Unpack a solution yy into meaningful variable names."""
    Om1 = yy[:, 0, :]
    rho = yy[:, 1:, :].reshape((Nt, Nrho, Nv, Nz))
    return rho, Om1


def pack_sol(rho, Om1, Nt, Nrho, Nv, Nz):
    r"""Pack solutions rho21, rho31, Om1 into an array yy."""
    rho = rho.reshape((Nt, Nrho*Nv, Nz))
    yy = np.zeros((Nt, 1+Nrho*Nv, Nz), complex)
    yy[:, 0, :] = Om1
    yy[:, 1:, :] = rho
    return yy


def unpack_slice(yyii, Nt, Nrho, Nv, Nz):
    r"""Unpack a time slice yyii into meaningful variable names."""
    Om1i = yyii[: Nz]
    rhoi = yyii[Nz:].reshape((Nrho, Nv, Nz))
    return rhoi, Om1i


def pack_slice(rhoi, Om1i, Nt, Nrho, Nv, Nz):
    r"""Pack slices rhoi, Om1i into an array yyii."""
    rhoi = rhoi.reshape((Nrho*Nv*Nz))
    yyii = np.zeros((1+Nrho*Nv)*Nz, complex)
    yyii[:Nz] = Om1i
    yyii[Nz:] = rhoi
    return yyii


def solve(params, plots=False, name="", integrate_velocities=False,
          input_signal=None):
    r"""Solve the equations for the given parameters."""
    def heaviside(x):
        return np.where(x <= 0, 0.0, 1.0) + np.where(x == 0, 0.5, 0.0)

    if True:
        e_charge = params["e_charge"]
        hbar = params["hbar"]
        c = params["c"]
        epsilon_0 = params["epsilon_0"]
        kB = params["kB"]
        Omega = params["Omega"]
        distance_unit = params["distance_unit"]
        element = params["element"]
        isotope = params["isotope"]
        Nt = params["Nt"]
        Nz = params["Nz"]
        Nv = params["Nv"]
        Nrho = params["Nrho"]
        T = params["T"]
        L = params["L"]
        sampling_rate = params["sampling_rate"]
        keep_data = params["keep_data"]
        mass = params["mass"]
        Temperature = params["Temperature"]
        Nsigma = params["Nsigma"]
        gamma21 = params["gamma21"]
        gamma32 = params["gamma32"]
        # omega21 = params["omega21"]
        # omega32 = params["omega32"]
        r1 = params["r1"]
        r2 = params["r2"]
        omega_laser1 = params["omega_laser1"]
        omega_laser2 = params["omega_laser2"]
        # n_atomic0 = params["n_atomic0"]
        delta1 = params["delta1"]
        # delta2 = params["delta2"]
        sigma_power1 = params["sigma_power1"]
        sigma_power2 = params["sigma_power2"]
        w1 = params["w1"]
        w2 = params["w2"]
        energy_pulse1 = params["energy_pulse1"]
        energy_pulse2 = params["energy_pulse2"]
        t0s = params["t0s"]
        t0w = params["t0w"]
        t0r = params["t0r"]
        alpha_rw = params["alpha_rw"]
        verbose = params["verbose"]

        #
        # Additions BB for Green's function calculations
        #

        ns = params["ns"]
        nw = params["nw"]
        nr = params["nr"]
        USE_HG_CTRL = params["USE_HG_CTRL"]
        USE_HG_SIG = params["USE_HG_SIG"]
    if input_signal is not None:
        USE_INPUT_SIGNAL = True
        USE_HG_SIG = False
    else:
        USE_INPUT_SIGNAL = False
        USE_HG_SIG = params["USE_HG_SIG"]
    ##################################################
    # We calculate the atomic density (ignoring the atoms in the lower
    #  hyperfine state)
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
    n_atomic0 = upper_fraction*n_atomic0

    ##################################################
    # We calculate the duration of the pulses from the standard deviations
    tau1 = 2*sqrt(2)*log(2)/pi / sigma_power1
    tau2 = 2*sqrt(2)*log(2)/pi / sigma_power2

    # We calculate the peak Rabi frequencies
    Omega1_peak = 4*2**(0.75)*np.sqrt(energy_pulse1)*e_charge*r1 *\
        (np.log(2.0))**(0.25)/(np.pi**(0.75)*hbar*w1*np.sqrt(c*epsilon_0*tau1))

    Omega2_peak = 4*2**(0.75)*np.sqrt(energy_pulse2)*e_charge*r2 *\
        (np.log(2.0))**(0.25)/(np.pi**(0.75)*hbar*w2*np.sqrt(c*epsilon_0*tau2))
    if verbose >= 2:
        print "The peak Rabi frequencies are"
        print Omega1_peak/2/np.pi*Omega*1e-6, "MHz,",
        print Omega2_peak/2/np.pi*Omega*1e-6, "MHz"

    ##################################################
    D = 1.05 * L
    t = np.linspace(0, T, Nt)

    # We calculate our Chebyshev matrix and mesh:
    zL = -0.5 * D
    cheb_diff_mat, cheb_mesh = cheb(Nz-1)
    cheb_diff_mat = cheb_diff_mat.T / zL
    Z = zL * cheb_mesh.T  # space axis
    t = np.linspace(0, T, Nt)
    dt = T/(Nt-1)

    ##################################################
    # We establish the needed functions:
    # n_atomic(t, Z), Omega2(t, Z), and .
    # The Maxwell-Boltzmann velocity distribution g(vZ), as given by the
    # Temperature.
    sigma_vZ = np.sqrt(kB*Temperature/mass)*1
    if Nv == 1:
        # We only consider vZ = 0.
        vZ = np.array([0.0])
    else:
        vZ = np.linspace(-Nsigma*sigma_vZ, Nsigma*sigma_vZ, Nv)

    p = np.sqrt(1/(2*np.pi))/sigma_vZ*np.exp(-(vZ/sigma_vZ)**2/2)
    # This is the continuous distribution. We can discretize it so that each
    # data point represents the probability of the atom being between
    # vZ - dvZ/2 and vZ + dvZ/2.
    # dvZ = vZ[1]-vZ[0]
    # p = p*dvZ/sum([p[i]*dvZ for i in range(Nv)])
    p = p/sum([p[i] for i in range(Nv)])
    # print sum(g)

    # The atomic density n_atomic(t, Z)
    n_atomic = n_atomic0*(-heaviside(Z - 0.5 * L) + heaviside(0.5 * L + Z))

    ##################################################
    # We initialize the solutions.
    if keep_data == "all":
        # t_sample = np.linspace(0, T, Nt)
        t_sample = t
        Nt_sample = Nt
        Om1 = np.zeros((Nt_sample, Nz), complex)
        rho = np.zeros((Nt_sample, Nrho, Nv, Nz), complex)

    elif keep_data == "sample":
        Nt_sample = Nt/sampling_rate
        t_sample = np.linspace(0, T, Nt_sample)
        Om1 = np.zeros((Nt_sample, Nz), complex)
        rho = np.zeros((Nt_sample, Nrho, Nv, Nz), complex)

    elif keep_data == "current":
        # t_sample = np.zeros(1)
        Om1 = np.zeros((1, Nz), complex)
        rho = np.zeros((1, Nrho, Nv, Nz), complex)
        # output = np.zeros(1, complex)
    else:
        raise ValueError(keep_data+" is not a valid value for keep_data.")

    ##################################################
    # The control Rabi frequency Omega(t, Z)
    def Omega2(ti, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw):
        Om2 = Omega2_peak*np.exp(
            -4*log(2.0)*(ti-t0w+Z/c)**2/tau2**2)
        Om2 += Omega2_peak*np.exp(
            -4*log(2.0)*(ti-t0r+Z/c)**2/tau2**2)*alpha_rw
        return Om2

    # We plot these functions n_atomic(t, z) and Omega2(t, z).
    if plots:
        fig, ax1 = plt.subplots()
        ax1.plot(vZ, p, "b-")
        ax1.set_xlabel(r"$ v_Z \ (\mathrm{m/s})$", fontsize=20)
        ax1.set_ylabel(r"$ p_i $", fontsize=20)
        ax1.set_ylim([0.0, None])
        ax1.set_xlim([-Nsigma*sigma_vZ*1.05, Nsigma*sigma_vZ*1.05])

        ax2 = ax1.twiny()
        doppler_delta = -omega_laser1*vZ/c*1e-9
        ax2.plot(doppler_delta, p, "r+")
        ax2.set_xlabel(r"$-\varpi_1 v_Z/c \ (\mathrm{GHz})$", fontsize=20)

        dd_lim = Nsigma*sigma_vZ*1.05*omega_laser1/c*1e-9
        ax2.set_xlim([dd_lim, -dd_lim])
        ax2.set_ylim([0.0, None])

        plt.savefig("params_vZ_"+name+".png", bbox_inches="tight")
        plt.close("all")

        plt.title(r"$\mathrm{Atomic \ density}$", fontsize=20)
        plt.plot(Z/distance_unit*100, n_atomic, "r+")
        plt.plot(Z/distance_unit*100, n_atomic, "b-")
        plt.xlabel(r"$ Z \ (\mathrm{cm})$", fontsize=20)
        plt.ylabel(r"$ n \ (\mathrm{m^{-3}})$", fontsize=20)
        plt.savefig("params_n_atomic_"+name+".png", bbox_inches="tight")
        plt.close("all")

        const2 = np.pi*c*epsilon_0*hbar*(w2/e_charge/r2)**2/16.0/omega_laser2
        if not USE_HG_CTRL:
            Om2_mesh = [Omega2(ti, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw)
                        for ti in t_sample]
        else:
            Om2_mesh = [Omega2_HG(Z, ti, sigma_power2, sigma_power2,
                                  Omega2_peak, t0w, t0r,
                                  alpha_rw, nw=nw, nr=nr) for ti in t_sample]
        Om2_mesh = np.array(Om2_mesh)
        Om2_mesh = const2*Om2_mesh**2

        cp = plt.pcolormesh(Z/distance_unit*100, t_sample*Omega*1e9,
                            Om2_mesh*1e-9)
        plt.colorbar(cp)
        plt.plot([-L/2/distance_unit*100, -L/2/distance_unit*100],
                 [0, T*Omega*1e9], "r-", linewidth=1)
        plt.plot([L/2/distance_unit*100, L/2/distance_unit*100],
                 [0, T*Omega*1e9], "r-", linewidth=1)
        plt.xlabel(r"$ Z \ (\mathrm{cm})$", fontsize=20)
        plt.ylabel(r"$ t \ (\mathrm{ns})$", fontsize=20)
        plt.savefig("params_Om2_"+name+".png", bbox_inches="tight")
        plt.close("all")

        del Om2_mesh

    ##################################################
    # We establish the boundary and initial conditions.
    Omega1_peak = 4*2**(0.75)*np.sqrt(energy_pulse1)*e_charge*r1 *\
        (np.log(2.0))**(0.25)/(np.pi**(0.75)*hbar*w1*np.sqrt(c*epsilon_0*tau1))

    if USE_HG_SIG:
        Omega1_peak_norm = (2**3*np.log(2.0)/np.pi)**(0.25)/np.sqrt(tau1)
        Omega1_boundary = Omega1_boundary_HG(t, 1*sigma_power1,
                                             Omega1_peak_norm,
                                             t0s, D, ns=ns)
        Omega1_initial = Omega1_initial_HG(Z, 1*sigma_power1, Omega1_peak_norm,
                                           t0s, ns=ns)
        dtt = t[1]-t[0]
        norm = sum([Omega1_boundary[i]*Omega1_boundary[i].conjugate()
                    for i in range(len(Omega1_boundary))])*dtt
        Omega1_boundary = Omega1_boundary/np.sqrt(norm)
        Omega1_initial = Omega1_initial/np.sqrt(norm)

    if USE_INPUT_SIGNAL:
        # We get the boundary by extending the input signal to
        # account for the sampling rate.
        Omega1_boundary = sum([[omi]*sampling_rate
                               for omi in input_signal], [])
        Omega1_boundary = np.array(Omega1_boundary)

        Omega1_initial = np.zeros(Nz, complex)
    if (not USE_HG_SIG) and (not USE_INPUT_SIGNAL):
        Omega1_boundary = Omega1_peak*np.exp(-4*np.log(2.0) *
                                             (t - t0s + D/2/c)**2/tau1**2)
        # The signal pulse at t = 0 is
        Omega1_initial = Omega1_peak*np.exp(-4*np.log(2.0) *
                                            (-t0s-Z/c)**2/tau1**2)

    Om1[0] = Omega1_initial

    # The coupling coefficient for the signal field.
    g1 = omega_laser1*(e_charge*r1)**2/(hbar*epsilon_0)
    # The detuning of the control field.
    delta2 = -delta1

    params = (delta1, delta2, gamma21, gamma32, g1,
              Omega2_peak, tau2, t0w, t0r, alpha_rw,
              p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv,
              omega_laser1, omega_laser2)

    # We define the equations that the Runge-Kutta method will solve.
    def rhs(rhoi, Om1i, ti, params):
        # We unpack the parameters.
        delta1, delta2, gamma21, gamma32, g1 = params[:5]
        Omega2_peak, tau2, t0w, t0r, alpha_rw = params[5:10]
        p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv = params[10:17]
        omega_laser1, omega_laser2 = params[17:]

        # We unpack the density matrix components.
        rho21i = rhoi[0]
        rho31i = rhoi[1]
        Nv = len(rho21i)

        # We calculate the control field at time ti.
        if not USE_HG_CTRL:
            Om2i = Omega2(ti, Z, Omega2_peak, tau2, t0w, t0r, alpha_rw)
        else:
            Om2i = Omega2_HG(Z, ti, sigma_power2, sigma_power2,
                             Omega2_peak, t0w, t0r, alpha_rw,
                             nw=nw, nr=nr)

        # We calculate the right-hand sides of equations 1 and 2 for all
        # velocity groups.
        eq1 = np.zeros((Nv, Nz), complex)
        eq2 = np.zeros((Nv, Nz), complex)
        for jj in range(Nv):
            rho21ij = rho21i[jj]
            rho31ij = rho31i[jj]

            eq1j = (1j*delta1-gamma21/2)*rho21ij
            eq1j += -1j/2.0*Om2i.conjugate()*rho31ij
            eq1j += -1j/2.0*(Om1i)
            # Doppler term:
            eq1j += -1j*vZ[jj]*omega_laser1/c*rho21ij

            eq2j = -1j/2.0*Om2i*rho21ij
            eq2j += (1j*delta1 + 1j*delta2 - gamma32/2)*rho31ij
            # Doppler terms:
            eq2j += (omega_laser2 - omega_laser1)*1j*vZ[jj]/c*rho31ij

            eq1[jj] = eq1j
            eq2[jj] = eq2j

        # We calculate the right-hand side of equation 3 taking rho21 as
        # the average of all velocity groups weighted by the p_i's. In other
        # words we use here the density matrix of the complete velocity
        # ensemble.
        rho21i_tot = sum([p[jj]*rho21i[jj] for jj in range(Nv)])

        eq3 = -1j*g1*n_atomic*rho21i_tot
        eq3 += - cDz(Om1i, c, cheb_diff_mat)

        return np.array([eq1, eq2]), eq3

    def f(ti, yyii):
        rhoi, Om1i = unpack_slice(yyii, Nt, Nrho, Nv, Nz)
        rhok, Om1k = rhs(rhoi, Om1i, ti, params)
        # We impose the boundary condition.
        # dt = t_sample[1]-t_sample[0]

        Om1k[0] = (Omega1_boundary[ii+1]-Omega1_boundary[ii])/dt
        kk = pack_slice(rhok, Om1k, Nt_sample, Nrho, Nv, Nz)
        return kk

    # def complex2real(yyii):
    #     Nvar = len(yyii)
    #     xxii = np.zeros(2*Nvar)
    #     xxii[:Nvar] = np.real(yyii)
    #     xxii[Nvar:] = np.imag(yyii)
    #     return xxii
    #
    # def real2complex(xxii):
    #     Nvar = len(xxii)/2
    #     yyii = np.zeros(Nvar, complex)
    #     yyii = xxii[:Nvar]+1j*xxii[Nvar:]
    #     return yyii
    #
    # def g(xxii, ti):
    #     print "ti", ti*1e9
    #     yyii = real2complex(xxii)
    #     kk = f(ti, yyii)
    #     return complex2real(kk)

    ii = 0
    # We carry out the Runge-Kutta method.
    ti = 0.0
    rhoii = rho[0]
    Om1ii = Om1[0]

    yyii = pack_slice(rhoii, Om1ii, Nt_sample, Nrho, Nv, Nz)

    # warnings.filterwarnings("error")

    solver = ode(f)
    # rk4  19.73
    dt = t_sample[1]-t_sample[0]
    # solver.set_integrator('lsoda', max_hnil=1000, ixpr=True)  # 10 min
    solver.set_integrator('dopri5')  # 6.66 s
    # solver.set_integrator('dop853')  # 7.7936398983 s.
    # solver.set_integrator("vode", method='bdf')  # 6.64 s
    solver.set_initial_value(yyii, ti)
    Omega1_boundary2 = Omega1_peak *\
        np.exp(-4*np.log(2.0)*(t_sample - t0s + D/2/c)**2/tau1**2)

    Omega1_boundary = Omega1_boundary2
    ii = 0
    # kkii = f(ti, yyii)
    # rhok1, Om1k1 = unpack_slice(kkii, Nt_sample, Nrho, Nv, Nz)

    # xxii = complex2real(yyii)
    # xx = odeint(g, xxii, t_sample)
    # for ii in range(len(t_sample)):
    #     rho[ii], Om1[ii] = unpack_slice(real2complex(xx[ii]),
    #                                     Nt_sample, Nrho, Nv, Nz)

    while solver.successful() and ii < Nt_sample-1:
        # We advance
        solver.integrate(solver.t+dt)
        yyii = solver.y
        rhoii, Om1ii = unpack_slice(yyii, Nt, Nrho, Nv, Nz)
        # print ii, Nt_sample, float(ii)/Nt_sample*100, solver.t*1e9
        # print Om1ii[: 2]
        # We impose the boundary condition.
        Om1ii[0] = Omega1_boundary[ii+1]
        # yyii = pack_slice(rhoii, Om1ii, Nt_sample, Nrho, Nv, Nz)

        ii += 1
        solver.t+dt
        rho[ii] = rhoii
        Om1[ii] = Om1ii

    # We plot the solution.
    if plots:
        const1 = np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1
        # We calculate the complete density matrix:
        rho_tot = sum([p[jj]*rho[:, :, jj, :] for jj in range(Nv)])
        rho21 = rho_tot[:, 0, :]
        rho31 = rho_tot[:, 1, :]

        simple_complex_plot(Z*100, t_sample*1e9, np.sqrt(const1*1e-9)*Om1,
                            "sol_Om1_"+name+".png", amount=r"\Omega_s",
                            modsquare=True)
        simple_complex_plot(Z*100, t_sample*1e9, rho21,
                            "sol_rho21_"+name+".png", amount=r"\rho_{21}")
        simple_complex_plot(Z*100, t_sample*1e9, rho31,
                            "sol_rho31_"+name+".png", amount=r"\rho_{31}")

    if integrate_velocities:
        rho_tot = sum(p[jj] * rho[:, :, jj, :] for jj in range(Nv))
        rho31 = rho_tot[:, 1, :]
        return t_sample, Z, rho31, Om1
    else:
        return t_sample, Z, vZ, rho, Om1


def efficiencies_r1r2t0w(energy_pulse2, p, explicit_decoherence=None, name="",
                         return_params=False):
    r"""Get the efficiencies for modified r1, r2, t0w."""
    # We unpack the errors.
    r1_error, r2_error, t0w_error = p

    # A name to use in the plots.
    name = "energy"+name
    # We get the default values.
    r1 = no_fit_params["r1"]
    r2 = no_fit_params["r2"]
    t0s = no_fit_params["t0s"]
    t0w = no_fit_params["t0w"]
    # print r1/a0, r2/a0, t0w-t0s

    # The modified parameters.
    r1 = r1*r1_error
    r2 = r2*r2_error
    t0w = t0s+t0w_error*1e-9

    params = set_parameters_ladder({"t0w": t0w, "r1": r1, "r2": r2,
                                    "energy_pulse2": energy_pulse2,
                                    "verbose": 0, "Nv": 1,
                                    "t_cutoff": 3.5e-9})
    if return_params:
        return params

    t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
    del rho
    # print "t_cutoff", params["t_cutoff"]
    eff_in, eff_out, eff = efficiencies(t, Om1, params, plots=True, name=name)

    # We explicitly introduce the measured decoherence.
    if explicit_decoherence is not None:
        eff_out = eff_out*explicit_decoherence
        eff = eff_in*eff_out

    return eff_in, eff_out, eff


def efficiencies_t0wenergies(p, explicit_decoherence=None, name=""):
    r"""Get the efficiencies for modified t0w-t0s, energy_write, energy_read.

    This is done while using the fitted r1, and r2.
    """
    # A name to use in the plots.
    name = "energy"+str(name)
    # We get the default values.
    r1 = default_params["r1"]
    r2 = default_params["r2"]
    t0s = default_params["t0s"]
    t0w = default_params["t0w"]
    t0r = default_params["t0r"]

    # We unpack the errors.
    # print "r1, r2:", r1/a0, r2/a0
    tmeet_error, energy_write, energy_read = p
    alpha_rw = np.sqrt(energy_read/energy_write)

    t0w = t0s+tmeet_error*1e-9
    t0r = t0w + 1.0*3.5e-9
    params = set_parameters_ladder({"t0w": t0w, "t0r": t0r,
                                    "r1": r1, "r2": r2,
                                    "energy_pulse2": energy_write,
                                    "alpha_rw": alpha_rw,
                                    "t_cutoff": 3.5e-9,
                                    "Nv": 1,
                                    "verbose": 0})

    t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
    del rho
    eff_in, eff_out, eff = efficiencies(t, Om1, params, plots=True, name=name)

    # We explicitly introduce the measured decoherence.
    if explicit_decoherence is not None:
        eff_out = eff_out*explicit_decoherence
        eff = eff_in*eff_out

    return eff_in, eff_out, eff


def num_integral(t, f):
    """We integrate using the trapezium rule."""
    dt = t[1]-t[0]
    F = sum(f[1:-1])
    F += (f[1] + f[-1])*0.5
    return np.real(F*dt)


def normalization(t, f, params):
    """Get the normalization constant for const1*|f|^2 to integrate to one."""
    const1 = photons_const(params)
    Nin = num_integral(t, const1*f*f.conjugate())
    return np.sqrt(np.abs(Nin))


def photons_const(params):
    """Get the constant to translate to photon number."""
    e_charge = params["e_charge"]
    hbar = params["hbar"]
    c = params["c"]
    epsilon_0 = params["epsilon_0"]
    r1 = params["r1"]
    omega_laser1 = params["omega_laser1"]
    w1 = params["w1"]
    return np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1


def rescale_input(t, mode, params):
    r"""Rescale an input mode so that its mod square integral is one photon."""
    return mode/normalization(t, mode, params)


def bra(v):
    """"Get a bra from an array."""
    return v.reshape((1, len(v))).conjugate()


def ket(v):
    """"Get a ket from an array."""
    return v.reshape((len(v), 1))


def rel_error(a, b):
    r"""Get the relative error between two quantities."""
    m = max([abs(a), abs(b)])
    n = min([abs(a), abs(b)])
    return 1 - float(n)/m


def greens(params, index, Nhg=5):
    r"""Calculate the Green's function using params."""
    # We build the Green's function.
    t_cutoff = params["t_cutoff"]

    t_sample = np.linspace(0, params["T"],
                           params["Nt"]/params["sampling_rate"])
    Nt = len(t_sample); dt = t_sample[1]-t_sample[0]
    t_out = np.array([t_sample[i] for i in range(Nt)
                      if t_sample[i] > t_cutoff])
    Ntout = len(t_out)

    phi = []; psi = []
    Gri = np.zeros((Ntout, Nt), complex)
    print "The size of Green's function", Gri.shape
    Nhg = 15
    Kprev = 1e6
    for ii in range(Nhg):
        print ("Mode order %i" % ii)
        params["ns"] = ii
        # We solve for the Hermite Gauss mode of order 0.
        aux = solve(params, integrate_velocities=True)
        t_sample, Z, rho31, Om1 = aux
        Nt = len(t_sample); dt = t_sample[1]-t_sample[0]

        # Extract input and output.
        Om1_in = Om1[:, 0]
        Om1_out = np.array([Om1[i, -1] for i in range(Nt)
                            if t_sample[i] > t_cutoff])
        Ntout = len(Om1_out)
        Gri += ket(Om1_out).dot(bra(Om1_in))*dt
        phi += [Om1_in]
        psi += [Om1_out]

        if ii >= 2:
            U, D, V = svd(Gri)
            DD = D/np.sqrt(sum(D**2))
            K = 1.0/(DD**4).sum()
            K_change = rel_error(K, Kprev)

            check = K_change <= 0.01
            print ii, K, Kprev, K_change, check
            if check:
                break
            Kprev = K
            # check = ii >= 8*K
            # print "check:", check, ii, 8*K, K, D[: 5]
            # if check:
            #     break
    # U, D, V = svd(Gri)

    print "Checking Green's function..."
    plt.close("all")
    Nhg = ii
    plt.figure()
    for i in range(Nhg):
        # print ".............."
        Nin = num_integral(t_sample, phi[i]*phi[i].conjugate())
        Nout = num_integral(t_out, psi[i]*psi[i].conjugate())
        psi_cal = Gri.dot(ket(phi[i])).reshape(Ntout)
        Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
        print i, Nin, Nout, Ncal
        plt.subplot(211)
        plt.plot(t_sample, np.abs(phi[i]*phi[i].conjugate()), label=str(i))
        plt.subplot(212)
        plt.plot(t_out, np.abs(psi[i]*psi[i].conjugate()), label=str(i))
    plt.subplot(211)
    plt.legend()
    plt.subplot(212)
    plt.legend()
    plt.savefig("a"+str(index)+".png")
    plt.close("all")

    print "testing singular modes..."
    plt.figure()
    for ii in range(Nhg):
        # print ".............."
        dt = t_sample[1]-t_sample[0]
        phii = V[ii, :].conjugate()/np.sqrt(dt)
        Nin = num_integral(t_sample, phii*phii.conjugate())
        Nout = D[ii]**2
        psi_cal = Gri.dot(ket(phii)).reshape(Ntout)
        Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
        print ii, Nin, Nout, Ncal
        plt.subplot(211)
        plt.plot(t_sample, 1e-9*np.abs(phii*phii.conjugate()),
                 label=str(ii))
        plt.subplot(212)
        plt.plot(t_out, np.abs(U[:, ii]*U[:, ii].conjugate()),
                 label=str(ii))
    plt.subplot(211)
    plt.legend()
    plt.subplot(212)
    plt.legend()
    plt.savefig("b"+str(index)+".png")
    plt.close("all")

    # Vhg = V
    # # We make the Green's function converge.
    # print "Using single modes..."
    # Gri = np.zeros((Ntout, Nt), complex)
    # Nmodes = Nhg
    # phi = []; psi = []
    # for ii in range(Nmodes):
    #     print ("Mode order %i" % ii)
    #     params["ns"] = ii
    #     norm = num_integral(t_sample, Vhg[ii, :]*Vhg[ii, :].conjugate())
    #     phi_i = Vhg[ii, :] / np.sqrt(np.real(norm))
    #     # We solve for the Hermite Gauss mode of order 0.
    #     aux = solve(params, integrate_velocities=True,
    #                 input_signal=phi_i)
    #     t_sample, Z, rho31, Om1 = aux
    #     Nt = len(t_sample); dt = t_sample[1]-t_sample[0]
    #
    #     # Extract input and output.
    #     Om1_in = Om1[:, 0]
    #     Om1_out = np.array([Om1[i, -1] for i in range(Nt)
    #                         if t_sample[i] > t_cutoff])
    #     Ntout = len(Om1_out)
    #     Gri += ket(Om1_out).dot(bra(Om1_in))*dt
    #     phi += [Om1_in]
    #     psi += [Om1_out]
    #     # if ii >= 4:
    #     #     U, D, V = svd(Gri)
    #     #     DD = D/np.sqrt(sum(D**2))
    #     #     K = 1.0/(DD**4).sum()
    #     #     check = ii >= 3*K
    #     #     print "check:", check, ii, 3*K, K, D[: 5]
    #     #     if check:
    #     #         break
    #
    # Nhg = ii
    # # We check that the Green's function does its job.
    # print "Checking Green's function..."
    #
    # plt.close("all")
    # plt.figure()
    # for i in range(Nmodes):
    #     # print ".............."
    #     Nin = num_integral(t_sample, phi[i]*phi[i].conjugate())
    #     Nout = num_integral(t_out, psi[i]*psi[i].conjugate())
    #     psi_cal = Gri.dot(ket(phi[i])).reshape(Ntout)
    #     Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
    #     print i, Nin, Nout, Ncal
    #     plt.subplot(211)
    #     plt.plot(t_sample, np.abs(phi[i]*phi[i].conjugate()), label=str(i))
    #     plt.subplot(212)
    #     plt.plot(t_out, np.abs(psi[i]*psi[i].conjugate()), label=str(i))
    # plt.subplot(211)
    # plt.legend()
    # plt.subplot(212)
    # plt.legend()
    # plt.savefig("b"+str(index)+".png")
    # plt.close("all")

    return Gri, t_sample, t_out


def optimize_signal(params, index, Nhg=5, plots=False, check=False,
                    name="optimal"):
    """Get the optimal signal modes and total efficiency."""
    Nhg = 25
    Gri, t_sample, t_out = greens(params, index, Nhg)
    Ntout = len(t_out)
    const1 = photons_const(params)

    U, D, V = svd(Gri)

    # We extract the optimal modes.
    optimal_input = rescale_input(t_sample, V[0, :].conjugate(), params)
    # print np.amax(np.real(V[0, :]*V[0, :].conjugate())),
    # print np.amax(np.real(optimal_input*optimal_input.conjugate()))
    ##########################################################################
    # We check that the Green's function actually returns the expected thing.
    if check:
        Om1_in_actual = rescale_input(t_sample, optimal_input, params)
        t_sample, Z, rho31, Om1 = solve(params, plots=True,
                                        name=name,
                                        integrate_velocities=True,
                                        input_signal=Om1_in_actual)

        GOm1_in_actual = Gri.dot(ket(Om1_in_actual)).reshape(Ntout)

        eff_cal = num_integral(t_out, const1 *
                               GOm1_in_actual*GOm1_in_actual.conjugate())

        eff_in, eff_out, eff = efficiencies(t_sample, Om1, params,
                                            plots=True, name=name+str(index))
        # DD = D/np.sqrt(sum(D**2))
        print "The SVD-calculated efficiency is", D[0]**2
        print "The Green's function-calculated efficiency is", eff_cal
        print "The actual efficiency is", eff
    ##########################################################################
    if plots:
        # Plotting.
        T, S = np.meshgrid(t_sample*1e9, t_out*1e9)
        plt.figure()
        cs = plt.contourf(T, S, abs(Gri)**2, 256)
        plt.tight_layout()
        plt.savefig("Greens"+str(index)+".png", bbox_inches="tight")
        plt.colorbar(cs)
        plt.xlabel(r"$t \ \mathrm{(ns)}$")
        plt.ylabel(r"$t \ \mathrm{(ns)}$")
        plt.close("all")

        # We plot the one-photon modes.
        ii = 0
        plt.figure()
        for ii in range(6):
            if ii == 0:
                label_out = "Optimal output"
                label_in = "Optimal input"
            else:
                label_out = "output mode "+str(ii)
                label_in = "intput mode "+str(ii)
            input_ii = rescale_input(t_sample, V[ii, :], params)
            output_ii = Gri.dot(ket(input_ii)).reshape(Ntout)

            plt.subplot(212)
            plt.plot(t_out*1e9, const1*np.abs(output_ii)**2*1e-9,
                     label=label_out)

            plt.subplot(211)
            plt.plot(t_sample*1e9, const1*np.abs(input_ii)**2*1e-9,
                     label=label_in)

        plt.subplot(211)
        plt.ylabel(r"$\mathrm{photons/ns}$", fontsize=15)
        plt.legend()
        # plt.ylim([0, 3])
        plt.subplot(212)
        plt.xlabel(r"$t \ (\mathrm{ns})$", fontsize=15)
        plt.ylabel(r"$\mathrm{photons/ns}$", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig("singular_modes.png", bbox_inches="tight")
        plt.close("all")
        # plt.show()

        plt.figure()
        plt.subplot(121)
        plt.bar(np.arange(5), D[:5])
        plt.subplot(122)
        plt.bar(np.arange(5), (D[:5])**2)
        plt.tight_layout()
        plt.savefig("singular_values.png", bbox_inches="tight")
        plt.close("all")

    ##########################################################################
    # The return.
    optimal_output = Gri.dot(ket(optimal_input)).reshape(Ntout)
    optimal_efficiency = D[0]**2
    # optimal_efficiency = eff_cal
    return optimal_input, optimal_output, optimal_efficiency, eff


no_fit_params = set_parameters_ladder(fitted_couplings=False)
default_params = set_parameters_ladder()
