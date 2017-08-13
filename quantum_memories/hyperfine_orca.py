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

from misc import cheb, cDz, set_parameters_ladder
from misc import simple_complex_plot
from misc import vapour_number_density
from misc import Omega2_HG, Omega1_initial_HG, Omega1_boundary_HG
from misc import efficiencies
from scipy.constants import physical_constants
# from scipy.linalg import svd
from scipy.integrate import complex_ode as ode
# from scipy.integrate import odeint
# import warnings

a0 = physical_constants["Bohr radius"][0]
# We set matplotlib to use a nice latex font.
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'


# def unpack_sol(yy, Nt, Nrho, Nv, Nz):
#     r"""Unpack a solution yy into meaningful variable names."""
#     Om1 = yy[:, 0, :]
#     rho = yy[:, 1:, :].reshape((Nt, Nrho, Nv, Nz))
#     return rho, Om1
#
#
# def pack_sol(rho, Om1, Nt, Nrho, Nv, Nz):
#     r"""Pack solutions rho21, rho31, Om1 into an array yy."""
#     rho = rho.reshape((Nt, Nrho*Nv, Nz))
#     yy = np.zeros((Nt, 1+Nrho*Nv, Nz), complex)
#     yy[:, 0, :] = Om1
#     yy[:, 1:, :] = rho
#     return yy


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
        Nrho = 20
        T = params["T"]
        L = params["L"]
        sampling_rate = params["sampling_rate"]
        sampling_rate = 50
        keep_data = params["keep_data"]
        mass = params["mass"]
        Temperature = params["Temperature"]
        Nsigma = params["Nsigma"]
        # gamma21 = params["gamma21"]
        # gamma32 = params["gamma32"]
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

        # We establish the hyperfine structure constants.
        r_1_3_1 = 1.11803398874989*r1
        r_1_4_1 = -1.14564392373896*r1
        r_1_4_2 = 0.661437827766148*r1
        r_1_5_1 = 0.968245836551854*r1
        r_1_5_2 = -1.14564392373896*r1
        r_1_6_2 = 1.6583123951777*r1

        r_2_7_6 = 1.47196014438797*r2
        r_2_8_5 = 1.14891252930761*r2
        r_2_8_6 = -0.716472842006823*r2
        r_2_9_4 = 0.82915619758885*r2
        r_2_9_5 = -0.861684396980704*r2
        r_2_9_6 = 0.264575131106459*r2
        r_2_10_3 = 0.5*r2
        r_2_10_4 = -0.853912563829966*r2
        r_2_10_5 = 0.433012701892219*r2
        r_2_11_3 = -0.707106781186548*r2
        r_2_11_4 = 0.577350269189626*r2
        r_2_12_3 = 0.707106781186548*r2

        gamma_3_1 = 32886191.8978
        gamma_4_1 = 24664643.9233
        gamma_4_2 = 8221547.97444
        gamma_5_1 = 13702579.9574
        gamma_5_2 = 19183611.9404
        gamma_6_2 = 32886191.8978

        gamma_7_6 = 14878582.8074
        gamma_8_5 = 10712579.6213
        gamma_8_6 = 4166003.18607
        gamma_9_4 = 6819350.45339
        gamma_9_5 = 7364898.48966
        gamma_9_6 = 694333.864345
        gamma_10_3 = 3188267.74444
        gamma_10_4 = 9299114.25463
        gamma_10_5 = 2391200.80833
        gamma_11_3 = 8927149.68444
        gamma_11_4 = 5951433.12296
        gamma_12_3 = 14878582.8074

        omega_1 = 0.0
        omega_2 = 57759008871.6
        omega_3 = 2.20998822144e+15
        omega_4 = 2.20998917161e+15
        omega_5 = 2.20999043634e+15
        omega_6 = 2.20999201399e+15
        omega_7 = 4.26305337164e+15
        omega_8 = 4.26305354441e+15
        omega_9 = 4.26305369061e+15
        omega_10 = 4.26305380902e+15
        omega_11 = 4.26305389868e+15
        omega_12 = 4.26305395885e+15
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

    E01_peak = 4*2**(0.75)*np.sqrt(energy_pulse1) *\
        (np.log(2.0))**(0.25)/(np.pi**(0.75)*w1*np.sqrt(c*epsilon_0*tau1))

    E02_peak = 4*2**(0.75)*np.sqrt(energy_pulse2) *\
        (np.log(2.0))**(0.25)/(np.pi**(0.75)*w2*np.sqrt(c*epsilon_0*tau2))

    if verbose >= 2:
        print "The peak field amplitudes are"
        print E01_peak/2/np.pi*Omega*1e-6, "MHz,",
        print E02_peak/2/np.pi*Omega*1e-6, "MHz"

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
        E01 = np.zeros((Nt_sample, Nz), complex)
        rho = np.zeros((Nt_sample, Nrho, Nv, Nz), complex)
    elif keep_data == "sample":
        Nt_sample = Nt/sampling_rate
        t_sample = np.linspace(0, T, Nt_sample)
        E01 = np.zeros((Nt_sample, Nz), complex)
        rho = np.zeros((Nt_sample, Nrho, Nv, Nz), complex)
    elif keep_data == "current":
        # t_sample = np.zeros(1)
        E01 = np.zeros((1, Nz), complex)
        rho = np.zeros((1, Nrho, Nv, Nz), complex)
        # output = np.zeros(1, complex)
    else:
        raise ValueError(keep_data+" is not a valid value for keep_data.")

    # ##################################################
    # The control Rabi frequency Omega(t, Z)
    def Omega2(ti, Z, E02_peak, tau2, t0w, t0r, alpha_rw):
        Om2 = E02_peak*np.exp(
            -4*log(2.0)*(ti-t0w+Z/c)**2/tau2**2)
        Om2 += E02_peak*np.exp(
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
            E02_mesh = [Omega2(ti, Z, E02_peak, tau2, t0w, t0r, alpha_rw)
                        for ti in t_sample]
        else:
            E02_mesh = [Omega2_HG(Z, ti, sigma_power2, sigma_power2,
                                  E02_peak, t0w, t0r,
                                  alpha_rw, nw=nw, nr=nr) for ti in t_sample]
        E02_mesh = np.array(E02_mesh)
        E02_mesh = const2*E02_mesh**2

        cp = plt.pcolormesh(Z/distance_unit*100, t_sample*Omega*1e9,
                            E02_mesh*1e-9)
        plt.colorbar(cp)
        plt.plot([-L/2/distance_unit*100, -L/2/distance_unit*100],
                 [0, T*Omega*1e9], "r-", linewidth=1)
        plt.plot([L/2/distance_unit*100, L/2/distance_unit*100],
                 [0, T*Omega*1e9], "r-", linewidth=1)
        plt.xlabel(r"$ Z \ (\mathrm{cm})$", fontsize=20)
        plt.ylabel(r"$ t \ (\mathrm{ns})$", fontsize=20)
        plt.savefig("params_Om2_"+name+".png", bbox_inches="tight")
        plt.close("all")

        del E02_mesh

    ##################################################
    # We establish the boundary and initial conditions.

    if USE_HG_SIG:
        E01_peak_norm = (2**3*np.log(2.0)/np.pi)**(0.25)/np.sqrt(tau1)
        E01_boundary = Omega1_boundary_HG(t, 1*sigma_power1,
                                          E01_peak_norm,
                                          t0s, D, ns=ns)
        E01_initial = Omega1_initial_HG(Z, 1*sigma_power1,
                                        E01_peak_norm, t0s, ns=ns)
        dtt = t[1]-t[0]
        norm = sum([E01_boundary[i]*E01_boundary[i].conjugate()
                    for i in range(len(E01_boundary))])*dtt
        E01_boundary = E01_boundary/np.sqrt(norm)
        E01_initial = E01_initial/np.sqrt(norm)
    if USE_INPUT_SIGNAL:
        # We get the boundary by extending the input signal to
        # account for the sampling rate.
        E01_boundary = sum([[omi]*sampling_rate
                            for omi in input_signal], [])
        E01_boundary = np.array(E01_boundary)

        E01_initial = np.zeros(Nz, complex)
    if (not USE_HG_SIG) and (not USE_INPUT_SIGNAL):
        E01_boundary = E01_peak*np.exp(-4*np.log(2.0) *
                                       (t - t0s + D/2/c)**2/tau1**2)
        # The signal pulse at t = 0 is
        E01_initial = E01_peak*np.exp(-4*np.log(2.0)*(-t0s-Z/c)**2/tau1**2)

    E01[0] = E01_initial

    # The coupling coefficient for the signal field.
    g1 = omega_laser1*(e_charge*r1)**2/(hbar*epsilon_0)
    # The detuning of the control field.
    delta2 = -delta1

    params = (delta1, delta2, g1,
              E02_peak, tau2, t0w, t0r, alpha_rw,
              p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv,
              omega_laser1, omega_laser2,
              gamma_3_1, gamma_4_1, gamma_4_2, gamma_5_1, gamma_5_2,
              gamma_6_2, gamma_7_6, gamma_8_5, gamma_8_6, gamma_9_4,
              gamma_9_5, gamma_9_6, gamma_10_3, gamma_10_4, gamma_10_5,
              gamma_11_3, gamma_11_4, gamma_12_3,
              r_1_3_1, r_1_4_1, r_1_4_2, r_1_5_1, r_1_5_2, r_1_6_2, r_2_7_6,
              r_2_8_5, r_2_8_6, r_2_9_4, r_2_9_5, r_2_9_6, r_2_10_3,
              r_2_10_4, r_2_10_5, r_2_11_3, r_2_11_4, r_2_12_3,
              omega_1, omega_2, omega_3, omega_4,
              omega_5, omega_6, omega_7, omega_8,
              omega_9, omega_10, omega_11, omega_12)

    # We define the equations that the Runge-Kutta method will solve.
    def rhs(rhoi, E01i, ti, params):
        # We unpack the parameters.
        delta1, delta2, g1 = params[:3]
        E02_peak, tau2, t0w, t0r, alpha_rw = params[3:8]
        p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv = params[8:15]
        omega_laser1, omega_laser2 = params[15:17]
        gamma_3_1, gamma_4_1, gamma_4_2, gamma_5_1 = params[17:21]
        gamma_5_2, gamma_6_2, gamma_7_6, gamma_8_5 = params[21:25]
        gamma_8_6, gamma_9_4, gamma_9_5, gamma_9_6 = params[25:29]
        gamma_10_3, gamma_10_4, gamma_10_5, gamma_11_3 = params[29:33]
        gamma_11_4, gamma_12_3 = params[33:35]
        r_1_3_1, r_1_4_1, r_1_4_2, r_1_5_1, r_1_5_2 = params[35: 40]
        r_1_6_2, r_2_7_6, r_2_8_5, r_2_8_6, r_2_9_4 = params[40: 45]
        r_2_9_5, r_2_9_6, r_2_10_3, r_2_10_4, r_2_10_5 = params[45: 50]
        r_2_11_3, r_2_11_4, r_2_12_3 = params[50: 53]
        omega_level_1, omega_level_2, omega_level_3 = params[53: 56]
        omega_level_4, omega_level_5, omega_level_6 = params[56: 59]
        omega_level_7, omega_level_8, omega_level_9 = params[59: 62]
        omega_level_10, omega_level_11, omega_level_12 = params[62:65]

        mu0 = 1.2566370614359173e-06

        Nv = len(rhoi[0])

        # We calculate the control field at time ti.
        if not USE_HG_CTRL:
            Om2i = Omega2(ti, Z, E02_peak, tau2, t0w, t0r, alpha_rw)
        else:
            Om2i = Omega2_HG(Z, ti, sigma_power2, sigma_power2,
                             E02_peak, t0w, t0r, alpha_rw,
                             nw=nw, nr=nr)

        omega_laser1 = delta1+(omega_3-omega_2)
        # We calculate the right-hand sides of equations 1 and 2 for all
        # velocity groups.
        delta1_0 = delta1
        delta2_0 = delta2
        eqs_rho = np.zeros((Nrho, Nv, Nz), complex)
        for jj in range(Nv):

            delta1 = delta1_0 + omega_laser1*vZ[jj]/c
            delta2 = delta2_0 - omega_laser2*vZ[jj]/c
            # rho_3_1 = rhoi[0, jj]
            # rho_4_1 = rhoi[1, jj]
            # rho_5_1 = rhoi[2, jj]
            # rho_6_1 = rhoi[3, jj]
            # rho_7_1 = rhoi[4, jj]
            # rho_8_1 = rhoi[5, jj]
            # rho_9_1 = rhoi[6, jj]
            # rho_10_1 = rhoi[7, jj]
            # rho_11_1 = rhoi[8, jj]
            # rho_12_1 = rhoi[9, jj]
            rho_3_2 = rhoi[10, jj]
            rho_4_2 = rhoi[11, jj]
            rho_5_2 = rhoi[12, jj]
            rho_6_2 = rhoi[13, jj]
            rho_7_2 = rhoi[14, jj]
            rho_8_2 = rhoi[15, jj]
            rho_9_2 = rhoi[16, jj]
            rho_10_2 = rhoi[17, jj]
            rho_11_2 = rhoi[18, jj]
            rho_12_2 = rhoi[19, jj]

            fact = 1j*e_charge/hbar
            E01 = E01i
            E02 = Om2i
            E02c = E02.conjugate()

            # The equations:
            # eqs_rho[0, jj] = fact*(-r_2_10_3*rho_10_1*E02c -
            #                        r_2_11_3*rho_11_1*E02c -
            #                        r_2_12_3*rho_12_1*E02c) +\
            #     (-gamma_3_1*0.5 + 1j*delta1 + 1j*omega_1 - 1j*omega_2) *\
            #     rho_3_1
            #
            # eqs_rho[1, jj] = fact*(-r_2_10_4*rho_10_1*E02c -
            #                        r_2_11_4*rho_11_1*E02c -
            #                        r_2_9_4*rho_9_1*E02c) +\
            #     (-gamma_4_1*0.5 - gamma_4_2*0.5 + 1j*delta1 + 1j*omega_1 -
            #      1j*omega_2 + 1j*omega_3 - 1j*omega_4)*rho_4_1
            #
            # eqs_rho[2, jj] = fact*(-r_2_10_5*rho_10_1*E02c -
            #                        r_2_8_5*rho_8_1*E02c -
            #                        r_2_9_5*rho_9_1*E02c) +\
            #     (-gamma_5_1*0.5 - gamma_5_2*0.5 + 1j*delta1 + 1j*omega_1 -
            #      1j*omega_2 + 1j*omega_3 - 1j*omega_5)*rho_5_1
            #
            # eqs_rho[3, jj] = fact*(-r_2_7_6*rho_7_1*E02c -
            #                        r_2_8_6*rho_8_1*E02c -
            #                        r_2_9_6*rho_9_1*E02c) +\
            #     (-gamma_6_2*0.5 + 1j*delta1 + 1j*omega_1 - 1j*omega_2 +
            #      1j*omega_3 - 1j*omega_6)*rho_6_1
            #
            # eqs_rho[4, jj] = -fact*r_2_7_6*E02*rho_6_1 +\
            #     (-gamma_7_6*0.5 + 1j*delta1 + 1j*delta2 + 1j*omega_1 -
            #      1j*omega_2 + 1j*omega_3 - 1j*omega_6)*rho_7_1
            #
            # eqs_rho[5, jj] = fact*(-r_2_8_5*E02*rho_5_1 -
            #                        r_2_8_6*E02*rho_6_1) +\
            #     (-gamma_8_5*0.5 - gamma_8_6*0.5 + 1j*delta1 + 1j*delta2 +
            #      1j*omega_1 - 1j*omega_2 + 1j*omega_3 - 1j*omega_6 +
            #      1j*omega_7 - 1j*omega_8)*rho_8_1
            #
            # eqs_rho[6, jj] = fact*(-r_2_9_4*E02*rho_4_1 -
            #                        r_2_9_5*E02*rho_5_1 -
            #                        r_2_9_6*E02*rho_6_1) +\
            #     (-gamma_9_4*0.5 - gamma_9_5*0.5 - gamma_9_6*0.5 + 1j*delta1 +
            #      1j*delta2 + 1j*omega_1 - 1j*omega_2 + 1j*omega_3 -
            #      1j*omega_6 + 1j*omega_7 - 1j*omega_9)*rho_9_1
            #
            # eqs_rho[7, jj] = fact*(-r_2_10_3*E02*rho_3_1 -
            #                        r_2_10_4*E02*rho_4_1 -
            #                        r_2_10_5*E02*rho_5_1) +\
            #     (-gamma_10_3*0.5 - gamma_10_4*0.5 - gamma_10_5*0.5 +
            #      1j*delta1 + 1j*delta2 + 1j*omega_1 - 1j*omega_10 -
            #      1j*omega_2 + 1j*omega_3 - 1j*omega_6 + 1j*omega_7)*rho_10_1
            #
            # eqs_rho[8, jj] = fact*(-r_2_11_3*E02*rho_3_1 -
            #                        r_2_11_4*E02*rho_4_1) +\
            #     (-gamma_11_3*0.5 - gamma_11_4*0.5 + 1j*delta1 + 1j*delta2 +
            #      1j*omega_1 - 1j*omega_11 - 1j*omega_2 + 1j*omega_3 -
            #      1j*omega_6 + 1j*omega_7)*rho_11_1
            #
            # eqs_rho[9, jj] = -fact*r_2_12_3*E02*rho_3_1 +\
            #     (-gamma_12_3*0.5 + 1j*delta1 + 1j*delta2 + 1j*omega_1 -
            #      1j*omega_12 - 1j*omega_2 + 1j*omega_3 - 1j*omega_6 +
            #      1j*omega_7)*rho_12_1

            eqs_rho[10, jj] = fact*(-r_2_10_3*rho_10_2*E02c -
                                    r_2_11_3*rho_11_2*E02c -
                                    r_2_12_3*rho_12_2*E02c) +\
                (-gamma_3_1*0.5 + 1j*delta1)*rho_3_2

            eqs_rho[11, jj] = fact*(-r_1_4_2*E01 -
                                    r_2_10_4*rho_10_2*E02c -
                                    r_2_11_4*rho_11_2*E02c -
                                    r_2_9_4*rho_9_2*E02c) +\
                (-gamma_4_1*0.5 - gamma_4_2*0.5 + 1j*delta1 + 1j*omega_3 -
                 1j*omega_4)*rho_4_2

            eqs_rho[12, jj] = fact*(-r_1_5_2*E01 -
                                    r_2_10_5*rho_10_2*E02c -
                                    r_2_8_5*rho_8_2*E02c -
                                    r_2_9_5*rho_9_2*E02c) +\
                (-gamma_5_1*0.5 - gamma_5_2*0.5 + 1j*delta1 + 1j*omega_3 -
                 1j*omega_5)*rho_5_2

            eqs_rho[13, jj] = fact*(-r_1_6_2*E01 -
                                    r_2_7_6*rho_7_2*E02c -
                                    r_2_8_6*rho_8_2*E02c -
                                    r_2_9_6*rho_9_2*E02c) +\
                (-gamma_6_2*0.5 + 1j*delta1 + 1j*omega_3 - 1j*omega_6)*rho_6_2

            eqs_rho[14, jj] = -fact*r_2_7_6*E02*rho_6_2 +\
                (-gamma_7_6*0.5 + 1j*delta1 + 1j*delta2 + 1j*omega_3 -
                 1j*omega_6)*rho_7_2

            eqs_rho[15, jj] = fact*(-r_2_8_5*E02*rho_5_2 -
                                    r_2_8_6*E02*rho_6_2) +\
                (-gamma_8_5*0.5 - gamma_8_6*0.5 + 1j*delta1 + 1j*delta2 +
                 1j*omega_3 - 1j*omega_6 + 1j*omega_7 - 1j*omega_8)*rho_8_2

            eqs_rho[16, jj] = fact*(-r_2_9_4*E02*rho_4_2 -
                                    r_2_9_5*E02*rho_5_2 -
                                    r_2_9_6*E02*rho_6_2) +\
                (-gamma_9_4*0.5 - gamma_9_5*0.5 - gamma_9_6*0.5 + 1j*delta1 +
                 1j*delta2 + 1j*omega_3 - 1j*omega_6 + 1j*omega_7 -
                 1j*omega_9)*rho_9_2

            eqs_rho[17, jj] = fact*(-r_2_10_3*E02*rho_3_2 -
                                    r_2_10_4*E02*rho_4_2 -
                                    r_2_10_5*E02*rho_5_2) +\
                (-gamma_10_3*0.5 - gamma_10_4*0.5 - gamma_10_5*0.5 +
                 1j*delta1 + 1j*delta2 - 1j*omega_10 + 1j*omega_3 -
                 1j*omega_6 + 1j*omega_7)*rho_10_2

            eqs_rho[18, jj] = fact*(-r_2_11_3*E02*rho_3_2 -
                                    r_2_11_4*E02*rho_4_2) +\
                (-gamma_11_3*0.5 - gamma_11_4*0.5 + 1j*delta1 + 1j*delta2 -
                 1j*omega_11 + 1j*omega_3 - 1j*omega_6 + 1j*omega_7)*rho_11_2

            eqs_rho[19, jj] = -fact*r_2_12_3*E02*rho_3_2 +\
                (-gamma_12_3*0.5 + 1j*delta1 + 1j*delta2 - 1j*omega_12 +
                 1j*omega_3 - 1j*omega_6 + 1j*omega_7)*rho_12_2

        # We calculate the right-hand side of of the E01 equation
        # taking rho21 as the average of all velocity groups weighted by
        # the p_i's. In other words we use here the density matrix of the
        # complete velocity ensemble.
        # rho_3_1_tot = sum([p[jj]*rhoi[0, jj] for jj in range(Nv)])
        # rho_4_1_tot = sum([p[jj]*rhoi[1, jj] for jj in range(Nv)])
        # rho_5_1_tot = sum([p[jj]*rhoi[2, jj] for jj in range(Nv)])
        rho_4_2_tot = sum([p[jj]*rhoi[11, jj] for jj in range(Nv)])
        rho_5_2_tot = sum([p[jj]*rhoi[12, jj] for jj in range(Nv)])
        rho_6_2_tot = sum([p[jj]*rhoi[13, jj] for jj in range(Nv)])

        eq_E01 = -(r_1_4_2*rho_4_2_tot +
                   r_1_5_2*rho_5_2_tot +
                   r_1_6_2*rho_6_2_tot) *\
            1j*c**2*e_charge*mu0*omega_laser1*n_atomic

        # eq_E01 = np.zeros(Nz, complex)
        eq_E01 += - cDz(E01i, c, cheb_diff_mat)

        return eqs_rho, eq_E01

    def f(ti, yyii):
        rhoi, E01i = unpack_slice(yyii, Nt, Nrho, Nv, Nz)
        rhok, E01k = rhs(rhoi, E01i, ti, params)
        # We impose the boundary condition.
        # dt = t_sample[1]-t_sample[0]

        E01k[0] = (E01_boundary[ii+1]-E01_boundary[ii])/dt
        kk = pack_slice(rhok, E01k, Nt_sample, Nrho, Nv, Nz)
        return kk

    ii = 0
    # We carry out the Runge-Kutta method.
    ti = 0.0
    rhoii = rho[0]
    E01ii = E01[0]

    yyii = pack_slice(rhoii, E01ii, Nt_sample, Nrho, Nv, Nz)

    # warnings.filterwarnings("error")

    solver = ode(f)
    # rk4  19.73
    dt = t_sample[1]-t_sample[0]
    # solver.set_integrator('lsoda', max_hnil=1000, ixpr=True)  # 10 min
    solver.set_integrator('dopri5')  # 6.66 s
    # solver.set_integrator('dop853', nsteps=10000)  # 7.7936398983 s.
    # solver.set_integrator("vode", method='bdf')  # 6.64 s
    solver.set_initial_value(yyii, ti)
    E01_boundary2 = E01_peak *\
        np.exp(-4*np.log(2.0)*(t_sample - t0s + D/2/c)**2/tau1**2)

    E01_boundary = E01_boundary2
    ii = 0
    kkii = f(ti, yyii)
    rhok1, Om1k1 = unpack_slice(kkii, Nt_sample, Nrho, Nv, Nz)

    while solver.successful() and ii < Nt_sample-1:
        # We advance
        solver.integrate(solver.t+dt)
        yyii = solver.y
        rhoii, E01ii = unpack_slice(yyii, Nt, Nrho, Nv, Nz)
        # We impose the boundary condition.
        E01ii[0] = E01_boundary[ii+1]

        ii += 1
        solver.t+dt
        rho[ii] = rhoii
        E01[ii, :] = E01ii

    # We plot the solution.
    if plots:
        const1 = np.pi*c*epsilon_0*(w1)**2/16.0/omega_laser1/hbar
        # We calculate the complete density matrix:
        rho_tot = sum([p[jj]*rho[:, :, jj, :] for jj in range(Nv)])
        # rho21 = rho_tot[:, 0, :]
        # rho31 = rho_tot[:, 1, :]
        # rho_3_1 = rho_tot[:, 0, :]
        # rho_4_1 = rho_tot[:, 1, :]
        # rho_5_1 = rho_tot[:, 2, :]
        # rho_6_1 = rho_tot[:, 3, :]
        # rho_7_1 = rho_tot[:, 4, :]
        # rho_8_1 = rho_tot[:, 5, :]
        # rho_9_1 = rho_tot[:, 6, :]
        # rho_10_1 = rho_tot[:, 7, :]
        # rho_11_1 = rho_tot[:, 8, :]
        # rho_12_1 = rho_tot[:, 9, :]
        rho_3_2 = rho_tot[:, 10, :]
        rho_4_2 = rho_tot[:, 11, :]
        rho_5_2 = rho_tot[:, 12, :]
        rho_6_2 = rho_tot[:, 13, :]
        rho_7_2 = rho_tot[:, 14, :]
        rho_8_2 = rho_tot[:, 15, :]
        rho_9_2 = rho_tot[:, 16, :]
        rho_10_2 = rho_tot[:, 17, :]
        rho_11_2 = rho_tot[:, 18, :]
        rho_12_2 = rho_tot[:, 19, :]

        simple_complex_plot(Z*100, t_sample*1e9, np.sqrt(const1*1e-9)*E01,
                            "sol_E01_"+name+".png", amount=r"E_{0s}",
                            modsquare=True)

        # simple_complex_plot(Z*100, t_sample*1e9, rho_3_1,
        #                     "sol_rho3_1_"+name+".png", amount=r"\rho_{3,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_4_1,
        #                     "sol_rho4_1_"+name+".png", amount=r"\rho_{4,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_5_1,
        #                     "sol_rho5_1_"+name+".png", amount=r"\rho_{5,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_6_1,
        #                     "sol_rho6_1_"+name+".png", amount=r"\rho_{6,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_7_1,
        #                     "sol_rho7_1_"+name+".png", amount=r"\rho_{7,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_8_1,
        #                     "sol_rho8_1_"+name+".png", amount=r"\rho_{8,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_9_1,
        #                     "sol_rho9_1_"+name+".png", amount=r"\rho_{9,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_10_1,
        #                     "sol_rho10_1_"+name+".png",
        #                     amount=r"\rho_{10,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_11_1,
        #                     "sol_rho11_1_"+name+".png",
        #                     amount=r"\rho_{11,1}")
        # simple_complex_plot(Z*100, t_sample*1e9, rho_12_1,
        #                     "sol_rho12_1_"+name+".png",
        #                     amount=r"\rho_{12,1}")

        simple_complex_plot(Z*100, t_sample*1e9, rho_3_2,
                            "sol_rho3_2_"+name+".png", amount=r"\rho_{3,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_4_2,
                            "sol_rho4_2_"+name+".png", amount=r"\rho_{4,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_5_2,
                            "sol_rho5_2_"+name+".png", amount=r"\rho_{5,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_6_2,
                            "sol_rho6_2_"+name+".png", amount=r"\rho_{6,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_7_2,
                            "sol_rho7_2_"+name+".png", amount=r"\rho_{7,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_8_2,
                            "sol_rho8_2_"+name+".png", amount=r"\rho_{8,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_9_2,
                            "sol_rho9_2_"+name+".png", amount=r"\rho_{9,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_10_2,
                            "sol_rho10_2_"+name+".png", amount=r"\rho_{10,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_11_2,
                            "sol_rho11_2_"+name+".png", amount=r"\rho_{11,2}")
        simple_complex_plot(Z*100, t_sample*1e9, rho_12_2,
                            "sol_rho12_2_"+name+".png", amount=r"\rho_{12,2}")

    if integrate_velocities:
        rho_tot = sum(p[jj] * rho[:, :, jj, :] for jj in range(Nv))
        rho31 = rho_tot[:, 1, :]
        return t_sample, Z, rho31, E01
    else:
        return t_sample, Z, vZ, rho, E01


def efficiencies_r1r2t0w(energy_pulse2, p, explicit_decoherence=None,
                         name="", return_params=False):
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
                                    "verbose": 0, "Nv": 11,
                                    "t_cutoff": 4.7e-9})
    if return_params:
        return params

    t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
    del rho
    # print "t_cutoff", params["t_cutoff"]
    aux = efficiencies(t, Om1, params, plots=True, name=name)
    eff_in, eff_out, eff = aux

    # We explicitly introduce the measured decoherence.
    if explicit_decoherence is not None:
        eff_out = eff_out*explicit_decoherence
        eff = eff_in*eff_out

    return eff_in, eff_out, eff
#
#
# def efficiencies_t0wenergies(p, explicit_decoherence=None, name=""):
#     r"""Get the efficiencies for modified t0w-t0s, energy_write, energy_read.
#
#     This is done while using the fitted r1, and r2.
#     """
#     # A name to use in the plots.
#     name = "energy"+str(name)
#     # We get the default values.
#     r1 = default_params["r1"]
#     r2 = default_params["r2"]
#     t0s = default_params["t0s"]
#     t0w = default_params["t0w"]
#     t0r = default_params["t0r"]
#
#     # We unpack the errors.
#     # print "r1, r2:", r1/a0, r2/a0
#     tmeet_error, energy_write, energy_read = p
#     alpha_rw = np.sqrt(energy_read/energy_write)
#
#     t0w = t0s+tmeet_error*1e-9
#     t0r = t0w + 1.0*3.5e-9
#     params = set_parameters_ladder({"t0w": t0w, "t0r": t0r,
#                                     "r1": r1, "r2": r2,
#                                     "energy_pulse2": energy_write,
#                                     "alpha_rw": alpha_rw,
#                                     "t_cutoff": 3.5e-9,
#                                     "Nv": 1,
#                                     "verbose": 0})
#
#     t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
#     del rho
#     aux =efficiencies(t, Om1, params, plots=True, name=name)
#     eff_in, eff_out, eff = aux
#
#     # We explicitly introduce the measured decoherence.
#     if explicit_decoherence is not None:
#         eff_out = eff_out*explicit_decoherence
#         eff = eff_in*eff_out
#
#     return eff_in, eff_out, eff
#
#
# def num_integral(t, f):
#     """We integrate using the trapezium rule."""
#     dt = t[1]-t[0]
#     F = sum(f[1:-1])
#     F += (f[1] + f[-1])*0.5
#     return np.real(F*dt)
#
#
# def normalization(t, f, params):
#     """Get the normalization for const1*|f|^2 to integrate to one."""
#     const1 = photons_const(params)
#     Nin = num_integral(t, const1*f*f.conjugate())
#     return np.sqrt(np.abs(Nin))
#
#
# def photons_const(params):
#     """Get the constant to translate to photon number."""
#     e_charge = params["e_charge"]
#     hbar = params["hbar"]
#     c = params["c"]
#     epsilon_0 = params["epsilon_0"]
#     r1 = params["r1"]
#     omega_laser1 = params["omega_laser1"]
#     w1 = params["w1"]
#     return np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1
#
#
# def rescale_input(t, mode, params):
#     r"""Rescale an input mode so that its mod square integral is 1 photon."""
#     return mode/normalization(t, mode, params)
#
#
# def bra(v):
#     """"Get a bra from an array."""
#     return v.reshape((1, len(v))).conjugate()
#
#
# def ket(v):
#     """"Get a ket from an array."""
#     return v.reshape((len(v), 1))
#
#
# def rel_error(a, b):
#     r"""Get the relative error between two quantities."""
#     m = max([abs(a), abs(b)])
#     n = min([abs(a), abs(b)])
#     return 1 - float(n)/m
#
#
# def greens(params, index, Nhg=5):
#     r"""Calculate the Green's function using params."""
#     # We build the Green's function.
#     t_cutoff = params["t_cutoff"]
#
#     t_sample = np.linspace(0, params["T"],
#                            params["Nt"]/params["sampling_rate"])
#     Nt = len(t_sample); dt = t_sample[1]-t_sample[0]
#     t_out = np.array([t_sample[i] for i in range(Nt)
#                       if t_sample[i] > t_cutoff])
#     Ntout = len(t_out)
#
#     phi = []; psi = []
#     Gri = np.zeros((Ntout, Nt), complex)
#     print "The size of Green's function", Gri.shape
#     Nhg = 15
#     Kprev = 1e6
#     for ii in range(Nhg):
#         print ("Mode order %i" % ii)
#         params["ns"] = ii
#         # We solve for the Hermite Gauss mode of order 0.
#         aux = solve(params, integrate_velocities=True)
#         t_sample, Z, rho31, Om1 = aux
#         Nt = len(t_sample); dt = t_sample[1]-t_sample[0]
#
#         # Extract input and output.
#         Om1_in = Om1[:, 0]
#         Om1_out = np.array([Om1[i, -1] for i in range(Nt)
#                             if t_sample[i] > t_cutoff])
#         Ntout = len(Om1_out)
#         Gri += ket(Om1_out).dot(bra(Om1_in))*dt
#         phi += [Om1_in]
#         psi += [Om1_out]
#
#         if ii >= 2:
#             U, D, V = svd(Gri)
#             DD = D/np.sqrt(sum(D**2))
#             K = 1.0/(DD**4).sum()
#             K_change = rel_error(K, Kprev)
#
#             check = K_change <= 0.01
#             print ii, K, Kprev, K_change, check
#             if check:
#                 break
#             Kprev = K
#             # check = ii >= 8*K
#             # print "check:", check, ii, 8*K, K, D[: 5]
#             # if check:
#             #     break
#     # U, D, V = svd(Gri)
#
#     print "Checking Green's function..."
#     plt.close("all")
#     Nhg = ii
#     plt.figure()
#     for i in range(Nhg):
#         # print ".............."
#         Nin = num_integral(t_sample, phi[i]*phi[i].conjugate())
#         Nout = num_integral(t_out, psi[i]*psi[i].conjugate())
#         psi_cal = Gri.dot(ket(phi[i])).reshape(Ntout)
#         Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
#         print i, Nin, Nout, Ncal
#         plt.subplot(211)
#         plt.plot(t_sample, np.abs(phi[i]*phi[i].conjugate()), label=str(i))
#         plt.subplot(212)
#         plt.plot(t_out, np.abs(psi[i]*psi[i].conjugate()), label=str(i))
#     plt.subplot(211)
#     plt.legend()
#     plt.subplot(212)
#     plt.legend()
#     plt.savefig("a"+str(index)+".png")
#     plt.close("all")
#
#     print "testing singular modes..."
#     plt.figure()
#     for ii in range(Nhg):
#         # print ".............."
#         dt = t_sample[1]-t_sample[0]
#         phii = V[ii, :].conjugate()/np.sqrt(dt)
#         Nin = num_integral(t_sample, phii*phii.conjugate())
#         Nout = D[ii]**2
#         psi_cal = Gri.dot(ket(phii)).reshape(Ntout)
#         Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
#         print ii, Nin, Nout, Ncal
#         plt.subplot(211)
#         plt.plot(t_sample, 1e-9*np.abs(phii*phii.conjugate()),
#                  label=str(ii))
#         plt.subplot(212)
#         plt.plot(t_out, np.abs(U[:, ii]*U[:, ii].conjugate()),
#                  label=str(ii))
#     plt.subplot(211)
#     plt.legend()
#     plt.subplot(212)
#     plt.legend()
#     plt.savefig("b"+str(index)+".png")
#     plt.close("all")
#
#     # Vhg = V
#     # # We make the Green's function converge.
#     # print "Using single modes..."
#     # Gri = np.zeros((Ntout, Nt), complex)
#     # Nmodes = Nhg
#     # phi = []; psi = []
#     # for ii in range(Nmodes):
#     #     print ("Mode order %i" % ii)
#     #     params["ns"] = ii
#     #     norm = num_integral(t_sample, Vhg[ii, :]*Vhg[ii, :].conjugate())
#     #     phi_i = Vhg[ii, :] / np.sqrt(np.real(norm))
#     #     # We solve for the Hermite Gauss mode of order 0.
#     #     aux = solve(params, integrate_velocities=True,
#     #                 input_signal=phi_i)
#     #     t_sample, Z, rho31, Om1 = aux
#     #     Nt = len(t_sample); dt = t_sample[1]-t_sample[0]
#     #
#     #     # Extract input and output.
#     #     Om1_in = Om1[:, 0]
#     #     Om1_out = np.array([Om1[i, -1] for i in range(Nt)
#     #                         if t_sample[i] > t_cutoff])
#     #     Ntout = len(Om1_out)
#     #     Gri += ket(Om1_out).dot(bra(Om1_in))*dt
#     #     phi += [Om1_in]
#     #     psi += [Om1_out]
#     #     # if ii >= 4:
#     #     #     U, D, V = svd(Gri)
#     #     #     DD = D/np.sqrt(sum(D**2))
#     #     #     K = 1.0/(DD**4).sum()
#     #     #     check = ii >= 3*K
#     #     #     print "check:", check, ii, 3*K, K, D[: 5]
#     #     #     if check:
#     #     #         break
#     #
#     # Nhg = ii
#     # # We check that the Green's function does its job.
#     # print "Checking Green's function..."
#     #
#     # plt.close("all")
#     # plt.figure()
#     # for i in range(Nmodes):
#     #     # print ".............."
#     #     Nin = num_integral(t_sample, phi[i]*phi[i].conjugate())
#     #     Nout = num_integral(t_out, psi[i]*psi[i].conjugate())
#     #     psi_cal = Gri.dot(ket(phi[i])).reshape(Ntout)
#     #     Ncal = num_integral(t_out, psi_cal*psi_cal.conjugate())
#     #     print i, Nin, Nout, Ncal
#     #     plt.subplot(211)
#     #     plt.plot(t_sample, np.abs(phi[i]*phi[i].conjugate()), label=str(i))
#     #     plt.subplot(212)
#     #     plt.plot(t_out, np.abs(psi[i]*psi[i].conjugate()), label=str(i))
#     # plt.subplot(211)
#     # plt.legend()
#     # plt.subplot(212)
#     # plt.legend()
#     # plt.savefig("b"+str(index)+".png")
#     # plt.close("all")
#
#     return Gri, t_sample, t_out
#
#
# def optimize_signal(params, index, Nhg=5, plots=False, check=False,
#                     name="optimal"):
#     """Get the optimal signal modes and total efficiency."""
#     Nhg = 25
#     Gri, t_sample, t_out = greens(params, index, Nhg)
#     Ntout = len(t_out)
#     const1 = photons_const(params)
#
#     U, D, V = svd(Gri)
#
#     # We extract the optimal modes.
#     optimal_input = rescale_input(t_sample, V[0, :].conjugate(), params)
#     # print np.amax(np.real(V[0, :]*V[0, :].conjugate())),
#     # print np.amax(np.real(optimal_input*optimal_input.conjugate()))
#     ##########################################################################
#     # We check that the Green's function actually returns the expected thing.
#     if check:
#         Om1_in_actual = rescale_input(t_sample, optimal_input, params)
#         t_sample, Z, rho31, Om1 = solve(params, plots=True,
#                                         name=name,
#                                         integrate_velocities=True,
#                                         input_signal=Om1_in_actual)
#
#         GOm1_in_actual = Gri.dot(ket(Om1_in_actual)).reshape(Ntout)
#
#         eff_cal = num_integral(t_out, const1 *
#                                GOm1_in_actual*GOm1_in_actual.conjugate())
#
#         eff_in, eff_out, eff = efficiencies(t_sample, Om1, params,
#                                             plots=True, name=name+str(index))
#         # DD = D/np.sqrt(sum(D**2))
#         print "The SVD-calculated efficiency is", D[0]**2
#         print "The Green's function-calculated efficiency is", eff_cal
#         print "The actual efficiency is", eff
#     ##########################################################################
#     if plots:
#         # Plotting.
#         T, S = np.meshgrid(t_sample*1e9, t_out*1e9)
#         plt.figure()
#         cs = plt.contourf(T, S, abs(Gri)**2, 256)
#         plt.tight_layout()
#         plt.savefig("Greens"+str(index)+".png", bbox_inches="tight")
#         plt.colorbar(cs)
#         plt.xlabel(r"$t \ \mathrm{(ns)}$")
#         plt.ylabel(r"$t \ \mathrm{(ns)}$")
#         plt.close("all")
#
#         # We plot the one-photon modes.
#         ii = 0
#         plt.figure()
#         for ii in range(6):
#             if ii == 0:
#                 label_out = "Optimal output"
#                 label_in = "Optimal input"
#             else:
#                 label_out = "output mode "+str(ii)
#                 label_in = "intput mode "+str(ii)
#             input_ii = rescale_input(t_sample, V[ii, :], params)
#             output_ii = Gri.dot(ket(input_ii)).reshape(Ntout)
#
#             plt.subplot(212)
#             plt.plot(t_out*1e9, const1*np.abs(output_ii)**2*1e-9,
#                      label=label_out)
#
#             plt.subplot(211)
#             plt.plot(t_sample*1e9, const1*np.abs(input_ii)**2*1e-9,
#                      label=label_in)
#
#         plt.subplot(211)
#         plt.ylabel(r"$\mathrm{photons/ns}$", fontsize=15)
#         plt.legend()
#         # plt.ylim([0, 3])
#         plt.subplot(212)
#         plt.xlabel(r"$t \ (\mathrm{ns})$", fontsize=15)
#         plt.ylabel(r"$\mathrm{photons/ns}$", fontsize=15)
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("singular_modes.png", bbox_inches="tight")
#         plt.close("all")
#         # plt.show()
#
#         plt.figure()
#         plt.subplot(121)
#         plt.bar(np.arange(5), D[:5])
#         plt.subplot(122)
#         plt.bar(np.arange(5), (D[:5])**2)
#         plt.tight_layout()
#         plt.savefig("singular_values.png", bbox_inches="tight")
#         plt.close("all")
#
#     ##########################################################################
#     # The return.
#     optimal_output = Gri.dot(ket(optimal_input)).reshape(Ntout)
#     optimal_efficiency = D[0]**2
#     # optimal_efficiency = eff_cal
#     return optimal_input, optimal_output, optimal_efficiency, eff


no_fit_params = set_parameters_ladder(fitted_couplings=False)
default_params = set_parameters_ladder()
