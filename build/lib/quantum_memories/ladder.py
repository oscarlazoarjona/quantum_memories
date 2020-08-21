# -*- coding: utf-8 -*-
# ***********************************************************************
#       Copyright (C) 2016 - 2017 Oscar Gerardo Lazo Arjona             *
#              <oscar.lazoarjona@physics.ox.ac.uk>                      *
# ***********************************************************************

"""This is a solver for Maxwell-Bloch equations for a three level ladder system
driven by two optical fields under the rotating wave approximation. An
arbitrary number of velocity groups are allowed. These are coupled, partial,
first order differential equations.

For a detailed derivation of the equations see the jupyter notebook included
"Ladder memory equations.ipynb".

References:
    [1] https://arxiv.org/abs/1704.00013
"""

import numpy as np
from math import pi, sqrt, log
from matplotlib import pyplot as plt
from matplotlib import rcParams

from misc import cheb, cDz, simple_complex_plot, set_parameters_ladder
from misc import efficiencies, vapour_number_density
import warnings

warnings.filterwarnings("error")

# We set matplotlib to use a nice latex font.
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.rm'] = 'serif'


def solve(params, plots=False, name=""):
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
    Nrho = 6
    if keep_data == "all":
        # t_sample = np.linspace(0, T, Nt)
        t_sample = t
        Om = np.zeros((Nt, 2, Nz), complex)
        rho = np.zeros((Nt, Nrho, Nv, Nz), complex)
        # output = np.zeros(Nt, complex)
    elif keep_data == "sample":
        # t_sample = np.linspace(0, T, Nt/sampling_rate)
        t_sample = np.linspace(0, T, Nt/sampling_rate)
        Om = np.zeros((Nt/sampling_rate, 2, Nz), complex)
        rho = np.zeros((Nt/sampling_rate, Nrho, Nv, Nz), complex)
        # output = np.zeros(Nt/sampling_rate, complex)
    elif keep_data == "current":
        # t_sample = np.zeros(1)
        Om = np.zeros((1, 2, Nz), complex)
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

    ##################################################
    # We establish the boundary and initial conditions.
    # The signal pulse at Z = -D/2
    Omega1_boundary = Omega1_peak*np.exp(-4*np.log(2.0) *
                                         (t - t0s + D/2/c)**2/tau1**2)

    # The control pulses at Z = D/2
    Omega2_boundary = Omega2_peak*np.exp(-4*np.log(2.0) *
                                         (t - t0w + D/2/c)**2/tau2**2)
    Omega2_boundary += alpha_rw*Omega2_peak * \
        np.exp(-4*np.log(2.0)*(t - t0r + D/2/c)**2/tau2**2)

    # The signal pulse at t = 0 is
    Omega1_initial = Omega1_peak*np.exp(-4*np.log(2.0)*(-t0s-Z/c)**2/tau1**2)
    # The control pulses at t = 0 is
    Omega2_initial = Omega2_peak*np.exp(-4*np.log(2.0)*(-t0w+Z/c)**2/tau2**2)
    Omega2_initial += alpha_rw*Omega2_peak * \
        np.exp(-4*np.log(2.0)*(-t0r+Z/c)**2/tau2**2)

    # We establish the Rabi frequencies.
    Om[0][0] = Omega1_initial
    Om[0][1] = Omega2_initial
    # All the population initially at |1> with no coherences.
    rho[0][0] = 1.0

    # The coupling coefficient for the signal field.
    g1 = omega_laser1*(e_charge*r1)**2/(hbar*epsilon_0)
    g2 = omega_laser2*(e_charge*r2)**2/(hbar*epsilon_0)

    const1 = np.pi*c*epsilon_0*hbar*(w1/e_charge/r1)**2/16.0/omega_laser1
    const2 = np.pi*c*epsilon_0*hbar*(w2/e_charge/r2)**2/16.0/omega_laser2
    # The detuning of the control field.
    delta2 = -delta1

    params = (delta1, delta2, gamma21, gamma32, g1, g2,
              Omega2_peak, tau2, t0w, t0r, alpha_rw,
              p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv,
              omega_laser1, omega_laser2)

    # We define the equations that the Runge-Kutta method will solve.
    # from scipy.constants import physical_constants
    # a0 = physical_constants["Bohr radius"][0]
    # print g1
    # print delta1/2/pi*1e-9, delta2/2/pi*1e-9
    # print gamma21/2/pi*1e-6, gamma32/2/pi*1e-6
    #
    # print c/(omega_laser1/2/pi)*1e9
    # print e_charge
    #
    # print r1/a0
    # print hbar, epsilon_0, c

    def f(rhoi, Omi, ti, params, flag=False):
        # We unpack the parameters.
        delta1, delta2, gamma21, gamma32, g1, g2 = params[:6]
        Omega2_peak, tau2, t0w, t0r, alpha_rw = params[6:11]
        p, vZ, Z, n_atomic, cheb_diff_mat, c, Nv = params[11:18]
        omega_laser1, omega_laser2 = params[18:]

        # We unpack the density matrix components.
        rho11i = rhoi[0]
        rho22i = rhoi[1]
        rho33i = rhoi[2]

        rho21i = rhoi[3]
        rho31i = rhoi[4]
        rho32i = rhoi[5]
        Nv = len(rho11i)

        # We unpack the Rabi frequencies.
        Om1i = Omi[0]
        Om2i = Omi[1]

        Om1ic = Om1i.conjugate()
        Om2ic = Om2i.conjugate()

        # We calculate the right-hand sides of equations 1 through 6 for all
        # velocity groups.
        eqs_rho = np.zeros((6, Nv, Nz), complex)
        eqs_Om = np.zeros((2, Nz), complex)
        for jj in range(Nv):
            rho11ij = rho11i[jj]
            rho22ij = rho22i[jj]
            rho33ij = rho33i[jj]

            rho21ij = rho21i[jj]
            rho31ij = rho31i[jj]
            rho32ij = rho32i[jj]

            rho21ijc = rho21ij.conjugate()
            # rho31ijc = rho31ij.conjugate()
            rho32ijc = rho32ij.conjugate()

            # Equation 1 (d rho11ij / dt)
            eqs_rho[0, jj, :] += 0.5j*(Om1i*rho21ijc-Om1ic*rho21ij)
            eqs_rho[0, jj, :] += gamma21*rho22ij

            # Equation 2 (d rho22ij / dt)
            eqs_rho[1, jj, :] += -0.5j*(Om1i*rho21ijc-Om1ic*rho21ij)
            eqs_rho[1, jj, :] += 0.5j*(Om2i*rho32ijc-Om2ic*rho32ij)
            eqs_rho[1, jj, :] += -gamma21*rho22ij + gamma32*rho33ij

            # Equation 3 (d rho33ij / dt)
            eqs_rho[2, jj, :] += -0.5j*(Om2i*rho32ijc-Om2ic*rho32ij)
            eqs_rho[2, jj, :] += -gamma32*rho33ij

            # Equation 4 (d rho21ij / dt)
            fact = (1j*delta1 - gamma21/2 - 1j*vZ[jj]*omega_laser1/c)
            eqs_rho[3, jj, :] += -0.5j*Om1i*rho11ij
            eqs_rho[3, jj, :] += 0.5j*Om1i*rho22ij
            eqs_rho[3, jj, :] += -0.5j*Om2ic*rho31ij
            eqs_rho[3, jj, :] += fact*rho21ij

            # Equation 5 (d rho31ij / dt)
            fact = 1j*(delta1+delta2+vZ[jj]*(omega_laser2-omega_laser1)/c)
            fact += -gamma32/2
            eqs_rho[4, jj, :] += 0.5j*Om1i*rho32ij
            eqs_rho[4, jj, :] += -0.5j*Om2i*rho21ij
            eqs_rho[4, jj, :] += fact*rho31ij

            # Equation 6 (d rho32ij / dt)
            fact = 1j*(delta2+vZ[jj]*omega_laser2/c) - (gamma21+gamma32)/2
            eqs_rho[5, jj, :] += -0.5j*Om2i*rho22ij
            eqs_rho[5, jj, :] += 0.5j*Om2i*rho33ij
            eqs_rho[5, jj, :] += 0.5j*Om1ic*rho31ij
            eqs_rho[5, jj, :] += fact*rho32ij

        rho21i_tot = sum([p[jj]*rho21i[jj] for jj in range(Nv)])
        rho32i_tot = sum([p[jj]*rho32i[jj] for jj in range(Nv)])

        # Equation 7 (d rho32ij / dt)
        eqs_Om[0] += -1j*g1*n_atomic*rho21i_tot
        eqs_Om[0] += - cDz(Om1i, c, cheb_diff_mat)

        # Equation 8 (d rho32ij / dt)
        eqs_Om[1] += -1j*g2*n_atomic*rho32i_tot
        eqs_Om[1] += + cDz(Om2i, c, cheb_diff_mat)

        return eqs_rho, eqs_Om

    ii = 0
    # We carry out the Runge-Kutta method.
    ti = 0.0
    rhoii = rho[0]
    Omii = Om[0]
    for ii in range(Nt-1):

        # print ii, np.amax(const2*Omii[1]*Omii[1].conjugate())

        rhok1, Omk1 = f(rhoii, Omii, ti, params, flag=ii)

        rhok2, Omk2 = f(rhoii+rhok1*dt/2.0, Omii+Omk1*dt/2.0,
                        ti+dt/2.0, params, flag=2)

        rhok3, Omk3 = f(rhoii+rhok2*dt/2.0, Omii+Omk2*dt/2.0,
                        ti+dt/2.0, params, flag=3)

        rhok4, Omk4 = f(rhoii+rhok3*dt, Omii+Omk3*dt,
                        ti+dt, params, flag=4)
        # The solution at time ti + dt:
        ti = ti + dt
        rhoii = rhoii + (rhok1+2*rhok2+2*rhok3+rhok4)*dt/6.0
        Omii = Omii + (Omk1+2*Omk2+2*Omk3+Omk4)*dt/6.0

        # We impose the boundary condition.
        Omii[0][0] = Omega1_boundary[ii+1]
        Omii[1][-1] = Omega2_boundary[ii+1]

        # We determine the index for the sampling.
        if keep_data == "all":
            sampling_index = ii+1
        if keep_data == "sample":
            if ii % sampling_rate == 0:
                sampling_index = ii/sampling_rate

        rho[sampling_index] = rhoii
        Om[sampling_index] = Omii

        # Om1 = np.zeros((Nt, Nz), complex)
        # rho = np.zeros((Nt, Nrho, Nv, Nz), complex)
        if verbose >= 1:
            if ii == 0:
                print "  0.0 % done..."
            elif ii % (Nt/10) == 0:
                print ' '+str(100.0*ii/Nt) + " % done..."
            if ii == Nt-2: print "100.0 % done."

    # We plot the solution.
    if plots:
        # We calculate the complete density matrix:
        rho_tot = sum([p[jj]*rho[:, :, jj, :] for jj in range(Nv)])
        rho11 = rho_tot[:, 0, :]
        rho22 = rho_tot[:, 1, :]
        rho33 = rho_tot[:, 2, :]
        rho21 = rho_tot[:, 3, :]
        rho31 = rho_tot[:, 4, :]
        rho32 = rho_tot[:, 5, :]

        simple_complex_plot(Z*100, t_sample*1e9,
                            np.sqrt(const1*1e-9)*Om[:, 0, :],
                            "sol_Om1_"+name+".png", amount=r"\Omega_s",
                            modsquare=True)
        simple_complex_plot(Z*100, t_sample*1e9,
                            np.sqrt(const2*1e-9)*Om[:, 1, :],
                            "sol_Om2_"+name+".png", amount=r"\Omega_c",
                            modsquare=True)

        simple_complex_plot(Z*100, t_sample*1e9, rho11,
                            "sol_rho11_"+name+".png", amount=r"\rho_{11}")
        simple_complex_plot(Z*100, t_sample*1e9, rho22,
                            "sol_rho22_"+name+".png", amount=r"\rho_{22}")
        simple_complex_plot(Z*100, t_sample*1e9, rho33,
                            "sol_rho33_"+name+".png", amount=r"\rho_{33}")
        simple_complex_plot(Z*100, t_sample*1e9, rho21,
                            "sol_rho21_"+name+".png", amount=r"\rho_{21}")
        simple_complex_plot(Z*100, t_sample*1e9, rho31,
                            "sol_rho31_"+name+".png", amount=r"\rho_{31}")
        simple_complex_plot(Z*100, t_sample*1e9, rho32,
                            "sol_rho32_"+name+".png", amount=r"\rho_{32}")

    return t_sample, Z, vZ, rho, Om


def efficiencies_r1r2t0w(energy_pulse2, p, explicit_decoherence=None, name=""):
    r"""Get the efficiencies for modified r1, r2, t0w."""
    # We unpack the errors.
    r1_error, r2_error, t0w_error = p

    # A name to use in the plots.
    name = "energy"+name
    # We get the default values.
    r1 = default_params["r1"]
    r2 = default_params["r2"]
    t0s = default_params["t0s"]
    t0w = default_params["t0w"]
    # print r1/a0, r2/a0, t0w-t0s

    # The modified parameters.
    r1 = r1*r1_error
    r2 = r2*r2_error
    t0w = t0s+t0w_error*1e-9
    # print r1/a0, r2/a0, t0w-t0s

    params = set_parameters_ladder({"t0w": t0w, "r1": r1, "r2": r2,
                                    "energy_pulse2": energy_pulse2,
                                    "verbose": 0})

    t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
    del rho
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
    r1 = default_params["r1"]*0.23765377
    r2 = default_params["r2"]*0.78910769
    t0s = default_params["t0s"]
    t0w = default_params["t0w"]
    t0r = default_params["t0r"]

    # We unpack the errors.
    tmeet_error, energy_write, energy_read = p
    alpha_rw = np.sqrt(energy_read/energy_write)

    t0w = t0s+tmeet_error*1e-9
    t0r = t0w + 3.5e-9
    params = set_parameters_ladder({"t0w": t0w, "t0r": t0r,
                                    "r1": r1, "r2": r2,
                                    "energy_pulse2": energy_write,
                                    "alpha_rw": alpha_rw,
                                    "t_cutoff": 3.5e-9,
                                    "verbose": 0})

    t, Z, vZ, rho, Om1 = solve(params, plots=False, name=name)
    del rho
    eff_in, eff_out, eff = efficiencies(t, Om1, params, plots=True, name=name)

    # We explicitly introduce the measured decoherence.
    if explicit_decoherence is not None:
        eff_out = eff_out*explicit_decoherence
        eff = eff_in*eff_out

    return eff_in, eff_out, eff


default_params = set_parameters_ladder()
