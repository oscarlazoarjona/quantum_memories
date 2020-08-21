# -*- coding: utf-8 -*-
# Compatible with Python 2.7.xx
# Copyright (C) 2020 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Finite difference routines."""
import numpy as np
import warnings
from time import time

from scipy.sparse import linalg, csr_matrix, spmatrix, bmat, spdiags
from scipy.sparse import kron as sp_kron
from scipy.sparse import eye as sp_eye

from scipy.constants import c
from matplotlib import pyplot as plt

from sympy import zeros, Matrix
from sympy import factorial as sym_fact
from math import factorial as num_fact

from misc import (interpolator, hermite_gauss, harmonic, rel_error, glo_error,
                  get_range)
from orca import (calculate_kappa, calculate_Gamma21, calculate_Gamma32,
                  calculate_Omega,
                  build_t_mesh, build_Z_mesh)

from graphical import plot_solution


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
