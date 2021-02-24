# -*- coding: utf-8 -*-
# Compatible with Python 3.8
# Copyright (C) 2020-2021 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Graphical routines."""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from quantum_memories.misc import build_t_mesh, build_Z_mesh
from scipy.constants import c


def sketch_frame_transform(params, folder="", name="sketch",
                           draw_readout=False, auxiliaries=False):
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
                         alpha=0.25, label=r"$\Xi$")
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
                         color="r", alpha=0.25, label=r"$\Xi$")
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
        plt.savefig(folder+name+".png", bbox_inches="tight")
        plt.close()


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
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(B), norm=get_lognorm(B),
                            shading="auto")
    else:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(B), shading="auto")
    if colorbar: plt.colorbar(cb)
    plt.ylabel(r"$\tau$ (ns)")
    plt.xlabel("$Z$ (cm)")

    plt.subplot(1, 2, 2)
    if log:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(S), norm=get_lognorm(S),
                            shading="auto")
    else:
        cb = plt.pcolormesh(Z*100, tau*1e9, np.abs(S), shading="auto")
    if ii is not None:
        plt.plot(Z[jj]*100, tau[ii]*1e9, "rx")
    if colorbar: plt.colorbar(cb)
    plt.ylabel(r"$\tau$ (ns)")
    plt.xlabel("$Z$ (cm)")
    plt.savefig(folder+name+".png", bbox_inches="tight")
    plt.close("all")


def plot_inout(tau, Z, Bw, Sw, Br, Sr, folder, name):
    r"""Make a plot of the input and output signal."""
    L = Z[-1] - Z[0]
    tau_iniS = tau[0]
    tau_iniQ = tau_iniS - L*2/c
    tauQ0 = (tau_iniS-tau_iniQ)/(Z[0]-Z[-1])*(Z-Z[0]) + tau_iniS

    tau_finS = tau[-1]
    tau_finQ = tau_finS + L*2/c
    tauQf = (tau_finS-tau_finQ)/(Z[-1]-Z[0])*(Z-Z[-1]) + tau_finS

    angle1 = np.unwrap(np.angle(Sw[:, 0]))/2/np.pi
    angle1Q = np.unwrap(np.angle(Sw[0, :]))/2/np.pi

    angle2 = np.unwrap(np.angle(Sw[:, -1]))/2/np.pi
    angle2Q = np.unwrap(np.angle(Sw[-1, :]))/2/np.pi

    angle3 = np.unwrap(np.angle(Sr[:, -1]))/2/np.pi

    ###############################################################
    fig, ax11 = plt.subplots(figsize=(8, 6))
    fs = 15
    ax32 = ax11.twinx()
    ax11.plot(tau*1e9, np.abs(Sw[:, 0])**2*1e-9, "b-",
              label=r"$S_{in}(\tau)$")

    ax11.plot(tauQ0*1e9, np.abs(Sw[0, :])**2*1e-9, "b-")

    ax11.plot(tau*1e9, np.abs(Sw[:, -1])**2*1e-9, "r-",
              label=r"$S_{leak}(\tau)$")
    ax11.plot(tauQf*1e9, np.abs(Sw[-1, :])**2*1e-9, "r-")
    tau_offset = tau[-1] - tau[0]
    ax11.plot((tau+tau_offset)*1e9, np.abs(Sr[:, -1])**2*1e-9,
              "g-", label=r"$S_{out}(\tau)$")

    ax32.plot(tau*1e9, angle1, "b:")
    ax32.plot(tauQ0*1e9, angle1Q, "b:")

    ax32.plot(tau*1e9, angle2, "r:")
    ax32.plot(tauQf*1e9, angle2Q, "r:")
    ax32.plot((tau+tau_offset)*1e9, angle3, "g:")

    ax11.set_xlabel(r"$\tau$ [ns]", fontsize=fs)
    ax11.set_ylabel(r"$|S|^2$  [1/ns]", fontsize=fs)
    ax32.set_ylabel(r"Phase  [revolutions]", fontsize=fs)
    # ax32.set_yticks([])
    ax11.legend(loc=9, fontsize=fs-2)

    fig.savefig(folder+name+".png", bbox_inches="tight")
