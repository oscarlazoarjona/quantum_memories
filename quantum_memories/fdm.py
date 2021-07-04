# -*- coding: utf-8 -*-
# Compatible with Python 3.8
# Copyright (C) 2020-2021 Oscar Gerardo Lazo Arjona
# mailto: oscar.lazoarjona@physics.ox.ac.uk
r"""Finite difference routines."""
import numpy as np
from time import time

from scipy.sparse import csc_matrix, spmatrix, bmat
from scipy.sparse import kron as sp_kron
from scipy.sparse import eye as sp_eye
from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt

from sympy import zeros, Matrix
from sympy import factorial as sym_fact
from math import factorial as num_fact

from quantum_memories.misc import harmonic, glo_error
import warnings
from pandas import unique as pdunique

# from misc import (interpolator, hermite_gauss, rel_error, glo_error)
# from orca import (calculate_kappa, calculate_Gamma21, calculate_Gamma32,
#                   calculate_Omega)

# from graphical import plot_solution


# def set_row(W, ii):
#     r"""Set a given row to zero and its diagonal element to -1."""
#     indr = np.where(W.row == ii)[0]
#     indc = np.where(W.col == ii)[0]
#
#     for ind in indr:
#         W.data[ind] = 0.0
#
#     indrc = list(set(indr).intersection(set(indc)))
#     if len(indrc) == 0:
#         W.data = np.append(W.data, -1.0)
#         W.row = np.append(W.row, ii)
#         W.col = np.append(W.col, ii)
#     elif len(indrc) == 1:
#         W.data[indrc[0]] = -1.0
#     else:
#         raise ValueError(str(indrc))
#
#     return W
#
#
# def set_col(W, ii):
#     r"""Set a given column to zero and its diagonal element to 1."""
#     indr = np.where(W.row == ii)[0]
#     indc = np.where(W.col == ii)[0]
#
#     for ind in indc:
#         W.data[ind] = 0.0
#
#     indrc = list(set(indr).intersection(set(indc)))
#     if len(indrc) == 0:
#         W.data = np.append(W.data, 1.0)
#         W.row = np.append(W.row, ii)
#         W.col = np.append(W.col, ii)
#     elif len(indrc) == 1:
#         W.data[indrc[0]] = 1.0
#     else:
#         raise ValueError(str(indrc))
#
#     return W

#
# def transform_system(Wp, xb, tau, boundary_indices=[0],
#                      symbolic=False, sparse=False, verbose=0):
#     r"""Transform a system of equations of the form $W' x = 0$ into $W x = b$
#     by imposing the values of $x_i$ for $i\in$ `boundary indices`.
#     """
#     if verbose > 0: t00 = time()
#     if hasattr(Wp, "__call__"):
#         # We have time-dependent equations.
#         W = Wp(tau)
#     else:
#         # We have time-independent equations.
#         W = Wp.copy()
#     N = W.shape[0]
#     if symbolic:
#         zero_row = zeros(1, N)
#         zero_col = zeros(N, 1)
#     else:
#         zero_row = 0.0
#         zero_col = 0.0
#     if verbose > 0: print("555 time: {}".format(time()-t00))
#
#     if verbose > 0: t00 = time()
#     if sparse:
#         W = W.tocoo()
#     if verbose > 0: print("666 time: {}".format(time()-t00))
#
#     if verbose > 0: t00 = time()
#     for i in boundary_indices:
#         if sparse:
#             W = set_row(W, i)
#         else:
#             W[i, :] = zero_row
#             W[i, i] = -1
#     if verbose > 0: print("777 time: {}".format(time()-t00))
#     # raise ValueError
#
#     if verbose > 0: t00 = time()
#     if symbolic:
#         b = -W*xb
#     elif sparse:
#         b = -W.dot(xb)
#     else:
#         b = -np.dot(W, xb)
#     if verbose > 0: print("888 time: {}".format(time()-t00))
#
#     if verbose > 0: t00 = time()
#     for i in boundary_indices:
#         if sparse:
#             W = set_col(W, i)
#         else:
#             W[:, i] = zero_col
#             W[i, i] = 1
#     if verbose > 0: print("999 time: {}".format(time()-t00))
#
#     if verbose > 0: t00 = time()
#     if sparse:
#         W = W.tocsr()
#     if verbose > 0: print("111 time: {}".format(time()-t00))
#
#     return W, b


# def impose_boundary(Wp, tau, S0t, S0z, B0z, P0z=None, sparse=False):
#     r"""Impose boudary conditions."""
#     # We unpack parameters.
#     if P0z is not None:
#         nv = 3
#     else:
#         nv = 2
#     # We unpack parameters.
#     if True:
#         Nt = len(S0t)
#         Nz = len(S0z)
#         nX = nv*Nt*Nz
#     # We build Xb.
#     if True:
#         Xb_ = np.zeros((nv, Nt, Nz), complex)
#         # An auxiliary array to find the boundary condition indices.
#         aux = np.zeros((nv, Nt, Nz), int)
#         zero_bound = np.zeros(Nt)
#
#         adiabatic = P0z is None
#         if not adiabatic:
#             ex = 1
#             # We set up P(tau=0, Z)
#             Xb_[0, 0, :] = P0z
#             aux[0, 0, :] = 1
#             # We set up P(tau, Z=-D/2)
#             Xb_[0, :, 0] = zero_bound
#             aux[0, :, 0] = 1
#             # We set up P(tau, Z=+D/2)
#             Xb_[0, :, -1] = zero_bound
#             aux[0, :, -1] = 1
#         else:
#             ex = 0
#
#         # We set up B(tau=0, Z)
#         Xb_[0+ex, 0, :] = B0z
#         aux[0+ex, 0, :] = 1
#         # We set up S(tau=0, Z)
#         Xb_[1+ex, 0, :] = S0z
#         aux[1+ex, 0, :] = 2
#
#         # We set up B(tau, Z=-D/2)
#         # Xb_[0+ex, :, 0] = zero_bound
#         # aux[0+ex, :, 0] = 3
#         # # We set up B(tau, Z=+D/2)
#         # Xb_[0+ex, :, -1] = zero_bound
#         # aux[0+ex, :, -1] = 4
#         # We set up S(tau, Z=-D/2)
#         Xb_[1+ex, :, 0] = S0t
#         aux[1+ex, :, 0] = 5
#
#         # We flatten Xb_.
#         Xb = np.reshape(Xb_, nX)
#         # We find the boundary_indices.
#         aux = np.reshape(aux, nX)
#         boundary_indices = [ind for ind, auxi in enumerate(aux) if auxi != 0]
#     # We transform the system.
#     if True:
#         W, b = transform_system(Wp, Xb, tau, sparse=sparse,
#                                 boundary_indices=boundary_indices)
#         return W, b, Xb_


#############################################
# Finite difference equations.


bfmt = "csc"
bfmtf = csc_matrix


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
    >>> x = np.array([0.0, 1.0, 5.0])
    >>> print(D_coefficients(2, 0, xaxis=x))
    [-1.2   1.25 -0.05]

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


def derivative_operator(xaxis, p=2, d=1, symbolic=False, sparse=False,
                        function=False):
    u"""Return a function to calculate the derivative of a discretized
    function for an arbitrary grid.

    INPUT:

    -  ``xaxis`` - an array, the grid on which the function is represented.

    -  ``p`` - int, the precission order of the approximation.

    -  ``symbolic`` - bool, whether to return symbolic coefficients.

    -  ``sparse`` - bool, whether to use a sparse matrix.

    -  ``matrix`` - bool, whether to return the matrix.

    OUTPUT:

    A 2-d array representation of the differential operator.

    Examples
    ========

    >>> from sympy import pprint
    >>> x = np.array([2, 4, 5, 7, 10])
    >>> f = 5*x
    >>> D = derivative_operator(x)
    >>> print(D(f))
    [5. 5. 5. 5. 5.]

    Getting the matrix representation of the derivatives

    >>> D = derivative_operator(range(5), matrix=True)
    >>> print(D)
    [[-1.5  2.  -0.5  0.   0. ]
     [-0.5  0.   0.5  0.   0. ]
     [ 0.  -0.5  0.   0.5  0. ]
     [ 0.   0.  -0.5  0.   0.5]
     [ 0.   0.   0.5 -2.   1.5]]

    >>> D = derivative_operator(range(5), p=4, matrix=True)
    >>> print(D)
    [[-2.08333333  4.         -3.          1.33333333 -0.25      ]
     [-0.25       -0.83333333  1.5        -0.5         0.08333333]
     [ 0.08333333 -0.66666667  0.          0.66666667 -0.08333333]
     [-0.08333333  0.5        -1.5         0.83333333  0.25      ]
     [ 0.25       -1.33333333  3.         -4.          2.08333333]]

    >>> D = derivative_operator(range(5), p=4, symbolic=True, matrix=True)
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

    For an arbitrary grid:

    >>> x = [1, 2, 4, 6, 7]
    >>> D = derivative_operator(x, p=2, symbolic=True, matrix=True)
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
    # We determine if the grid is uniform.
    hlist = [xaxis[i+1] - xaxis[i] for i in range(N-1)]
    uniform_grid = not np.any([rel_dif(hlist[i], h) >= 1e-5
                               for i in range(N-1)])
    if uniform_grid:
        coefficients = [D_coefficients(p, i, d=d, symbolic=symbolic)
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

        D = D/h**d
    else:
        # We generate a p + 1 long list for each of the N rows.
        for i in range(N):
            if i < p//2:
                a = 0
                jj = i
            elif i >= N - p//2:
                a = N - p - 1
                jj = (i - (N - p - 1))
            else:
                a = i - p//2
                jj = p//2
            b = a + p + 1
            D[i, a: b] = D_coefficients(p, jj, xaxis=xaxis[a:b], d=d,
                                        symbolic=symbolic)

    if sparse:
        return bfmtf(D)
    elif function:
        def deri(f):
            return np.dot(D, f)
        return deri
    else:
        return D


def fdm_derivative_operators(tau, Z, pt=4, pz=4, sparse=False,
                             plots=False, folder=""):
    r"""Calculate the block-matrix representation of the derivatives."""
    # We unpack parameters.
    if True:
        Nt = len(tau)
        Nz = len(Z)

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
    nv = int(nX/Ntz)
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


#############################################


def build_permutation(N, i0_list, if_list, verbose=0):
    r"""Return a permutation matrix and its reflexive generalized inverse that
    reorders the input, intermediate, and output points.
    """
    t0 = time()
    N0 = len(i0_list)
    Nf = len(if_list)
    i_list = i0_list + if_list

    Nbig = N0 + Nf - len(set(i0_list).union(set(if_list)))
    Nbig = N + Nbig
    # R = lil_matrix((Nbig, N))
    # Rrgi = lil_matrix((N, Nbig))
    # used = [0 for i in range(N)]
    data = []; rows = []; cols = []
    if verbose > 0: print(11, time()-t0)

    t0 = time()
    for i in range(N0+Nf):
        j = i_list[i]
        data.append(1)
        rows.append(i)
        cols.append(j)
        # if used[j] != 1:
        #     Rrgi[j, i] = 1
        # used[j] = 1

    data += list(np.ones(N - N0 - Nf))
    rows += range(N0+Nf, Nbig)
    if verbose > 0: print(12, time()-t0)
    t0 = time()
    # cols += [i for i in range(Nbig) if i not in i_list]
    cols += list(np.delete(range(Nbig), i_list))
    # print(rows)
    if verbose > 0: print(13, time()-t0)

    t0 = time()
    data = np.array(data, float)
    rows = np.array(rows)
    cols = np.array(cols)
    R = bfmtf((data, (rows, cols)), shape=(Nbig, N))
    if verbose > 0: print(14, time()-t0)
    # print(R.todense())
    return R


def solve_fdm(A, input_indices, output_indices, input=None,
              full_propagator=False, verbose=0,
              plots=False, folder="", name="G"):
    r"""Solve finite difference equations `Ax=0` with boundary conditions."""
    def filter_out_indices(i0_list_, if_list_):
        i0_list = pdunique(i0_list_)
        if_list = pdunique(if_list_)
        if_list = np.setdiff1d(if_list, i0_list, assume_unique=True)
        return i0_list.tolist(), if_list.tolist()
    # def filter_out_indices(i0_list_, if_list_):
    #     i0_list = []
    #     for i in i0_list_:
    #         if i not in i0_list:
    #             i0_list += [i]
    #     if_list = []
    #     for i in if_list_:
    #         if i not in if_list and (i not in i0_list_):
    #             if_list += [i]
    #     return i0_list, if_list

    # Let Y be Y be a solution to AY = 0. We first will rearrange the elements
    # of A using a permutation matrix P that in Z = PY they are
    # oredered as (inputs, final, intermediate). These blocks (or subspaces)
    # in y are labeled Z0, Zf, Zm respectively.
    #
    # We specify the order in which the elements of Y will be arranged in Z
    # by providing lists of indices `input_indices`, `output_indices` of Y
    # elements. A given index may appear more than once in any of these lists,
    # or appear in both, since it could be useful to have subblocks within
    # Z0 or Zf that have some overlap, or to have some elements of Y be both
    # an input and an output. This is especially true when the
    # input is a boundary in space and a boundary in time, and the output is
    # the solution at the opposite boundaries; so that inputs and outputs
    # together form the edge of space-time, and necessary touch.

    t00 = time()
    NN = A.shape[0]

    # We label y indices with a _ and x indices without it.
    i0_list_ = input_indices[:]
    if_list_ = output_indices[:]
    # We get index lists where i0_list has no internal repetitions.
    # And if_list has no internal repetitions or repetitions with i0_list
    # In other words, we get index lists with no repetitions at all,
    # and where indices that are repeated between the lists are kept in
    # i0_list.

    i0_list, if_list = filter_out_indices(i0_list_, if_list_)

    N0 = len(i0_list)
    Nf = len(if_list)
    Nm = NN - N0 - Nf
    ini0 = 0; fin0 = N0
    inif = N0; finf = N0+Nf
    inim = finf; finm = NN

    N0_ = len(i0_list_)
    Nf_ = len(if_list_)

    if verbose > 0: print(10, time()-t00)

    t0 = time()
    # We build the permutation matrix, which since it uses non repeated
    # indices, is a true permutation matrix.
    P = build_permutation(NN, i0_list, if_list)
    if verbose > 0: print(111, time()-t0)

    # We transform A (without expanding it).
    t0 = time()
    A = P*A*P.T
    if verbose > 0: print(222, time()-t0)

    # We build the W matrix.
    t0 = time()
    No = Nf+Nm
    W = bmat([[sp_eye(N0), bfmtf((N0, No))],
              [bfmtf((No, N0)), A[inif:finm, inif:finm]]], format=bfmt)
    if verbose > 0: print(333, time()-t0)

    # If we are given an input, proceed naively.
    if input is not None:
        t0 = time()
        B = bmat([[sp_eye(N0), bfmtf((N0, Nf)), bfmtf((N0, Nm))],
                  [-A[inif:finf, ini0:fin0], bfmtf((Nf, Nf)), bfmtf((Nf, Nm))],
                  [-A[inim:finm, ini0:fin0], bfmtf((Nm, Nf)),
                  bfmtf((Nm, Nm))]],
                 format=bfmt)

        ####################################################################
        # We bring the input from the original indices i0_list_ to
        # i0_list indices.

        Pi00_ = np.zeros((N0, N0_))
        used = []
        for j, indj in enumerate(i0_list_):
            i = i0_list.index(indj)

            if i not in used:
                Pi00_[i, j] = 1
            used += [i]
        ####################################################################
        # We solve the system.
        if len(input.shape) == 1:
            Ncols = 1
            input = np.reshape(input, (N0, 1))
        else:
            Ncols = input.shape[1]
        Yb = np.zeros((NN, Ncols), dtype=input.dtype)
        Yb[:N0, :] = np.dot(Pi00_, input)
        b = B*Yb

        if verbose > 0: print(444, time()-t0)

        t0 = time()
        Y = spsolve(W, b)
        if len(Y.shape) == 1:
            Y = np.reshape(Y, (Y.shape[0], 1))
        if verbose > 0: print(555, time()-t0)

        t0 = time()
        Y = P.T * Y

        ####################################################################
        # We bring the output from the if_list indices to the if_list_ indices.
        Yf = Y[np.array(if_list_)]
        if verbose > 0: print(666, time()-t0)
        return Yf

    #########################################################################
    # If we are not given an input, calculate the Green's function.
    #########################################################################
    # OLD METHOD
    # t0 = time()
    # # ids = lil_matrix((NN, Nf))
    # # ids[inif:finf, :] = sp_eye(Nf)
    #
    # # We build the 1_s.
    # ids = bmat([[bfmtf((N0, Nf))],
    #             [sp_eye(Nf, format=bfmt)],
    #             [bfmtf((Nm, Nf))]],
    #            format=bfmt)
    # ids = ids.todense()
    # if verbose > 0:
    #     t1 = time()-t0
    #     print()
    #     print(444, t1)
    #
    # # We solve the system!
    # t0 = time()
    # block_row = spsolve(W.T, ids).T
    # if verbose > 0:
    #     t2 = time()-t0
    #     print(555, t2)
    #
    # # We extract the matrix B (it doesn't matter that the A_00 block is not
    # # the identity, since the block_row_f0 block is zero). It is slightly
    # # wasteful though to multiply them, because they are dense matrices.
    # t0 = time()
    # block_col = A[:, ini0:fin0]
    # if verbose > 0:
    #     t3 = time()-t0
    #     print(666, t3)
    #
    # # We do the matrix multiplication.
    # t0 = time()
    # G = -block_row*block_col
    # if verbose > 0:
    #     t4 = time()-t0
    #     print(777, t4, t1+t2+t3+t4)
    # OLD METHOD
    #########################################################################
    # We build the B matrix.
    t0 = time()
    B = bmat([[bfmtf((N0, N0))],
              [-A[inif:finm, ini0:fin0]]], format=bfmt)
    B = B.todense()
    # W = csc_matrix(W)
    if verbose > 0:
        t1 = time()-t0
        print()
        print(8888, t1)

    t0 = time()
    G_full = spsolve(W, B)
    if verbose > 0:
        t2 = time()-t0
        print(9999, t2)

    t0 = time()
    G = G_full[inif: finf, :]
    if verbose > 0:
        t3 = time()-t0
        print(1111, t3, t1+t2+t3)

    if full_propagator:
        raise NotImplementedError

    if i0_list != i0_list_ or if_list != if_list_:
        # We expand G using copy-making matrices to account for the indices
        # that were eliminated when we transformed
        #
        # i0_list_  --->  i0_list
        # if_list_  --->  if_list
        #
        # So that instead of a map between the reduced indices, we have a map
        # between the original indices i0_list_, if_list_. This mapping takes
        # the for
        #
        # G_ = (Pf_f*G + Pf_0)*Pi00_

        t0 = time()
        ##################################################
        Pf_f = np.zeros((Nf_, Nf))
        for i, indi in enumerate(if_list_):
            try:
                j = if_list.index(indi)
            except:
                j = None
            # print(i, indi, j)
            if j is not None:
                Pf_f[i, j] = 1
        ##################################################
        Pf_0 = np.zeros((Nf_, N0))
        for i, indi in enumerate(if_list_):
            try:
                j = i0_list.index(indi)
            except:
                j = None
            # print(i, indi, j)
            if j is not None:
                Pf_0[i, j] = 1
        ##################################################
        Pi00_ = np.zeros((N0, N0_))
        used = []
        for j, indj in enumerate(i0_list_):
            i = i0_list.index(indj)

            if i not in used:
                Pi00_[i, j] = 1
            used += [i]
        ##################################################
        # print(Pf_f)
        # print(Pf_0)
        # print(Pi00_)
        if verbose > 0: print(888, time()-t0)

        t0 = time()
        G = np.dot((np.dot(Pf_f, G) + Pf_0), Pi00_)
        if verbose > 0: print(999, time()-t0)

    if plots:
        plt.imshow(np.abs(G))
        plt.savefig(folder+name+".png", bbox_inches="tight")
        plt.close()
    return G


def input_harmonics(s, Ds, Ds_exact, threshold=1e-2,
                    plots=False, folder="", name=""):
    r"""Return an array of input harmonics over domain `s` that are well
    such that a derivative operator of order `d` and precission degree `p`
    represent them well.
    """
    Ns = s.shape[0]
    L = s[-1] - s[0]
    h = np.zeros((Ns, Ns))
    Dsall = np.zeros((Ns, Ns))
    Dsall_exact = np.zeros((Ns, Ns))
    s0 = s[0]
    sf = s[-1]
    sm = (s0+sf)/2
    sp = s - sm

    for j in range(Ns-2):
        hj = harmonic(j+1, sp, L)
        # print(hj)
        Dshj_exact = Ds_exact(j+1, sp, L)
        Dshj = np.dot(Ds, hj)

        errj = np.mean(glo_error(Dshj_exact, Dshj))
        Nred = j

        if errj <= threshold:
            h[:, j] = hj
            Dsall[:, j] = Dshj
            Dsall_exact[:, j] = Dshj_exact
        else:
            break

    flag = False
    if Nred == 0:
        warnings.warn("The grid does not allow precise differentiation.")
        flag = True
        Nred = 1

    h = h[:, :Nred]
    Dsall = Dsall[:, :Nred]
    Dsall_exact = Dsall_exact[:, :Nred]
    # print(Nred, h.shape)

    if plots and not flag:
        fs = 15
        fig, ax = plt.subplots(3, 1, figsize=(15, 8))
        try:
            vmax = np.amax(np.array([Dsall, Dsall_exact]))*1.2
            vmin = np.amin(np.array([Dsall, Dsall_exact]))*1.2
        except:
            vmax = None
            vmin = None
        ax2 = plt.twinx(ax[1])
        if Nred > 4:
            indices = [0, 1, -2, -1]
        else:
            indices = range(Nred)

        for j in indices:
            ax[0].plot(s, h[:, j])
            ax[1].plot(s, Dsall[:, j], "x", ms=10)
            ax2.plot(s, Dsall_exact[:, j], "+", ms=10)
            aux = np.abs(glo_error(Dsall[:, j], Dsall_exact[:, j]))
            ax[2].semilogy(s, aux)

        ax[1].set_ylim(vmin, vmax)
        ax[2].set_ylim(1e-6, 1)
        ax[2].grid(True)

        ax[0].set_ylabel("Harmonics", fontsize=fs)
        ax[1].set_ylabel("Derivative", fontsize=fs)
        ax[2].set_ylabel("Derivative Error", fontsize=fs)

        ax2.set_ylim(vmin, vmax)
        plt.savefig(folder+"input_harmonics"+name+".png", bbox_inches="tight")
        plt.close("all")

    return h


def harmonic_prime(n, x, L):
    r"""Return the derivative of a harmonic."""
    omega = np.pi/L
    hp = n*omega*np.cos(n*omega*(x+L/2))/np.sqrt(L/2)
    return hp


if __name__ == "__main__":
    import doctest
    print(doctest.testmod(verbose=False))
