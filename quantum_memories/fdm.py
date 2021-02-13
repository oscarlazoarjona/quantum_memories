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

from matplotlib import pyplot as plt

from sympy import zeros, Matrix
from sympy import factorial as sym_fact
from math import factorial as num_fact

# from misc import (interpolator, hermite_gauss, rel_error, glo_error)
# from orca import (calculate_kappa, calculate_Gamma21, calculate_Gamma32,
#                   calculate_Omega)

# from graphical import plot_solution


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


def impose_boundary(Wp, tau, S0t, S0z, B0z, P0z=None, sparse=False):
    r"""Impose boudary conditions."""
    # We unpack parameters.
    if P0z is not None:
        nv = 3
    else:
        nv = 2
    # We unpack parameters.
    if True:
        Nt = len(S0t)
        Nz = len(S0z)
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


if __name__ == "__main__":
    import doctest
    print(doctest.testmod(verbose=False))
