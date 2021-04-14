import random

import numpy as nmp


def __optimised_mul__(
        q, a_r, k
):
    """
        Parameters
        ----------
        q : numpy.ndarray
            Q Square matrix (almost E)
        a_r : numpy.ndarray
            Square matrix, opposite to A
        k : int
            Number of replaced column in Q, starting from 0

        Return
        ------
        O(n^2) q * a_r
        """
    n = len(q)
    return nmp.array([
        [
            q[i][k] * a_r[k][j] if i == k
            else a_r[i][j] + q[i][k] * a_r[k][j]
            for j in range(n)]
        for i in range(n)])


def fast_revers(
        x,
        a_r,
        i
):
    """
    Parameters
    ----------
    x : numpy.ndarray
        Replacement vector
    a_r : numpy.ndarray
        Square matrix, opposite to A
    i : int
        Number of column to replace, starting from 0

    Return
    ------
    opposite matrix if exists else None
    """

    l = a_r.dot(x)
    if l[i] == 0:
        print("Opposite matrix does not exists")
        return None

    e = l.copy()
    e[i] = -1
    l_r = -1 / l[i] * e
    q = nmp.eye(len(x))
    q[:, i] = l_r
    return __optimised_mul__(q, a_r, i)


# M = nmp.random.randint(-10, 10, size=(4, 4))
# print(M)
# M_R = nmp.linalg.inv(M)
# print(M_R)
# D = M.copy()
# i = random.randint(0, 3)
# print(i)
# D[:, i] = nmp.random.randint(-5, 5, 4)
# print(D)
# D_R = nmp.linalg.inv(D)
# D_R_A = fast_revers(D[:, i], M_R, i)
# print(D_R)
# print(D_R_A)
# print((nmp.absolute(D_R - D_R_A) < 0.00001).all())
