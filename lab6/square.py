import numpy as nmp


def solve_square(a: nmp.ndarray, b: nmp.ndarray, c: nmp.ndarray, d: nmp.ndarray, x_0: nmp.ndarray, j_b_0: nmp.ndarray):
    j_base = j_b_0
    j_s = j_b_0
    x = x_0

    while True:
        a_b_r = nmp.linalg.inv(a[:, j_base])
        j_n = nmp.setdiff1d(nmp.arange(len(c)), j_s)

        c_x = c + d.dot(x)
        u = -c_x[j_base].dot(a_b_r)

        d_n = u.dot(a[:, j_n]) + c_x[j_n]

        if (d_n >= 0).all():
            return x

        j_k = nmp.argmax(d_n < 0)
        j_0 = j_n[j_k]

        l = nmp.zeros(len(j_n))
        l[j_k] = 1

        h = nmp.vstack((
            nmp.hstack((d[j_s][:, j_s], a[:, j_s])),
            nmp.hstack((a[:, j_s], nmp.zeros((len(j_s), len(j_s))))),
        ))

        bb = nmp.hstack((d[j_s, j_0], a[:, j_0]))

        l_dy = nmp.linalg.inv(h).dot(bb)

        l = nmp.hstack((l, l_dy[:len(j_s)]))
        l_i = nmp.hstack((j_n, j_s)).argsort()
        l = l[l_i]

        theta = nmp.array([nmp.inf if l[j_s[i]] >= 0 else -x[j_s[i]] / l[j_s[i]] for i in range(len(j_s))])
        delta = l.dot(d).dot(l)
        if delta == 0:
            theta = nmp.append(theta, [nmp.inf])
        else:
            theta = nmp.append(theta, [abs(d_n[j_k]) / delta])

        j_t = theta.argmin()

        if theta[j_t] == nmp.inf:
            print("Целевая функция не ограничена снизу на множестве планов")
            return None

        x = x + theta[j_t] * l

        if j_t == len(theta) - 1:
            j_t = j_0
        else:
            j_t = j_s[j_t]

        if j_t == j_0:
            j_s = nmp.append(j_s, [j_0])
            continue
        elif j_t in nmp.setdiff1d(j_s, j_base):
            j_s = nmp.setdiff1d(j_s, [j_t])
            continue
        return


def check_case_c(j_t: int, j_base: nmp.ndarray, j_s: nmp.ndarray, a_b_r: nmp.ndarray, a: nmp.ndarray) -> bool:
    part = nmp.setdiff1d(j_s, j_base)
    if len(part) == 0:
        return False

    mask = nmp.in1d(j_base, [j_t])
    s_l = nmp.arange(len(mask))[mask]

    if len(s_l) == 0:
        return False

    for s in s_l:
        s_v = a_b_r[s]
        j_plus_v = s_v.dot(a[:, part])
        if (j_plus_v != 0).any():
            return True

    return False


_a = nmp.array([
    [6, 6, 0],
    [3, 0, 1]
], dtype=float)

_b = nmp.array([3, 1], dtype=float)

_c = nmp.array([-1, 0, 0], dtype=float)

_d = nmp.array([
    [4, -2, 0],
    [-2, 4, 0],
    [0, 0, 1],
], dtype=float)

_x_0 = nmp.array([0, 0.5, 1], dtype=float)

_j_b_0 = nmp.array([1, 2])

solve_square(_a, _b, _c, _d, _x_0, _j_b_0)
