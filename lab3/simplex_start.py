from typing import Optional

import numpy as nmp

from lab1.fast_revers import fast_revers


def simplex_solve_min(a, c, b_0, x_0):
    x = x_0
    b = b_0
    a_b_r = nmp.eye(len(b))
    n = nmp.setdiff1d(range(len(c)), b)

    while True:
        dn = (c[n] - c[b].dot(a_b_r).dot(a[:, n]))
        in_i = nmp.where(dn < 0)[0]
        in_i = in_i[0] if len(in_i) > 0 else 0

        # If there is no positive variables exit
        if dn[in_i] >= 0:
            break

        abr_a = a_b_r.dot(a[:, n[in_i]])
        thetas = nmp.array([x[b[i]] / abr_a[i] if abr_a[i] != 0 else -1 for i in range(len(b))])

        valid_theta_idx = nmp.where(thetas > 0)[0]
        # Index of variable to exit the basis
        out_i = valid_theta_idx[thetas[valid_theta_idx].argmin()]

        theta = thetas[out_i]

        x_b = x[b.astype(int)] - abr_a * theta
        x[n[in_i]] += theta
        x = nmp.array([x_b[nmp.where(b == i)[0][0]] if i in b else x[i] for i in range(len(x))])

        # update B
        tmp = b[out_i]
        b[out_i] = n[in_i]
        n[in_i] = tmp

        a_b_r = fast_revers(a[:, b[out_i]], a_b_r, out_i)

    return x


def simplex_first_step(a: nmp.ndarray, b: nmp.ndarray) -> Optional[nmp.ndarray]:
    eye = nmp.eye(len(b))

    a_1 = nmp.hstack((a, eye))

    c_1 = nmp.hstack((nmp.zeros(a.shape[1]), nmp.ones(len(b))))

    x_1 = nmp.hstack((nmp.zeros(a.shape[1]), b))

    solve_res = simplex_solve_min(a_1, c_1, nmp.array([4, 5, 6]), x_1)

    z_v = solve_res[a.shape[1]:].sum()
    if z_v == 0:
        return solve_res[:a.shape[1]]
    else:
        return None


a_0 = nmp.array([
    [-1, 3, 1, 0],
    [4, 3, 0, 1],
    [2, 9, 2, 1]
])

b_0 = nmp.array([9, 24, 42])

res = simplex_first_step(a_0, b_0)

if res is None:
    print('Несовместна')
else:
    print(res)
