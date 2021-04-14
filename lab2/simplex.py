import numpy as nmp

from lab1.fast_revers import fast_revers

# A0 = nmp.array([
#     [1, 2, 1, 0, 0, 0],
#     [2, 1, 0, 1, 0, 0],
#     [1, 0, 0, 0, 1, 0],
#     [0, 1, 0, 0, 0, 1]
# ])
#
# c0 = nmp.array([20, 26, 0, 0, 0, 1])
#
# B0 = nmp.array([2, 3, 4, 5], dtype=int)
# x0 = nmp.array([0, 0, 10, 11, 5, 4])


def simplex_solve(A, c, b_0, x_0):
    x = x_0
    b = b_0
    a_b_r = nmp.eye(len(b))
    n = nmp.setdiff1d(range(len(c)), b)

    while True:
        dn = (c[n] - c[b].dot(a_b_r).dot(A[:, n]))
        # print("DN: ", dn)
        # Index of variable to enter the basis
        in_i = nmp.where(dn > 0)[0]
        in_i = in_i[0] if len(in_i) > 0 else 0
        # print("In_i: " , in_i)

        # If there is no positive variables exit
        if dn[in_i] <= 0:
            break

        abr_a = a_b_r.dot(A[:, n[in_i]])
        thetas = nmp.array([x[b[i]] / abr_a[i] if abr_a[i] != 0 else -1 for i in range(len(b))])

        # print("Thetas: ", thetas)

        valid_theta_idx = nmp.where(thetas > 0)[0]
        # Index of variable to exit the basis
        out_i = valid_theta_idx[thetas[valid_theta_idx].argmin()]
        # print("Thetas_i: ", out_i)

        theta = thetas[out_i]

        x_b = x[b.astype(int)] - abr_a * theta
        # print("X_b: ", x_b)
        x[n[in_i]] += theta
        x = nmp.array([x_b[nmp.where(b == i)[0][0]] if i in b else x[i] for i in range(len(x))])

        # print("New_x: ", x)

        # print(b)
        # update B
        tmp = b[out_i]
        b[out_i] = n[in_i]
        n[in_i] = tmp
        # print(b)
        # print(out_i, tmp)

        a_b_r = fast_revers(A[:, b[out_i]], a_b_r, out_i)
        # print("New_abr: ", a_b_r)

    return x


# print("A: ")
# print(A0)
# print("c: ", c0)
# print("x: ", x0)
#
# print("Result:", simplex_solve(A0, c0, B0, x0))
