import numpy as nmp


def twin_simplex(a: nmp.ndarray, b: nmp.ndarray, c: nmp.ndarray, j_b_0: nmp.ndarray):
    j_b = j_b_0
    j_n = nmp.setdiff1d(range(len(c)), j_b)

    a_b_r = nmp.linalg.inv(a[:, j_b])

    d_n = c[j_b].dot(a_b_r).dot(a[:, j_n]) - c[j_n]

    while True:
        ksi_b = a_b_r.dot(b)
        if (ksi_b >= 0).all():
            ksi = nmp.hstack((nmp.zeros(len(j_n)), ksi_b))
            ksi_ind = nmp.argsort(nmp.hstack((j_n, j_b)))
            return ksi[ksi_ind]

        k = nmp.nanargmax(ksi_b < 0)
        mu = a_b_r[k].dot(a[:, j_n])

        sigma = nmp.array([d_n[j] / mu[j] if mu[j] < 0 else nmp.inf for j in range(len(mu))])

        sigma_0_i = sigma.argmin()

        if sigma[sigma_0_i] == nmp.inf:
            print('Ограничения прямой задачи несовместны')
            return None

        # swap k and sigma_0_i
        temp = j_b[k]
        j_b[k] = j_n[sigma_0_i]
        j_n[sigma_0_i] = temp

        d_n = d_n + sigma[sigma_0_i] * mu
        d_n[sigma_0_i] = sigma[sigma_0_i]

        a_b_r = nmp.linalg.inv(a[:, j_b])


a_0 = nmp.array([
    [1, -5, 1, 0],
    [-3, 1, 0, 1]
])

b_0 = nmp.array([-10, -12])

c_0 = nmp.array([0, -6, 1, 0])

j_b_0 = nmp.array([2, 3])

print(twin_simplex(a_0, b_0, c_0, j_b_0))
