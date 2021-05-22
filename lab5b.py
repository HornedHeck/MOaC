import numpy as nmp


def horizontal_move(u_m: nmp.ndarray, i_0: int, j_0: int) -> int:
    j = j_0 - 1
    while j >= 0:
        if u_m[i_0, j] != 0:
            return j
        j -= 1
    j = j_0 + 1
    while j < u_m.shape[1]:
        if u_m[i_0, j] != 0:
            return j
        j += 1
    raise Exception("horizontal move error")


def vertical_move(u_m: nmp.ndarray, i_0: int, j_0: int) -> int:
    i = i_0 - 1
    while i >= 0:
        if u_m[i, j_0] != 0:
            return i
        i -= 1
    i = i_0 + 1
    while i < u_m.shape[0]:
        if u_m[i, j_0] != 0:
            return i
        i += 1
    raise Exception("vertical move error")


def get_cycle(u_b: nmp.ndarray, i_0: int, j_0: int, shape: tuple) -> tuple:
    u_m = nmp.zeros(shape)
    for i, j in u_b:
        u_m[i, j] = 1

    u_m[i_0, j_0] = 2
    is_any_changed = True
    while is_any_changed:
        is_any_changed = False
        rows = nmp.count_nonzero(u_m, 1)
        for i in range(rows.shape[0]):
            if rows[i] == 1:
                is_any_changed = True
                u_m[i] = nmp.zeros(shape[1])

        columns = nmp.count_nonzero(u_m, 0)
        for i in range(columns.shape[0]):
            if columns[i] == 1:
                is_any_changed = True
                u_m[:, i] = nmp.zeros(shape[0])

    j = horizontal_move(u_m, i_0, j_0)
    i = vertical_move(u_m, i_0, j)

    u_h = [[i_0, j]]
    u_v = [[i, j]]

    while u_m[i, j] != 2:
        j = horizontal_move(u_m, i, j)
        u_h.append([i, j])
        i = vertical_move(u_m, i, j)
        u_v.append([i, j])

    return u_h, u_v


def is_basis(u_b: nmp.ndarray, i_0: int, j_0: int) -> bool:
    for i in range(u_b.shape[0]):
        if i_0 == u_b[i, 0] and j_0 == u_b[i, 1]:
            return True
    return False


def apply_u(a: nmp.ndarray, i_j_l: nmp.ndarray) -> nmp.ndarray:
    return nmp.array([a[i, j] for i, j in i_j_l])


def solve_transport(c: nmp.ndarray, x_0: nmp.ndarray, d: nmp.ndarray, u_b_0: nmp.ndarray) -> nmp.ndarray:
    x = x_0
    u_b = u_b_0

    while True:
        print(x)
        u, v = get_uv(c, u_b, c.shape[0], c.shape[1], d)

        deltas = nmp.array([
            [u[i] + v[j] - c[i, j] for j in range(c.shape[1])]
            for i in range(c.shape[0])
        ])

        res, i_0, j_0 = is_solved(deltas, u_b, d, x)
        if res:
            return x

        if x[i_0, j_0] == 0:
            k = 1
        else:
            k = -1

        # u_h == u_-
        # u_v == u_+
        u_h, u_v = get_cycle(u_b, i_0, j_0, x.shape)

        if k == -1:
            u_h, u_v = u_v, u_h

        theta_h = apply_u(x, u_h)
        theta_v = apply_u(d, u_v) - apply_u(x, u_v)

        h_0 = theta_h.argmin()
        v_0 = theta_v.argmin()

        if theta_h[h_0] < theta_v[v_0]:
            theta_0 = theta_h[h_0]
            i_k, j_k = u_h[h_0]
        else:
            theta_0 = theta_v[v_0]
            i_k, j_k = u_v[v_0]

        for i, j in u_h:
            x[i, j] -= theta_0

        for i, j in u_v:
            x[i, j] += theta_0

        i_b_replace = nmp.all(u_b == [i_k, j_k], 1).argmax()
        u_b[i_b_replace] = nmp.array([i_0, j_0])


def is_delta_wrong(delta: float, x: float, d: float) -> bool:
    return x == 0 and delta > 0 or x == d and delta < 0


def is_solved(deltas: nmp.ndarray, u_b: nmp.ndarray, x: nmp.ndarray, d: nmp.ndarray):
    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            if not is_basis(u_b, i, j) and is_delta_wrong(deltas[i, j], x[i, j], d[i, j]):
                return False, i, j
    return True, -1, -1


def get_max_potential_with_indexes(u: nmp.ndarray, v: nmp.ndarray, c: nmp.ndarray):
    max_delta = 0
    max_i = -1
    max_j = -1
    for i in range(len(u)):
        for j in range(len(v)):
            potential = u[i] + v[j]
            if potential > c[i][j] and potential - c[i][j] > max_delta:
                max_delta = potential - c[i][j]
                max_i = i
                max_j = j
    return max_delta, max_i, max_j


def get_uv_line(i, j, size, a_size) -> nmp.ndarray:
    res = nmp.zeros(size)

    res[i] = 1

    res[j + a_size] = 1
    return res


def get_uv(c: nmp.ndarray, u_b: nmp.ndarray, a_size, b_size, d: nmp.ndarray):
    equations = []
    b = []
    for i, j in u_b:
        equations.append(get_uv_line(i, j, a_size + b_size, a_size))
        b.append(c[i][j])

    u_0 = nmp.zeros(a_size + b_size)
    u_0[0] = 1
    equations.append(u_0)

    b.append(1)

    equations = nmp.array(equations)
    b = nmp.array(b)

    if len(equations) == 0:
        res = nmp.zeros(a_size + b_size - 1)
    else:
        res = nmp.linalg.solve(equations, b)
    u = [1]
    for i in range(a_size):
        u.append(res[i])

    u = nmp.array(u)

    v = res[a_size:]

    return u, v


x_1 = nmp.array([
    [4, 3, 10, 0, 10, 0],
    [10, 0, 0, 2, 0, 10],
    [0, 0, 6, 0, 5, 9],
    [0, 0, 10, 7, 0, 0],
    [10, 5, 0, 0, 10, 4]
])

u_b = nmp.array([
    [0, 0], [0, 1], [1, 3], [2, 2], [2, 4], [2, 5], [3, 3], [3, 5], [4, 5], [4, 1]
])

d_1 = nmp.ones(x_1.shape) * 10
# a_1 = nmp.array([27, 22, 20, 17, 29])
# b_1 = nmp.arra
c_1 = nmp.array([
    [1, 1, 20, -4, 15, -1],
    [10, -5, -3, 1, -1, 2],
    [-1, 2, 4, -5, 2, 1],
    [4, 3, 40, 6, -6, -20],
    [5, 10, -10, -3, 15, 5]
])

print(solve_transport(c_1, x_1, d_1, u_b))

# print(get_cycle(u_b, 0, 2, x_1.shape))
