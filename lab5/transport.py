import numpy as nmp

a_0 = nmp.array([50, 50, 100])
b_0 = nmp.array([40, 90, 70])
c_0 = nmp.array([
    [2, 5, 3],
    [4, 3, 2],
    [5, 1, 2],
])

a_1 = nmp.array([0, 0, 0])
b_1 = nmp.array([0, 0, 0])
c_1 = nmp.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
])


def solve_transport(a: nmp.ndarray, b: nmp.ndarray, c: nmp.ndarray) -> nmp.ndarray:
    if a.sum() != b.sum():
        print("Can't be solved")
        return nmp.zeros(0)

    x = nw_corner(a, b)

    u, v = get_uv(c, x, len(a), len(b))

    d, d_i, d_j = get_max_potential_with_indexes(u, v, c)

    while d != 0:

        lt_j = d_j - 1
        while x[d_i][lt_j] == 0:
            lt_j -= 1

        rb_i = d_i + 1
        while x[rb_i][d_j] == 0:
            rb_i += 1

        delta = min(x[d_i, lt_j], x[rb_i, d_j])
        x[d_i][lt_j] -= delta
        x[rb_i][d_j] -= delta
        x[d_i][d_j] += delta
        x[rb_i][lt_j] += delta

        u, v = get_uv(c, x, len(a), len(b))
        d, d_i, d_j = get_max_potential_with_indexes(u, v, c)

    return x


def is_solved(u: nmp.ndarray, v: nmp.ndarray, c: nmp.ndarray) -> bool:
    for i in range(len(u)):
        for j in range(len(v)):
            if u[i] + v[j] > c[i][j]:
                return False
    return True


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


def nw_corner(a: nmp.ndarray, b: nmp.ndarray) -> nmp.ndarray:
    _a = a.copy()
    _b = b.copy()
    res = nmp.zeros((len(_a), len(_b)))
    i = 0
    j = 0
    while i < len(_a) and j < len(_b):
        if _a[i] < _b[j]:
            _b[j] -= _a[i]
            res[i][j] = _a[i]
            _a[i] = 0
            i += 1
        else:
            _a[i] -= _b[j]
            res[i][j] = _b[j]
            _b[j] = 0
            j += 1
    return res


def get_uv_line(i, j, size, a_size) -> nmp.ndarray:
    res = nmp.zeros(size)
    if i != 0:
        res[i - 1] = 1

    res[j + a_size - 1] = 1
    return res


def get_uv(c: nmp.ndarray, x: nmp.ndarray, a_size, b_size):
    equations = []
    b = []
    for i in range(a_size):
        for j in range(b_size):
            if x[i][j] != 0:
                equations.append(get_uv_line(i, j, a_size + b_size - 1, a_size))
                b.append(c[i][j])

    equations = nmp.array(equations)
    b = nmp.array(b)

    if len(equations) == 0:
        res = nmp.zeros(a_size + b_size - 1)
    else:
        res = nmp.linalg.solve(equations, b)
    u = [0]
    for i in range(a_size - 1):
        u.append(res[i])

    u = nmp.array(u)

    v = res[a_size - 1:]

    return u, v


# x_r = nw_corner(a_0, b_0)

# u_r, v_r = get_uv(c_0, x_r, 3, 3)

print(solve_transport(a_1, b_1, c_1))
