import cmath
from math import cos, sin, pi

import numpy as np


def generate_function_values(function, beginning_value, amount, step):
    return [function(x) for x in (np.arange(0, amount) * step + beginning_value)]


###########################
# for(int m = 0; m < N;m++){
#    w = -j*2*pi*m/N
#    for(int n = 0; n < N; n++){
#        res += data[n]*e^(w*n); 
#    }
#    res_array.add(res);
# }

# Discrete fourier transform
def dft(input_data, length):
    if length:
        N = length
    else:
        N = len(input_data)

    output = []
    for m in range(N):
        res = complex(0)
        w = -2j * np.pi * m / N
        for k in range(N):
            res += input_data[k] * cmath.exp(-(w * k))
        output.append(res)
    return output


# Inverse discrete fourier transform
def inverse_dft(input_data, length):
    if length:
        N = length
    else:
        N = len(input_data)

    output = []
    for m in range(N):
        res = 0
        w = 2j * cmath.pi * m / N
        for k in range(N):
            res += input_data[k] * cmath.exp(w * k)
        res /= N
        output.append(res)
    return output


# Configurable dft(dft+inverse_dft)
def conf_dft(input_data, length, direction):
    if length:
        n = length
    else:
        n = len(input_data)

    output = []
    for m in range(n):
        s = complex(0)
        w = -2j * cmath.pi * m / n
        for k in range(n):
            angle = w * k
            s += input_data[k] * cmath.exp(direction * angle)
        if direction == -1:
            s /= n
        output.append(s)
    return output


# dif-fft implementation
def conf_dif_fft(input_data, input_length, direction):
    fft_result = dif_fft(input_data, input_length, direction)

    if direction == -1:
        for i in range(input_length):
            fft_result[i] /= input_length

    return fft_result


##def dif_fft(input_data, input_length, direction=1):
#   if is_power_of_two(input_length):
#       length = input_length
#       bits_in_length = int(np.log2(length))
#   else:
#       bits_in_length = np.log2(input_length)
#       length = 1 << bits_in_length
#
#   data = []
#   for i in range(length):
#       data.append(complex(input_data[i]))
#
#   for ldm in range(bits_in_length, 0, -1):  #старт, стоп, шаг
#       m = 2 ** ldm  #N = m
#       mh = int(m / 2)
#       for k in range(mh):
#           w = np.exp(direction * -2j * np.pi * k / m)
#           for r in range(0, length, m):
#               u = data[r + k]
#               v = data[r + k + mh]
#
#               data[r + k] = u + v
#               data[r + k + mh] = (u - v) * w
#
#   for i in range(length):
#       j = reverse_bits(i, bits_in_length)
#       if j > i:
#           temp = data[j]
#           data[j] = data[i]
#           data[i] = temp
#
#   return data


def dif_fft(a, n, direction=1):
    if len(a) == 1:
        return a
    w_n = complex(cos(2 * pi / n), direction * sin(2 * pi / n))
    w = 1
    b = []
    c = []
    for j in range(n // 2):
        b.append(a[j] + a[j + n // 2])
        c.append((a[j] - a[j + n // 2]) * w)
        w = w * w_n

    yb = dif_fft(b, n // 2, direction)
    yc = dif_fft(c, n // 2, direction)

    y = []
    for i in range(n // 2):
        y.append(yb[i])
        y.append(yc[i])

    return y


def fft_shift(fft_result):
    length = len(fft_result)

    first_half = fft_result[0:int(length / 2)]
    second_half = first_half[:]
    first_half.reverse()
    shifted_result = first_half + second_half

    return shifted_result


# возвращает степень 2
def is_power_of_two(n):
    return n > 1 and (n & (n - 1)) == 0


def reverse_bits(n, bits_count):
    reversed_value = 0

    for i in range(bits_count):
        next_bit = n & 1
        n >>= 1

        reversed_value <<= 1
        reversed_value |= next_bit

    return reversed_value
