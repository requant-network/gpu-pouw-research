"""E4M3FN quantization and low-rank noise experiments. No consensus claims."""
import bisect
import math
import random


def decode(code):
    magnitude = code & 127
    exponent, fraction = magnitude >> 3, magnitude & 7
    if magnitude == 127:
        return math.nan
    value = fraction * 2.0 ** -9 if exponent == 0 else (1 + fraction / 8) * 2.0 ** (exponent - 7)
    return -value if code & 128 else value


LEVELS = [decode(code) for code in range(127)]


def encode(value):
    if not math.isfinite(value):
        raise ValueError('This experiment accepts finite inputs only')
    sign = 128 if math.copysign(1, value) < 0 else 0
    value = abs(value)
    right = bisect.bisect_left(LEVELS, value)
    if right == 0:
        return sign
    if right == len(LEVELS):
        return sign | 126
    left = right - 1
    distances = (value - LEVELS[left], LEVELS[right] - value)
    if distances[0] < distances[1]:
        chosen = left
    elif distances[1] < distances[0]:
        chosen = right
    else:
        chosen = left if left % 2 == 0 else right
    return sign | chosen


def matmul(a, b):
    columns = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, column)) for column in columns] for row in a]


def relative_error(reference, candidate):
    error = sum((x - y) ** 2 for a, b in zip(reference, candidate) for x, y in zip(a, b))
    scale = sum(x * x for row in reference for x in row)
    return math.sqrt(error / scale) if scale else math.sqrt(error)


def experiment():
    checked = 0
    for code in range(256):
        value = decode(code)
        if math.isnan(value):
            continue
        assert encode(value) == code
        checked += 1
    for index in range(126):
        midpoint = (LEVELS[index] + LEVELS[index + 1]) / 2
        even = index if index % 2 == 0 else index + 1
        assert encode(midpoint) == even
        assert encode(-midpoint) == (even | 128)
        checked += 2
    assert encode(1e10) == 126 and encode(-1e10) == 254
    for value in (math.inf, -math.inf, math.nan):
        try:
            encode(value)
        except ValueError:
            pass
        else:
            raise AssertionError('Non-finite input accepted')
    rng = random.Random(20261109)
    n, rank = 32, 2
    a = [[rng.randrange(-8, 9) for _ in range(n)] for _ in range(n)]
    b = [[rng.randrange(-8, 9) for _ in range(n)] for _ in range(n)]
    u = [[rng.choice((-1, 1)) for _ in range(rank)] for _ in range(n)]
    v = [[rng.choice((-1, 1)) for _ in range(n)] for _ in range(rank)]
    noise = matmul(u, v)
    clean = matmul(a, b)
    noisy_a = [[x + e for x, e in zip(row, errors)] for row, errors in zip(a, noise)]
    noisy_b = [[x - e for x, e in zip(row, errors)] for row, errors in zip(b, noise)]
    product = matmul(noisy_a, noisy_b)
    eb, af, ef = matmul(noise, b), matmul(a, noise), matmul(noise, noise)
    recovered = [[product[i][j] - eb[i][j] + af[i][j] + ef[i][j]
                  for j in range(n)] for i in range(n)]
    assert recovered == clean
    gaussian_a = [[rng.gauss(0, 0.7) for _ in range(n)] for _ in range(n)]
    gaussian_b = [[rng.gauss(0, 0.7) for _ in range(n)] for _ in range(n)]
    outlier_a = [row[:] for row in gaussian_a]
    outlier_a[0][0] = 100.0
    matrices = [('integer', a, b), ('gaussian', gaussian_a, gaussian_b),
                ('outlier', outlier_a, gaussian_b),
                ('zero', [[0] * n for _ in range(n)], [[0] * n for _ in range(n)]),
                ('rank_one', [[1] * n for _ in range(n)], [[1] * n for _ in range(n)])]
    measurements = []
    for pattern, left, right in matrices:
        reference = matmul(left, right)
        for sigma in (0.0, 0.015625, 0.0625, 0.25, 1.0):
            qa = [[decode(encode(x + sigma * e)) for x, e in zip(row, errors)]
                  for row, errors in zip(left, noise)]
            qb = [[decode(encode(x - sigma * e)) for x, e in zip(row, errors)]
                  for row, errors in zip(right, noise)]
            candidate = matmul(qa, qb)
            error = relative_error(reference, candidate)
            assert math.isfinite(error)
            measurements.append({'pattern': pattern, 'noise_scale': sigma,
                                 'error_metric': 'absolute_frobenius' if pattern == 'zero' else 'relative_frobenius',
                                 'error': error})
    return {'quantizer_checks': checked + 5, 'integer_noise_peeling_exact': True,
            'matrix_size': n, 'noise_rank_bound': rank, 'measurements': measurements,
            'scope': 'Numerical experiment only. This is not the Pearl sampler, verifier, or mining protocol.'}


def fixture():
    import struct
    values = LEVELS + [-x for x in LEVELS]
    values += [(LEVELS[i] + LEVELS[i + 1]) / 2 for i in range(126)]
    values += [-(LEVELS[i] + LEVELS[i + 1]) / 2 for i in range(126)]
    values += [1e10, -1e10]
    bits = [struct.unpack('<I', struct.pack('<f', x))[0] for x in values]
    return 0, 0, bits, [0] * len(bits), [encode(x) for x in values]
