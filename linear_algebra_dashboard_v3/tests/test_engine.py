"""Tests for the calculation engine. Only numpy is needed (it comes with Streamlit).

Run:  python -m pytest -q        (or)      python tests/test_engine.py
"""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linear_algebra import (Fmt, as_matrix, det_value, identity, mat_mul, parse_number,  # noqa: E402
                            solve_det, solve_inverse, solve_mul, solve_reduce, solve_system)
from samples import EXAMPLES, random_matrix  # noqa: E402

F3 = Fmt(3)


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def _np(m):
    return np.array([[float(x) for x in r] for r in m])


def _rank(m):
    return int(np.linalg.matrix_rank(_np(m)))


def _all_text(sol):
    out = [sol.title]
    for c in sol.chips:
        out += [c.label, c.value]
    for s in sol.steps:
        out += [s.tag, s.text, s.why, s.phase, *s.formulas]
    out += list(sol.lines)
    return out


# ----------------------------------------------------------------- input ----
def test_parser_accepts():
    cases = [('', F(0)), ('  ', F(0)), ('۱۲٫۵', F(25, 2)), ('−٢', F(-2)), ('.125', F(1, 8)),
             ('1e-12', F(1, 10**12)), ('-.5', F(-1, 2)), ('+3', F(3)), ('0.1', F(1, 10)), ('5.', F(5))]
    for text, want in cases:
        assert parse_number(text) == want, text


def test_parser_rejects():
    for text in ['1/0', '1/2', '2/4', 'nan', 'inf', '<script>', '2**2', '1,2', '1،5', '1e99', '1e-99',
                 '__import__("os")', '+', '--1', '1.2.3', 'x', '1 2', 'a' * 41]:
        assert _raises(parse_number, text), text


def test_decimal_input_is_exact():
    a = as_matrix([[parse_number('0.1'), parse_number('0.2')], [parse_number('0.3'), parse_number('0.4')]])
    assert det_value(a) == F(-1, 50)
    assert F3.num(det_value(a)) == '−0.02'


# --------------------------------------------------------------- display ----
def test_decimal_display():
    assert F3.num(F(1, 3)) == '0.333'
    assert F3.num(F(-1, 3)) == '−0.333'
    assert F3.num(F(2, 3)) == '0.667'
    assert F3.num(2) == '2' and F3.num(F(5, 2)) == '2.5' and F3.num(0) == '0'
    assert F3.num(F(-1, 2000)) == '−0.001'                      # half away from zero
    assert Fmt(2).num(F('0.005')) == '0.01'
    assert F3.num(F(10**12)) == '1000000000000'
    assert Fmt(1, True).num(F(5, 2)) == '۲٫۵'
    assert Fmt(6).num(F(1, 7)) == '0.142857'


def test_tiny_numbers_are_never_shown_as_zero():
    assert F3.num(F('0.00049')) == '4.9e-4'
    assert F3.num(F(1, 10**6)) != '0'
    assert F3.num(F(-1, 10**5)).startswith('−') and F3.num(F(-1, 10**5)) != '−0'
    assert F3.num(0) == '0'
    assert '-0' not in F3.num(F(-1, 10**9)).replace('−', '-')[:2]


# ---------------------------------------------------------- determinant -----
def test_determinant_value_matches_numpy():
    rng = random.Random(1)
    for n in range(1, 8):
        for _ in range(6):
            a = as_matrix(random_matrix(n, n, rng))
            want = np.linalg.det(_np(a))
            assert abs(float(det_value(a)) - want) < 1e-6 * max(1.0, abs(want)), (n, a)


def test_laplace_every_row_and_column():
    rng = random.Random(2)
    for n in range(1, 6):
        for _ in range(3):
            a = as_matrix(random_matrix(n, n, rng))
            want = det_value(a)
            for axis in ('row', 'col'):
                for idx in range(n):
                    sol = solve_det(a, axis, idx, F3)
                    assert sol.meta['value'] == want
                    if n > 1:
                        assert sum(sol.meta['terms'], F(0)) == want
                        assert len(sol.steps) == n + 2
                    assert sol.scalar == ('det(A)', F3.num(want))


def test_laplace_bad_input():
    assert _raises(solve_det, as_matrix([[1, 2, 3], [4, 5, 6]]), 'row', 0, F3)
    assert _raises(solve_det, as_matrix([[1, 2], [3, 4]]), 'row', 2, F3)
    assert _raises(solve_det, as_matrix([[1, 2], [3, 4]]), 'diag', 0, F3)


# ------------------------------------------------------- multiplication -----
def test_multiplication_matches_numpy():
    rng = random.Random(3)
    for _ in range(25):
        m, n, q = rng.randint(1, 7), rng.randint(1, 7), rng.randint(1, 7)
        a = as_matrix(random_matrix(m, n, rng))
        b = as_matrix(random_matrix(n, q, rng))
        sol = solve_mul(a, b, F3)
        assert np.array_equal(_np(sol.meta['C']), _np(a) @ _np(b))
        assert len(sol.steps) == m + 2
        assert sol.csv == sol.meta['C']
    assert _raises(solve_mul, as_matrix([[1, 2]]), as_matrix([[1, 2]]), F3)


def test_multiplication_with_decimals_is_exact():
    a = as_matrix([[F('0.1'), F('0.2')]])
    b = as_matrix([[3], [F('0.5')]])
    assert solve_mul(a, b, F3).meta['C'] == ((F('0.4'),),)


# ---------------------------------------------------- row reduction (REF) ---
def _check_echelon(final, pivots, reduced):
    cols = [c for _, c in pivots]
    assert cols == sorted(set(cols))
    for r, c in pivots:
        assert final[r][c] == 1
        assert all(v == 0 for v in final[r][:c])
        for i in range(len(final)):
            if i > r or (reduced and i != r):
                assert final[i][c] == 0
    assert all(not any(row) for row in final[len(pivots):])


def _same_row_space(a, b):
    ra, rb = _rank(a), _rank(b)
    both = _rank(tuple(a) + tuple(b))
    assert ra == rb == both


def test_reduction_properties_random():
    rng = random.Random(4)
    for _ in range(60):
        m, n = rng.randint(1, 6), rng.randint(1, 7)
        base = random_matrix(m, n, rng, zero_p=0.35)
        if m > 1 and rng.random() < 0.4:      # force dependent rows
            base[-1] = [2 * x for x in base[0]]
        a = as_matrix(base)
        for strategy in ('first', 'max'):
            for reduced in (False, True):
                sol = solve_reduce(a, reduced, strategy, F3)
                final, piv = sol.meta['final'], sol.meta['pivots']
                _check_echelon(final, piv, reduced)
                _same_row_space(a, final)
                assert sol.meta['rank'] == _rank(a) == len(piv)
                # every intermediate matrix is row-equivalent to the input
                for step in sol.steps:
                    for v in step.views[:1]:
                        if len(v.m) == m and len(v.m[0]) == n and step.kind not in ('intro', 'result'):
                            _same_row_space(a, v.m)
                # the last step shows the final matrix
                assert sol.steps[-1].views[0].m == final


def test_rref_is_unique_for_any_pivot_strategy():
    rng = random.Random(5)
    for _ in range(40):
        a = as_matrix(random_matrix(rng.randint(2, 5), rng.randint(2, 6), rng, zero_p=0.3))
        one = solve_reduce(a, True, 'first', F3).meta['final']
        two = solve_reduce(a, True, 'max', F3).meta['final']
        assert one == two


def test_known_rref_values():
    assert solve_reduce(as_matrix([[1, 2, 3], [4, 5, 6]]), True, 'first', F3).meta['final'] == \
        as_matrix([[1, 0, -1], [0, 1, 2]])
    zero = solve_reduce(as_matrix([[0, 0], [0, 0]]), True, 'first', F3)
    assert zero.meta['rank'] == 0 and zero.meta['final'] == as_matrix([[0, 0], [0, 0]])
    already = solve_reduce(as_matrix([[1, 0], [0, 1]]), True, 'first', F3)
    assert already.meta['ops'] == 0 and already.meta['rank'] == 2


# --------------------------------------------------------------- inverse ----
def test_inverse_random():
    rng = random.Random(6)
    for n in range(1, 8):
        for _ in range(4):
            a = as_matrix(random_matrix(n, n, rng, invertible=True))
            for strategy in ('first', 'max'):
                sol = solve_inverse(a, strategy, F3)
                assert sol.meta['invertible'] and sol.meta['det'] != 0
                inv = sol.meta['inverse']
                assert mat_mul(a, inv) == identity(n) and mat_mul(inv, a) == identity(n)
                assert sol.meta['product'] == identity(n)


def test_inverse_singular_is_detected():
    rng = random.Random(7)
    for n in range(2, 7):
        base = random_matrix(n, n, rng)
        base[-1] = list(base[0])                      # two equal rows -> singular
        sol = solve_inverse(as_matrix(base), 'first', F3)
        assert not sol.meta['invertible'] and sol.meta['det'] == 0
        assert sol.results == () and sol.csv is None
    assert _raises(solve_inverse, as_matrix([[1, 2, 3], [4, 5, 6]]), 'first', F3)


# --------------------------------------------------------------- systems ----
def test_systems_random():
    rng = random.Random(8)
    seen = set()
    for _ in range(120):
        m, n = rng.randint(1, 5), rng.randint(1, 5)
        a_list = random_matrix(m, n, rng, zero_p=0.3)
        if m > 1 and rng.random() < 0.4:
            a_list[-1] = [x + y for x, y in zip(a_list[0], a_list[-1])]
        b_list = [rng.randint(-6, 9) for _ in range(m)]
        a = as_matrix(a_list)
        b = tuple(F(x) for x in b_list)
        aug = tuple(tuple(r) + (b[i],) for i, r in enumerate(a))
        ra, rab = _rank(a), _rank(aug)
        sol = solve_system(a, b, 'first', F3)
        kind = sol.meta['kind']
        seen.add(kind)
        assert sol.meta['rank'] == ra
        if rab > ra:
            assert kind == 'none'
        elif ra == n:
            assert kind == 'unique'
            x = sol.meta['x']
            assert all(sum((a[i][j] * x[j] for j in range(n)), F(0)) == b[i] for i in range(m))
            assert sol.meta['verified']
        else:
            assert kind == 'infinite'
            free, solution = sol.meta['free'], sol.meta['solution']
            assert len(free) == n - ra
            particular = [F(0)] * n
            for c, (const, _) in solution.items():
                particular[c] = const
            assert all(sum((a[i][j] * particular[j] for j in range(n)), F(0)) == b[i] for i in range(m))
            for fc in free:                            # every direction solves the homogeneous system
                v = [F(0)] * n
                v[fc] = F(1)
                for c, (_, coefs) in solution.items():
                    v[c] = coefs[fc]
                assert all(sum((a[i][j] * v[j] for j in range(n)), F(0)) == 0 for i in range(m))
    assert seen == {'unique', 'infinite', 'none'}


# ------------------------------------------------------------ examples ------
def test_examples_do_what_they_promise():
    d = [solve_det(as_matrix(e['A']), 'row', 0, F3).meta['value'] for e in EXAMPLES['det']]
    assert d[0] == -9 and d[3] == 0
    assert solve_mul(as_matrix(EXAMPLES['mul'][0]['A']), as_matrix(EXAMPLES['mul'][0]['B']), F3).meta['C'] == \
        as_matrix([[58, 64], [139, 154]])
    inv = [solve_inverse(as_matrix(e['A']), 'first', F3).meta['invertible'] for e in EXAMPLES['inv']]
    assert inv == [True, True, False]
    kinds = []
    for e in EXAMPLES['sys']:
        sol = solve_system(as_matrix(e['A']), tuple(F(x) for x in e['b']), 'first', F3)
        kinds.append(sol.meta['kind'])
    assert kinds == ['unique', 'infinite', 'none']
    first = solve_system(as_matrix(EXAMPLES['sys'][0]['A']), tuple(F(x) for x in EXAMPLES['sys'][0]['b']),
                         'first', F3)
    assert first.meta['x'] == (2, 3, -1)
    for op in ('ref', 'rref'):
        for e in EXAMPLES[op]:
            assert solve_reduce(as_matrix(e['A']), op == 'rref', 'first', F3).steps


# ------------------------------------------------------ no fractions shown --
def test_no_fraction_is_ever_displayed():
    rng = random.Random(9)
    for _ in range(25):
        n = rng.randint(2, 4)
        a = as_matrix(random_matrix(n, n, rng, invertible=True))
        b = tuple(F(rng.randint(-5, 5)) for _ in range(n))
        sols = [solve_det(a, 'row', 0, F3), solve_mul(a, a, F3), solve_reduce(a, True, 'first', F3),
                solve_reduce(a, False, 'max', F3), solve_inverse(a, 'first', F3),
                solve_system(a, b, 'first', F3)]
        for sol in sols:
            for t in _all_text(sol):
                assert '/' not in t, (sol.op, t)
            for st in sol.steps:
                for v in st.views:
                    for row in v.m:
                        for x in row:
                            assert isinstance(x, str) or isinstance(x, F)   # rendered by Fmt.num only


def test_persian_digits_option():
    sol = solve_reduce(as_matrix([[2, 4], [1, 3]]), False, 'first', Fmt(3, True))
    text = ' '.join(_all_text(sol))
    assert '۲' in text and '1' not in text.replace('R₁', '').replace('R₂', '')


if __name__ == '__main__':       # tiny runner so no pytest is required
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('ok  ', name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print('FAIL', name, '->', type(exc).__name__, exc)
    sys.exit(1 if failed else 0)
