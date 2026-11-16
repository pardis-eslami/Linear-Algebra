"""Linear-algebra engine for the dashboard. No Streamlit imports here.

All arithmetic is exact (fractions.Fraction), so ranks, pivots and singularity
are never wrong because of rounding. The user, however, only ever sees decimals:
numbers are rounded for display by `Fmt` and are never shown as fractions.

Every `solve_*` function returns a `Solution`: a final answer plus a list of
`Step` objects (a short Persian explanation, optional formulas and matrices).
Text may contain two tiny markers that the HTML layer understands:
  **bold**   and   ⟦math⟧   (a left-to-right island inside Persian text).
A formula line that starts with '#' is a plain Persian note, not math.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
import functools
from fractions import Fraction
import re

Matrix = tuple[tuple[Fraction, ...], ...]

MAX_DIM = 7
MINUS = '\u2212'
_DIGITS_IN = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩−٫', '01234567890123456789-.')
_SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
_FA = str.maketrans('0123456789.', '۰۱۲۳۴۵۶۷۸۹٫')
_TO_FA = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
_TO_EN = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
_NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?\Z')


# ----------------------------------------------------------------- parsing --
def parse_number(text: str) -> Fraction:
    """Blank -> 0. Accepts Persian/Arabic digits and decimals. No fractions."""
    s = (text or '').translate(_DIGITS_IN).strip()
    if not s:
        return Fraction(0)
    if len(s) > 40:
        raise ValueError('حداکثر طول هر عدد ۴۰ نویسه است.')
    if '/' in s:
        raise ValueError('کسر پشتیبانی نمی‌شود؛ عدد اعشاری بنویسید، مثلاً ⟦0.5⟧.')
    if ',' in s or '،' in s:
        raise ValueError('برای اعشار از نقطه استفاده کنید، مثلاً ⟦2.5⟧.')
    if not _NUMBER.fullmatch(s):
        raise ValueError('عدد معتبر وارد کنید، مانند ⟦3⟧ یا ⟦-2.5⟧.')
    value = Fraction(s)
    if value and not (Fraction(1, 10**12) <= abs(value) <= 10**12):
        raise ValueError('قدرمطلق عدد باید بین ⟦1e-12⟧ و ⟦1e12⟧ باشد.')
    return value


@dataclass(frozen=True)
class Fmt:
    """Display rules: decimals only, trailing zeros trimmed, optional Persian digits."""
    decimals: int = 3
    persian: bool = False

    def num(self, x) -> str:
        q = Fraction(x)
        d = self.decimals
        scaled = q * 10 ** d
        r = int(scaled + (Fraction(1, 2) if scaled >= 0 else Fraction(-1, 2)))
        if r == 0:
            if q == 0:
                s = '0'
            else:  # non-zero but it rounds to zero: keep it visible
                s = format(Decimal(abs(q.numerator)) / Decimal(q.denominator), '.1e')
                if q < 0:
                    s = MINUS + s
        else:
            digits = str(abs(r)).rjust(d + 1, '0')
            ip, fp = (digits[:-d], digits[-d:]) if d else (digits, '')
            fp = fp.rstrip('0')
            s = (MINUS if r < 0 else '') + ip + ('.' + fp if fp else '')
        return s.translate(_FA) if self.persian else s

    def i(self, k: int) -> str:
        s = str(k)
        return s.translate(_FA) if self.persian else s

    def m(self, x) -> str:   # number inside Persian prose
        return '⟦' + self.num(x) + '⟧'

    def mi(self, k: int) -> str:   # integer inside Persian prose
        return '⟦' + self.i(k) + '⟧'

    def t(self, s: str) -> str:    # make every digit in a sentence follow the chosen style
        return s.translate(_TO_FA if self.persian else _TO_EN)


def sub(k: int) -> str:
    return str(k).translate(_SUB)


def _p(f: Fmt, x) -> str:
    """Number for formulas: negatives get parentheses."""
    s = f.num(x)
    return f'({s})' if Fraction(x) < 0 else s


# ------------------------------------------------------------- data model --
@dataclass(frozen=True)
class Hl:
    """Which cells/rows/columns of a matrix to highlight."""
    pivot: frozenset = frozenset()      # {(r, c)}  leading entries (green)
    rows: frozenset = frozenset()       # rows changed / active (amber)
    cols: frozenset = frozenset()       # columns active (amber)
    src: frozenset = frozenset()        # rows used as the source (blue)
    dim_rows: frozenset = frozenset()   # crossed-out rows
    dim_cols: frozenset = frozenset()   # crossed-out columns
    focus: frozenset = frozenset()      # {(r, c)} emphasised cells


@dataclass(frozen=True)
class MView:
    m: tuple                 # rows; cells are Fraction or str
    label: str = ''
    hl: Hl = Hl()
    sep: int | None = None   # dashed divider before this column (augmented matrix)
    small: bool = False


@dataclass(frozen=True)
class Step:
    kind: str
    tag: str
    text: str
    formulas: tuple = ()
    why: str = ''
    views: tuple = ()
    phase: str = ''


@dataclass(frozen=True)
class Chip:
    label: str
    value: str
    tone: str = 'info'       # ok | bad | warn | info


@dataclass(frozen=True)
class Solution:
    op: str
    title: str
    steps: tuple
    chips: tuple = ()
    scalar: tuple | None = None       # (label, value)
    results: tuple = ()               # result MViews
    lines: tuple = ()                 # math lines in the answer card
    csv: tuple | None = None          # matrix to export
    inputs: tuple = ()                # input MViews (for the report)
    meta: dict = field(default_factory=dict)


def _localize(sol: 'Solution', f: Fmt) -> 'Solution':
    steps = tuple(replace(st, tag=f.t(st.tag), text=f.t(st.text), why=f.t(st.why), phase=f.t(st.phase))
                  for st in sol.steps)
    chips = tuple(replace(c, label=f.t(c.label), value=f.t(c.value)) for c in sol.chips)
    return replace(sol, title=f.t(sol.title), steps=steps, chips=chips)


def _localized(fn):
    """Public solve_* functions take Fmt as the last argument; unify digits in all prose."""
    @functools.wraps(fn)
    def wrapper(*args):
        return _localize(fn(*args), args[-1])
    return wrapper


# ----------------------------------------------------------- matrix basics --
def as_matrix(rows) -> Matrix:
    m = tuple(tuple(Fraction(x) for x in r) for r in rows)
    if not m or not m[0] or any(len(r) != len(m[0]) for r in m):
        raise ValueError('ماتریس باید مستطیلی و غیرتهی باشد.')
    return m


def shape(m) -> tuple[int, int]:
    return len(m), len(m[0])


def minor(m, i: int, j: int):
    return tuple(tuple(x for k, x in enumerate(r) if k != j)
                 for t, r in enumerate(m) if t != i)


def identity(n: int) -> Matrix:
    return tuple(tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n))


def mat_mul(a, b) -> Matrix:
    if len(a[0]) != len(b):
        raise ValueError('تعداد ستون‌های ماتریس اول باید با تعداد سطرهای ماتریس دوم برابر باشد.')
    return tuple(tuple(sum((x * y for x, y in zip(row, col)), Fraction(0))
                       for col in zip(*b)) for row in a)


def det_value(m) -> Fraction:
    n = len(m)
    if n == 0:
        return Fraction(1)
    w = [list(r) for r in m]
    det = Fraction(1)
    for c in range(n):
        p = next((i for i in range(c, n) if w[i][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            w[c], w[p] = w[p], w[c]
            det = -det
        det *= w[c][c]
        for i in range(c + 1, n):
            if w[i][c] != 0:
                k = w[i][c] / w[c][c]
                w[i] = [a - k * b for a, b in zip(w[i], w[c])]
    return det


def zeros_in_line(m, axis: str, idx: int) -> int:
    line = m[idx] if axis == 'row' else [r[idx] for r in m]
    return sum(1 for x in line if x == 0)


def best_line(m) -> tuple[str, int, int]:
    """Row/column with the most zeros (earliest on ties): the cheapest line to expand."""
    best = None
    for axis in ('row', 'col'):
        for k in range(len(m)):
            z = zeros_in_line(m, axis, k)
            if best is None or z > best[2]:
                best = (axis, k, z)
    return best


def _snap(w) -> Matrix:
    return tuple(tuple(r) for r in w)


def _word(axis: str) -> str:
    return 'سطر' if axis == 'row' else 'ستون'


# ----------------------------------------------------------- determinant ---
def _mini(f: Fmt, m) -> str:
    return '|' + '; '.join(' '.join(f.num(x) for x in r) for r in m) + '|'


def _small_det(m, f: Fmt, allow_nested: bool, label: str):
    """Determinant of a minor with the working shown as formula lines."""
    k = len(m)
    if k == 1:
        return m[0][0], [f'{label} = {f.num(m[0][0])}']
    if k == 2:
        a, b, c, d = m[0][0], m[0][1], m[1][0], m[1][1]
        val = a * d - b * c
        return val, [f'{label} = {_mini(f, m)} = ({_p(f, a)}×{_p(f, d)}) − ({_p(f, b)}×{_p(f, c)}) = {f.num(val)}']
    if k == 3 and allow_nested:
        axis, idx, _ = best_line(m)
        lines = [f'# این مینور ⟦3×3⟧ را از {_word(axis)} {f.mi(idx + 1)} بسط می‌دهیم:']
        contribs = []
        for t in range(3):
            i, j = (idx, t) if axis == 'row' else (t, idx)
            a = m[i][j]
            sc = '+' if (i + j) % 2 == 0 else '−'
            sm = minor(m, i, j)
            if a == 0:
                lines.append(f'{sc} 0 × {_mini(f, sm)} = 0')
                contribs.append(Fraction(0))
                continue
            v = sm[0][0] * sm[1][1] - sm[0][1] * sm[1][0]
            contrib = (1 if sc == '+' else -1) * a * v
            lines.append(f'{sc} {_p(f, a)} × {_mini(f, sm)} = {sc} {_p(f, a)} × {_p(f, v)} = {f.num(contrib)}')
            contribs.append(contrib)
        total = sum(contribs, Fraction(0))
        lines.append(f'{label} = ' + ' + '.join(_p(f, c) for c in contribs) + f' = {f.num(total)}')
        return total, lines
    val = det_value(m)
    return val, [f'{label} = {f.num(val)}']


@_localized
def solve_det(A: Matrix, axis: str, idx: int, f: Fmt) -> Solution:
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError('دترمینان فقط برای ماتریس مربعی تعریف می‌شود.')
    if axis not in ('row', 'col') or not 0 <= idx < n:
        raise ValueError('سطر یا ستون بسط معتبر نیست.')
    value = det_value(A)
    inputs = (MView(A, 'A'),)
    invertible = value != 0
    chips = (
        Chip('وضعیت', 'وارون‌پذیر' if invertible else 'تکین (وارون ندارد)', 'ok' if invertible else 'bad'),
        Chip('روش', 'بسط لاپلاس' if n > 1 else 'ماتریس ۱×۱', 'info'),
    )
    scalar = ('det(A)', f.num(value))
    if n == 1:
        steps = (Step(
            'result', 'نتیجه',
            'برای ماتریس ⟦1×1⟧ دترمینان همان تنها عنصر ماتریس است.',
            (f'det(A) = {f.num(value)}',), '',
            (MView(A, 'A', Hl(focus=frozenset({(0, 0)}))),)),)
        return Solution('det', 'دترمینان ماتریس A', steps, chips, scalar, (), (), None, inputs,
                        {'value': value, 'axis': axis, 'idx': idx})

    word = _word(axis)
    z_here = zeros_in_line(A, axis, idx)
    b_axis, b_idx, z_best = best_line(A)
    hint = ''
    if z_best > z_here:
        hint = (f' **نکته:** {_word(b_axis)} {f.mi(b_idx + 1)} صفر بیشتری دارد '
                f'({f.mi(z_best)} صفر)؛ بسط از آن‌جا کوتاه‌تر می‌شود.')

    active = Hl(rows=frozenset({idx})) if axis == 'row' else Hl(cols=frozenset({idx}))
    sign_cells = tuple(tuple('+' if (i + j) % 2 == 0 else '−' for j in range(n)) for i in range(n))
    if axis == 'row':
        a_terms = ' + '.join(f'a{sub(idx + 1)}{sub(j + 1)}·C{sub(idx + 1)}{sub(j + 1)}' for j in range(n))
    else:
        a_terms = ' + '.join(f'a{sub(i + 1)}{sub(idx + 1)}·C{sub(i + 1)}{sub(idx + 1)}' for i in range(n))
    steps = [Step(
        'intro', 'شروع',
        f'دترمینان را با **بسط لاپلاس** از {word} {f.mi(idx + 1)} حساب می‌کنیم: هر عنصر این {word} '
        f'را در «همسازه»اش ضرب می‌کنیم و حاصل‌ها را با هم جمع می‌زنیم.' + hint,
        (f'det(A) = {a_terms}', 'C_ij = (−1)^(i+j) · det(M_ij)'),
        'مینور M_ij ماتریسی است که با خط زدن سطر i و ستون j به دست می‌آید. علامت (−1)^(i+j) '
        'مثل صفحهٔ شطرنج یکی در میان + و − است' + ('؛ ماتریس «علامت‌ها» علامت هر خانه را نشان می‌دهد.' if n <= 4 else '.'),
        (MView(A, 'A', active),) + ((MView(sign_cells, 'علامت‌ها', active, small=True),) if n <= 4 else ()))]

    nested_ok = n <= 4
    contribs = []
    for k in range(n):
        i, j = (idx, k) if axis == 'row' else (k, idx)
        a = A[i][j]
        sgn = 1 if (i + j) % 2 == 0 else -1
        sname = f'{sub(i + 1)}{sub(j + 1)}'
        cross = Hl(dim_rows=frozenset({i}), dim_cols=frozenset({j}), focus=frozenset({(i, j)}))
        if a == 0:
            contribs.append(Fraction(0))
            steps.append(Step(
                'term', f'جملهٔ {f.i(k + 1)} از {f.i(n)}',
                f'عنصر **⟦a{sname} = 0⟧** است؛ پس جملهٔ آن هم صفر می‌شود و نیازی به محاسبهٔ مینور نیست.',
                (f't{sub(k + 1)} = 0 × ({"+1" if sgn > 0 else "−1"}) × det(M{sname}) = 0',),
                'هر جا عنصر صفر باشد، کار کمتر می‌شود. برای همین بهتر است از سطر یا ستونی بسط بدهیم که صفر دارد.',
                (MView(A, 'A', cross),)))
            continue
        M = minor(A, i, j)
        mval, lines = _small_det(M, f, nested_ok, f'det(M{sname})')
        term = a * sgn * mval
        contribs.append(term)
        lines = list(lines)
        lines.append(f't{sub(k + 1)} = a{sname} × ({"+1" if sgn > 0 else "−1"}) × det(M{sname}) = '
                     f'{_p(f, a)} × ({"+1" if sgn > 0 else "−1"}) × {_p(f, mval)} = {f.num(term)}')
        extra = '' if (len(M) <= 3 and nested_ok) or len(M) <= 2 else ' دترمینان این مینور را هم با همین روش (یا حذف گاوسی) حساب کرده‌ایم.'
        steps.append(Step(
            'term', f'جملهٔ {f.i(k + 1)} از {f.i(n)}',
            f'عنصر **⟦a{sname} = {f.num(a)}⟧** را برمی‌داریم. سطر {f.mi(i + 1)} و ستون {f.mi(j + 1)} را خط می‌زنیم؛ '
            f'ماتریسِ باقی‌مانده «مینور» **⟦M{sname}⟧** است. دترمینان مینور را حساب می‌کنیم و در '
            f'عنصر و علامت ضرب می‌کنیم.' + extra,
            tuple(lines),
            'علامت این جمله ' + ('مثبت (+) است' if sgn > 0 else 'منفی (−) است') +
            f' چون مجموع شمارهٔ سطر و ستون ({f.mi(i + 1)} + {f.mi(j + 1)}) ' +
            ('زوج' if sgn > 0 else 'فرد') + ' است.',
            (MView(A, 'A', cross), MView(M, f'M{sname}', small=True))))

    total = sum(contribs, Fraction(0))
    assert total == value
    sum_line = 'det(A) = ' + ' + '.join(f't{sub(k + 1)}' for k in range(n))
    num_line = '= ' + ' + '.join(_p(f, c) for c in contribs) + f' = {f.num(total)}'
    steps.append(Step(
        'result', 'جمع جمله‌ها',
        f'همهٔ جمله‌ها را جمع می‌کنیم: **⟦det(A) = {f.num(total)}⟧**. ' +
        ('چون دترمینان صفر نیست، ماتریس وارون‌پذیر است.' if invertible else
         'چون دترمینان صفر است، ماتریس تکین است و وارون ندارد.'),
        (sum_line, num_line), '',
        (MView(A, 'A'),)))
    return Solution('det', 'دترمینان ماتریس A', tuple(steps), chips, scalar, (), (), None, inputs,
                    {'value': value, 'axis': axis, 'idx': idx, 'terms': tuple(contribs)})


# ---------------------------------------------------------- multiplication --
@_localized
def solve_mul(A: Matrix, B: Matrix, f: Fmt) -> Solution:
    m, n = shape(A)
    n2, q = shape(B)
    if n != n2:
        raise ValueError('تعداد ستون‌های ماتریس اول باید با تعداد سطرهای ماتریس دوم برابر باشد.')
    C = mat_mul(A, B)
    steps = [Step(
        'intro', 'بررسی ابعاد',
        f'ماتریس A دارای {f.mi(m)} سطر و {f.mi(n)} ستون و ماتریس B دارای {f.mi(n)} سطر و {f.mi(q)} ستون است. '
        f'تعداد ستون‌های A با تعداد سطرهای B برابر است؛ پس ضرب ممکن است و حاصل یک ماتریس '
        f'⟦{f.i(m)}×{f.i(q)}⟧ می‌شود.',
        (f'({f.i(m)}×{f.i(n)}) · ({f.i(n)}×{f.i(q)}) = ({f.i(m)}×{f.i(q)})',
         f'c_ij = a_i1·b_1j + a_i2·b_2j + … + a_i{sub(n)}·b_{sub(n)}j'),
        'درایهٔ سطر i و ستون j از C برابر است با جمع حاصل‌ضرب عنصرهای هم‌شمارهٔ «سطر i از A» و «ستون j از B».',
        (MView(A, 'A'), MView(B, 'B')))]
    for i in range(m):
        formulas = []
        for j in range(q):
            terms = ' + '.join(f'{_p(f, A[i][k])}×{_p(f, B[k][j])}' for k in range(n))
            formulas.append(f'c{sub(i + 1)}{sub(j + 1)} = {terms} = {f.num(C[i][j])}')
        steps.append(Step(
            'row', f'سطر {f.i(i + 1)} از C',
            f'سطر {f.mi(i + 1)} از A را با هر ستون B ضرب داخلی می‌کنیم تا سطر {f.mi(i + 1)} از C ساخته شود.',
            tuple(formulas), '',
            (MView(A, 'A', Hl(src=frozenset({i}))), MView(B, 'B'),
             MView((C[i],), f'سطر {f.i(i + 1)} از C'))))
    steps.append(Step(
        'result', 'نتیجه',
        f'همهٔ سطرها ساخته شد؛ ماتریس حاصل‌ضرب **⟦C = A × B⟧** به اندازهٔ ⟦{f.i(m)}×{f.i(q)}⟧ آماده است.',
        (), '', (MView(C, 'C = A × B'),)))
    chips = (Chip('اندازهٔ نتیجه', f'{f.i(m)}×{f.i(q)}', 'info'),)
    return Solution('mul', 'ضرب ماتریس‌ها', tuple(steps), chips, None, (MView(C, 'C = A × B'),),
                    (), C, (MView(A, 'A'), MView(B, 'B')), {'C': C})


# ----------------------------------------------------- row reduction core --
@dataclass
class Reduction:
    final: Matrix
    pivots: tuple
    steps: list
    ops: int
    stopped: int | None


WHY_SWAP_ZERO = 'عنصر پیشرو نباید صفر باشد؛ پس سطری را که در این ستون عدد غیرصفر دارد بالا می‌آوریم. جابه‌جایی دو سطر جواب را عوض نمی‌کند.'
WHY_SWAP_MAX = 'انتخاب بزرگ‌ترین عدد به‌عنوان عنصر پیشرو خطای گرد کردن را کمتر می‌کند. جابه‌جایی دو سطر جواب را عوض نمی‌کند.'
WHY_SCALE = 'ضرب یا تقسیم یک سطر در عددی غیرصفر جواب را عوض نمی‌کند؛ این‌جا برای این است که عنصر پیشرو دقیقاً ۱ شود.'
WHY_ELIM = 'اضافه یا کم کردن مضربی از یک سطر به سطر دیگر ارزش ماتریس را حفظ می‌کند و خانهٔ هدف را صفر می‌کند.'
WHY_INFO = 'ستونی که عنصر پیشرو ندارد به یک متغیر «آزاد» مربوط است و رتبه را کمتر می‌کند.'
PHASE_FWD = 'مرحلهٔ ۱ — رو به جلو (ساختن پله‌ها)'
PHASE_BWD = 'مرحلهٔ ۲ — رو به عقب (صفر کردن بالای عنصرهای پیشرو)'
PHASE_ONE = 'عملیات سطری'


def _elim_formula(f: Fmt, i: int, r: int, k) -> str:
    mult = '' if abs(k) == 1 else f'{f.num(abs(k))}·'
    sign = '−' if k > 0 else '+'
    return f'R{sub(i + 1)} ← R{sub(i + 1)} {sign} {mult}R{sub(r + 1)}'


def _elim_text(f: Fmt, i: int, r: int, c: int, k, up: bool) -> str:
    amount = f'سطر {f.mi(r + 1)}' if abs(k) == 1 else f'{f.m(abs(k))} برابر سطر {f.mi(r + 1)}'
    verb, prep = ('کم', 'از') if k > 0 else ('اضافه', 'به')
    where = (f'خانهٔ بالای عنصر پیشرو (سطر {f.mi(i + 1)}، ستون {f.mi(c + 1)})' if up
             else f'خانهٔ سطر {f.mi(i + 1)} و ستون {f.mi(c + 1)}')
    return f'برای صفر کردن {where}، {amount} را {prep} سطر {f.mi(i + 1)} {verb} می‌کنیم.'


def reduce_matrix(M, *, reduced: bool, strategy: str, pivot_cols: int, f: Fmt, name: str,
                  sep: int | None = None, info_cols: int | None = None,
                  stop_if_no_pivot: bool = False) -> Reduction:
    """Gauss (reduced=False) or Gauss-Jordan (reduced=True) with a full operation log.

    strategy: 'first' = first non-zero entry, 'max' = largest absolute value.
    pivot_cols: only the first `pivot_cols` columns may hold pivots.
    """
    w = [list(r) for r in M]
    rows = len(w)
    info_cols = pivot_cols if info_cols is None else info_cols
    pivots: list[tuple[int, int]] = []
    steps: list[Step] = []
    ops = 0
    stopped = None
    phase1 = PHASE_FWD if reduced else PHASE_ONE

    def view(rows_=(), src=(), focus=()):
        return MView(_snap(w), name, Hl(pivot=frozenset(pivots), rows=frozenset(rows_),
                                        src=frozenset(src), focus=frozenset(focus)), sep)

    r = 0
    for c in range(pivot_cols):
        if r >= rows:
            break
        cand = [i for i in range(r, rows) if w[i][c] != 0]
        if not cand:
            if c < info_cols:
                steps.append(Step(
                    'info', 'ستون بدون عنصر پیشرو',
                    f'در ستون {f.mi(c + 1)} از سطر {f.mi(r + 1)} به پایین فقط صفر داریم؛ پس این ستون '
                    f'عنصر پیشرو ندارد و به ستون بعدی می‌رویم.',
                    (), WHY_INFO, (view(focus=[(i, c) for i in range(r, rows)]),), phase1))
            if stop_if_no_pivot:
                stopped = c
                break
            continue
        p = cand[0] if strategy == 'first' else max(cand, key=lambda i: (abs(w[i][c]), -i))
        if p != r:
            was_zero = w[r][c] == 0
            w[r], w[p] = w[p], w[r]
            ops += 1
            pivots.append((r, c))
            text = (f'خانهٔ سطر {f.mi(r + 1)} و ستون {f.mi(c + 1)} صفر است و عنصر پیشرو نباید صفر باشد؛ '
                    f'پس سطر {f.mi(r + 1)} و سطر {f.mi(p + 1)} را جابه‌جا می‌کنیم.') if was_zero else (
                    f'برای دقت بیشتر، سطر {f.mi(p + 1)} که بزرگ‌ترین عدد (از نظر قدرمطلق) ستون '
                    f'{f.mi(c + 1)} را دارد، با سطر {f.mi(r + 1)} جابه‌جا می‌شود.')
            steps.append(Step('swap', 'جابه‌جایی دو سطر', text,
                              (f'R{sub(r + 1)} ↔ R{sub(p + 1)}',),
                              WHY_SWAP_ZERO if was_zero else WHY_SWAP_MAX,
                              (view(rows_=[r, p]),), phase1))
        else:
            pivots.append((r, c))
        pv = w[r][c]
        if pv != 1:
            w[r] = [x / pv for x in w[r]]
            ops += 1
            if pv == -1:
                text = f'سطر {f.mi(r + 1)} را در {f.m(-1)} ضرب می‌کنیم تا عنصر پیشرو آن ۱ شود.'
                formula = f'R{sub(r + 1)} ← −R{sub(r + 1)}'
            else:
                text = f'سطر {f.mi(r + 1)} را بر {f.m(pv)} تقسیم می‌کنیم تا عنصر پیشرو آن ۱ شود.'
                formula = f'R{sub(r + 1)} ← R{sub(r + 1)} ÷ {_p(f, pv)}'
            steps.append(Step('scale', 'ساختن عنصر پیشرو ۱', text, (formula,), WHY_SCALE,
                              (view(rows_=[r]),), phase1))
        for i in range(r + 1, rows):
            if w[i][c] != 0:
                k = w[i][c]
                w[i] = [x - k * y for x, y in zip(w[i], w[r])]
                ops += 1
                steps.append(Step('elim', 'صفر کردن زیر عنصر پیشرو', _elim_text(f, i, r, c, k, False),
                                  (_elim_formula(f, i, r, k),), WHY_ELIM,
                                  (view(rows_=[i], src=[r]),), phase1))
        r += 1

    if reduced and stopped is None:
        for (pr, pc) in reversed(pivots):
            for i in range(pr - 1, -1, -1):
                if w[i][pc] != 0:
                    k = w[i][pc]
                    w[i] = [x - k * y for x, y in zip(w[i], w[pr])]
                    ops += 1
                    steps.append(Step('elim_up', 'صفر کردن بالای عنصر پیشرو',
                                      _elim_text(f, i, pr, pc, k, True),
                                      (_elim_formula(f, i, pr, k),), WHY_ELIM,
                                      (view(rows_=[i], src=[pr]),), PHASE_BWD))
    return Reduction(_snap(w), tuple(pivots), steps, ops, stopped)


def _cols_text(f: Fmt, pivots, limit: int) -> str:
    cols = [c for (_, c) in pivots if c < limit]
    return '، '.join(f.i(c + 1) for c in cols) if cols else '—'


INTRO_WHY = ('سه عمل مجاز داریم: جابه‌جایی دو سطر؛ ضرب یک سطر در عددی غیرصفر؛ و اضافه کردن مضربی '
             'از یک سطر به سطری دیگر. هیچ‌کدام جواب دستگاه را عوض نمی‌کنند.')


@_localized
def solve_reduce(A: Matrix, reduced: bool, strategy: str, f: Fmt) -> Solution:
    m, n = shape(A)
    name = 'RREF' if reduced else 'REF'
    red = reduce_matrix(A, reduced=reduced, strategy=strategy, pivot_cols=n, f=f, name='ماتریس فعلی')
    rank = len(red.pivots)
    if reduced:
        goal = ('عنصر پیشرو هر سطر ۱ است و بالا و پایین آن فقط صفر است. '
                'هر ماتریس دقیقاً یک RREF دارد.')
        form = 'فرم پلکانی کاهش‌یافته (⟦RREF⟧)'
    else:
        goal = ('عنصر پیشرو هر سطر ۱ است، زیر آن فقط صفر است و پله‌ها از بالا به پایین '
                'به سمت راست می‌روند.')
        form = 'فرم پلکانی (⟦REF⟧)'
    steps = [Step('intro', 'شروع',
                  f'ماتریس را با عملیات سطری به **{form}** می‌بریم. در این فرم {goal}',
                  (), INTRO_WHY, (MView(A, 'ماتریس A'),))]
    steps += red.steps
    if red.ops == 0:
        end = 'ماتریس از همان ابتدا در این فرم بود و به عملیات سطری نیازی نداشت. '
    else:
        end = ''
    if rank == 0:
        end += 'ماتریس صفر است؛ عنصر پیشرو ندارد و رتبه‌اش ⟦0⟧ است.'
    else:
        end += (f'عنصرهای پیشرو (خانه‌های سبز) را می‌شمریم: **رتبهٔ ماتریس = {f.i(rank)}**.'
                + (' فرم پلکانی یکتا نیست؛ با انتخاب دیگرِ سطر محوری ممکن است REF دیگری، ولی درست، به دست آید.'
                   if not reduced else ''))
    final_view = MView(red.final, name, Hl(pivot=frozenset(red.pivots)))
    steps.append(Step('result', 'نتیجه', end, (), '', (final_view,)))
    chips = (
        Chip('رتبه', f.i(rank), 'info'),
        Chip('ستون‌های پیشرو', _cols_text(f, red.pivots, n), 'ok' if rank else 'warn'),
        Chip('تعداد عملیات', f.i(red.ops), 'info'),
    )
    title = 'فرم پلکانی کاهش‌یافته (RREF)' if reduced else 'فرم پلکانی (REF)'
    return Solution('rref' if reduced else 'ref', title, tuple(steps), chips, None, (final_view,), (),
                    red.final, (MView(A, 'A'),),
                    {'rank': rank, 'pivots': red.pivots, 'final': red.final, 'ops': red.ops})


# ------------------------------------------------------------------ inverse --
@_localized
def solve_inverse(A: Matrix, strategy: str, f: Fmt) -> Solution:
    n = len(A)
    if n == 0 or any(len(r) != n for r in A):
        raise ValueError('وارون فقط برای ماتریس مربعی تعریف می‌شود.')
    eye = identity(n)
    aug = tuple(tuple(A[i]) + eye[i] for i in range(n))
    red = reduce_matrix(aug, reduced=True, strategy=strategy, pivot_cols=n, f=f, name='[A | I]',
                        sep=n, info_cols=n, stop_if_no_pivot=True)
    det = det_value(A)
    steps = [Step(
        'intro', 'شروع',
        'ماتریس همانی I را کنار A می‌نویسیم و ماتریس افزودهٔ **⟦[A | I]⟧** را می‌سازیم. با عملیات سطری '
        'سمت چپ را به I می‌رسانیم؛ همان عملیات، سمت راست را به **وارونِ A** تبدیل می‌کند.',
        (), INTRO_WHY, (MView(aug, '[A | I]', sep=n),))]
    steps += red.steps
    inputs = (MView(A, 'A'),)
    if red.stopped is not None:
        steps.append(Step(
            'result', 'ماتریس تکین است',
            f'ستون {f.mi(red.stopped + 1)} عنصر پیشرو ندارد؛ پس سمت چپ هرگز به I تبدیل نمی‌شود. '
            f'نتیجه: A تکین است (دترمینان آن صفر است) و **وارون ندارد**.',
            (), '', (MView(red.final, '[A | I]', Hl(pivot=frozenset(red.pivots)), n),)))
        chips = (Chip('وضعیت', 'تکین (وارون ندارد)', 'bad'), Chip('det(A)', f.num(det), 'info'))
        return Solution('inv', 'وارون ماتریس A', tuple(steps), chips, None, (), (), None, inputs,
                        {'invertible': False, 'det': det})
    inv = tuple(tuple(row[n:]) for row in red.final)
    steps.append(Step(
        'result', 'خواندن وارون',
        'سمت چپ به ماتریس همانی تبدیل شد؛ پس سمت راست همان **وارونِ A** است.',
        (), '', (MView(red.final, '[I | A⁻¹]', Hl(pivot=frozenset(red.pivots)), n), MView(inv, 'A⁻¹'))))
    prod = mat_mul(A, inv)
    steps.append(Step(
        'check', 'راستی‌آزمایی',
        'برای اطمینان A را در A⁻¹ ضرب می‌کنیم؛ باید ماتریس همانی شود.',
        ('A × A⁻¹ = I',), '', (MView(A, 'A'), MView(inv, 'A⁻¹'), MView(prod, 'A × A⁻¹'))))
    chips = (Chip('وضعیت', 'وارون‌پذیر', 'ok'), Chip('det(A)', f.num(det), 'info'),
             Chip('تعداد عملیات', f.i(red.ops), 'info'))
    return Solution('inv', 'وارون ماتریس A', tuple(steps), chips, None, (MView(inv, 'A⁻¹'),), (), inv,
                    inputs, {'invertible': True, 'det': det, 'inverse': inv, 'product': prod})


# ------------------------------------------------------------------ systems --
def _lin_expr(f: Fmt, const, terms) -> str:
    parts = []
    if const != 0 or not terms:
        parts.append(('+' if const >= 0 else '−', f.num(abs(const))))
    for coef, sym in terms:
        if coef == 0:
            continue
        mag = '' if abs(coef) == 1 else f.num(abs(coef)) + '·'
        parts.append(('+' if coef > 0 else '−', f'{mag}{sym}'))
    if not parts:
        return '0'
    s = ('' if parts[0][0] == '+' else '−') + parts[0][1]
    for sg, body in parts[1:]:
        s += f' {sg} {body}'
    return s


def equation_lines(A, b, f: Fmt) -> tuple:
    return tuple(f'{_lin_expr(f, Fraction(0), [(a, f"x{sub(j + 1)}") for j, a in enumerate(row)])} = {f.num(b[i])}'
                 for i, row in enumerate(A))


@_localized
def solve_system(A: Matrix, b: tuple, strategy: str, f: Fmt) -> Solution:
    m, n = shape(A)
    aug = tuple(tuple(A[i]) + (b[i],) for i in range(m))
    red = reduce_matrix(aug, reduced=True, strategy=strategy, pivot_cols=n + 1, f=f, name='[A | b]',
                        sep=n, info_cols=n)
    piv_cols = [c for (_, c) in red.pivots]
    inconsistent = n in piv_cols
    rank = len([c for c in piv_cols if c < n])
    steps = [Step(
        'intro', 'شروع',
        'دستگاه را با **ماتریس افزوده** نشان می‌دهیم: ضرایب هر معادله یک سطر می‌شود و عددهای سمت راست '
        '(ثابت‌ها) ستون آخر. بعد با عملیات سطری آن را ساده می‌کنیم.',
        equation_lines(A, b, f),
        'حل دستگاه با عملیات سطری همان جواب اصلی را می‌دهد؛ فقط ظاهر معادله‌ها ساده‌تر می‌شود.',
        (MView(aug, '[A | b]', sep=n),))]
    steps += red.steps
    final = red.final
    inputs = (MView(A, 'A'), MView(tuple((x,) for x in b), 'b'))
    xs = tuple(f'x{sub(j + 1)}' for j in range(n))

    if inconsistent:
        r = next(r for (r, c) in red.pivots if c == n)
        zeros = ' + '.join(f'0·{x}' for x in xs)
        steps.append(Step(
            'result', 'دستگاه جواب ندارد',
            f'سطر {f.mi(r + 1)} به معادله‌ای مثل ⟦0 = {f.num(final[r][n])}⟧ تبدیل شده که هرگز درست نیست؛ '
            f'پس دستگاه **جواب ندارد** (ناسازگار است).',
            (f'{zeros} = {f.num(final[r][n])}',), '',
            (MView(final, '[A | b]', Hl(pivot=frozenset(red.pivots), rows=frozenset({r})), n),)))
        chips = (Chip('نوع جواب', 'ناسازگار (بدون جواب)', 'bad'), Chip('رتبهٔ A', f.i(rank), 'info'))
        return Solution('sys', 'حل دستگاه معادلات', tuple(steps), chips, None, (), (), None, inputs,
                        {'kind': 'none', 'rank': rank})

    if rank == n:
        x = tuple(final[r][n] for r in range(n))
        lines = tuple(f'{xs[j]} = {f.num(x[j])}' for j in range(n))
        steps.append(Step(
            'result', 'خواندن جواب',
            'هر ستونِ ضرایب یک عنصر پیشرو دارد و ستون ثابت‌ها ندارد؛ پس دستگاه **جواب یکتا** دارد '
            'و مقدار هر مجهول همان ستون آخر است.',
            lines, '', (MView(final, '[A | b]', Hl(pivot=frozenset(red.pivots)), n),)))
        check = []
        ok = True
        for i in range(m):
            terms = ' + '.join(f'{_p(f, A[i][j])}×{_p(f, x[j])}' for j in range(n))
            total = sum((A[i][j] * x[j] for j in range(n)), Fraction(0))
            good = total == b[i]
            ok = ok and good
            check.append(f'{terms} = {f.num(total)} {"✓" if good else "✗"}')
        steps.append(Step(
            'check', 'راستی‌آزمایی',
            'جواب را در هر معادلهٔ اصلی می‌گذاریم؛ باید همان عدد سمت راست به دست آید.',
            tuple(check), '', ()))
        chips = (Chip('نوع جواب', 'یکتا', 'ok'), Chip('رتبهٔ A', f.i(rank), 'info'))
        return Solution('sys', 'حل دستگاه معادلات', tuple(steps), chips, None, (), lines,
                        tuple((v,) for v in x), inputs, {'kind': 'unique', 'x': x, 'rank': rank, 'verified': ok})

    # infinitely many solutions
    pivot_set = {c for (_, c) in red.pivots}
    free = [c for c in range(n) if c not in pivot_set]
    tname = {c: f't{sub(k + 1)}' for k, c in enumerate(free)}
    by_col = {}
    solution = {}
    for (r, c) in red.pivots:
        terms = [(-final[r][fc], tname[fc]) for fc in free]
        by_col[c] = f'{xs[c]} = {_lin_expr(f, final[r][n], terms)}'
        solution[c] = (final[r][n], {fc: -final[r][fc] for fc in free})
    for fc in free:
        by_col[fc] = f'{xs[fc]} = {tname[fc]}'
    lines = [by_col[c] for c in range(n)]
    free_names = '، '.join(xs[c] for c in free)
    if len(free) == 1:
        free_phrase = f'مجهول ⟦{xs[free[0]]}⟧ «آزاد» است و می‌تواند هر عددی باشد؛ بقیهٔ مجهول‌ها بر حسب آن نوشته می‌شوند.'
    else:
        names = '، '.join(f'⟦{xs[c]}⟧' for c in free)
        free_phrase = f'مجهول‌های {names} «آزاد» هستند و می‌توانند هر عددی باشند؛ بقیهٔ مجهول‌ها بر حسب آن‌ها نوشته می‌شوند.'
    steps.append(Step(
        'result', 'جواب‌های بی‌شمار',
        f'تعداد عنصرهای پیشرو ({f.mi(rank)}) از تعداد مجهول‌ها ({f.mi(n)}) کمتر است؛ پس **بی‌شمار جواب** '
        f'داریم. {free_phrase}',
        tuple(lines), 'هر انتخابِ دلخواه برای پارامترهای t یک جواب درست می‌دهد؛ مثلاً با t = 0 یک جواب خاص به دست می‌آید.',
        (MView(final, '[A | b]', Hl(pivot=frozenset(red.pivots)), n),)))
    chips = (Chip('نوع جواب', 'بی‌شمار جواب', 'warn'), Chip('رتبهٔ A', f.i(rank), 'info'),
             Chip('متغیر آزاد', free_names, 'info'))
    return Solution('sys', 'حل دستگاه معادلات', tuple(steps), chips, None, (), tuple(lines), None, inputs,
                    {'kind': 'infinite', 'rank': rank, 'free': tuple(free), 'solution': solution,
                     'final': final})
