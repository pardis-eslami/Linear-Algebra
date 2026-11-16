"""Tests for the HTML builders: they must be safe and must survive st.markdown.

Run:  python -m pytest -q        (or)      python tests/test_views.py
"""
from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import views  # noqa: E402
from linear_algebra import (Chip, Fmt, Hl, MView, as_matrix, solve_det, solve_inverse, solve_mul,  # noqa: E402
                            solve_reduce, solve_system)
from samples import EXAMPLES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F3 = Fmt(3)


def _solutions(f=F3):
    A = as_matrix(EXAMPLES['ref'][1]['A'])
    sq = as_matrix(EXAMPLES['det'][0]['A'])
    b = tuple(F(x) for x in EXAMPLES['sys'][0]['b'])
    return [solve_det(sq, 'row', 0, f), solve_det(sq, 'col', 1, f),
            solve_det(as_matrix(EXAMPLES['det'][2]['A']), 'row', 0, f),
            solve_mul(as_matrix(EXAMPLES['mul'][0]['A']), as_matrix(EXAMPLES['mul'][0]['B']), f),
            solve_reduce(A, False, 'first', f), solve_reduce(A, True, 'max', f),
            solve_inverse(sq, 'first', f), solve_inverse(as_matrix(EXAMPLES['inv'][2]['A']), 'first', f),
            solve_system(as_matrix(EXAMPLES['sys'][0]['A']), b, 'first', f),
            solve_system(as_matrix(EXAMPLES['sys'][1]['A']), (F(4), F(9)), 'first', f),
            solve_system(as_matrix(EXAMPLES['sys'][2]['A']), (F(1), F(3)), 'first', f)]


def _is_markdown_safe(html: str) -> bool:
    """One line, no blank line, no indentation: otherwise st.markdown may show it as code."""
    return '\n' not in html and not html.startswith(' ') and html.lstrip().startswith(('<div', '<style'))


def test_rich_escapes_and_marks():
    out = views.rich('<script>alert(1)</script> **bold** ⟦x < y⟧')
    assert '<script>' not in out and '&lt;script&gt;' in out
    assert '<b>bold</b>' in out and '<span class="la-m">x &lt; y</span>' in out


def test_every_block_is_markdown_safe():
    for f in (F3, Fmt(2, True)):
        for sol in _solutions(f):
            blocks = [views.answer_html(sol, f), views.legend_html(sol) or '<div></div>',
                      views.all_steps_html(sol, f, True), views.all_steps_html(sol, f, False)]
            blocks += [views.step_html(st, i, len(sol.steps), f, i % 2 == 0) for i, st in enumerate(sol.steps)]
            for html in blocks:
                assert _is_markdown_safe(html), html[:120]
                assert '$' not in html
    for html in (views.brand_html(), views.credits_html(), views.side_title_html('عملیات'),
                 views.header_html('عنوان'), views.empty_html('t', 'h ⟦0⟧'),
                 views.errors_html(['a', 'b']), views.hint_html('x **y**'), views.help_html(['a', 'b']),
                 views.mx_title_html('ماتریس A', 3, 3, F3), views.invalid_cells_css(['A_0_1'])):
        assert _is_markdown_safe(html), html[:120]


def test_untrusted_text_is_escaped():
    evil = Chip('<img src=x onerror=alert(1)>', '<b>', 'ok')
    html = views.chips_html([evil])
    assert '<img' not in html and '&lt;img' in html
    assert '<script' not in views.errors_html(['<script>alert(1)</script>'])


def test_matrix_classes():
    hl = Hl(pivot=frozenset({(0, 0)}), rows=frozenset({1}), src=frozenset({2}),
            dim_cols=frozenset({1}), focus=frozenset({(2, 2)}))
    m = as_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    html = views.matrix_html(MView(m, 'A', hl, sep=2), F3)
    assert html.count('class="la-c') == 9
    classes = set(re.findall(r'class="la-c([^"]*)"', html))
    flat = ' '.join(classes).split()
    for name in ('p', 'h', 's', 'd', 'f', 'sp'):
        assert name in flat, (name, classes)
    assert 'repeat(3,auto)' in html and '3×3' in html
    big = views.matrix_html(MView(as_matrix([[1] * 7] * 7)), F3)
    assert 'la-mx dn' in big
    sign = views.matrix_html(MView((('+', '−'), ('−', '+')), small=True), F3)
    assert 'la-mx sm' in sign and '>−<' in sign


def test_numbers_are_decimals_in_html():
    sol = solve_inverse(as_matrix([[4, 7], [2, 6]]), 'first', F3)
    html = views.all_steps_html(sol, F3, True) + views.answer_html(sol, F3)
    assert '0.6' in html and '−0.7' in html and '/' not in re.sub(r'</?[a-z][^>]*>', '', html)
    fa = views.answer_html(solve_inverse(as_matrix([[4, 7], [2, 6]]), 'first', Fmt(3, True)), Fmt(3, True))
    assert '۰٫۶' in fa


def test_progress_and_counts():
    sol = solve_reduce(as_matrix([[1, 2], [3, 4]]), True, 'first', F3)
    last = views.step_html(sol.steps[-1], len(sol.steps) - 1, len(sol.steps), F3, True)
    assert 'width:100%' in last and f'مرحلهٔ {len(sol.steps)} از {len(sol.steps)}' in last
    first = views.step_html(sol.steps[0], 0, len(sol.steps), F3, False)
    assert 'چرا؟' not in first
    assert 'چرا؟' in views.step_html(sol.steps[0], 0, len(sol.steps), F3, True)


def test_css_is_valid_after_minify():
    css = (ROOT / 'styles.css').read_text(encoding='utf-8')
    out = views.minify_css(css)
    assert '/*' not in out and '\n' not in out
    assert out.count('{') == out.count('}') and out.count('(') == out.count(')')
    for needle in ('color-mix(in srgb', '.la-c.p', '@media (max-width:900px)',
                   '.st-key-panel_out'):
        assert needle in out, needle
    assert '.st-key-navigation' in out
    assert 'prefers-color-scheme' not in out     # colours must follow Streamlit's theme, not the OS


def test_report_is_self_contained():
    css = (ROOT / 'styles.css').read_text(encoding='utf-8')
    for sol in _solutions():
        html = views.report_html(sol, F3, css, '@font-face{font-family:Vazirmatn}')
        assert html.startswith('<!doctype html>') and '<html lang="fa" dir="rtl">' in html
        assert sol.title in html and html.count('class="la-step compact"') == len(sol.steps)
        assert 'مرحلهٔ اول' in html and '@page{size:A4;' in html
        assert 'http://' not in html and 'https://' not in html          # nothing is loaded from the internet


def test_invalid_cell_css():
    css = views.invalid_cells_css(['A_0_1', 'b_2'])
    assert '.st-key-A_0_1' in css and '.st-key-b_2' in css and css.startswith('<style>')
    assert views.invalid_cells_css([]) == ''


if __name__ == '__main__':
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
