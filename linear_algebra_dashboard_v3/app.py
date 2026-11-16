"""آزمایشگاه جبر خطی. Run with: python -m streamlit run app.py."""
from __future__ import annotations

import base64
from functools import partial
from pathlib import Path

import streamlit as st

import views
from linear_algebra import Fmt, MAX_DIM, MINUS, parse_number, solve_det, solve_inverse, solve_mul, solve_reduce
from report_pdf import pdf_bytes
from samples import random_matrix

ROOT = Path(__file__).resolve().parent
S = st.session_state

OPS = {
    'det': ('دترمینان', 'دترمینان با بسط لاپلاس'),
    'mul': ('ضرب ماتریس‌ها', 'ضرب دو ماتریس'),
    'ref': ('فرم پلکانی (REF)', 'فرم پلکانی با حذف گاوسی'),
    'rref': ('فرم پلکانی کاهش‌یافته (RREF)', 'فرم پلکانی کاهش‌یافته (گاوس–جردن)'),
    'inv': ('وارون ماتریس', 'وارون ماتریس با گاوس–جردن'),
}
NUMS = list(range(1, MAX_DIM + 1))


def store() -> dict:
    """Remember committed entries when dimensions or operations hide a widget."""
    return S.setdefault('store', {})


def dims() -> dict:
    return S.setdefault('dims', {'m': 3, 'n': 3, 'q': 3})


def put_matrix(prefix: str, values: list) -> None:
    width = len(values[0]) if values else 0
    for i in range(MAX_DIM):
        for j in range(MAX_DIM):
            key = f'{prefix}_{i}_{j}'
            value = str(values[i][j]) if i < len(values) and j < width else ''
            store()[key] = value
            if key in S:
                S[key] = value


def preset(op: str, action: str, rows: int, cols: int, q: int) -> None:
    S.pop('submission', None)
    if action == 'clear':
        put_matrix('A', [])
        put_matrix('B', [])
    elif action == 'identity':
        put_matrix('A', [[int(i == j) for j in range(cols)] for i in range(rows)])
        if op == 'mul':
            put_matrix('B', [[int(i == j) for j in range(q)] for i in range(cols)])
    else:
        put_matrix('A', random_matrix(rows, cols, invertible=op == 'inv'))
        if op == 'mul':
            put_matrix('B', random_matrix(cols, q))


def grid(prefix: str, rows: int, cols: int, f: Fmt) -> list[list[str]]:
    values = []
    st.html(views.mx_title_html(f'ماتریس {prefix}', rows, cols, f))
    with st.container(key=f'grid_{prefix}'):
        for i in range(rows):
            line = []
            for j, column in enumerate(st.columns(cols)):
                key = f'{prefix}_{i}_{j}'
                if key not in S:
                    S[key] = store().get(key, '')
                with column:
                    value = st.text_input(f'{prefix} سطر {i + 1} ستون {j + 1}', key=key,
                                          label_visibility='collapsed', placeholder=f.i(0),
                                          max_chars=40, autocomplete='off')
                store()[key] = value
                line.append(value)
            values.append(line)
    return values


def read_matrix(prefix: str, values: list[list[str]], errors: list, bad: list) -> tuple:
    matrix = []
    for i, row in enumerate(values):
        line = []
        for j, value in enumerate(row):
            try:
                line.append(parse_number(value))
            except ValueError as exc:
                errors.append(f'ماتریس {prefix}، سطر ⟦{i + 1}⟧، ستون ⟦{j + 1}⟧: {exc}')
                bad.append(f'{prefix}_{i}_{j}')
        matrix.append(tuple(line))
    return tuple(matrix)


def commit_dim(key: str, fields: tuple[str, ...]) -> None:
    for field in fields:
        dims()[field] = S[key]
    S.pop('submission', None)


def dimension(key: str, label: str, value: int, f: Fmt, *fields: str, help: str | None = None) -> None:
    S[key] = value
    st.selectbox(label, NUMS, key=key, format_func=f.i, disabled=not fields,
                 on_change=commit_dim if fields else None, args=(key, fields), help=help)


def expansion(k: int, f: Fmt) -> tuple[str, int]:
    # Keep rows and columns as distinct choices when the matrix order changes.
    options = [(axis, i) for axis in ('row', 'col') for i in range(k)]
    if 'line' not in S or S['line'] not in options:
        saved = store().get('line', ('row', 0))
        S['line'] = saved if saved in options else ('row', 0)
    choice = st.selectbox('بسط از', options, key='line',
                          format_func=lambda v: f"{'سطر' if v[0] == 'row' else 'ستون'} {f.i(v[1] + 1)}")
    store()['line'] = choice
    return choice


def input_panel(op: str, f: Fmt) -> tuple:
    d = dims()
    rows, cols = d['m'], d['n']
    q = d.get('q', 3)
    axis, idx = 'row', 0
    with st.container(key='dimensions'):
        if op in ('det', 'inv'):
            rows = cols = min(rows, cols)
            if op == 'det':
                order, line = st.columns(2)
                with order:
                    dimension('dim_sq', 'مرتبهٔ ماتریس', rows, f, 'm', 'n')
                with line:
                    axis, idx = expansion(rows, f)
            else:
                dimension('dim_sq', 'مرتبهٔ ماتریس', rows, f, 'm', 'n')
        else:
            a_rows, a_cols = st.columns(2)
            with a_rows:
                dimension('dim_m', 'سطرهای A', rows, f, 'm')
            with a_cols:
                dimension('dim_n', 'ستون‌های A', cols, f, 'n')
            if op == 'mul':
                b_rows, b_cols = st.columns(2)
                with b_rows:
                    dimension('dim_b_rows', 'سطرهای B', cols, f, help='از تعداد ستون‌های A تعیین می‌شود.')
                with b_cols:
                    dimension('dim_b_cols', 'ستون‌های B', q, f, 'q')

    problem = (op, rows, cols, axis, q if op == 'mul' else idx)
    with st.form('matrix_input', border=False, enter_to_submit=False):
        matrices = {'A': grid('A', rows, cols, f)}
        if op == 'mul':
            matrices['B'] = grid('B', cols, q, f)
        with st.container(key='preset_buttons'):
            buttons = st.columns(3)
            for column, (label, action) in zip(buttons, (
                ('تصادفی', 'random'), ('همانی', 'identity'), ('پاک‌کردن', 'clear'),
            )):
                with column:
                    st.form_submit_button(label, on_click=preset, args=(op, action, rows, cols, q),
                                          disabled=action == 'identity' and rows != cols, width='stretch')
        with st.container(key='calculate_button'):
            submitted = st.form_submit_button('محاسبه', type='primary', width='stretch')

    if submitted:
        errors, bad, parsed = [], [], {}
        for prefix, values in matrices.items():
            if not any(cell.strip() for row in values for cell in row):
                errors.append(f'ماتریس {prefix} را وارد کنید.')
            parsed[prefix] = read_matrix(prefix, values, errors, bad)
        S['submission'] = {'problem': problem, 'matrices': parsed, 'errors': errors, 'bad': bad}
    elif S.get('submission', {}).get('problem') != problem:
        S.pop('submission', None)

    if bad_cells := S.get('submission', {}).get('bad'):
        st.html(views.invalid_cells_css(bad_cells))
    return problem


def compute(problem: tuple, matrices: dict, f: Fmt):
    op, _, _, axis, idx = problem
    A = matrices['A']
    if op == 'det':
        return solve_det(A, axis, idx, f)
    if op == 'mul':
        return solve_mul(A, matrices['B'], f)
    if op == 'inv':
        return solve_inverse(A, 'first', f)
    return solve_reduce(A, op == 'rref', 'first', f)


def csv_bytes(matrix) -> bytes:
    f = Fmt()
    body = '\n'.join(','.join(f.num(x).replace(MINUS, '-') for x in row) for row in matrix)
    return (body + '\n').encode('utf-8-sig')


@st.cache_data(show_spinner=False)
def load_css() -> tuple[str, str]:
    css = (ROOT / 'styles.css').read_text(encoding='utf-8')
    font = base64.b64encode((ROOT / 'assets' / 'Vazirmatn-Regular.woff2').read_bytes()).decode()
    font_css = ('@font-face{font-family:Vazirmatn;src:url(data:font/woff2;base64,' + font +
                ') format("woff2");font-weight:400;font-style:normal;font-display:swap}')
    return font_css, css


def result_panel(problem: tuple, f: Fmt, font_css: str, css: str) -> None:
    submission = S.get('submission')
    if not submission:
        st.html(views.empty_html('نتیجهٔ محاسبه', 'ماتریس را وارد کنید و «محاسبه» را بزنید.'))
        return
    if submission['errors']:
        errors = [f.t(message) for message in submission['errors'][:3]]
        if len(submission['errors']) > 3:
            errors.append('…')
        st.html(views.errors_html(errors))
        return

    if submission.get('fmt') != f:
        try:
            submission['solution'] = compute(problem, submission['matrices'], f)
        except ValueError as exc:
            st.html(views.errors_html([f.t(str(exc))]))
            return
        submission['fmt'] = f
    sol = submission['solution']

    st.html(views.answer_html(sol, f))
    show_steps = st.checkbox('نمایش مراحل راه‌حل', key='show_steps')
    with st.container(key='export_buttons'):
        pdf, csv = st.columns(2)
        with pdf:
            st.download_button('دانلود راه‌حل PDF', data=partial(pdf_bytes, sol, f, css, font_css),
                               file_name=f'{sol.op}-solution.pdf', mime='application/pdf',
                               on_click='ignore', icon=':material/download:', width='stretch')
        if sol.csv is not None:
            with csv:
                st.download_button('نتیجهٔ CSV', data=csv_bytes(sol.csv), file_name=f'{sol.op}-result.csv',
                                   mime='text/csv', on_click='ignore', width='stretch')
    if show_steps:
        if legend := views.legend_html(sol):
            st.html(legend)
        st.html(views.all_steps_html(sol, f, True))


def navigation() -> tuple[str, Fmt]:
    with st.container(key='navigation', border=True):
        st.html(views.brand_html())
        st.html(views.side_title_html('عملیات'))
        op = st.radio('عملیات', list(OPS), key='op', format_func=lambda k: OPS[k][0],
                      label_visibility='collapsed', on_change=lambda: S.pop('submission', None))
        st.html(views.side_title_html('نمایش اعداد'))
        persian = st.toggle('ارقام فارسی', key='persian')
        with st.expander('راهنمای سریع'):
            st.html(views.help_html((
                'هر خانهٔ خالی برابر ⟦0⟧ حساب می‌شود.',
                'اعشار را با نقطه بنویسید، مثلاً ⟦2.5⟧.',
                'پس از تغییر ورودی، «محاسبه» را بزنید.',
            )))
        st.html(views.credits_html())
    return op, Fmt(persian=persian)


def main() -> None:
    st.set_page_config(page_title='آزمایشگاه جبر خطی', page_icon='📐', layout='wide',
                       initial_sidebar_state='collapsed')
    font_css, css = load_css()
    st.html(f'<style>{font_css}{views.minify_css(css)}</style>')
    st.html(ROOT / 'appearance.html', unsafe_allow_javascript=True)

    with st.container(key='dashboard'):
        menu, workspace = st.columns([3, 12], gap='large', vertical_alignment='top')
        with menu:
            op, f = navigation()
        with workspace:
            st.html(views.header_html(OPS[op][1]))
            with st.container(key='workspace'):
                inputs, output = st.columns([5, 6], gap='medium', vertical_alignment='top')
                with inputs:
                    with st.container(key='panel_in', border=True):
                        problem = input_panel(op, f)
                with output:
                    with st.container(key='panel_out', border=True):
                        result_panel(problem, f, font_css, css)


main()
