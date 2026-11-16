"""HTML builders for the dashboard. No Streamlit imports.

Every function returns ONE line of HTML (no newlines, no blank lines, no indentation):
st.markdown would otherwise treat indented or split HTML as a code block.
Only <div> and <span> are used, so Streamlit's own paragraph/heading styles never leak in.
"""
from __future__ import annotations

from html import escape
import re

from linear_algebra import Fmt, MView, Solution, Step

_MATH = re.compile(r'⟦(.*?)⟧')
_BOLD = re.compile(r'\*\*(.+?)\*\*')
_ORDINALS = ('اول', 'دوم', 'سوم', 'چهارم', 'پنجم', 'ششم', 'هفتم', 'هشتم', 'نهم', 'دهم')


def rich(text: str) -> str:
    """Escape, then turn ⟦math⟧ into an LTR island and **bold** into <b>."""
    out = escape(text, quote=False)
    out = _MATH.sub(lambda m: f'<span class="la-m">{m.group(1)}</span>', out)
    return _BOLD.sub(r'<b>\1</b>', out)


def minify_css(css: str) -> str:
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    css = re.sub(r'\s+', ' ', css)
    return re.sub(r'\s*([{};,])\s*', r'\1', css).replace(': ', ':').strip()


# ----------------------------------------------------------------- matrices --
def matrix_html(v: MView, f: Fmt) -> str:
    rows, cols = len(v.m), len(v.m[0])
    h = v.hl
    cells = []
    for i, row in enumerate(v.m):
        for j, x in enumerate(row):
            cls = ['la-c']
            if (i, j) in h.pivot:
                cls.append('p')
            if i in h.rows or j in h.cols:
                cls.append('h')
            if i in h.src:
                cls.append('s')
            if i in h.dim_rows or j in h.dim_cols:
                cls.append('d')
            if (i, j) in h.focus:
                cls.append('f')
            if v.sep is not None and j == v.sep:
                cls.append('sp')
            text = x if isinstance(x, str) else f.num(x)
            cells.append(f'<span class="{" ".join(cls)}">{escape(text)}</span>')
    cap = ''
    if v.label:
        cap = (f'<div class="la-mx-cap"><span dir="auto">{rich(v.label)}</span>'
               f'<span class="la-dim">{f.i(rows)}×{f.i(cols)}</span></div>')
    size = (' sm' if v.small else '') + (' dn' if rows >= 6 or cols >= 6 else '')
    return (f'<div class="la-mx{size}">{cap}<div class="la-mx-body"><div class="la-br">'
            f'<div class="la-grid" style="grid-template-columns:repeat({cols},auto)">{"".join(cells)}</div>'
            f'</div></div></div>')


def chips_html(chips) -> str:
    items = ''.join(
        f'<span class="la-chip {escape(c.tone)}"><span class="k">{escape(c.label)}</span>'
        f'<b dir="auto">{escape(c.value)}</b></span>' for c in chips)
    return f'<div class="la-chips">{items}</div>' if items else ''


def formulas_html(lines) -> str:
    if not lines:
        return ''
    rows = []
    for ln in lines:
        if ln.startswith('#'):
            rows.append(f'<div class="la-note">{rich(ln[1:].strip())}</div>')
        else:
            rows.append(f'<div class="la-ln">{escape(ln)}</div>')
    return f'<div class="la-math">{"".join(rows)}</div>'


# ------------------------------------------------------------------ answer ---
def answer_html(sol: Solution, f: Fmt) -> str:
    main = []
    if sol.scalar:
        k, val = sol.scalar
        main.append(f'<div class="la-big"><div class="la-ans-k">{escape(k)} =</div>'
                    f'<div class="la-ans-num">{escape(val)}</div></div>')
    main += [matrix_html(mv, f) for mv in sol.results]
    if sol.lines:
        main.append('<div class="la-lines">' + ''.join(f'<div>{escape(ln)}</div>' for ln in sol.lines) + '</div>')
    side = chips_html(sol.chips)
    body = ''.join(main)
    return (f'<div class="la-ans"><div class="la-ans-title">{escape(sol.title)}</div>'
            f'<div class="la-ans-row"><div class="la-ans-main">{body}</div>{side}</div></div>')


# -------------------------------------------------------------------- steps --
_TAG_CLASS = {'result': 'res', 'check': 'res', 'info': 'inf', 'intro': 'int'}


def _step_inner(step: Step, f: Fmt, show_why: bool) -> str:
    views = ''.join(matrix_html(v, f) for v in step.views)
    views_block = f'<div class="la-views">{views}</div>' if views else ''
    why = f'<div class="la-why"><b>چرا؟</b> {rich(step.why)}</div>' if (show_why and step.why) else ''
    return f'<div class="la-text">{rich(step.text)}</div>{formulas_html(step.formulas)}{views_block}{why}'


def step_html(step: Step, index: int, total: int, f: Fmt, show_why: bool) -> str:
    pct = round((index + 1) / total * 100)
    tag_cls = _TAG_CLASS.get(step.kind, '')
    phase = f'<span class="la-phase-inline">{escape(step.phase)}</span>' if step.phase else ''
    return (f'<div class="la-step"><div class="la-step-head"><span class="la-tag {tag_cls}">{escape(step.tag)}</span>'
            f'{phase}<span class="la-count">مرحلهٔ {f.i(index + 1)} از {f.i(total)}</span></div>'
            f'<div class="la-bar"><i style="width:{pct}%"></i></div>{_step_inner(step, f, show_why)}</div>')


def all_steps_html(sol: Solution, f: Fmt, show_why: bool) -> str:
    out = []
    last_phase = None
    for k, step in enumerate(sol.steps, start=1):
        if step.phase and step.phase != last_phase:
            out.append(f'<div class="la-phase">{escape(step.phase)}</div>')
        last_phase = step.phase or last_phase
        tag_cls = _TAG_CLASS.get(step.kind, '')
        ordinal = _ORDINALS[k - 1] if k <= len(_ORDINALS) else f.i(k)
        out.append(f'<div class="la-step compact"><div class="la-step-head"><span class="la-tag {tag_cls}">{escape(step.tag)}</span>'
                   f'<span class="la-count">مرحلهٔ {ordinal}</span></div>{_step_inner(step, f, show_why)}</div>')
    return f'<div class="la-all">{"".join(out)}</div>'


def legend_html(sol: Solution) -> str:
    seen = set()
    for st in sol.steps:
        for v in st.views:
            if v.hl.pivot:
                seen.add('p')
            if v.hl.rows or v.hl.cols:
                seen.add('h')
            if v.hl.src:
                seen.add('s')
            if v.hl.dim_rows or v.hl.dim_cols:
                seen.add('d')
            if v.hl.focus and not v.hl.pivot:
                seen.add('f')
    names = {
        'p': ('p', 'عنصر پیشرو'),
        'h': ('h', 'سطر یا ستون فعال'),
        's': ('s', 'سطر مرجع'),
        'd': ('d', 'سطر و ستون خط‌خورده'),
        'f': ('f', 'عنصر انتخاب‌شده'),
    }
    items = ''.join(f'<span class="la-lg"><i class="{names[k][0]}"></i>{names[k][1]}</span>'
                    for k in ('p', 'h', 's', 'd', 'f') if k in seen)
    return f'<div class="la-legend">{items}</div>' if items else ''


# ------------------------------------------------------------ small pieces ---
def empty_html(title: str, hint: str) -> str:
    return (f'<div class="la-empty">'
            f'<div class="la-empty-t">{escape(title)}</div><div>{rich(hint)}</div></div>')


def errors_html(messages) -> str:
    items = ''.join(f'<div>{rich(m)}</div>' for m in messages)
    return f'<div class="la-err">{items}</div>'


def header_html(title: str) -> str:
    return f'<div class="la-head"><h1 class="la-title">{escape(title)}</h1></div>'


def brand_html() -> str:
    return ('<div class="la-brandcard"><div class="la-logo" aria-hidden="true"></div>'
            '<div class="la-brand-t">آزمایشگاه جبر خطی</div></div>')


def mx_title_html(name: str, rows: int, cols: int, f: Fmt) -> str:
    return (f'<div class="la-mxt"><span dir="auto">{rich(name)}</span>'
            f'<span class="la-dim">{f.i(rows)}×{f.i(cols)}</span></div>')


def hint_html(text: str) -> str:
    return f'<div class="la-hint">{rich(text)}</div>'


def help_html(lines) -> str:
    return '<div class="la-help">' + ''.join(f'<div>{rich(t)}</div>' for t in lines) + '</div>'


def side_title_html(text: str) -> str:
    return f'<div class="la-sec">{escape(text)}</div>'


def credits_html() -> str:
    return ('<div class="la-credit"><div>استاد راهنما: آقای عباس رضایی</div>'
            '<div>تهیه‌کننده: پردیس جدید اسلامی</div></div>')


def invalid_cells_css(keys) -> str:
    """Red outline for the text boxes that hold an invalid number."""
    if not keys:
        return ''
    sel = ','.join(f'.st-key-{k} [data-testid="stTextInputRootElement"],.st-key-{k} input' for k in keys)
    return f'<style>{sel}{{border-color:#dc2626!important;box-shadow:0 0 0 1px #dc2626!important;}}</style>'


# ------------------------------------------------------------------ report ---
def report_html(sol: Solution, f: Fmt, css: str, font_css: str) -> str:
    """Self-contained HTML of the whole solution, with A4 print styling."""
    inputs = ''.join(matrix_html(v, f) for v in sol.inputs)
    steps = all_steps_html(sol, f, True)
    result = answer_html(sol, f)
    return ('<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(sol.title)}</title>'
            f'<style>{font_css}{minify_css(css)}'
            '@page{size:A4;margin:14mm 13mm}'
            'body{margin:0;background:#fff;color:#0f172a;font-family:Vazirmatn,Tahoma,sans-serif;direction:rtl;font-size:15px}'
            '.la-report{max-width:920px;margin:0 auto;padding:24px 20px}'
            '.la-report .la-views{display:block}'
            '.la-report .la-views .la-mx{margin:4px 0 4px 16px}'
            '.la-report .la-all{display:block;max-height:none;overflow:visible;padding:0}'
            '.la-report .la-step{margin-top:10px}'
            '@media print{'
            'body{-webkit-print-color-adjust:exact;print-color-adjust:exact}'
            '.la-report{padding:0}'
            '.la-report .la-step,.la-report .la-ans,.la-report .la-mx{break-inside:avoid}'
            '.la-report .la-phase,.la-report .la-title{break-after:avoid}'
            '.la-report .la-mx-body{overflow:visible}'
            '.la-report .la-c{display:block;text-align:center;min-width:0;white-space:normal;overflow-wrap:anywhere}'
            '.la-report .la-step.compact{padding:8px 10px}'
            '.la-report .la-text{font-size:14px;line-height:1.75}'
            '.la-report .la-math,.la-report .la-why{font-size:12.5px;line-height:1.7}'
            '.la-report .la-mx.dn .la-grid{gap:2px}'
            '.la-report .la-mx.dn .la-c{font-size:11.5px;line-height:1.35;padding:1px 3px}'
            '.la-report .la-mx.dn .la-c.sp{margin-left:4px;padding-left:5px}'
            '}'
            f'</style></head><body><div class="la-report"><div class="la-title">{escape(sol.title)}</div>'
            f'<div class="la-views" style="margin:12px 0">{inputs}</div>{result}'
            f'<div class="la-phase" style="margin-top:14px">راه‌حل گام‌به‌گام</div>{steps}</div></body></html>')
