"""Render the self-contained solution report as an A4 PDF."""
from __future__ import annotations

import os

from playwright.sync_api import sync_playwright

from linear_algebra import Fmt, Solution
from views import report_html


def pdf_bytes(solution: Solution, fmt: Fmt, css: str, font_css: str) -> bytes:
    document = report_html(solution, fmt, css, font_css)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=os.getenv('LINEAR_ALGEBRA_CHROMIUM') or None)
        try:
            page = browser.new_page()
            page.set_content(document)
            page.evaluate('document.fonts.ready')
            return page.pdf(format='A4', prefer_css_page_size=True, print_background=True)
        finally:
            browser.close()
