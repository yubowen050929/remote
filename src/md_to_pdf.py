"""Convert the flagship report to a styled PDF (CJK + embedded exhibits).

Markdown -> HTML (with tables) -> PDF via WeasyPrint. Uses a print stylesheet
with a CJK font, McKinsey-ish typography, and page breaks before each major
section. Image paths are resolved relative to the repo root.
"""
import os
import re

import markdown
from weasyprint import HTML

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = os.path.join(ROOT, "DEEP_RESEARCH_REPORT.md")
PDF = os.path.join(ROOT, "DEEP_RESEARCH_REPORT.pdf")

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm;
  @bottom-right { content: counter(page) " / " counter(pages);
                  font-size: 8pt; color: #999; } }
body { font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", sans-serif;
  font-size: 10.5pt; line-height: 1.55; color: #222; }
h1 { color: #1f3a5f; font-size: 20pt; border-bottom: 3px solid #1f3a5f;
  padding-bottom: 6px; margin-top: 8px; }
h2 { color: #1f3a5f; font-size: 14.5pt; border-left: 5px solid #4a7ba6;
  padding-left: 10px; margin-top: 22px; page-break-after: avoid; }
h3 { color: #2c3e50; font-size: 11.5pt; margin-top: 16px; page-break-after: avoid; }
h2:contains("第") { page-break-before: always; }
blockquote { background: #f4f8fb; border-left: 4px solid #4a7ba6;
  margin: 12px 0; padding: 8px 14px; color: #2c3e50; font-size: 10pt; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt;
  page-break-inside: avoid; }
th { background: #1f3a5f; color: #fff; padding: 6px 8px; text-align: left; }
td { border-bottom: 1px solid #ddd; padding: 5px 8px; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fb; }
img { max-width: 100%; margin: 10px 0; page-break-inside: avoid;
  border: 1px solid #eee; }
code { background: #f0f2f5; padding: 1px 4px; border-radius: 3px;
  font-family: "DejaVu Sans Mono", monospace; font-size: 9pt; }
pre { background: #1f3a5f; color: #e8eef5; padding: 12px; border-radius: 4px;
  font-size: 8.5pt; line-height: 1.4; page-break-inside: avoid; }
pre code { background: none; color: inherit; }
strong { color: #1a1a1a; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
"""


def main():
    text = open(SRC, encoding="utf-8").read()
    # force a page break before each numbered major section
    text = re.sub(r"\n(## 第[一二三四五六]+部分)", r"\n<div class='pb'></div>\n\1", text)
    html_body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"])
    html_body = html_body.replace("<div class='pb'></div>",
                                  "<div style='page-break-before: always;'></div>")
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>" \
           f"<body>{html_body}</body></html>"
    HTML(string=html, base_url=ROOT).write_pdf(PDF)
    size = os.path.getsize(PDF) / 1024
    print(f"wrote {PDF}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
