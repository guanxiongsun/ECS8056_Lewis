#!/usr/bin/env python3
"""Package the BtP draft for Overleaf.

Writes, next to this script:
  open_in_overleaf.html  an auto-submitting form that POSTs every project file to
                         https://www.overleaf.com/docs as base64 data URLs; opening it in a
                         browser that is logged in to Overleaf creates a new private project.
  btp_paper_overleaf.zip the same files, for Overleaf's "New Project > Upload Project".

Nothing is uploaded anywhere else. Usage: python3 make_overleaf.py [--open]
"""
import base64, html, subprocess, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
FILES = ["main.tex", "refs.bib"] + sorted(
    str(p.relative_to(PAPER)) for p in (PAPER / "figures").glob("*")
    if p.is_file() and p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})
MIME = {".tex": "application/x-tex", ".bib": "application/x-bibtex", ".pdf": "application/pdf",
        ".png": "image/png", ".jpg": "image/jpeg"}

inputs = []
for name in FILES:
    data = (PAPER / name).read_bytes()
    uri = f"data:{MIME.get(Path(name).suffix, 'application/octet-stream')};base64,{base64.b64encode(data).decode()}"
    inputs.append(f'<input type="hidden" name="snip_uri[]" value="{html.escape(uri)}">')
    inputs.append(f'<input type="hidden" name="snip_name[]" value="{html.escape(name)}">')

page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Open BtP draft in Overleaf</title></head>
<body style="font-family: sans-serif; max-width: 40em; margin: 3em auto;">
<p>Creating a new private Overleaf project from {len(FILES)} files: {', '.join(FILES)}.</p>
<form id="f" action="https://www.overleaf.com/docs" method="post">
{chr(10).join(inputs)}
<input type="hidden" name="engine" value="pdflatex">
<input type="hidden" name="main_document" value="main.tex">
<button type="submit">Open in Overleaf</button>
</form>
<script>document.getElementById('f').submit();</script>
</body></html>
"""
(HERE / "open_in_overleaf.html").write_text(page, encoding="utf-8")

with zipfile.ZipFile(HERE / "btp_paper_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in FILES:
        z.write(PAPER / name, name)

print(f"wrote open_in_overleaf.html ({len(page) // 1024} KB) and btp_paper_overleaf.zip with: {FILES}")
if "--open" in sys.argv:
    subprocess.run(["open", str(HERE / "open_in_overleaf.html")], check=True)
