#!/bin/bash
# Build the BtP draft with tectonic and print the page on which the main text ends.
# TECTONIC defaults to the session scratchpad copy; override with TECTONIC=/path/to/tectonic.
set -euo pipefail
cd "$(dirname "$0")"
SP=/private/tmp/claude-501/-Users-s3057498-code-ECS8056-Lewis/1edbc43c-058b-467f-9a1a-41a03169f872/scratchpad
TECTONIC=${TECTONIC:-$SP/bin/tectonic}
"$TECTONIC" --keep-logs main.tex 2>&1 | grep -v -E "^note: (downloading|Running|Rerunning)" || true
"$SP/pdfvenv/bin/python" - <<'PY'
import fitz
d = fitz.open("main.pdf")
print(f"pages: {d.page_count}")
for i, page in enumerate(d, 1):
    text = page.get_text()
    for marker in ("References", "REFERENCES", "Appendix", "APPENDIX"):
        if marker in text:
            print(f"  '{marker}' first appears on page {i}")
            break
PY
