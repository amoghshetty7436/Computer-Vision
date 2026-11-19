"""Build the shareable reports: inline every image as a data URI so each HTML file is self-contained."""
import base64, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SEARCH = [ROOT / "docs/report_assets", ROOT / "results/figures", ROOT / "docs"]
def uri(name):
    for d in SEARCH:
        p = d / name
        if p.exists():
            mime = "image/png" if p.suffix == ".png" else "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    raise FileNotFoundError(name)
def render(src):
    return re.sub(r"\{\{([\w.\-]+)\}\}", lambda m: uri(m.group(1)), src.read_text())
def standalone(body):
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
if __name__ == "__main__":
    src, artifact_out, standalone_out = map(Path, sys.argv[1:4])
    body = render(src)
    artifact_out.write_text(body)
    standalone_out.write_text(standalone(body))
    print(standalone_out, len(body) // 1024, "KB")


# ---------------------------------------------------------------- detailed report (PDF)
def summary_tables(md_path):
    """Map section keys (E1, E1-seq, E1s, E2a, ...) to the markdown tables of results/summary.md as HTML."""
    out, key, buf, seq = {}, None, [], False
    def flush():
        nonlocal buf
        if buf and key:
            k = key + ("-seq" if seq else "")
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in buf if not set(l.replace("|", "").strip()) <= set("-: ")]
            h = "<table><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>"
            h += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>"
            out[k] = h
        buf = []
    for line in Path(md_path).read_text().splitlines():
        if line.startswith("## "):
            flush(); key = line[3:].split()[0]; seq = False
        elif line.startswith("Per sequence"):
            flush(); seq = True
        elif line.startswith("|"):
            buf.append(line)
        else:
            flush()
    flush()
    return out

def refs_html(md_path):
    txt = Path(md_path).read_text().split("## References", 1)[1]
    lines = [l for l in txt.splitlines() if l.startswith("[")]
    lines.append("[Cai23]      Cai, Tung, Wang, Averbuch-Elor, Hariharan, Snavely. Doppelgangers: Learning to Disambiguate Images of Similar Structures. ICCV 2023. https://arxiv.org/abs/2309.02420")
    lines.sort(key=str.lower)
    import html
    return "\n".join(f"<p>{html.escape(l)}</p>" for l in lines)

def render_detailed(src):
    tabs = summary_tables(ROOT / "results/summary.md")
    s = src.read_text()
    s = re.sub(r"\{\{tbl:([\w\-]+)\}\}", lambda m: tabs[m.group(1)], s)
    s = s.replace("{{refs}}", refs_html(ROOT / "docs/related_work.md"))
    return re.sub(r"\{\{([\w.\-]+)\}\}", lambda m: uri(m.group(1)), s)
