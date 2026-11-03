"""Inline pricing.js + each league's data into the calculator: ../build/kalkulator_<league>.html."""
import glob, os, re

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
app = open(d + "web/app.html", encoding="utf-8").read()
js = open(d + "web/pricing.js", encoding="utf-8").read()
for path in sorted(glob.glob(d + "build/data_*.json")):
    key = re.search(r"data_(.+)\.json$", path).group(1)
    html = app.replace("/*__PRICING__*/", js).replace("/*__DATA__*/", open(path, encoding="utf-8").read())
    out = d + f"build/kalkulator_{key}.html"
    open(out, "w", encoding="utf-8").write(html)
    print(os.path.basename(out), len(html))
