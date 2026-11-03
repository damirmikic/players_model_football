"""Inline pricing.js + each league's data into ../build/kalkulator_<league>.html and unified ../build/index.html."""
import glob, json, os, re

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..") + "/"
app = open(d + "web/app.html", encoding="utf-8").read()
js = open(d + "web/pricing.js", encoding="utf-8").read()
all_data = {}
for path in sorted(glob.glob(d + "build/data_*.json")):
    key = re.search(r"data_(.+)\.json$", path).group(1)
    raw = open(path, encoding="utf-8").read()
    all_data[key] = json.loads(raw)
    html = app.replace("/*__PRICING__*/", js).replace("/*__DATA__*/", raw)
    out = d + f"build/kalkulator_{key}.html"
    open(out, "w", encoding="utf-8").write(html)
    print(os.path.basename(out), len(html))

if all_data:
    default_key = "EPL" if "EPL" in all_data else next(iter(all_data))
    bundle = json.dumps({"__bundle__": True, "default": default_key, "data": all_data},
                        ensure_ascii=False, separators=(",", ":"))
    index_html = app.replace("/*__PRICING__*/", js).replace("/*__DATA__*/", bundle)
    index_out = d + "build/index.html"
    open(index_out, "w", encoding="utf-8").write(index_html)
    print(os.path.basename(index_out), len(index_html))

