import sys
d='/home/claude/epl/'
html=open(d+'web/app.html').read()
html=html.replace('/*__PRICING__*/',open(d+'web/pricing.js').read()).replace('/*__DATA__*/',open(d+'build/epl_data.json').read())
open(d+'build/epl_kalkulator.html','w').write(html); print(len(html))
