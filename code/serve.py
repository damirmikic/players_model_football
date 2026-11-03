"""Local web app server for the calculator with live FotMob lineup refresh.

    python serve.py [PORT]     # serves ../build/ at http://127.0.0.1:8000 and handles POST /api/refresh"""
import http.server, json, os, subprocess, sys, urllib.parse
import leagues

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "build"))
VENV_PY = os.path.abspath(os.path.join(CODE_DIR, "..", ".venv", "bin", "python"))
PY = VENV_PY if os.path.exists(VENV_PY) else sys.executable


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BUILD_DIR, **kwargs)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/refresh":
            self.send_error(404)
            return
        qs = urllib.parse.parse_qs(u.query)
        lg = (qs.get("league") or ["EPL"])[0]
        if lg not in leagues.LEAGUES:
            self.send_error(400, f"Unknown league: {lg}")
            return
        try:
            subprocess.run([PY, "lineups.py", lg], cwd=CODE_DIR, check=True)
            subprocess.run([PY, "export.py", lg], cwd=CODE_DIR, check=True)
            subprocess.run([PY, "build.py"], cwd=CODE_DIR, check=True)
            data_path = os.path.join(BUILD_DIR, f"data_{lg}.json")
            payload = open(data_path, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            msg = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if not os.path.exists(os.path.join(BUILD_DIR, "index.html")):
        subprocess.run([sys.executable, "build.py"], cwd=CODE_DIR, check=True)
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"Serving web app at http://127.0.0.1:{port} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

