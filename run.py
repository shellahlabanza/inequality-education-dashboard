"""One command: rebuild the data from the CSV, then serve the dashboard at http://localhost:8000.
Usage:  python run.py"""
import subprocess, sys, threading, webbrowser, http.server, socketserver, functools
from pathlib import Path

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "build_data.py")], check=True)
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root / "public"))
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", 8000), handler) as srv:
    print("\nDashboard running at http://localhost:8000   (Ctrl+C to stop)")
    threading.Timer(1, lambda: webbrowser.open("http://localhost:8000")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
