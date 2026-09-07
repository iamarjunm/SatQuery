"""Stand-in for the four model services, serving the fixtures in mock_data/
over HTTP in the team contract: POST /<tool> with a JSON body, returning
{"status": "success", "result": {...}}.

This is the last step before real models: the controller talks to it the
same way it will talk to the AI team's services (tools._http), so it proves
the HTTP adapter, the envelope handling and the response validation, not
just the in-process mocks.

Run:   .venv/Scripts/python mock_service.py            # ports 8001-8004
       .venv/Scripts/python mock_service.py --port 9000  # single port, set SATQUERY_<TOOL>_URL
Then:  SATQUERY_MOCK=0 .venv/Scripts/python e2e.py

Fixtures are keyed by the image id(s) in the request:
    vqa.json            image_id
    ground.json         image_id
    change_detect.json  "image_id_t1>image_id_t2"
    cross_modal.json    "optical_image_id+sar_image_id"
An unknown key returns 404 with an error envelope, which is what a real
service should do for an image it has not loaded.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools import TOOLS

FIXTURE_DIR = Path(__file__).with_name("mock_data")

# How each tool's request body maps onto a fixture key.
_KEY = {
    "vqa": lambda a: a["image_id"],
    "ground": lambda a: a["image_id"],
    "change_detect": lambda a: f"{a['image_id_t1']}>{a['image_id_t2']}",
    "cross_modal": lambda a: f"{a['optical_image_id']}+{a['sar_image_id']}",
}


def load_fixtures() -> dict[str, dict]:
    return {name: json.loads((FIXTURE_DIR / f"{name}.json").read_text(encoding="utf-8"))
            for name in _KEY}


def respond(fixtures: dict[str, dict], tool: str, args: dict) -> tuple[int, dict]:
    """Pure function so tests can call it without a socket."""
    if tool not in _KEY:
        return 404, {"status": "error", "error": f"unknown tool {tool!r}"}
    try:
        key = _KEY[tool](args)
    except (KeyError, TypeError) as exc:
        return 400, {"status": "error", "error": f"{tool}: missing argument {exc}"}
    result = fixtures[tool].get(key)
    if result is None:
        return 404, {"status": "error", "error": f"{tool}: no fixture for {key!r}"}
    result = json.loads(json.dumps(result))  # copy; never hand out the cached dict
    if tool == "ground":
        # A real grounding model labels what it was asked for.
        for obj in result["objects"]:
            obj["label"] = str(args.get("phrase", obj["label"]))
    return 200, {"status": "success", "result": result}


class Handler(BaseHTTPRequestHandler):
    fixtures: dict[str, dict] = {}

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            status, body = 400, {"status": "error", "error": "body is not JSON"}
        else:
            status, body = respond(self.fixtures, self.path.strip("/"), args)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args) -> None:  # one line per call, no timestamps
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def serve(ports: list[int]) -> None:
    Handler.fixtures = load_fixtures()
    servers = [ThreadingHTTPServer(("127.0.0.1", p), Handler) for p in ports]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("mock services on " + ", ".join(f"http://127.0.0.1:{p}/" for p in ports) + "  (Ctrl+C to stop)", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, help="serve every tool on this one port instead of each tool's default")
    ns = ap.parse_args()
    serve([ns.port] if ns.port else sorted({spec.port for spec in TOOLS.values() if spec.port}))
