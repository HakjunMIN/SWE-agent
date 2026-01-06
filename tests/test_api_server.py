from __future__ import annotations

from fastapi.testclient import TestClient

from sweagent.api.server import create_app


def test_health() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_help() -> None:
    client = TestClient(create_app())
    resp = client.post("/run", json={"command": "run", "extra_args": ["--help"]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["returncode"] == 0
    # The CLI help is printed to stdout by argparse/rich.
    assert "sweagent" in (data["stdout"] + data["stderr"]).lower()
