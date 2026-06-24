#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

MAX_LOG_CHARS = 120000


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tail(text):
    text = text or ""
    return text[-MAX_LOG_CHARS:]


def find_job_id():
    direct = (os.environ.get("ADMIN_AUDIT_JOB_ID") or "").strip()
    if direct:
        return direct

    raw = (os.environ.get("INCOMING_HOOK_BODY") or "").strip()
    if not raw:
        return ""

    try:
        payload = json.loads(raw)
    except Exception:
        return ""

    return str(payload.get("job_id") or payload.get("jobId") or "").strip()


def supabase_patch(job_id, patch):
    supabase_url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    supabase_key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    patch = dict(patch)
    patch["updated_at"] = now_iso()

    req = urllib.request.Request(
        f"{supabase_url}/rest/v1/admin_audit_jobs?id=eq.{job_id}",
        data=json.dumps(patch).encode("utf-8"),
        method="PATCH",
        headers={
            "apikey": supabase_key,
            "authorization": f"Bearer {supabase_key}",
            "content-type": "application/json",
            "prefer": "return=representation",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Supabase patch failed HTTP {exc.code}: {body}") from exc


def run_audit():
    started = time.time()
    proc = subprocess.run(
        ["bash", "scripts/run_green_baseline_audit.sh"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "CI": "1"},
    )

    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "duration_ms": int((time.time() - started) * 1000),
        "stdout": tail(proc.stdout),
        "stderr": tail(proc.stderr),
    }


def main():
    job_id = find_job_id()

    if not job_id:
        print("No ADMIN_AUDIT_JOB_ID or INCOMING_HOOK_BODY job_id found; running green baseline audit without admin job wrapper.")
        result = run_audit()
        if result["ok"]:
            print("Green baseline audit passed in Netlify build runtime without admin job wrapper.")
            return 0
        print(f"Green baseline audit failed in Netlify build runtime without admin job wrapper: exit_code={result['exit_code']}")
        return result["exit_code"]

    print(f"Admin audit job detected: {job_id}")

    supabase_patch(job_id, {
        "status": "running",
        "started_at": now_iso(),
        "message": "Green baseline audit started in Netlify build runtime.",
        "result_payload": {
            "runner": {
                "surface": "signals_netlify_build_runtime",
                "action": "green_baseline",
                "started_at": now_iso(),
            }
        },
    })

    result = run_audit()

    supabase_patch(job_id, {
        "status": "succeeded" if result["ok"] else "failed",
        "completed_at": now_iso(),
        "message": "Green baseline audit passed." if result["ok"] else "Green baseline audit failed. Review stdout/stderr payload.",
        "error": None if result["ok"] else f"Green baseline audit exited with code {result['exit_code']}.",
        "result_payload": {
            "runner": {
                "surface": "signals_netlify_build_runtime",
                "action": "green_baseline",
                "completed_at": now_iso(),
                "ok": result["ok"],
                "exit_code": result["exit_code"],
                "duration_ms": result["duration_ms"],
            },
            "stdout_tail": result["stdout"],
            "stderr_tail": result["stderr"],
        },
    })

    print(f"Admin audit job {job_id} final status: {'succeeded' if result['ok'] else 'failed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
