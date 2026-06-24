from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
failures = []

private_markers = [
    "SUPABASE_SERVICE_ROLE_KEY",
    "SERVICE_ROLE",
    "DATABASE_URL",
    "DIRECT_URL",
    "POSTGRES_URL",
    "POSTGRES_PRISMA_URL",
    "SUPABASE_DB_PASSWORD",
    "PGPASSWORD",
    "ADMIN_RUN_TOKEN",
    "YAHOO_CLIENT_SECRET",
    "YAHOO_REFRESH_TOKEN",
]

static_roots = [
    ROOT / "dist",
    ROOT / "public",
]

allowed_private_script_paths = [
    "scripts/run_aaa_weekly_ingest.sh",
    "scripts/run_admin_green_baseline_from_build.py",
    "scripts/verify_aaa_weekly_ingest.sh",
    "scripts/audit_signal_wall_env_boundary.py",
    "scripts/audit_strict_environment_preflight.py",
]

def run(cmd):
    return subprocess.run(
        cmd,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

tracked = run('git ls-files ".env" ".env.*"')
if tracked.stdout.strip():
    for line in tracked.stdout.strip().splitlines():
        failures.append(f"env file is tracked by git: {line}")

gitignore = ROOT / ".gitignore"
if not gitignore.exists():
    failures.append(".gitignore missing")
else:
    ignore_text = gitignore.read_text(errors="ignore")
    if ".env" not in ignore_text:
        failures.append(".gitignore does not protect .env")
    if ".env.*" not in ignore_text and ".env*" not in ignore_text:
        failures.append(".gitignore does not protect .env.*")

for static_root in static_roots:
    if not static_root.exists():
        continue
    for path in static_root.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size > 3_000_000:
            continue
        text = path.read_text(errors="ignore")
        for marker in private_markers:
            if marker in text:
                failures.append(f"private env marker leaked into deployed/static output: {path.relative_to(ROOT)} :: {marker}")

for path in (ROOT / "src").rglob("*") if (ROOT / "src").exists() else []:
    if not path.is_file():
        continue
    if path.suffix not in {".js", ".jsx", ".ts", ".tsx", ".html"}:
        continue
    text = path.read_text(errors="ignore")
    for marker in private_markers:
        if marker in text:
            failures.append(f"private env marker found in client/source surface: {path.relative_to(ROOT)} :: {marker}")

for path in (ROOT / "scripts").rglob("*") if (ROOT / "scripts").exists() else []:
    if not path.is_file():
        continue
    rel = str(path.relative_to(ROOT))
    text = path.read_text(errors="ignore")
    for marker in private_markers:
        if marker in text and rel not in allowed_private_script_paths:
            failures.append(f"private env marker found in unapproved script: {rel} :: {marker}")

if failures:
    print("FINAL_STATUS: FAIL_SIGNAL_WALL_ENV_BOUNDARY")
    for failure in failures:
        print(f" - {failure}")
    sys.exit(1)

print("FINAL_STATUS: PASS_SIGNAL_WALL_ENV_BOUNDARY")
