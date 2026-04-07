from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

SCRIPTS = [
    ROOT / "build_dashboard.py",
    ROOT / "build_call_up_v2.py",
    ROOT / "build_ivb_heat_map.py",

]

def run_script(path: Path) -> None:
    print(f"\n=== Running {path.name} ===")
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{path.name} failed with exit code {result.returncode}")

def main() -> None:
    for script in SCRIPTS:
        run_script(script)
    print("\nAll dashboard builds completed successfully.")

if __name__ == "__main__":
    main()