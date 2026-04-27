import subprocess
import sys

PREBUILD_COMMANDS = [
    [sys.executable, "scripts/build_aaa_hitter_refresh.py"],
    [sys.executable, "scripts/test_aaa_transactions.py"],
]

MODULES = [
    "dashboard.build_dashboard",
    "dashboard.build_call_up_live",
    "dashboard.build_mlb_extraction",
    "dashboard.build_apex_extraction",
    "dashboard.build_ivb_heat_map",
    "dashboard.build_velocity_decay",
    "dashboard.build_stuff_disruption",
]

def run_module(module_name: str) -> None:
    print(f"\n=== Running {module_name} ===")
    result = subprocess.run([sys.executable, "-m", module_name], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{module_name} failed with exit code {result.returncode}")

def run_command(cmd: list[str], label: str) -> None:
    print(f"\n=== Running {label} ===")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")

def main() -> None:
    for cmd in PREBUILD_COMMANDS:
        run_command(cmd, "scripts/build_aaa_hitter_refresh.py")
    for module_name in MODULES:
        run_module(module_name)
    print("\nAll dashboard builds completed successfully.")

if __name__ == "__main__":
    main()