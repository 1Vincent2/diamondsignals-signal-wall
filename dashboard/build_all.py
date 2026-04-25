import subprocess
import sys

MODULES = [
    "dashboard.build_dashboard",
    "dashboard.build_call_up_live",
    "dashboard.build_hidden_gems",
    "dashboard.build_ivb_heat_map",
    "dashboard.build_velocity_decay",
    "dashboard.build_stuff_disruption",
]

def run_module(module_name: str) -> None:
    print(f"\n=== Running {module_name} ===")
    result = subprocess.run([sys.executable, "-m", module_name], check=False)
    if result.returncode != 0:
        raise SystemExit(f"{module_name} failed with exit code {result.returncode}")

def main() -> None:
    for module_name in MODULES:
        run_module(module_name)
    print("\nAll dashboard builds completed successfully.")

if __name__ == "__main__":
    main()