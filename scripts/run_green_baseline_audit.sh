#!/usr/bin/env bash
set -euo pipefail

echo "--- DiamondSignals green baseline audit ---"

echo
echo "--- route/domain contract ---"
python3 scripts/test_route_domain_contract.py

echo
echo "--- IVB Heat Map contract ---"
python3 scripts/test_ivb_heat_map_contract.py

echo
echo "--- Promotion Watch payload contract ---"
python3 scripts/test_promotion_watch_payload_contract.py

echo
echo "--- Kinetic Drift contract ---"
python3 scripts/test_kinetic_drift_contract.py

echo
echo "--- Velocity Decay contract ---"
python3 scripts/test_velocity_decay_contract.py

echo
echo "--- report data-mode hardening inspection ---"
python3 dashboard/audit_report_data_modes.py

echo
echo "--- legacy route/name contract audit ---"
python3 scripts/audit_legacy_route_names.py
echo
echo "--- stale artifact containment audit ---"
python3 scripts/audit_stale_artifact_containment.py


echo
echo "--- mobile header/menu obstruction audit ---"
python3 scripts/audit_mobile_header_menu_contract.py

echo
echo "--- tracking regression audit ---"
python3 scripts/test_tracking_actions.py

echo
echo "--- GREEN BASELINE AUDIT PASSED ---"
