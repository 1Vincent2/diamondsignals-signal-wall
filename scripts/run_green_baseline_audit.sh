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
echo "--- Promotion Watch live rebuild ---"
python3 -m dashboard.build_call_up_live

echo
echo "--- Dashboard / dossier rebuild for Promotion Watch context ---"
python3 -m dashboard.build_dashboard

echo
echo "--- Promotion Watch payload contract ---"
python3 scripts/test_promotion_watch_payload_contract.py

echo
echo "--- Promotion Watch integrity audit ---"
python3 dashboard/audit_promotion_watch_integrity.py

echo
echo "--- AAA UI hardening audit ---"
python3 scripts/test_aaa_ui_hardening.py


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
echo "--- stale live-v2 artifact audit ---"
python3 scripts/audit_no_stale_live_v2_artifact.py
echo
echo "--- fallback snapshot seeder ---"
python3 scripts/seed_fallback_snapshots.py

echo
echo "--- report inventory truth audit ---"
python3 scripts/audit_report_inventory_truth.py

echo "--- scout dossier route integrity audit ---"
python3 scripts/audit_scout_dossier_route_integrity.py
echo "--- canonical player identity contract audit ---"
python3 scripts/audit_canonical_player_identity_contract.py
python3 scripts/audit_crawl_index_boundaries.py

echo
echo "--- build entrypoint coverage audit ---"
python3 scripts/audit_build_entrypoint_coverage.py

echo
echo "--- build output freshness audit ---"
python3 scripts/audit_build_output_freshness.py

echo
echo "--- status contract audit ---"
python3 scripts/audit_status_contract.py

echo
echo "--- refinement queue closure audit ---"
python3 scripts/audit_refinement_queue_closure.py

echo
echo "--- green baseline audit registry audit ---"
python3 scripts/audit_green_baseline_registry.py
echo
echo "--- Netlify green baseline deploy gate audit ---"
python3 scripts/audit_netlify_green_baseline_gate.py
python3 scripts/write_netlify_headers.py
python3 scripts/audit_cache_busting_cdn_contract.py
python3 scripts/audit_admin_build_runtime_job_contract.py
python3 scripts/audit_admin_readiness_truth.py

echo
echo "--- admin audit manifest builder ---"
python3 scripts/build_admin_audit_manifest.py

echo
echo "--- admin audit manifest contract audit ---"
python3 scripts/audit_admin_audit_manifest.py



echo
echo "--- mobile header/menu obstruction audit ---"
python3 scripts/audit_mobile_header_menu_contract.py
echo "--- desktop/mobile deferred contract audit ---"
python3 scripts/audit_desktop_mobile_deferred_contract.py
python3 scripts/audit_pre_mobile_readiness_seal.py
python3 scripts/audit_supabase_pooling_contract.py
python3 scripts/audit_strict_environment_preflight.py

echo
echo "--- tracking regression audit ---"
python3 scripts/test_tracking_actions.py
python3 scripts/audit_tracking_write_idempotency_contract.py

echo
echo "--- GREEN BASELINE AUDIT PASSED ---"
