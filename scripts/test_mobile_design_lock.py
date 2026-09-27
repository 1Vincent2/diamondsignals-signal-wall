"""Semantic contracts for presentation adapters. Run with python3 -B."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dashboard'))
from mobile_hybrid_cards import adapt

class CardTruth(unittest.TestCase):
    def test_fry_has_no_invented_barrel_baseline(self):
        s=adapt(dict(player_name='David Fry',player_type='hitter',edge_score=94.1,metric_1=92.6,metric_2=14.3,metric_3=108,why='Avg EV 92.6 mph (+10.6 vs baseline), barrel-like rate 14.3%.',badges=['EV Burst','Barrel Jump']),'signal-wall')['story']
        self.assertEqual(s['changed'],'Exit velocity and barrel quality have both jumped.')
        self.assertEqual(len(s['evidence']),3)
        self.assertIn('+10.6',s['evidence'][0]['context'])
        self.assertNotIn('baseline',s['evidence'][1]['context'])
        self.assertEqual(s['evidence'][1]['label'],'Barrel-like rate')
    def test_ivb_is_peer_comparison(self):
        s=adapt(dict(ivb_raw='21.4"',ivb_vs_avg='+9.5"'),'ivb-heat-map')['story']
        self.assertIn('velocity-peer baseline',s['changed'])
        self.assertNotIn('rose',s['changed'])
        self.assertEqual(len(s['evidence']),2)
        self.assertEqual(s['detail'][0]['value'],'Not retained')
    def test_missing_values_are_not_zero(self):
        s=adapt({},'velocity-decay')['story']
        self.assertIn('unavailable',s['changed'])
        self.assertEqual(s['evidence'][0]['value'],'Not retained')
    def test_mlb_fraction_to_percentage_points(self):
        s=adapt(dict(kind='pitcher',raw={'recent_whiff_rate':.30,'whiff_delta':.12,'recent_fb_velo':97}),'mlb-extraction')['story']
        self.assertEqual(s['evidence'][1]['value'],'+12.0 pp')
        self.assertIn('Share of all pitches',s['evidence'][0]['context'])
    def test_mlb_ledger_scores_are_not_velocity(self):
        s=adapt(dict(raw={'trait_score_raw':.6,'surface_pressure_raw':.7,'market_score_raw':.8}),'mlb-extraction')['story']
        self.assertTrue(all('Model component' in x['context'] for x in s['evidence']))
    def test_shape_is_not_stuff_rating(self):
        s=adapt(dict(ivb_delta=2,vaa_delta=-.5,movement_delta=1),'stuff-disruption')['story']
        self.assertIn('not a numeric Stuff+ rating',s['context'])
        self.assertNotIn('Stuff+',s['changed'])
    def test_drift_does_not_diagnose_cause(self):
        s=adapt(dict(metrics={'release_speed_delta':-2,'release_extension_delta':-.1,'ivb_delta':-1},recent_appearances=3,baseline_appearances=8),'kinetic-drift')['story']
        self.assertIn('do not establish fatigue',s['why'])
        self.assertIn('3 appearances versus 8',s['context'])
        self.assertEqual(len(s['evidence']),3)
    def test_apex_category_uses_supported_range(self):
        s=adapt({'signal_family':'APEX BAT','forensic_metrics':[{'code':'LA_CONSISTENCY','value':'Surgical'}]},'apex-extraction')['story']
        self.assertEqual(s['evidence'][0]['value'],'15–25° band')
    def test_no_mutation_of_canonical_row(self):
        r={'player_name':'Test','ivb_vs_avg':'+3.0"'}; before=dict(r);adapt(r,'ivb-heat-map');self.assertEqual(r,before)

if __name__=='__main__':unittest.main()
