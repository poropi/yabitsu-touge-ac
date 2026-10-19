"""区間端・カバレッジ・工程の混同防止を確認する。"""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blender_scripts'))
from landscape_section_coverage import station_plan, coverage_cells, summarize, record_visual_review

class SectionCoverageTests(unittest.TestCase):
    """描画完了のみで品質を完了扱いにしないことを検査する。"""
    def test_entire_s1_and_endpoint(self):
        """全335駅と334セルが長さを過不足なく覆う。"""
        length = 3335.9881731028836
        stations = station_plan(length)
        self.assertEqual(len(stations), 335)
        self.assertEqual(stations[-1], length)
        self.assertLessEqual(max(b-a for a,b in zip(stations, stations[1:])), 10)
        self.assertAlmostEqual(summarize(coverage_cells(length), length)['total_length_m'], length)

    def test_rendering_does_not_mean_quality(self):
        """両方向を描画しても視覚確認率は0のまま。"""
        cells = coverage_cells(20)
        for cell in cells:
            cell['rendered'] = {'up': True, 'down': True}
            cell['implementation'] = 'verified'
        result = summarize(cells, 20)['stages']
        self.assertEqual(result['rendered_both']['ratio'], 1)
        self.assertEqual(result['visual_pass_both']['ratio'], 0)
        record_visual_review(cells, 'validation', 0, 10, 'pass', ['actual_contact.png'])
        self.assertEqual(summarize(cells, 20)['stages']['visual_pass_both']['ratio'], 0)
        record_visual_review(cells, 'reference_audit', 0, 10, 'pass', ['actual_contact.png'])
        self.assertEqual(summarize(cells, 20)['stages']['visual_pass_both']['ratio'], .5)

    def test_gaps_are_rejected(self):
        """区間表の穴を完了率の分母から落とさない。"""
        cells = coverage_cells(20)
        cells[1]['start_m'] = 11
        with self.assertRaises(ValueError):
            summarize(cells, 20)

if __name__ == '__main__':
    unittest.main()

