"""No research code is imported or executed at module scope."""
import ast
import csv
import itertools
import json
import math
import statistics
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]

def evaluation_functions():
    tree = ast.parse((ROOT/'scripts/evaluate_final_holdout_repeat00_04.py').read_text(encoding='utf-8'))
    names = {'dice_score','get_surface','hd95_mm','exact_wilcoxon_two_sided','parse_filename','filename_at_z'}
    module = ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[])
    namespace = {'np':np,'binary_erosion':binary_erosion,'distance_transform_edt':distance_transform_edt,'itertools':itertools}
    import re
    namespace['re'] = re
    exec(compile(module,'public_evaluator_functions','exec'),namespace)
    return namespace

class ScientificChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.f = evaluation_functions()

    def test_dice_empty_and_overlap(self):
        empty = np.zeros((4,4,4),dtype=bool)
        first = empty.copy(); first[1,1,1] = True
        second = first.copy(); second[2,1,1] = True
        self.assertEqual(self.f['dice_score'](empty,empty),1)
        self.assertEqual(self.f['dice_score'](empty,first),0)
        self.assertAlmostEqual(self.f['dice_score'](first,second),2/3)

    def test_hd95_empty_and_physical_spacing(self):
        empty = np.zeros((5,5,5),dtype=bool)
        first = empty.copy(); first[1,1,1] = True
        second = empty.copy(); second[1,2,1] = True
        self.assertEqual(self.f['hd95_mm'](empty,empty,(1,2,3)),(0.0,'both_empty'))
        value,status = self.f['hd95_mm'](empty,first,(1,2,3))
        self.assertTrue(math.isnan(value)); self.assertEqual(status,'prediction_empty')
        value,status = self.f['hd95_mm'](first,empty,(1,2,3))
        self.assertTrue(math.isnan(value)); self.assertEqual(status,'ground_truth_empty')
        self.assertEqual(self.f['hd95_mm'](first,second,(1,2,3)),(2.0,'valid'))

    def test_paired_aggregate_and_wilcoxon(self):
        with (ROOT/'results/published_per_repeat.csv').open(newline='',encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        result = json.loads((ROOT/'results/published_aggregate.json').read_text())
        self.assertEqual(len(rows),5)
        for column,label,w,p,gain in [('positive3d','Positive 3D DSC',4,.4375,2.86),('positive2d','Positive 2D DSC',3,.3125,3.34)]:
            a = [float(row[f'single_{column}_dsc']) for row in rows]
            b = [float(row[f'cross_{column}_dsc']) for row in rows]
            calculated = self.f['exact_wilcoxon_two_sided'](a,b)
            self.assertEqual(calculated['W'],w)
            self.assertEqual(calculated['p_exact_two_sided'],p)
            self.assertAlmostEqual(statistics.mean(a),result['metrics'][label]['single_mean'])
            self.assertAlmostEqual(statistics.stdev(b),result['metrics'][label]['cross_sd_across_repeats'])
            self.assertEqual(round(100*statistics.mean([y-x for x,y in zip(a,b)]),2),gain)

    def test_protocol_and_metric_tradeoffs(self):
        spec = json.loads((ROOT/'configs/locked_protocol.json').read_text())
        self.assertEqual(spec['training_center_slices'],5)
        self.assertEqual(spec['validation_center_slices'],5)
        self.assertEqual(spec['encoder_adapter_depths'],[0,1,10,11])
        self.assertEqual(spec['inference_threshold'],'logit > 0')
        result = json.loads((ROOT/'results/published_aggregate.json').read_text())['metrics']
        self.assertGreater(result['FP negative slices']['paired_delta_mean'],0)
        self.assertGreater(result['HD95 (mm)']['paired_delta_mean'],0)
        self.assertLess(result['Missed positive slices']['paired_delta_mean'],0)
        self.assertLess(abs(result['Full DSC']['paired_delta_mean']),.001)

    def test_source_syntax_without_import_side_effects(self):
        for path in (ROOT/'scripts').glob('*.py'):
            compile(path.read_text(encoding='utf-8'),str(path),'exec')

    def test_all_documented_plans_are_read_only(self):
        before = (ROOT/'.runtime').exists()
        commands = [['preprocess'],['preprocess-final'],['prepare-evaluation'],['evaluate']]
        commands += [['train','--method',method,'--repeat',str(repeat)] for method in ('single','cross') for repeat in range(5)]
        for options in commands:
            proc = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),*options],cwd=ROOT,capture_output=True,text=True,check=True)
            self.assertFalse(json.loads(proc.stdout)['execute'])
        self.assertEqual((ROOT/'.runtime').exists(),before)

    def test_launcher_help_needs_no_gpu_dependencies(self):
        proc = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run.py'),'--help'],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertIn('--execute',proc.stdout)

if __name__=='__main__': unittest.main()
