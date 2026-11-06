import json
import tempfile
import unittest
from pathlib import Path
from ood.generate import generate, noise_sample, distance_to_support, SORT_REGIONS, STACK_REGIONS, valid_separation

class OODTests(unittest.TestCase):
    def test_generation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            ma=generate(a,2,3,2);mb=generate(b,2,3,2)
            self.assertEqual(ma,mb)
            ids=set()
            for path in a.glob('*.jsonl'):
                self.assertEqual(path.read_bytes(),(b/path.name).read_bytes())
                for line in path.read_text().splitlines():
                    c=json.loads(line);self.assertNotIn(c['id'],ids);ids.add(c['id'])
                    self.assertFalse(c['validation']['rollout_ready']);self.assertTrue(valid_separation(c['blocks']))
                    changed=c['perturbation'].get('block_index')
                    for i,block in enumerate(c['blocks']):
                        regions=SORT_REGIONS if c['task'].startswith('sorting') else [STACK_REGIONS[i]]
                        d=distance_to_support(*block['xy'],regions)
                        self.assertAlmostEqual(d,c['perturbation']['distance_outside_support_m'] if i==changed else 0.)
                    self.assertEqual(noise_sample(c,0,3,'b','x'),noise_sample(c,0,3,'b','x'))
                    if c['noise']['sigma']:
                        self.assertNotEqual(noise_sample(c,0,3,'b','x'),noise_sample(c,0,4,'b','x'))
            with self.assertRaises(FileExistsError):generate(a)
    def test_union_not_single_rectangle(self):
        self.assertEqual(distance_to_support(.6,-.125,SORT_REGIONS),0)
        self.assertAlmostEqual(distance_to_support(.525,-.125,SORT_REGIONS),.025)

if __name__=='__main__':unittest.main()
