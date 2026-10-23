import unittest
import numpy as np
from visualizer.analytics import enrich

class AnalyticsTest(unittest.TestCase):
    def rollout(self, boxes, rule='any_permutation', xyz=None):
        r=dict(key='c000_t0',steps=2,dt=.1,c=[[0,0,0],[.1,0,0]],des=[[.01,0],[.11,0]],boxes=[boxes,boxes],
               act=[[.01,0],[.03,0]],success=False,goal_spec=dict(goals=[[0,0,0],[1,0,0]],threshold=.05,rule=rule,description='test'))
        if xyz is not None:r['boxes_xyz']=[xyz,xyz]
        return r
    def test_crossed_goals_valid(self):
        r=enrich(self.rollout([1,0,0,0,0,0]))
        self.assertEqual(r['series']['reached_count'],[2,2])
        self.assertEqual(r['series']['goal_assignment'][0],[1,0])
    def test_two_blocks_same_goal_not_complete(self):
        r=enrich(self.rollout([0,0,0,0,0,0]))
        self.assertEqual(r['series']['reached_count'],[1,1])
        self.assertEqual(r['series']['mean_distance_mm'],[0,0]) # Official nearest-goal mean is not success.
    def test_inserting_fixed_assignment(self):
        r=enrich(self.rollout([1,0,0,0,0,0],rule='matched'))
        self.assertEqual(r['series']['reached_count'],[0,0])
    def test_z_distance_not_silently_ignored(self):
        r=enrich(self.rollout([0,0,0,1,0,0],xyz=[0,0,.1,1,0,.1]))
        self.assertEqual(r['distance_kind'],'3D')
        self.assertEqual(r['series']['goal_distances_mm'],[[100,100],[100,100]])
    def test_units_and_missing_transitions(self):
        r=enrich(self.rollout([0,0,0,1,0,0]))
        self.assertAlmostEqual(r['series']['tracking_mm'][0],10)
        self.assertEqual(r['series']['speed_m_s'],[None,1])
        self.assertIsNone(r['series']['action_delta_mm'][0])
        self.assertAlmostEqual(r['series']['action_delta_mm'][1],20)
        self.assertEqual(r['series']['joint_speed_rad_s'],[None,None])
    def test_unknown_task_has_no_invented_goals(self):
        r=self.rollout([0,0,0,1,0,0]);r.pop('goal_spec');r.pop('act');enrich(r)
        self.assertIsNone(r['goal_spec']);self.assertEqual(r['series']['mean_distance_mm'],[None,None])
        self.assertEqual(r['series']['action_mm'],[None,None])
