"""CPU-only fixed plan, duplicate and full video-row decoding guards."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from prepare_libero_source_pair import (
    PLAN_SHA256, decode_episode, duplicate_audit, matching_runs, read_json,
    transition_signatures, validate_plan,
)


class SourcePairTests(unittest.TestCase):
    def setUp(self):
        self.state = np.arange(80, dtype=np.float32).reshape(10,8)/100
        self.action = np.arange(70, dtype=np.float32).reshape(10,7)/100
        self.pixels = [hashlib.sha256(str(i).encode()).hexdigest()*2 for i in range(10)]

    def audit(self, state=None, action=None, pixels=None):
        return duplicate_audit((self.state, self.action, self.pixels),
                               (self.state+2 if state is None else state,
                                self.action/2 if action is None else action,
                                ["f"+h[1:] for h in self.pixels] if pixels is None else pixels))

    def test_fixed_plan_hash_and_fields(self):
        p = Path(__file__).resolve().parents[1]/"docs/libero-feature-pair-plan-v2.json"
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), PLAN_SHA256)
        self.assertEqual(validate_plan(read_json(p))["episodes"][0]["episode_index"],1400)

    def test_plan_cannot_switch_split_or_authorize_training(self):
        p = read_json(Path(__file__).resolve().parents[1]/"docs/libero-feature-pair-plan-v2.json")
        for key, value in [("training_authorized",True),("horizon",4),("orientation","flipped")]:
            changed=copy.deepcopy(p);changed[key]=value
            with self.assertRaises(ValueError):validate_plan(changed)
        p["episodes"][0]["partition"]="validation"
        with self.assertRaises(ValueError):validate_plan(p)

    def test_distinct_passes_without_independence_claim(self):
        result=self.audit()
        self.assertEqual(result["status"],"passed")
        self.assertFalse(result["family_independence_verified"])

    def test_previous_wrong_task_plan_rejected(self):
        p = read_json(Path(__file__).resolve().parents[1]/"docs/libero-feature-pair-plan-v1.json")
        with self.assertRaises(ValueError):validate_plan(p)

    def test_behavior_duplicates_despite_different_pixels(self):
        self.assertEqual(self.audit(self.state.copy(),self.action.copy())["status"],"rejected")

    def test_pixel_duplicates_despite_different_behavior(self):
        self.assertEqual(self.audit(pixels=self.pixels.copy())["status"],"rejected")

    def test_quantized_near_duplicate_guard(self):
        report=self.audit(self.state+1e-7,self.action+1e-7)
        self.assertEqual(report["cross_split_matching_windows"]["exact_transition_windows"],0)
        self.assertGreater(report["cross_split_matching_windows"]["quantized_transition_windows"],0)
        self.assertEqual(report["status"],"rejected")

    def test_final_unpaired_action_cannot_hide_duplicate(self):
        changed=self.action.copy();changed[-1]=-.9
        one=transition_signatures(self.state[-7:],self.action[-7:])
        two=transition_signatures(self.state[-7:],changed[-7:])
        self.assertEqual(one,two)

    def test_connecting_action_difference_is_visible(self):
        changed=self.action.copy();changed[-2]=-.9
        self.assertNotEqual(transition_signatures(self.state[-7:],self.action[-7:]),
                            transition_signatures(self.state[-7:],changed[-7:]))

    def test_shared_suffix_rejected(self):
        state=np.concatenate((self.state[:3]+10,self.state[3:]))
        action=np.concatenate((self.action[:3]/2,self.action[3:]))
        report=self.audit(state,action)
        self.assertEqual(report["status"],"rejected")
        self.assertEqual(report["state_runs"]["common_suffix"],7)

    def test_nonfinite_and_short_shapes_rejected(self):
        state=self.state.copy();state[0,0]=np.nan
        with self.assertRaises(ValueError):transition_signatures(state,self.action)
        with self.assertRaises(ValueError):transition_signatures(self.state[:6],self.action[:6])

    def test_hash_row_count_mismatch(self):
        with self.assertRaises(ValueError):self.audit(pixels=self.pixels[:-1])

    def test_matching_runs(self):
        self.assertEqual(matching_runs([1,2,3,4],[1,8,2,3,4]),
                         {"longest_exact_contiguous_match":3,"common_prefix":1,"common_suffix":3})


class DecodeTests(unittest.TestCase):
    def decode(self, pts):
        class Frame:
            def __init__(self,n):self.pts=n
            def to_ndarray(self,format):return np.full((256,256,3),self.pts,dtype=np.uint8)
        class Container:
            streams=SimpleNamespace(video=[SimpleNamespace(width=256,height=256,average_rate=10,time_base=.1)])
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def decode(self,stream):return iter(Frame(n) for n in pts)
        md={f"videos/{key}/{bound}":val for key in ("observation.images.image","observation.images.image2")
            for bound,val in (("from_timestamp",1.),("to_timestamp",1.8))}
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{"av":SimpleNamespace(open=lambda path:Container())}):
            hashes=decode_episode([Path("v0"),Path("v1")],md,8,10.,Path(temp))
            array=np.load(Path(temp)/"images.npy")
            self.assertEqual(array.shape,(8,2,256,256,3))
            self.assertEqual(int(array[0,0,0,0,0]),10)
            self.assertEqual(int(array[-1,1,0,0,0]),17)
            self.assertEqual(len(hashes),8)

    def test_complete_decode_uses_offset_and_all_frames(self):self.decode(list(range(20)))
    def test_missing_frame_rejected(self):
        with self.assertRaises(ValueError):self.decode([n for n in range(20) if n!=13])
    def test_duplicate_frame_rejected(self):
        with self.assertRaises(ValueError):self.decode(list(range(13))+[12]+list(range(13,20)))


if __name__=="__main__":unittest.main()
