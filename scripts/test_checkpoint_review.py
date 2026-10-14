"""Regression tests for checkpoint monitoring decisions and numeric evaluation."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from review_stage2_checkpoints import training_review,reference_metrics,decision,read_complete_jsonl

def rows(means):
    return [{'step':i+1,'loss':v,'lr':1e-4,'grad_norm':1.,'seconds':100.}
            for i,v in enumerate([value for value in means for _ in range(500)])]

def record(step,ce):
    return {'checkpoint':{'step':step},'reference':{'ce':ce,'top1_accuracy':.2}}

class MonitorTests(unittest.TestCase):
    def test_truncated_tail_and_boundary(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs') as d:
            p=Path(d)/'m.jsonl';p.write_bytes(b'{"step":1}\n{"step":2')
            self.assertEqual(read_complete_jsonl(p),[{'step':1}])
        data=rows([8.,7.,6.])
        data[1200]['loss']=float('nan')
        self.assertEqual(training_review(data,1000)['evaluated_through_step'],1000)
        with self.assertRaises(ValueError):training_review(data,1500)
    def test_missing_or_duplicate_metrics_fail_closed(self):
        data=rows([8.,7.]);del data[0]
        with self.assertRaises(ValueError):training_review(data,1000)
        data=rows([8.,7.]);data[2]['step']=2
        with self.assertRaises(ValueError):training_review(data,1000)
    def test_confirmed_deterioration_and_single_spike(self):
        train=training_review(rows([6.,6.2,6.4]),1500)
        d=decision(train,{'ce':6.4,'top1_accuracy':.1},[record(500,6.),record(1000,6.2)],1500)
        self.assertEqual(d['action'],'review_before_continuing')
        self.assertFalse(d['auto_stop'])
        train=training_review(rows([8.,7.,7.2]),1500)
        d=decision(train,{'ce':7.2,'top1_accuracy':.1},[record(500,8.),record(1000,7.)],1500)
        self.assertEqual(d['review_flags'],[])
    def test_plateau_and_gaps(self):
        train=training_review(rows([6.,6.0001,6.0002]),1500)
        d=decision(train,{'ce':6.0002,'top1_accuracy':.1},[record(500,6.),record(1000,6.0001)],1500)
        self.assertIn('training_loss_plateau',d['review_flags'])
        self.assertIn('fixed_train_reference_plateau',d['review_flags'])
        d=decision(train,{'ce':6.0002,'top1_accuracy':.1},[record(500,6.),record(1500,6.0001)],2500)
        self.assertFalse(d['reference_confirmation_available'])
    def test_numeric_reference_metrics(self):
        torch.set_num_threads(1)
        logits=torch.zeros(1,5,4096);target=torch.zeros(1,5,dtype=torch.int64)
        value=reference_metrics(logits,target,[1,2])
        self.assertAlmostEqual(value['ce'],float(torch.tensor(4096.).log()),places=5)
        self.assertAlmostEqual(value['mean_predictive_entropy'],value['ce'],places=5)
        self.assertEqual(value['top1_accuracy'],1.)
        self.assertEqual([r['tokens'] for r in value['per_scale']],[1,4])
        logits[0,0,0]=float('nan')
        with self.assertRaises(ValueError):reference_metrics(logits,target,[1,2])

if __name__=='__main__':unittest.main()
