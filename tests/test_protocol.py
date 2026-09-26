import json
from pathlib import Path
import tempfile
import unittest
from engintervene.protocol import rubric_score, t1_score, valid_judgment, prompt, target_answer, emitted_answer
from engintervene.io import shard, start_output, write_rows
from engintervene.data import normalize
from engintervene.inference import visual_mapping
from engintervene.summarize import aggregate
from engintervene.prepare_sft import prepare
from engintervene.io import write_json, read_rows


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.item = {'item_id': 'example', 'task_id': 'T1', 'domain': 'toy',
                     'input': {'question_benchmark': 'Select the marked option.',
                               'choices': [{'choice_id': 'A', 'text': 'one'}, {'choice_id': 'B', 'text': 'two'}],
                               'response_contract': ['Return the option ID.'], 'asset_ids': []}}
        self.rubric = {'criteria': [{'criterion_id': 'C1', 'description': 'goal', 'type': 'required', 'weight': 2},
                                   {'criterion_id': 'C2', 'description': 'constraint', 'type': 'required', 'weight': 1},
                                   {'criterion_id': 'E1', 'description': 'error', 'type': 'critical_error', 'weight': -2}],
                       'maximum_score': 3, 'strict_success_required': ['C1', 'C2'], 'strict_success_forbidden': ['E1']}

    def judgments(self, a, b, e):
        return [{'criterion_id': key, 'raw': json.dumps({'criterion_id': key, 'met': yes,
                'evidence': word if yes else ''})} for key, yes, word in [('C1', a, 'goal'), ('C2', b, 'safe'), ('E1', e, 'unsafe')]]

    def test_atomic_partial_strict_false(self):
        r = rubric_score('goal only', self.rubric, self.judgments(True, False, False))
        self.assertAlmostEqual(r['score_fraction'], 2 / 3)
        self.assertFalse(r['strict_success'])

    def test_strict_all(self):
        r = rubric_score('goal safe', self.rubric, self.judgments(True, True, False))
        self.assertEqual(r['score_fraction'], 1)
        self.assertTrue(r['strict_success'])

    def test_penalty_and_clipping(self):
        r = rubric_score('goal safe unsafe', self.rubric, self.judgments(True, True, True))
        self.assertAlmostEqual(r['score_fraction'], 1 / 3)
        self.assertFalse(r['strict_success'])
        r = rubric_score('unsafe', self.rubric, self.judgments(False, False, True))
        self.assertEqual(r['score_fraction'], 0)

    def test_evidence_must_be_candidate_span(self):
        raw = json.dumps({'criterion_id': 'C1', 'met': True, 'evidence': 'absent'})
        self.assertFalse(valid_judgment(raw, 'C1', 'goal safe'))
        raw = json.dumps({'criterion_id': 'C1', 'met': False, 'evidence': 'goal'})
        self.assertFalse(valid_judgment(raw, 'C1', 'goal safe'))

    def test_t1_unicode(self):
        label = {'gold_structure': {'correct_choice_id': 'A'}}
        score = t1_score(self.item, label, '{"choice_ids":["Ａ"]}')
        self.assertEqual(score['score_fraction'], 1)
        self.assertEqual(t1_score(self.item, label, '')['score_fraction'], 0)

    def test_inputs_exclude_private_fields(self):
        row = {**self.item, 'reference_answer': 'secret', 'rubric': self.rubric}
        self.assertNotIn('reference_answer', normalize(row))
        self.assertNotIn('rubric', normalize(row))
        self.assertNotIn('secret', prompt(normalize(row)))

    def test_targets_reference_not_generated(self):
        row = {**self.item, 'task_id': 'T4'}
        self.assertEqual(json.loads(target_answer(row, {'benchmark_reference_answer': 'fixed answer'})), {'answer': 'fixed answer'})

    def test_shards_disjoint_complete(self):
        values = list(range(11))
        pieces = [shard(values, i, 3) for i in range(3)]
        self.assertEqual(sorted(sum(pieces, [])), values)
        with self.assertRaises(ValueError):
            shard(values, 3, 3)

    def test_output_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'out.jsonl'
            self.assertEqual(start_output(p, {'run': 1}), [])
            write_rows(p, [{'item_id': 'x'}])
            self.assertEqual(len(start_output(p, {'run': 1})), 1)
            with self.assertRaises(RuntimeError):
                start_output(p, {'run': 2})

    def test_truncated_text_not_replaced_with_empty(self):
        self.assertEqual(emitted_answer('partial reasoning', 'raw')[0], 'partial reasoning')

    def test_visual_modes_do_not_change_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d) / f'{i}.png' for i in range(2)]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]))
            rows = [{'item_id': str(i), 'domain': 'toy', 'task_id': 'T4', 'images': [str(path)]}
                    for i, path in enumerate(paths)]
            original = json.dumps(rows)
            self.assertEqual(visual_mapping(rows, 'text-only'), {'0': [], '1': []})
            mapping = visual_mapping(rows, 'shuffled-image')
            self.assertEqual(mapping['0'], rows[1]['images'])
            self.assertEqual(json.dumps(rows), original)
            with self.assertRaises(ValueError):
                visual_mapping(rows[:1], 'shuffled-image')

    def test_summary_rejects_missing_and_reports_na(self):
        score = {'item_id': 'example', 'task_id': 'T1', 'domain': 'toy',
                 'score': {'score_fraction': 1, 'strict_success': True, 'score_valid': True}}
        r = aggregate([self.item], [score], set())
        self.assertIsNone(r['by_task']['T4']['mean_score'])
        self.assertIsNone(r['task_macro_average'])
        with self.assertRaises(ValueError):
            aggregate([self.item], [], set())

    def test_frozen_split_and_schedule(self):
        root = Path(__file__).resolve().parents[1]
        members = json.loads((root / 'splits/membership.json').read_text())
        self.assertEqual(len(members), 3229)
        from collections import Counter
        self.assertEqual(Counter(v['split'] for v in members.values()), {'train': 1615, 'validation': 308, 'test': 1306})
        groups = {}
        for value in members.values():
            self.assertEqual(groups.setdefault(value['group_id'], value['split']), value['split'])
        self.assertEqual(len(groups), 227)
        schedule = json.loads((root / 'splits/sampling_schedule.json').read_text())
        for epoch in schedule['epochs']:
            self.assertEqual(len(epoch), 1615)
            self.assertTrue(all(members[i]['split'] == 'train' for i in epoch))
        challenge = json.loads((root / 'splits/challenge_test_ids.json').read_text())
        self.assertEqual(len(challenge), 249)
        self.assertTrue(all(members[i]['split'] == 'test' for i in challenge))

    def test_export_order_and_test_target_isolation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            members = {iid: {'split': split, 'group_id': gid} for iid, split, gid in
                       [('b', 'train', 'group-b'), ('c', 'train', 'group-a'),
                        ('a', 'train', 'group-a'), ('v', 'validation', 'group-v'), ('z', 'test', 'group-z')]}
            items, labels = [], []
            for iid in members:
                image = root / f'{iid}.png'
                image.write_bytes(iid.encode())
                items.append({**self.item, 'item_id': iid, 'images': [str(image)]})
                labels.append({'item_id': iid, 'gold_structure': {'correct_choice_id': 'A'}})
            write_rows(root / 'items.jsonl', items)
            write_rows(root / 'labels.jsonl', labels)
            write_json(root / 'membership.json', members)
            write_json(root / 'schedule.json', {'epochs': [['a', 'b', 'c'], ['b', 'a', 'c']]})
            prepare(root / 'items.jsonl', root / 'labels.jsonl', root / 'membership.json',
                    root / 'schedule.json', root / 'export', 'test system')
            self.assertEqual([r['sample_id'] for r in read_rows(root / 'export/train.jsonl')], ['a', 'c', 'b'])
            self.assertFalse((root / 'export/test.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
