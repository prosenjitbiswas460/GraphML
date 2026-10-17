"""Checks for the exposure mask, the metrics, and a short training run."""

from __future__ import annotations

import unittest

import numpy as np

from esb.data import make_synthetic, prepare
from esb.metrics import evaluate_ranking, linear_cka, mean_jaccard
from esb.sampling import (
    apply_exposure,
    edge_keys,
    exposure_buckets,
    inverse_propensity_keep_probs,
    ips_weights,
    item_propensity,
    resample_by_inverse_propensity,
    split_by_user,
)
from run_experiment import parser


def config(extra=None):
    args = [
        "--dataset",
        "synthetic",
        "--epochs",
        "2",
        "--patience",
        "5",
        "--dim",
        "16",
        "--layers",
        "1",
        "--batch-size",
        "512",
        "--device",
        "cpu",
        "--results-dir",
        "results/test",
        "--no-resampled",
    ]
    if extra:
        args.extend(extra)
    cfg = parser().parse_args(args)
    cfg.ks = tuple(int(part) for part in cfg.ks.split(","))
    return cfg


class SamplingTests(unittest.TestCase):
    def test_split_is_a_partition(self):
        rng = np.random.default_rng(0)
        edges = np.array([[user, item] for user in range(20) for item in range(8)], dtype=np.int64)
        train, val, test = split_by_user(edges, 0.1, 0.2, rng, min_train=2)
        keys = lambda rows: {tuple(row) for row in rows.tolist()}
        self.assertEqual(keys(train) | keys(val) | keys(test), keys(edges))
        self.assertEqual(len(keys(train) & keys(val)), 0)
        self.assertEqual(len(keys(train) & keys(test)), 0)
        self.assertGreater(len(test), 0)

    def test_exposure_keeps_head_items(self):
        rng = np.random.default_rng(1)
        edges = np.column_stack(
            (np.arange(400), np.array([0] * 200 + [1] * 200, dtype=np.int64))
        ).astype(np.int64)
        propensity = np.array([1.0, 0.1])
        kept, kept_p, _ = apply_exposure(edges, propensity, rng, n_items=2, min_keep=0)
        head = kept[:, 1] == 0
        self.assertGreater(head.mean(), 0.75)
        self.assertTrue(np.all(kept_p[head] == 1.0))

    def test_resample_reduces_head_share(self):
        rng = np.random.default_rng(2)
        popular = np.column_stack((np.arange(270), np.zeros(270, dtype=np.int64)))
        rare = np.column_stack((np.arange(270, 300), np.ones(30, dtype=np.int64)))
        edges = np.concatenate([popular, rare]).astype(np.int64)
        propensity = np.array([1.0, 0.1])
        edge_p = propensity[edges[:, 1]]
        resampled = resample_by_inverse_propensity(
            edges, edge_p, rng, n_items=2, target_mean=0.5, min_keep=0
        )
        raw_rare = float((edges[:, 1] == 1).mean())
        new_rare = float((resampled[:, 1] == 1).mean())
        self.assertGreater(new_rare, raw_rare + 0.05)

    def test_ips_upweights_rare_items(self):
        weights = ips_weights(np.array([1.0, 0.1]), clip=10)
        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(item_propensity(np.array([10.0, 1.0]), 1.0, 0.1)[1]), 0.1)
        keep = inverse_propensity_keep_probs(np.array([1.0, 1.0, 1.0]), 0.4)
        self.assertTrue(np.allclose(keep, 0.4, atol=0.05))


class MetricTests(unittest.TestCase):
    def test_ndcg_known_ranking(self):
        user = np.array([[1.0, 0.0]], dtype=np.float32)
        items = np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.2, 0.0], [0.0, 0.2]],
            dtype=np.float32,
        )
        ban = np.zeros((1, 4), dtype=bool)
        relevant = np.zeros((1, 4), dtype=bool)
        relevant[0, 2] = True
        buckets = np.array([0, 0, 0, 0], dtype=np.int8)
        metrics = evaluate_ranking(user, items, ban, relevant, buckets, ks=(1, 2), device="cpu")
        self.assertAlmostEqual(metrics["ndcg@1"], 0.0)
        self.assertAlmostEqual(metrics["ndcg@2"], 1.0 / np.log2(3.0))
        self.assertAlmostEqual(metrics["recall@2"], 1.0)

    def test_cka_is_one_for_a_scaled_copy(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(40, 6))
        self.assertAlmostEqual(linear_cka(matrix, 3.0 * matrix), 1.0, places=6)

    def test_interaction_neighbors_are_real_cooccurrence(self):
        from esb.metrics import interaction_knn

        edges = np.array(
            [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]],
            dtype=np.int64,
        )
        neighbors, valid = interaction_knn(edges, n_users=4, n_items=4, k=3, min_degree=1)
        self.assertTrue(valid[0])
        self.assertEqual(set(neighbors[0][neighbors[0] >= 0].tolist()), {1})

    def test_jaccard_of_identical_neighbors(self):
        neighbors = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        valid = np.array([True, True, True])
        self.assertAlmostEqual(mean_jaccard(neighbors, neighbors, valid), 1.0)

    def test_buckets_leave_items_outside_the_training_split(self):
        counts = np.array([0, 0, 1, 2, 3, 4, 5, 6])
        present = counts > 0
        buckets = exposure_buckets(counts, present)
        self.assertTrue(np.all(buckets[:2] == -1))
        self.assertEqual(buckets[2], 0)
        self.assertEqual(buckets[-1], 2)
        self.assertEqual(set(buckets[2:].tolist()), {0, 1, 2})


class ExperimentTests(unittest.TestCase):
    def test_synthetic_exposure_moves_edges_onto_the_head(self):
        split = prepare(config(["--dry-run"]), seed=0)
        shares = split.meta["head_edge_share"]
        self.assertGreater(shares["raw"], shares["counterfactual"] + 0.05)
        self.assertLess(shares["resampled"], shares["raw"] - 0.05)
        self.assertGreater(split.meta["n_test_edges"], 0)
        self.assertIsNotNone(split.uniform_edges)
        self.assertEqual(split.meta["n_forced_keeps"], 0)
        self.assertFalse(split.meta["test_labels_in_training_loss"])
        n_items = split.n_items
        test_keys = set(edge_keys(split.test_edges, n_items).tolist())
        val_keys = set(edge_keys(split.val_edges, n_items).tolist())
        self.assertTrue(test_keys.isdisjoint(val_keys))
        for edges in (
            split.raw_edges,
            split.uniform_edges,
            split.resampled_edges,
            split.counterfactual_edges,
        ):
            keys = set(edge_keys(edges, n_items).tolist())
            self.assertTrue(keys.isdisjoint(test_keys))
            self.assertTrue(keys.isdisjoint(val_keys))
        train_items = set(split.counterfactual_edges[:, 1].tolist())
        self.assertTrue(set(split.test_edges[:, 1].tolist()).issubset(train_items))
        self.assertGreater(split.meta["n_shift_edges"], 0)
        shift_keys = set(edge_keys(split.shift_edges, n_items).tolist())
        self.assertTrue(shift_keys.issubset(test_keys))
        self.assertTrue(shift_keys.isdisjoint(set(edge_keys(split.raw_edges, n_items).tolist())))

    def test_alignment_moves_the_exposure_head_into_one_community(self):
        _, _, independent = make_synthetic(np.random.default_rng(0), align=0.0)
        _, _, aligned = make_synthetic(np.random.default_rng(1), align=1.0)
        self.assertLess(independent["head_in_aligned_community"], 0.85)
        self.assertEqual(aligned["head_in_aligned_community"], 1.0)

    def test_frontier_is_the_first_alpha_where_leakage_flips(self):
        from esb.phase import summarize_phase

        def row(alpha, raw, uniform):
            return {
                "alpha": alpha,
                "exposure_align": 0.0,
                "layers": 3,
                "seed": 0,
                "head_edge_share": {"raw": 0.8, "uniform_matched": 0.2},
                "leakage": {"raw": raw, "uniform_matched": uniform, "ips_graph": raw - 0.01},
                "ndcg": {"raw": 0.1, "uniform_matched": 0.11},
                "shift_ndcg": {"raw": 0.05, "uniform_matched": 0.08},
                "probe_gaps": {
                    "1": {"raw_minus_uniform_matched": 0.02, "raw_minus_ips_graph": 0.01},
                    "3": {"raw_minus_uniform_matched": 0.08, "raw_minus_ips_graph": 0.015},
                },
                "selector": {
                    "spearman_val_vs_shift": 0.1,
                    "spearman_leakage_vs_shift": -0.8,
                    "leakage_ranks_shift_better_than_val": True,
                },
            }

        summary = summarize_phase([row(0.25, -0.02, -0.03), row(1.0, 0.05, -0.01)])
        self.assertEqual(summary["frontier"][0]["alpha_when_neighborhood_flips"], 1.0)
        self.assertTrue(summary["propagation"][1]["support_gap_grew"])
        self.assertTrue(summary["propagation"][1]["ips_gap_grew_less_than_support"])
        self.assertEqual(summary["selector"]["n_cells_leakage_ranks_shift"], 2)

    def test_short_training_returns_finite_metrics(self):
        from esb.experiment import run_one

        result = run_one(config(), seed=0, device="cpu")
        self.assertIn("raw", result["variants"])
        self.assertIn("ips_graph", result["variants"])
        self.assertIn("ips_loss", result["variants"])
        self.assertNotIn("ips", result["variants"])
        self.assertIn("counterfactual", result["variants"])
        for name, record in result["variants"].items():
            self.assertTrue(np.isfinite(record["test"]["ndcg@20"]), name)
            self.assertTrue(np.isfinite(record["structure"]["leakage"]), name)
        self.assertIn("leakage_raw_minus_counterfactual", result["contrasts"])
        self.assertIn("spearman_leakage_vs_shift", result["selector"])
        for record in result["variants"].values():
            self.assertTrue(np.isfinite(record["shift"]["ndcg@20"]))
        self.assertIn("3", result["propagation_probe"]["gaps"])


if __name__ == "__main__":
    unittest.main()
