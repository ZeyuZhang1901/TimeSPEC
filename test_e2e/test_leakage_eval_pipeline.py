"""Stepwise tests for the LeakageEvaluationFramework pipeline."""

from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from LeakageEvaluationFramework.leakage_eval.config import ShapleyConfig, load_config
from LeakageEvaluationFramework.leakage_eval.evaluators.openrouter_client import OpenRouterConfig, OpenRouterTextClient
from LeakageEvaluationFramework.leakage_eval.io.loaders import build_eval_inputs, load_completion_rows
from LeakageEvaluationFramework.leakage_eval.io.metadata import (
    export_quality_gated_eval_metadata,
    prompt_safe_metadata,
    validate_metadata_for_completions,
)
from LeakageEvaluationFramework.leakage_eval.io.writers import ResultWriter
from LeakageEvaluationFramework.leakage_eval.metrics.aggregation import aggregate_instance_rows
from LeakageEvaluationFramework.leakage_eval.metrics.dclr import leakage_rate, shapley_dclr
from LeakageEvaluationFramework.leakage_eval.metrics.leakage import decision_labels
from LeakageEvaluationFramework.leakage_eval.metrics.performance import performance_score
from LeakageEvaluationFramework.leakage_eval.schemas import ClaimRecord, EvalInput, EvidenceItem
from LeakageEvaluationFramework.leakage_eval.shapley.coalition_sampler import (
    compute_budget,
    exact_coalitions,
    leverage_shap_coalitions,
)
from LeakageEvaluationFramework.leakage_eval.shapley.exact import exact_shapley
from LeakageEvaluationFramework.leakage_eval.shapley.leverage_regression import solve_projected_regression
from LeakageEvaluationFramework.leakage_eval.shapley.runner import ShapleyRunner
from LeakageEvaluationFramework.leakage_eval.shapley.value_function import reconstruction_value
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import local_claim_decision
from LeakageEvaluationFramework.leakage_eval.taxonomy.date_verifier import LlmDateVerifier
from LeakageEvaluationFramework.leakage_eval.pipeline import faithfulness_status


ROOT = Path(__file__).resolve().parent.parent
COMPLETIONS_PATH = (
    ROOT
    / "QualityGatedExperiment"
    / "runtime"
    / "results"
    / "qwen3_30b"
    / "eval_legal_timespec"
    / "completions.json"
)


class LengthBasedLegalPredictor:
    """Deterministic closed-world predictor for no-cost Shapley tests."""

    async def predict(self, instance: EvalInput, evidence_subset: list[EvidenceItem]) -> dict:
        probability = min(0.95, 0.2 + 0.1 * len(evidence_subset))
        return {"probability_petitioner": probability}


class FakeTaxonomyClient:
    async def call_text(self, prompt: str, system: str | None = None) -> str:
        return (
            '{"results":[{"id":1,"category":"C5","effective_source_date":"2024-01-01",'
            '"date_precision":"declared","is_leaked":true,"confidence":"high",'
            '"reason":"Fake verifier response for test."}]}'
        )


def _example_input() -> EvalInput:
    row = load_completion_rows(COMPLETIONS_PATH)[0]
    metadata = {
        row["instance_id"]: {
            "task": "legal",
            "question": "Predict whether the petitioner wins in Digital Realty Trust, Inc. v. Somers.",
            "cutoff_date": "2017-01-01",
            "ground_truth": {"outcome": "RESPONDENT"},
            "target_description": "Whether the petitioner wins.",
            "case_name": "Digital Realty Trust, Inc. v. Somers",
            "petitioner": "Digital Realty Trust, Inc.",
            "respondent": "Somers",
        }
    }
    return build_eval_inputs(
        [row],
        metadata,
        model="Qwen/Qwen3-30B-A3B",
        method="timespec",
    )[0]


class TestInputAdapter(unittest.TestCase):
    def test_real_timespec_completion_builds_eval_input(self) -> None:
        instance = _example_input()

        self.assertEqual(instance.instance_id, "eval_0001")
        self.assertEqual(instance.task, "legal")
        self.assertEqual(instance.model, "Qwen/Qwen3-30B-A3B")
        self.assertEqual(instance.method, "timespec")
        self.assertEqual(len(instance.evidence), 9)
        self.assertEqual(instance.evidence[0].id, 1)
        self.assertEqual(instance.evidence[0].source_date, "2016-05-12")
        self.assertEqual(instance.prediction, {"probability_petitioner": 0.3})

    def test_all_real_timespec_completions_are_parseable(self) -> None:
        rows = load_completion_rows(COMPLETIONS_PATH)
        metadata = {
            row["instance_id"]: {
                "task": row["task"],
                "question": f"Test question for {row['instance_id']}",
                "cutoff_date": "2025-01-01",
                "ground_truth": {"outcome": "RESPONDENT"},
                "case_name": f"Case {row['instance_id']}",
                "petitioner": "Petitioner",
                "respondent": "Respondent",
            }
            for row in rows
        }

        instances = build_eval_inputs(
            rows,
            metadata,
            model="Qwen/Qwen3-30B-A3B",
            method="timespec",
        )

        self.assertEqual(len(instances), 100)
        self.assertTrue(all(instance.evidence for instance in instances))
        self.assertTrue(all(instance.reasoning for instance in instances))
        self.assertTrue(all("probability_petitioner" in instance.prediction for instance in instances))

    def test_default_config_resolves_with_cli_overrides(self) -> None:
        config = load_config(
            ROOT / "LeakageEvaluationFramework" / "configs" / "leakage_eval_default.yaml",
            input_path=COMPLETIONS_PATH,
            dataset_path=ROOT / "synthetic_metadata.json",
            output_dir=ROOT / "LeakageEvaluationFramework" / "runtime" / "test_output",
        )

        self.assertEqual(config.input_path, COMPLETIONS_PATH)
        self.assertEqual(config.shapley.evaluator_model_policy, "same_as_generator")
        self.assertEqual(config.shapley.coalition_budget_c, 15.0)
        self.assertEqual(config.shapley.coalition_budget_cap, 1200)
        self.assertTrue(config.shapley.skip_if_no_leakage)

    def test_exported_metadata_validates_real_legal_completions(self) -> None:
        metadata = export_quality_gated_eval_metadata(ROOT / "dataset", tasks=["legal"])
        rows = load_completion_rows(COMPLETIONS_PATH)
        errors = validate_metadata_for_completions(rows, metadata)

        self.assertEqual(errors, [])
        row = metadata["eval_0001"]
        self.assertEqual(row["case_name"], "Digital Realty Trust, Inc. v. Somers")
        self.assertIn("ground_truth", row)
        self.assertNotIn("outcome", row["prompt_safe"])
        self.assertNotIn("conclusion", row["prompt_safe"])

    def test_prompt_safe_metadata_excludes_target_fields(self) -> None:
        safe = prompt_safe_metadata(
            {
                "case_name": "Example",
                "petitioner": "A",
                "respondent": "B",
                "outcome": "PETITIONER",
                "decision_date": "2024-01-01",
                "aav": 10,
                "ranking": ["A", "B"],
            }
        )

        self.assertEqual(safe["case_name"], "Example")
        self.assertNotIn("outcome", safe)
        self.assertNotIn("decision_date", safe)
        self.assertNotIn("aav", safe)
        self.assertNotIn("ranking", safe)


class TestLeakageStage(unittest.TestCase):
    def test_local_leakage_decisions_are_binary_and_aligned(self) -> None:
        instance = _example_input()
        decisions = [
            local_claim_decision(
                ClaimRecord(
                    instance_id=instance.instance_id,
                    claim_id=item.id,
                    claim_text=item.fact,
                    declared_source_date=item.source_date,
                    cutoff_date=instance.cutoff_date,
                    task=instance.task,
                    target_description=instance.target_description,
                    model=instance.model,
                    method=instance.method,
                )
            )
            for item in instance.evidence
        ]

        labels = decision_labels(decisions)
        self.assertEqual(len(labels), len(instance.evidence))
        self.assertTrue(all(isinstance(label, bool) for label in labels))
        self.assertEqual(sum(labels), 3)
        self.assertAlmostEqual(leakage_rate(labels), 3 / 9)

    def test_missing_date_for_time_verifiable_claim_is_conservative(self) -> None:
        decision = local_claim_decision(
            ClaimRecord(
                instance_id="x",
                claim_id=1,
                claim_text="The Supreme Court granted certiorari in the case.",
                declared_source_date=None,
                cutoff_date="2020-01-01",
                task="legal",
                target_description="",
                model="model",
                method="method",
            )
        )

        self.assertTrue(decision.is_leaked)
        self.assertEqual(decision.verification_method, "missing_date_conservative")

    def test_llm_date_verifier_maps_structured_response(self) -> None:
        verifier = LlmDateVerifier(FakeTaxonomyClient(), batch_size=10)  # type: ignore[arg-type]
        decisions = asyncio.run(
            verifier.verify(
                [
                    ClaimRecord(
                        instance_id="x",
                        claim_id=1,
                        claim_text="The court granted certiorari.",
                        declared_source_date="2024-01-01",
                        cutoff_date="2023-01-01",
                        task="legal",
                        target_description="",
                        model="model",
                        method="method",
                    )
                ]
            )
        )

        self.assertEqual(decisions[0].verification_method, "llm_taxonomy_date")
        self.assertEqual(decisions[0].category, "C5")
        self.assertTrue(decisions[0].is_leaked)


class TestShapleyStage(unittest.TestCase):
    def test_exact_shapley_recovers_additive_claim_values(self) -> None:
        weights = [0.2, 0.3, -0.1]
        masks = exact_coalitions(len(weights))
        values = [sum(weight for bit, weight in zip(mask, weights) if bit) for mask in masks]

        phi = exact_shapley(masks, values)

        self.assertEqual(len(phi), len(weights))
        for actual, expected in zip(phi, weights):
            self.assertAlmostEqual(actual, expected)

    def test_leverage_sampler_and_regression_preserve_efficiency(self) -> None:
        n_claims = 12
        budget = compute_budget(n_claims, c=1.5, cap=200, exact_threshold_n=3)
        masks = leverage_shap_coalitions(n_claims, budget, seed=7)
        self.assertIn(tuple([0] * n_claims), masks)
        self.assertIn(tuple([1] * n_claims), masks)
        self.assertLessEqual(len(masks), budget)

        values = [sum(mask) / n_claims for mask in masks]
        phi = solve_projected_regression(masks, values, v_full=1.0)

        self.assertEqual(len(phi), n_claims)
        self.assertAlmostEqual(float(phi.sum()), 1.0)

    def test_runner_computes_shapley_on_real_completion_subset(self) -> None:
        full = _example_input()
        instance = EvalInput(
            **{
                **asdict(full),
                "evidence": full.evidence[:3],
            }
        )
        runner = ShapleyRunner(
            ShapleyConfig(exact_threshold_n=3),
            LengthBasedLegalPredictor(),
        )

        result = asyncio.run(runner.run(instance))

        self.assertEqual(result.shapley_status, "computed")
        self.assertEqual(result.estimator, "exact")
        self.assertEqual(result.n_claims, 3)
        self.assertEqual(len(result.claim_shapley), 3)
        self.assertIsNotNone(result.full_reconstruction_gap)
        self.assertAlmostEqual(
            sum(item.shapley_value for item in result.claim_shapley),
            result.v_full or 0.0,
        )
        self.assertAlmostEqual(result.v_full or 0.0, 1.0 - (result.full_reconstruction_gap or 0.0))

    def test_reconstruction_value_is_normalized(self) -> None:
        self.assertAlmostEqual(
            reconstruction_value("legal", {"probability_petitioner": 0.7}, {"probability_petitioner": 0.5}),
            0.8,
        )
        self.assertAlmostEqual(
            reconstruction_value("salary", {"predicted_salary": 8.0}, {"predicted_salary": 10.0}),
            0.8,
        )
        self.assertAlmostEqual(
            reconstruction_value("stock", {"ranking": ["A", "B", "C"]}, {"ranking": ["A", "B", "C"]}),
            1.0,
        )


class TestMetricsAndOutputs(unittest.TestCase):
    def test_performance_dclr_aggregation_and_writers(self) -> None:
        instance = _example_input()
        perf = performance_score(instance.task, instance.prediction, instance.ground_truth)
        dclr = shapley_dclr([0.5, -0.25, 0.25], [True, False, True])
        summary = aggregate_instance_rows(
            [
                {
                    "olr": 1 / 3,
                    "shapley_dclr": dclr["shapley_dclr"],
                    "performance": perf,
                    "shapley_status": "computed",
                    "n_claims": 3,
                    "leaked_claims": 1,
                    "full_reconstruction_gap": 0.04,
                    "faithfulness_status": "high_faithfulness",
                    "m_total": 8,
                    "num_tinker_calls": 7,
                }
            ]
        )

        self.assertAlmostEqual(perf, 0.91)
        self.assertAlmostEqual(dclr["shapley_dclr"], 0.75)
        self.assertEqual(summary["n_instances"], 1)
        self.assertEqual(summary["shapley_computed"], 1)
        self.assertEqual(summary["faithfulness_counts"]["high_faithfulness"], 1)
        self.assertAlmostEqual(summary["faithfulness_rate_high_or_medium"], 1.0)
        self.assertEqual(summary["total_coalitions_evaluated"], 8)
        self.assertEqual(summary["total_tinker_calls"], 7)

        with tempfile.TemporaryDirectory() as tmp:
            writer = ResultWriter(Path(tmp))
            writer.write_summary(summary)
            writer.append_instance({"instance_id": instance.instance_id, "olr": 1 / 3})
            writer.append_low_faithfulness({"instance_id": instance.instance_id})
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "instance_results.jsonl").exists())
            self.assertTrue((Path(tmp) / "audit" / "low_faithfulness_instances.jsonl").exists())

    def test_faithfulness_status_thresholds(self) -> None:
        self.assertEqual(faithfulness_status(None), "not_applicable")
        self.assertEqual(faithfulness_status(0.05), "high_faithfulness")
        self.assertEqual(faithfulness_status(0.10), "medium_faithfulness")
        self.assertEqual(faithfulness_status(0.20), "low_faithfulness")


class TestApiBoundaries(unittest.TestCase):
    def test_env_keys_can_be_loaded_without_exposing_values(self) -> None:
        load_dotenv(ROOT / ".env", override=False)
        key_names = ("OPENROUTER_API_KEY", "TINKER_API_KEY", "PERPLEXITY_API_KEY")
        self.assertTrue(any(os.getenv(name) for name in key_names))

    def test_openrouter_client_can_be_constructed_without_calling_api(self) -> None:
        client = OpenRouterTextClient(
            OpenRouterConfig(model="google/gemini-3.1-flash-lite"),
            api_key="test-key",
        )

        self.assertEqual(client.config.model, "google/gemini-3.1-flash-lite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
