import json
import unittest
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt_files():
    return sorted(PROMPTS_DIR.glob("*.json"))


def key_point_names(key_points):
    names = set()
    for point in key_points:
        if isinstance(point, str):
            names.add(point)
        elif isinstance(point, dict):
            names.update(
                value
                for value in (point.get("id"), point.get("construct"), point.get("label"))
                if isinstance(value, str) and value.strip()
            )
    return names


class PromptInventoryTests(unittest.TestCase):
    def test_prompt_files_are_valid_json(self):
        self.assertTrue(load_prompt_files(), "Expected at least one prompt JSON file")

        for prompt_file in load_prompt_files():
            with self.subTest(prompt=prompt_file.name):
                with prompt_file.open(encoding="utf-8") as handle:
                    json.load(handle)

    def test_prompt_ids_are_unique_and_required_fields_are_present(self):
        seen_ids = set()

        for prompt_file in load_prompt_files():
            with self.subTest(prompt=prompt_file.name):
                prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
                prompt_id = prompt.get("id")

                self.assertIsInstance(prompt_id, str)
                self.assertNotIn(prompt_id, seen_ids)
                seen_ids.add(prompt_id)

                for field in ("title", "description", "prompt_text", "expert_answers", "metadata"):
                    self.assertIn(field, prompt)
                self.assertTrue(prompt["prompt_text"].strip())
                self.assertIsInstance(prompt["expert_answers"], list)
                self.assertTrue(prompt["expert_answers"])

    def test_expert_answers_have_scoring_evidence(self):
        for prompt_file in load_prompt_files():
            prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
            with self.subTest(prompt=prompt_file.name):
                for answer in prompt["expert_answers"]:
                    self.assertTrue(answer.get("answer", "").strip())
                    self.assertIsInstance(answer.get("key_points"), list)
                    self.assertTrue(answer["key_points"])
                    self.assertIsInstance(answer.get("rubric"), dict)
                    self.assertTrue(answer["rubric"])

                    names = key_point_names(answer["key_points"])
                    self.assertTrue(names)
                    missing = set(answer["rubric"]) - names
                    self.assertFalse(
                        missing,
                        f"Rubric keys must match key point text, ids, labels, or constructs: {sorted(missing)}",
                    )

    def test_phase_1b_circuit_prompts_have_research_metadata(self):
        circuit_prompts = list(PROMPTS_DIR.glob("circuit_*.json"))
        self.assertTrue(circuit_prompts, "Expected Phase 1b circuit prompts to exist")
        allowed_domains = {"digital_logic", "dc_circuits"}

        for prompt_file in circuit_prompts:
            prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
            metadata = prompt.get("metadata", {})

            with self.subTest(prompt=prompt_file.name):
                self.assertEqual(metadata.get("phase"), "1b")
                self.assertIn(metadata.get("domain"), allowed_domains)
                self.assertEqual(metadata.get("type"), "descriptive_circuit_analysis")
                self.assertIsInstance(metadata.get("knowledge_components"), list)
                self.assertTrue(metadata["knowledge_components"])


if __name__ == "__main__":
    unittest.main()
