from pydantic import BaseModel, Field

# --- Pydantic Model for Structured Flaw Evaluation ---
# This class defines the JSON structure we expect from the evaluation LLM.

class FlawEvaluation(BaseModel):
    """Represents the evaluation of a single generated review against a single planted flaw."""
    flaw_id: str = Field(
        description="The unique identifier for the flaw being evaluated (e.g., 'missing_state_encoder_description')."
    )
    is_flaw_mentioned: bool = Field(
        description="True if the review mentions, discusses, or alludes to the specific flaw described in the ground truth. False otherwise."
    )
    mention_reasoning: str = Field(
        description="Provide a brief justification for your decision. If the flaw was mentioned, quote the relevant sentences from the review. If not, state that it was absent."
    )
    is_reasoning_correct: bool = Field(
        description="True if the review not only mentions the flaw but also correctly explains *why* it is a flaw, with reasoning that aligns with the ground truth description. False if the reasoning is missing, incorrect, or superficial."
    )
    reasoning_analysis: str = Field(
        description="Analyze the depth and accuracy of the review's reasoning. Compare it to the ground truth. For example, did the reviewer just note the omission, or did they explain its negative impact on reproducibility and scope, as detailed in the ground truth?"
    )

# --- Prompt Generation for Evaluation Task ---

class EvaluationPrompts:
    """Manages and generates the prompts for the review evaluation task."""

    EVALUATION_PERSONA = """
    You are a meticulous meta-reviewer. Your task is to evaluate an AI-generated academic review against a known, "ground truth" description of a flaw that was intentionally planted in a research paper. You must be objective and precise.
    """

    @staticmethod
    def get_evaluation_prompt(review_text: str, flaw_id: str, flaw_description: str, json_schema: dict) -> str:
        """
        Constructs the full prompt for the evaluation model.

        Args:
            review_text: The full text of the generated review.
            flaw_id: The identifier of the flaw.
            flaw_description: The ground truth description of the flaw.
            json_schema: The Pydantic model's JSON schema for the output.

        Returns:
            A complete prompt string for the evaluation task.
        """
        return f"""
You are a meticulous meta-reviewer. Your task is to evaluate an AI-generated academic review against a known, "ground truth" description of a flaw that was intentionally planted in a research paper.

**Your Goal:**
Assess whether the generated review successfully identified and correctly reasoned about the planted flaw.

**Ground Truth Flaw Description:**
- **Flaw ID:** `{flaw_id}`
- **Description:** "{flaw_description}"

---

**Generated Review to Evaluate:**
```text
{review_text}
```

---

**Evaluation Instructions:**
Based on the ground truth and the provided review, answer the following questions. Your response MUST be a single, valid JSON object that conforms exactly to the schema below.

1.  **`is_flaw_mentioned`**: Did the review mention this specific flaw in any capacity? Look for direct mentions or clear allusions.
2.  **`is_reasoning_correct`**: If the flaw was mentioned, did the review's reasoning for *why* it is a flaw align with the ground truth description? For example, did it just state something was missing, or did it correctly identify the negative implications (e.g., on reproducibility, scope, etc.)?

**Required JSON Schema:**
```json
{json_schema}
```

Now, provide your complete evaluation as a single JSON object.
"""
