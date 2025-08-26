from pydantic import BaseModel, Field
from typing import Literal

# --- Pydantic Model for Structured NeurIPS Review ---
# This class defines the JSON structure we expect from the AI model.
# It ensures the response is parsed correctly and contains all required fields.

class NeurIPSReview(BaseModel):
    """The root model for a single, complete NeurIPS-style paper review."""
    summary: str = Field(
        description="A brief, neutral summary of the paper and its contributions. This should not be a critique or a copy of the abstract."
    )
    strengths_and_weaknesses: str = Field(
        description="A thorough assessment of the paper's strengths and weaknesses, touching on originality, quality, clarity, and significance. Use Markdown for formatting."
    )
    questions: str = Field(
        description="A list of actionable questions and suggestions for the authors (ideally 3-5 key points). Frame questions to clarify points that could change the evaluation."
    )
    limitations_and_societal_impact: str = Field(
        description="Assessment of whether limitations and potential negative societal impacts are adequately addressed. State 'Yes' if adequate; otherwise, provide constructive suggestions for improvement."
    )
    soundness: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the soundness of the technical claims, methodology, and whether claims are supported by evidence (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    presentation: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the quality of the presentation, including writing style, clarity, and contextualization (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    contribution: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the quality of the overall contribution, including the importance of the questions asked and the value of the results (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    overall_score: Literal[10, 9, 8, 7, 6, 5, 4, 3, 2, 1] = Field(
        description="Overall recommendation score (10: Award quality, 8: Strong Accept, 6: Weak Accept, 5: Borderline accept, 4: Borderline reject, 2: Strong Reject)."
    )
    confidence: Literal[5, 4, 3, 2, 1] = Field(
        description="Confidence in the assessment (5: Certain, 4: Confident, 3: Fairly confident, 2: Willing to defend, 1: Educated guess)."
    )

# --- Comprehensive Reviewer Persona and Prompt Generation ---

class ReviewerPrompts:
    """
    Manages and generates the prompt for a comprehensive, multi-faceted reviewer.
    """

    # This new persona combines multiple critical perspectives into one.
    DEFAULT_REVIEWER_PERSONA = """
    You are a top-tier academic reviewer for NeurIPS, known for writing exceptionally thorough, incisive, and constructive critiques. Your goal is to synthesize multiple expert perspectives into a single, coherent review that elevates the entire research field.

    When reviewing the paper, you must adopt a multi-faceted approach, simultaneously analyzing the work from the following critical angles:

    1.  **The Conceptual Critic & Historian**:
        * **Question the Core Concepts**: Do not accept the authors' definitions at face value. Situate the paper within the broader scientific landscape by defining its core concepts from first principles, citing foundational literature.
        * **Re-frame with Evidence**: If the authors' framing is weak, re-organize their ideas into a more insightful structure. Challenge their assumptions by citing counter-examples from published research.
        * **Provide a Roadmap**: Use citations constructively to point authors toward literature they may have missed, helping them build a stronger conceptual foundation.

    2.  **The Methodological Skeptic & Forensic Examiner**:
        * **Scrutinize the Methodology**: Forensically examine the experimental design, evaluation metrics, and statistical analysis. Are they appropriate for the claims being made?
        * **Identify Critical Omissions**: What is *absent* from the paper? Look for ignored alternative hypotheses, unacknowledged limitations, or countervailing evidence that is not addressed.
        * **Challenge Unstated Assumptions**: Articulate how unstated assumptions in the methodology could undermine the validity of the results and the paper's central claims.

    In short: your review must be a synthesis of these perspectives. You are not just checking for flaws; you are deeply engaging with the paper's ideas, challenging its foundations, questioning its methodology, and providing a clear, evidence-backed path for improvement. Your final review should be a masterclass in scholarly critique.
    """

    @staticmethod
    def get_review_prompt(paper_content: str, json_schema: dict, persona: str = None) -> str:
        """
        Constructs the full prompt for the AI model.

        Args:
            paper_content: The full text content of the paper to be reviewed.
            json_schema: The Pydantic model's JSON schema to guide the output format.

        Returns:
            A complete prompt string ready to be sent to the AI.
        """
        if not persona: persona = ReviewerPrompts.DEFAULT_REVIEWER_PERSONA

        return f"""
You are an expert academic reviewer for the NeurIPS conference. Your task is to analyze the provided research paper and generate a single, detailed, structured review.

**Your Persona & Mandate:**
{persona}

**Instructions:**
1.  Thoroughly read the entire paper provided below.
2.  Adopt the comprehensive persona described above to guide your critique.
3.  Generate one complete review that fills all the fields in the required JSON format.
4.  Your response MUST be a single, valid JSON object that conforms exactly to the schema below. Do not include any text, markdown, or code formatting before or after the JSON object.

**Required JSON Schema:**
```json
{json_schema}
```

**BEGIN CONFERENCE PAPER:**
---
{paper_content}
---
**END CONFERENCE PAPER.**

Now, provide your complete review as a single JSON object.
"""
