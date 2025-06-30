from pydantic import BaseModel, Field
from typing import Literal

# --- Pydantic Model for Structured NeurIPS Review ---
# This class defines the JSON structure we expect from the AI model.
# It ensures the response is parsed correctly and contains all required fields
# with the correct data types, especially for numerical ratings.

class NeurIPSReview(BaseModel):
    """The root model for a single, complete NeurIPS-style paper review."""
    summary: str = Field(
        description="A brief, neutral summary of the paper and its contributions. This should not be a critique or a copy of the abstract."
    )
    strengths_and_weaknesses: str = Field(
        description="A thorough assessment of the paper's strengths and weaknesses, touching on quality, clarity, significance, and originality. Use Markdown for formatting."
    )
    questions: str = Field(
        description="A list of actionable questions and suggestions for the authors (ideally 3-5 key points). Frame questions to clarify points that could change the evaluation."
    )
    limitations_and_societal_impact: str = Field(
        description="Assessment of whether limitations and potential negative societal impacts are adequately addressed. State 'Yes' if adequate; otherwise, provide constructive suggestions for improvement."
    )
    quality: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the work's quality (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    clarity: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the paper's clarity (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    significance: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the paper's significance (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    originality: Literal[4, 3, 2, 1] = Field(
        description="Numerical rating for the paper's originality (4: excellent, 3: good, 2: fair, 1: poor)."
    )
    overall_score: Literal[6, 5, 4, 3, 2, 1] = Field(
        description="Overall recommendation score (6: Strong Accept, 5: Accept, 4: Borderline accept, 3: Borderline reject, 2: Reject, 1: Strong Reject)."
    )
    confidence: Literal[5, 4, 3, 2, 1] = Field(
        description="Confidence in the assessment (5: Certain, 4: Confident, 3: Fairly confident, 2: Willing to defend, 1: Educated guess)."
    )

# --- Reviewer Personas and Prompt Generation ---

class ReviewerPrompts:
    """
    Manages and generates the prompts for different reviewer personas.
    """

    # This is the persona for Reviewer A, focusing on conceptual soundness and literature context.
    PROMPT_REVIEWER_A_PERSONA = """
    You are a top-tier academic reviewer, known for writing incisive yet constructive critiques that help elevate the entire research field. Your reviews are powerful because they are grounded in a deep and broad knowledge of the relevant literature.
    When reviewing a paper, your goal is to write a scholarly critique that does the following:
    1.  **Define the Terms from First Principles**: Do not accept the authors' definitions at face value. Start by establishing the canonical, accepted definition of the paper's core concepts, citing foundational sources or key literature from your knowledge base to support your definition.
    2.  **Re-frame with Evidence**: Don't just point out flaws in the authors' categories. Actively re-organize their ideas into a more insightful framework. For each new category you propose, provide evidence for your reasoning. This includes citing counter-examples from published research that challenge the authors' assumptions and using clear, powerful analogies to make your critique more concrete.
    3.  **Be Constructive Through Citations**: Your critique should not just tear the paper down; it should provide the authors with a roadmap for improvement. Use your citations to point them toward the literature they may have missed, helping them build a stronger conceptual foundation for their future work.

    In short: don't just critique the paper, situate it within the broader scientific landscape. Use external knowledge to prove your points, expose the paper's blind spots, and offer a path forward.
    """

    # This is the persona for Reviewer B, focusing on methodological rigor and identifying omissions.
    PROMPT_REVIEWER_B_PERSONA = """
    You are to assume the persona of an exceptionally discerning academic reviewer, known for a meticulous and forensic examination of scientific papers. Your primary role is to identify not only the flaws present in the text but, more importantly, the critical omissions and unstated assumptions that undermine the paper's validity.
    You must be especially critical of what is absent from the discussion—the alternative hypotheses the authors ignored, the limitations they failed to acknowledge, or the countervailing evidence they did not address. Your review should clearly articulate how these omissions fundamentally challenge the paper's claimed contribution and the soundness of its methodology, thereby determining if the work is suitable for publication.
    """

    @staticmethod
    def get_review_prompt(paper_content: str, reviewer_persona: str, json_schema: dict) -> str:
        """
        Constructs the full prompt for the AI model.

        Args:
            paper_content: The full text content of the paper to be reviewed.
            reviewer_persona: The specific persona text (e.g., PROMPT_REVIEWER_A_PERSONA).
            json_schema: The Pydantic model's JSON schema to guide the output format.

        Returns:
            A complete prompt string ready to be sent to the AI.
        """
        return f"""
You are an expert academic reviewer for the NeurIPS conference. Your task is to analyze the provided research paper and generate a detailed, structured review.

**Your Persona:**
{reviewer_persona}

**Instructions:**
1.  Thoroughly read the entire paper provided below.
2.  Adopt the persona described above to guide your critique.
3.  Generate a review that fills all the fields in the required JSON format.
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
