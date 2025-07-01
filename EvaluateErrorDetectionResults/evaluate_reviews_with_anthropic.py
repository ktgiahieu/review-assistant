import os
import glob
import argparse
import json
import pandas as pd
from pathlib import Path
import concurrent.futures
import time
import re

from tqdm import tqdm
import anthropic
from dotenv import load_dotenv

# Import the prompts and Pydantic model for evaluation
from evaluation_prompts import FlawEvaluation, EvaluationPrompts

# --- Environment & API Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Configure the Anthropic API with the key from environment variables
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Error: ANTHROPIC_API_KEY not found in environment variables or .env file.")
    exit(1)


# --- Exception Handling for Retries ---
# Define the exceptions from the Anthropic API client that are considered retryable
_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)

def _sanitize_json_string(json_str):
    """A simple function to clean common JSON errors from LLM output."""
    # Removes trailing commas before closing braces or brackets
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    # Strips markdown code block fences, which Claude often uses
    json_str = json_str.strip().strip("```json").strip("```")
    return json_str

def evaluate_single_flaw(
    anthropic_client,
    model_name,
    review_content,
    flaw_id,
    flaw_description,
    verbose=False
):
    """
    Calls the Anthropic API to evaluate a single flaw against a review.

    Args:
        anthropic_client: The configured Anthropic client instance.
        model_name (str): The name of the Anthropic model to use.
        review_content (dict): The content of the review.
        flaw_id (str): The ID of the flaw being evaluated.
        flaw_description (str): The description of the flaw.
        verbose (bool): If True, prints detailed logs.

    Returns:
        A dictionary containing the validated evaluation data or an error message.
    """
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print
    
    # Get the Pydantic model schema and format it for the prompt
    evaluation_schema = FlawEvaluation.model_json_schema()
    
    # The user prompt contains the specific data for the task.
    user_prompt = EvaluationPrompts.get_evaluation_prompt(
        review_text=json.dumps(review_content, indent=2),
        flaw_id=flaw_id,
        flaw_description=flaw_description,
        json_schema=evaluation_schema
    )

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            # Call the Anthropic API using the messages endpoint
            response_obj = anthropic_client.messages.create(
                model=model_name,
                max_tokens=10000, # Set a reasonable max token limit for the JSON output
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the text content from the response
            raw_json_content = response_obj.content[0].text
            
            # Clean the JSON string before validation
            cleaned_json = _sanitize_json_string(raw_json_content)
            
            # Validate the JSON structure and types with the Pydantic model
            validated_evaluation = FlawEvaluation.model_validate_json(cleaned_json)
            return validated_evaluation.model_dump()

        except Exception as e:
            last_exception = e
            if isinstance(e, _RETRYABLE_EXCEPTIONS):
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                if verbose: _print_method(f"Retryable API error ({type(e).__name__}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Handle other exceptions, like validation errors, as non-retryable
                if verbose: _print_method(f"Non-retryable error during evaluation: {type(e).__name__} - {e}")
                break
    
    # If all retries fail, return an error dictionary
    return {
        "flaw_id": flaw_id,
        "error": f"Failed to get a valid evaluation from LLM after {MAX_RETRIES} attempts.",
        "last_exception": str(last_exception)
    }


def process_review_evaluation(
    review_json_path,
    ground_truth_dir,
    anthropic_client,
    model_name,
    verbose
):
    """
    Processes a single review file, evaluating it against all flaws in its corresponding CSV.
    """
    try:
        review_path = Path(review_json_path)
        paper_id = review_path.parent.name

        # Construct the path to the corresponding ground truth CSV file
        ground_truth_csv_path = str(Path(ground_truth_dir) / paper_id / f"{'_'.join(paper_id.split('_')[:-2])}_modifications_summary.csv")

        if not os.path.exists(ground_truth_csv_path):
            return paper_id, f"ERROR: Ground truth CSV not found at {ground_truth_csv_path}"

        # --- Load review and ground truth data ---
        with open(review_json_path, 'r', encoding='utf-8') as f:
            review_content = json.load(f)
        
        ground_truth_df = pd.read_csv(ground_truth_csv_path)

        evaluations = []
        # Iterate over each flaw in the ground truth file and evaluate it
        for _, row in ground_truth_df.iterrows():
            flaw_id = row['flaw_id']
            flaw_description = row['flaw_description']
            
            if pd.isna(flaw_id) or pd.isna(flaw_description):
                continue

            evaluation_result = evaluate_single_flaw(
                anthropic_client,
                model_name,
                review_content,
                flaw_id,
                flaw_description,
                verbose
            )
            evaluations.append(evaluation_result)
        
        return paper_id, evaluations

    except Exception as e:
        # Catch any fatal errors during file processing
        return review_path.parent.name, f"FATAL ERROR processing {review_json_path}: {type(e).__name__} - {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated reviews against ground truth flaws using the Anthropic API.")
    parser.add_argument("--reviews_dir", type=str, required=True, help="Directory containing the generated review JSON files (e.g., './json_reviews/').")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Base directory of the flawed papers containing the ground truth CSVs (e.g., './content/output/ICLR2024_latest_flawed_papers_v1').")
    parser.add_argument("--output_file", type=str, default="./review_evaluations_anthropic.json", help="Path to save the final aggregated evaluation JSON file.")
    parser.add_argument("--model_name", type=str, required=True, help="Anthropic model name for evaluation (e.g., 'claude-3-sonnet-20240229').")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads for concurrent processing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = parser.parse_args()

    # Initialize the Anthropic client
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}"); exit(1)

    # Find all review files to be processed
    review_files = glob.glob(os.path.join(args.reviews_dir, "**/*_review.json"), recursive=True)
    if not review_files:
        print(f"No review files found in {args.reviews_dir}")
        exit(0)

    print(f"Found {len(review_files)} reviews to evaluate with model: {args.model_name}")
    
    all_evaluations = {}
    # Use a ThreadPoolExecutor to process reviews in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Create a future for each review file processing task
        future_to_review = {
            executor.submit(
                process_review_evaluation, review_path, args.ground_truth_dir,
                client, args.model_name, args.verbose
            ): review_path for review_path in review_files
        }

        # Use tqdm to show a progress bar as tasks are completed
        progress = tqdm(concurrent.futures.as_completed(future_to_review), total=len(review_files), desc="Evaluating Reviews")
        for future in progress:
            paper_id, result = future.result()
            all_evaluations[paper_id] = result

    # Save the final aggregated results to the specified output file
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, ensure_ascii=False, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Results for {len(all_evaluations)} papers saved to {args.output_file}")
    print("---------------------------\n")
