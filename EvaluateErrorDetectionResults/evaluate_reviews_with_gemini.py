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
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

# Import the prompts and Pydantic model for evaluation
from gemini_evaluation_prompts import FlawEvaluation, EvaluationPrompts

# --- Environment & API Configuration ---
# Load environment variables from a .env file
load_dotenv()

# Configure the Gemini API with the key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Error: GEMINI_API_KEY not found in environment variables or .env file.")
    exit(1)


# --- Exception Handling for Retries ---
# Define the exceptions from the Google API client that are considered retryable
_RETRYABLE_EXCEPTIONS = (
    google_exceptions.ResourceExhausted,  # Corresponds to RateLimitError
    google_exceptions.ServiceUnavailable,
    google_exceptions.InternalServerError,
    google_exceptions.GatewayTimeout,
)

def _sanitize_json_string(json_str: str) -> str:
    """
    A robust function to clean common JSON errors from LLM output.
    It finds the main JSON object within a string that might contain extra text
    by locating the first '{' and the last '}'.
    """
    try:
        # Find the first opening curly brace
        start_index = json_str.find('{')
        # Find the last closing curly brace
        end_index = json_str.rfind('}')
        
        # If a valid JSON object is likely present, extract it
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = json_str[start_index : end_index + 1]
        else:
            # If no valid braces are found, return original string for error logging
            return json_str

        # Remove trailing commas before closing braces or brackets, a common LLM error
        json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
        return json_str
    except Exception:
        # In case of any unexpected error, return the original string
        return json_str

def evaluate_single_flaw(
    gemini_model,
    review_content,
    flaw_id,
    flaw_description,
    verbose=False
):
    """
    Calls the Gemini API to evaluate a single flaw against a review.
    If a validation error occurs, it makes a second attempt with a corrective prompt.
    """
    MAX_API_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print

    # Initial prompt for the first attempt
    prompt = EvaluationPrompts.get_evaluation_prompt(
        review_text=json.dumps(review_content, indent=2),
        flaw_id=flaw_id,
        flaw_description=flaw_description,
    )

    # Configure the generation settings to strictly enforce a JSON response
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=FlawEvaluation
    )

    last_exception = None
    
    # We make up to two main attempts: one initial, and one corrective if the first fails validation.
    for attempt in range(2):
        # If this is the second attempt, create a corrective prompt.
        if attempt == 1:
            if verbose: _print_method(f"Initial attempt failed for flaw {flaw_id}. Retrying with corrective prompt.")
            error_message = str(last_exception)
            # Create a new prompt that includes the error message from the previous failed attempt.
            prompt = (
                f"Your previous response failed with a validation error. Please correct your mistake.\n\n"
                f"ERROR: {error_message}\n\n"
                f"Please provide a new response that is ONLY a single, valid JSON object matching the required schema. "
                f"Do not include any other text, thoughts, or explanations outside of the JSON object.\n\n"
                f"Here is the original request again for your reference:\n\n{prompt}"
            )

        # Inner loop for handling retryable API errors (e.g., rate limits, network issues)
        for api_retry_attempt in range(MAX_API_RETRIES):
            try:
                # Call the Gemini API
                response_obj = gemini_model.generate_content(
                    contents=prompt,
                    generation_config=generation_config
                )
                
                raw_json_content = response_obj.text
                cleaned_json = _sanitize_json_string(raw_json_content)
                
                # Validate the JSON structure and types with Pydantic
                validated_evaluation = FlawEvaluation.model_validate_json(cleaned_json)
                return validated_evaluation.model_dump() # Success!

            except _RETRYABLE_EXCEPTIONS as e:
                last_exception = e
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** api_retry_attempt)
                if verbose: _print_method(f"Retryable API error ({type(e).__name__}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                # Continue the inner loop to retry the API call

            except Exception as e:
                # This is a non-retryable error (like a ValidationError).
                last_exception = e
                if verbose:
                    _print_method(f"Attempt {attempt + 1} failed with non-retryable error: {type(e).__name__}")
                # Break the inner loop to proceed to the corrective attempt (if any).
                break
        
        # If the inner loop completed all its retries for a retryable error, we break the outer loop too.
        if isinstance(last_exception, _RETRYABLE_EXCEPTIONS):
            break

    # If we've exhausted all attempts (initial and corrective), return the final error.
    return {
        "flaw_id": flaw_id,
        "error": f"Failed to get a valid evaluation from LLM after a corrective attempt.",
        "last_exception": str(last_exception)
    }


def process_review_evaluation(
    review_json_path,
    ground_truth_dir,
    gemini_model,
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
                gemini_model,
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
    parser = argparse.ArgumentParser(description="Evaluate generated reviews against ground truth flaws using the Gemini API.")
    parser.add_argument("--reviews_dir", type=str, required=True, help="Directory containing the generated review JSON files (e.g., './json_reviews/').")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Base directory of the flawed papers containing the ground truth CSVs (e.g., './content/output/ICLR2024_latest_flawed_papers_v1').")
    parser.add_argument("--output_file", type=str, default="./review_evaluations_gemini.json", help="Path to save the final aggregated evaluation JSON file.")
    parser.add_argument("--model_name", type=str, required=True, help="Gemini model name for evaluation (e.g., 'gemini-1.5-flash').")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads for concurrent processing.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")
    args = parser.parse_args()

    # Initialize the Gemini model
    try:
        model = genai.GenerativeModel(args.model_name)
    except Exception as e:
        print(f"Error initializing Gemini model '{args.model_name}': {e}"); exit(1)

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
                model, args.verbose
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
