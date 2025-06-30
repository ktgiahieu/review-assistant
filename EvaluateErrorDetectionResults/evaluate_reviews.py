import os
import glob
import argparse
import json
import pandas as pd
from pathlib import Path
import concurrent.futures
import time

from tqdm import tqdm
from openai import AzureOpenAI, APIError, RateLimitError, APIConnectionError, InternalServerError
from dotenv import load_dotenv

# Import the new prompts and Pydantic model for evaluation
from evaluation_prompts import FlawEvaluation, EvaluationPrompts

# --- Environment & API Configuration ---
load_dotenv()

KEY = os.environ.get("KEY")
API_VERSION = os.environ.get("API_VERSION")
ENDPOINT = os.environ.get("ENDPOINT")

# --- Exception Handling for Retries ---
_RETRYABLE_EXCEPTIONS = (APIError, RateLimitError, APIConnectionError, InternalServerError)

def _sanitize_json_string(json_str):
    """A simple function to clean common JSON errors from LLM output."""
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    json_str = json_str.strip().strip("```json").strip("```")
    return json_str

def evaluate_single_flaw(
    openai_client,
    deployment_name,
    review_content,
    flaw_id,
    flaw_description,
    verbose=False
):
    """
    Calls the LLM to evaluate a single flaw against a review.
    """
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print
    
    # Get the prompt and schema
    evaluation_schema = FlawEvaluation.model_json_schema()
    prompt = EvaluationPrompts.get_evaluation_prompt(
        review_text=json.dumps(review_content, indent=2),
        flaw_id=flaw_id,
        flaw_description=flaw_description,
        json_schema=evaluation_schema
    )

    completion_args = {
        "model": deployment_name,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "timeout": 300.0,
    }

    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            response_obj = openai_client.chat.completions.create(**completion_args)
            
            raw_json_content = response_obj.choices[0].message.content
            # Validate with Pydantic
            validated_evaluation = FlawEvaluation.model_validate_json(raw_json_content)
            return validated_evaluation.model_dump()

        except Exception as e:
            last_exception = e
            if isinstance(e, _RETRYABLE_EXCEPTIONS):
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                if verbose: _print_method(f"Retryable API error ({type(e).__name__}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                if verbose: _print_method(f"Non-retryable error during evaluation: {type(e).__name__} - {e}")
                break
    
    # If all retries fail
    return {
        "flaw_id": flaw_id,
        "error": f"Failed to get a valid evaluation from LLM after {MAX_RETRIES} attempts.",
        "last_exception": str(last_exception)
    }


def process_review_evaluation(
    review_json_path,
    ground_truth_dir,
    openai_client,
    deployment_name,
    verbose
):
    """
    Processes a single review file, evaluating it against all flaws in its corresponding CSV.
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    _print_method = tqdm.write if not verbose else print

    try:
        review_path = Path(review_json_path)
        paper_id = review_path.parent.name

        # --- Find the corresponding ground truth CSV ---
        # Assumes CSV is in a path like .../{status}/{paper_id}/{paper_id}_modifications_summary.csv
        # We need to search for it since we don't know the {status} folder.
        # csv_search_pattern = str(Path(ground_truth_dir) / paper_id / f"{paper_id}_modifications_summary.csv")
        # found_csv_files = glob.glob(csv_search_pattern, recursive=True)

        # if not found_csv_files:
        #     return paper_id, f"ERROR: Ground truth CSV not found for paper ID {paper_id} using pattern {csv_search_pattern}"

        # ground_truth_csv_path = found_csv_files[0]
        ground_truth_csv_path = str(Path(ground_truth_dir) / paper_id / f"{'_'.join(paper_id.split('_')[:-2])}_modifications_summary.csv")

        # if verbose: _print_method(f"Worker {worker_id}: Matched review {review_path.name} with ground truth {ground_truth_csv_path}")

        # --- Load data ---
        with open(review_json_path, 'r', encoding='utf-8') as f:
            review_content = json.load(f)
        
        ground_truth_df = pd.read_csv(ground_truth_csv_path)

        evaluations = []
        for _, row in ground_truth_df.iterrows():
            flaw_id = row['flaw_id']
            flaw_description = row['flaw_description']
            
            if pd.isna(flaw_id) or pd.isna(flaw_description):
                continue

            evaluation_result = evaluate_single_flaw(
                openai_client,
                deployment_name,
                review_content,
                flaw_id,
                flaw_description,
                verbose
            )
            evaluations.append(evaluation_result)
        
        return paper_id, evaluations

    except Exception as e:
        return review_path.parent.name, f"FATAL ERROR processing {review_json_path}: {type(e).__name__} - {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated reviews against ground truth flaws.")
    parser.add_argument("--reviews_dir", type=str, required=True, help="Directory containing the generated review JSON files (e.g., './json_reviews/').")
    parser.add_argument("--ground_truth_dir", type=str, required=True, help="Base directory of the flawed papers containing the ground truth CSVs (e.g., './content/output/ICLR2024_latest_flawed_papers_v1').")
    parser.add_argument("--output_file", type=str, default="./review_evaluations.json", help="Path to save the final aggregated evaluation JSON file.")
    parser.add_argument("--deployment_name", type=str, required=True, help="Azure OpenAI deployment name for the evaluation model.")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    if not all([KEY, API_VERSION, ENDPOINT, args.deployment_name]):
        print("Error: Missing Azure credentials or deployment name in .env file or environment variables."); exit(1)
    try:
        azure_openai_client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY, api_version=API_VERSION)
    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {e}"); exit(1)

    review_files = glob.glob(os.path.join(args.reviews_dir, "**/*_review.json"), recursive=True)
    if not review_files:
        print(f"No review files found in {args.reviews_dir}")
        exit(0)

    print(f"Found {len(review_files)} reviews to evaluate.")
    
    all_evaluations = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_review = {
            executor.submit(
                process_review_evaluation, review_path, args.ground_truth_dir,
                azure_openai_client, args.deployment_name, args.verbose
            ): review_path for review_path in review_files
        }

        progress = tqdm(concurrent.futures.as_completed(future_to_review), total=len(review_files), desc="Evaluating Reviews")
        for future in progress:
            paper_id, result = future.result()
            all_evaluations[paper_id] = result

    # Save the final aggregated results
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, ensure_ascii=False, indent=2)

    print(f"\n--- Evaluation Complete ---")
    print(f"Results for {len(all_evaluations)} papers saved to {args.output_file}")
    print("---------------------------\n")
