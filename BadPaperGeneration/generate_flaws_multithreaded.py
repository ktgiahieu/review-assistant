import openreview
import os
import json
import csv
import re
import argparse
import concurrent.futures
import time
import threading
from pathlib import Path
from openai import AzureOpenAI, APIError, Timeout, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Tuple
from tqdm import tqdm

from generation_prompt import ConsensusExtractionPrompt, FlawInjectionPrompt

# --- Pydantic Schemas for Structured LLM Responses ---

class Flaw(BaseModel):
    flaw_id: str = Field(..., description="A short, snake_case identifier for the flaw (e.g., 'limited_scope', 'missing_baseline').")
    description: str = Field(..., description="A detailed description of the flaw, explaining its significance based on the review-rebuttal consensus.")

class FlawExtractionResponse(BaseModel):
    critical_flaws: List[Flaw] = Field(..., description="A list of critical flaws identified from the review process.")

class Modification(BaseModel):
    target_heading: str = Field(..., description="The exact markdown heading of the section to replace.")
    new_content: str = Field(..., description="The complete, rewritten text for the entire section, including the heading.")
    reasoning: str = Field(..., description="A brief explanation for the modification.")

class ModificationGenerationResponse(BaseModel):
    modifications: List[Modification] = Field(..., description="A list of sections to modify to re-introduce the flaw.")

# --- Configuration ---
load_dotenv()

# OpenReview Credentials
OPENREVIEW_USERNAME = os.environ.get('OPENREVIEW_USERNAME')
OPENREVIEW_PASSWORD = os.environ.get('OPENREVIEW_PASSWORD')

# Azure OpenAI Credentials
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

# --- Helper Functions & Classes ---

def get_openreview_client():
    """Initializes and returns a connected OpenReview client."""
    try:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=OPENREVIEW_USERNAME,
            password=OPENREVIEW_PASSWORD
        )
        return client
    except Exception as e:
        print(f"Failed to connect to OpenReview: {e}")
        return None

def format_reviews_for_llm(note_details: dict) -> str:
    """Formats OpenReview note details into a clean text block for LLM analysis."""
    output_text = []
    if not note_details or 'replies' not in note_details:
        return "No review or comment data found."

    for reply in sorted(note_details['replies'], key=lambda x: x.get('number', 0)):
        signatures = ", ".join(reply.get('signatures', ['Unknown']))
        content = reply.get('content', {})

        if any(key in content for key in ['strengths', 'weaknesses', 'rating']):
            output_text.append(f"--- Review by {signatures} ---\n")
            for key, value in content.items():
                if isinstance(value, dict) and 'value' in value:
                    value = value['value']
                if isinstance(value, str) and value.strip():
                    output_text.append(f"## {key.replace('_', ' ').title()}\n{value.strip()}\n")
        elif 'comment' in content:
            output_text.append(f"--- Comment by {signatures} ---\n")
            comment_value = content['comment'].get('value', content['comment'])
            if isinstance(comment_value, str) and comment_value.strip():
                output_text.append(f"{comment_value.strip()}\n")

    return "\n".join(output_text)

def try_apply_modifications(original_markdown: str, modifications: List[Modification]) -> Tuple[str, bool]:
    """
    Tries to apply a list of modifications. Returns the modified text and a boolean success flag.
    Success is False if any target heading cannot be found.
    """
    current_lines = original_markdown.split('\n')
    
    for mod in modifications:
        target_heading = mod.target_heading.strip()
        if not target_heading:
            continue

        start_index = -1
        
        # Pass 1: Try an exact match
        for i, line in enumerate(current_lines):
            if line.strip() == target_heading:
                start_index = i
                break
        
        # Pass 2: If exact match fails, try a flexible match (ignoring hash levels)
        if start_index == -1:
            cleaned_target = target_heading.lstrip('#').strip()
            for i, line in enumerate(current_lines):
                if line.strip().lstrip('#').strip() == cleaned_target:
                    start_index = i
                    break

        if start_index == -1:
            tqdm.write(f"Failure: Could not find target heading '{target_heading}'.")
            return original_markdown, False # Return failure

        found_heading_line = current_lines[start_index].strip()
        heading_level = found_heading_line.count('#') if found_heading_line.startswith('#') else 0

        end_index = len(current_lines)
        
        if heading_level > 0:
            for i in range(start_index + 1, len(current_lines)):
                line = current_lines[i].strip()
                if line.startswith('#'):
                    current_level = line.count('#')
                    if current_level <= heading_level:
                        end_index = i
                        break
        else:
            for i in range(start_index + 1, len(current_lines)):
                if current_lines[i].strip().startswith('#'):
                    end_index = i
                    break

        pre_section = current_lines[:start_index]
        post_section = current_lines[end_index:]
        new_content_lines = mod.new_content.split('\n')
        current_lines = pre_section + new_content_lines + post_section
            
    return "\n".join(current_lines), True # Return success


def call_llm_with_retries(client, prompt, response_model, purpose="LLM Call"):
    """Generic function to call the Azure OpenAI API with retries and Pydantic parsing."""
    max_retries = 3
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            response = client.beta.chat.completions.parse(
                model=AZURE_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
                timeout=300.0,
            )
            json_response = response.choices[0].message.content
            return response_model.model_validate_json(json_response)
        
        except (json.JSONDecodeError, ValidationError) as e:
            return None
            
        except Exception as e:
            tqdm.write(f"Worker {threading.get_ident()}: [{purpose}] Retryable API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff_factor ** attempt)
            else:
                tqdm.write(f"Worker {threading.get_ident()}: [{purpose}] All API retries failed.")
                raise

    return None

def process_paper(paper_md_path: Path, input_base_dir: Path, output_base_dir: Path, or_client, azure_client):
    """Main worker function to process a single paper."""
    worker_id = threading.get_ident()
    MAX_MODIFICATION_ATTEMPTS = 3
    try:
        paper_folder = paper_md_path.parent.parent
        openreview_id = paper_folder.name.split('_')[0]
        
        with open(paper_md_path, 'r', encoding='utf-8') as f:
            original_paper_text = f.read()

        note = or_client.get_note(openreview_id, details='replies')
        review_text = format_reviews_for_llm(note.details)

        flaw_extraction_prompt = ConsensusExtractionPrompt.generate_prompt_for_review_process(
            review_process=review_text, paper_text=original_paper_text
        )
        flaw_response = call_llm_with_retries(azure_client, flaw_extraction_prompt, FlawExtractionResponse, "Flaw Extraction")

        if not flaw_response or not flaw_response.critical_flaws:
            return []

        results_for_global_csv = []

        for flaw in flaw_response.critical_flaws:
            was_successful = False
            for attempt in range(MAX_MODIFICATION_ATTEMPTS):
                tqdm.write(f"Worker {worker_id}: Flaw '{flaw.flaw_id}' in {openreview_id}, attempt {attempt + 1}/{MAX_MODIFICATION_ATTEMPTS}...")

                modification_prompt = FlawInjectionPrompt.generate_prompt_for_flaw_injection(
                    flaw_description=flaw.description, original_paper_text=original_paper_text
                )
                mod_response = call_llm_with_retries(azure_client, modification_prompt, ModificationGenerationResponse, "Modification Generation")

                if not mod_response or not mod_response.modifications:
                    tqdm.write(f"Worker {worker_id}: LLM returned no modifications. Not retrying for this flaw.")
                    break # Stop trying for this flaw if LLM gives up

                flawed_paper_text, success = try_apply_modifications(original_paper_text, mod_response.modifications)

                if success:
                    tqdm.write(f"Worker {worker_id}: Successfully applied modifications for flaw '{flaw.flaw_id}'.")
                    
                    relative_paper_folder = paper_folder.relative_to(input_base_dir)
                    output_paper_folder = output_base_dir / relative_paper_folder / "flawed_papers"
                    output_paper_folder.mkdir(parents=True, exist_ok=True)
                    
                    output_filename = f"{openreview_id}_{flaw.flaw_id}.md"
                    output_filepath = output_paper_folder / output_filename

                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(flawed_paper_text)
                    
                    modifications_json_string = json.dumps([mod.model_dump() for mod in mod_response.modifications], indent=2)

                    results_for_global_csv.append({
                        'openreview_id': openreview_id,
                        'flaw_id': flaw.flaw_id,
                        'flaw_description': flaw.description,
                        'output_path': str(output_filepath),
                        'num_modifications': len(mod_response.modifications),
                        'llm_generated_modifications': modifications_json_string
                    })
                    was_successful = True
                    break # Success, so break the attempt loop
                else:
                    tqdm.write(f"Worker {worker_id}: Modification application failed on attempt {attempt + 1}. Retrying LLM call.")
            
            if not was_successful:
                tqdm.write(f"Worker {worker_id}: All {MAX_MODIFICATION_ATTEMPTS} attempts failed for flaw '{flaw.flaw_id}'. Skipping.")
            
        return results_for_global_csv
    
    except Exception as e:
        tqdm.write(f"Worker {worker_id}: ERROR processing {paper_md_path.parent.parent.name}: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_paper(paper_md_path: Path, input_base_dir: Path, output_base_dir: Path, or_client, azure_client):
    """Main worker function to process a single paper."""
    worker_id = threading.get_ident()
    MAX_MODIFICATION_ATTEMPTS = 3
    try:
        paper_folder = paper_md_path.parent.parent
        openreview_id = paper_folder.name.split('_')[0]
        
        with open(paper_md_path, 'r', encoding='utf-8') as f:
            original_paper_text = f.read()

        note = or_client.get_note(openreview_id, details='replies')
        review_text = format_reviews_for_llm(note.details)

        flaw_extraction_prompt = ConsensusExtractionPrompt.generate_prompt_for_review_process(
            review_process=review_text, paper_text=original_paper_text
        )
        flaw_response = call_llm_with_retries(azure_client, flaw_extraction_prompt, FlawExtractionResponse, "Flaw Extraction")

        if not flaw_response or not flaw_response.critical_flaws:
            return []

        
        results_for_global_csv = []
        all_modifications_for_local_csv = []


        for flaw in flaw_response.critical_flaws:
            
            was_successful = False
            for attempt in range(MAX_MODIFICATION_ATTEMPTS):
            
                modification_prompt = FlawInjectionPrompt.generate_prompt_for_flaw_injection(
                    flaw_description=flaw.description, original_paper_text=original_paper_text
                )
                mod_response = call_llm_with_retries(azure_client, modification_prompt, ModificationGenerationResponse, "Modification Generation")

                if not mod_response or not mod_response.modifications:
                    continue

                flawed_paper_text, success = try_apply_modifications(original_paper_text, mod_response.modifications)

                if success:
                    relative_paper_folder = paper_folder.relative_to(input_base_dir)
                    output_paper_folder = output_base_dir / relative_paper_folder / "flawed_papers"
                    output_paper_folder.mkdir(parents=True, exist_ok=True)
                    
                    output_filename = f"{openreview_id}_{flaw.flaw_id}.md"
                    output_filepath = output_paper_folder / output_filename

                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(flawed_paper_text)
                    
                    modifications_json_string = json.dumps([mod.model_dump() for mod in mod_response.modifications], indent=2)

                    results_for_global_csv.append({
                        'openreview_id': openreview_id,
                        'flaw_id': flaw.flaw_id,
                        'flaw_description': flaw.description,
                        'output_path': str(output_filepath),
                        'num_modifications': len(mod_response.modifications),
                        'llm_generated_modifications': modifications_json_string
                    })
                    
                    
                    # Collect modifications for the local CSV
                    all_modifications_for_local_csv.append({
                        'flaw_id': flaw.flaw_id,
                        'flaw_description': flaw.description,
                        'num_modifications': len(mod_response.modifications),
                        'llm_generated_modifications': modifications_json_string
                    })
                    was_successful = True
                    break # Success, so break the attempt loop
                else:
                    tqdm.write(f"Worker {worker_id}: Modification application failed on attempt {attempt + 1}. Retrying LLM call.")
            if not was_successful:
                tqdm.write(f"Worker {worker_id}: All {MAX_MODIFICATION_ATTEMPTS} attempts failed for flaw '{flaw.flaw_id}'. Skipping.")
            
        
        # Write the local modifications CSV for this paper
        if all_modifications_for_local_csv:
            local_csv_path = output_paper_folder / f"{openreview_id}_modifications.csv"
            with open(local_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_modifications_for_local_csv[0].keys())
                writer.writeheader()
                writer.writerows(all_modifications_for_local_csv)

        return results_for_global_csv

    except Exception as e:
        tqdm.write(f"Worker {worker_id}: ERROR processing {paper_md_path.parent.parent.name}: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- Main Workflow Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate flawed paper versions based on OpenReview discussions.")
    parser.add_argument("--input_dir", type=str, required=True, help="Base input directory (e.g., 'ICLR2024_latest').")
    parser.add_argument("--output_dir", type=str, default="flawed_papers_v1", help="Directory to save the flawed papers and results CSV.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel threads to run.")
    args = parser.parse_args()

    if not all([OPENREVIEW_USERNAME, OPENREVIEW_PASSWORD, AZURE_ENDPOINT, AZURE_API_KEY, AZURE_DEPLOYMENT_NAME]):
        print("Error: One or more required environment variables are not set. Exiting.")
        return

    input_base_dir = Path(args.input_dir)
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input directory: {input_base_dir.resolve()}")
    print(f"Output directory: {output_base_dir.resolve()}")

    paper_paths = list(input_base_dir.glob("**/paper.md"))
    if not paper_paths:
        print(f"No 'paper.md' files found in {input_base_dir}. Please check the path and folder structure.")
        return
    
    print(f"Found {len(paper_paths)} papers to process.")

    or_client = get_openreview_client()
    if not or_client:
        print("Failed to connect to OpenReview. Check credentials. Exiting.")
        return

    azure_client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION
    )

    all_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_path = {
            executor.submit(process_paper, path, input_base_dir, output_base_dir, or_client, azure_client): path
            for path in paper_paths
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_path), total=len(paper_paths), desc="Processing Papers"):
            try:
                result = future.result()
                if result:
                    all_results.extend(result)
            except Exception as e:
                path = future_to_path[future]
                tqdm.write(f"Main Thread: An unexpected error occurred for paper at {path}: {e}")

    if not all_results:
        print("\nWorkflow finished, but no flawed papers were generated.")
        return

    results_csv_path = output_base_dir / "flawed_papers_summary.csv"
    print(f"\nWriting global summary of {len(all_results)} generated files to {results_csv_path}...")
    
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print("Workflow complete.")

if __name__ == "__main__":
    main()