import os
import glob
import base64
import argparse
import re
import concurrent.futures
import time
import math
import urllib.parse
from pathlib import Path
import io
import json
from typing import List, Literal
import tiktoken

try:
    from PIL import Image
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image
except ImportError:
    print("ERROR: Pillow library not found. Please install it (`pip install Pillow`) to use image resizing features.")
    Image = None

from tqdm import tqdm
from openai import AzureOpenAI, APIError, RateLimitError, APIConnectionError, InternalServerError
from dotenv import load_dotenv
from checklist_prompts import (
    MainPrompt,
    CompliancePrompt,
    ContributionPrompt,
    SoundnessPrompt,
    PresentationPrompt
)
from pydantic import BaseModel, Field, ValidationError

CONTEXT_LENGTH_LIMIT = 160000  # The context limit of the model being used
TOKEN_BUFFER = 10000 # Buffer for system prompts, response format, etc.
# --- Helper Functions & Classes ---

def get_tokenizer():
    """Initializes and returns a tiktoken tokenizer."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return tiktoken.encoding_for_model("gpt-4")

tokenizer = get_tokenizer()

def truncate_text(text: str, max_tokens: int) -> str:
    """Truncates text to a maximum number of tokens."""
    tokens = tokenizer.encode(text, allowed_special="all")
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)
    return text

# --- Pydantic Models for Structured Output ---
# These classes define the structure we expect from the AI model.
# This ensures the response is parsed correctly and contains all required fields.

class ChecklistItem(BaseModel):
    """Represents a single question, its answer, and the reasoning."""
    question_id: str = Field(description="The unique identifier for the question (e.g., 'A1', 'B5').")
    question_text: str = Field(description="The full text of the checklist question.")
    answer: Literal["Yes", "No", "NA"] = Field(description="The answer to the question.")
    reasoning: str = Field(description="A brief, clear justification for the answer provided, citing evidence from the paper.")

# class ChecklistSection(BaseModel):
#     """Represents a category of checklist items (e.g., Compliance, Soundness)."""
#     section_title: str = Field(description="The title of the checklist section.")
#     items: List[ChecklistItem] = Field(description="A list of all questions and answers for this section.")

class FormattedAbstract(BaseModel):
    """Structured summary of the paper's abstract."""
    background_and_objectives: str
    material_and_methods: str
    results: str
    conclusion: str

# Update checklist review to separate checklists from it
# class MainReview(BaseModel):
#     """The root model for the entire paper review."""
#     paper_title: str = Field(description="The verbatim title of the research paper.")
#     formatted_abstract: FormattedAbstract

# --- Environment & Configuration ---
load_dotenv()

KEY = os.environ.get("KEY")
API_VERSION = os.environ.get("API_VERSION")
LOCATION = os.environ.get("LOCATION") # Not directly used in client init, but good practice
ENDPOINT = os.environ.get("ENDPOINT")
# DEPLOYMENT_NAME is now an arg

# --- Pre-filter OpenAI Exception classes for robust retry handling ---
_POTENTIAL_OPENAI_RETRY_EXCEPTIONS = [
    APIError, RateLimitError, APIConnectionError, InternalServerError
]
_VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE = tuple(
    exc for exc in _POTENTIAL_OPENAI_RETRY_EXCEPTIONS
    if isinstance(exc, type) and issubclass(exc, BaseException)
)

if len(_VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE) != len(_POTENTIAL_OPENAI_RETRY_EXCEPTIONS):
    print("WARNING: Not all expected OpenAI exception types were found to be valid exception classes.")
    _expected_names = {exc.__name__ for exc in _POTENTIAL_OPENAI_RETRY_EXCEPTIONS if hasattr(exc, '__name__')}
    _valid_names = {exc.__name__ for exc in _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE if hasattr(exc, '__name__')}
    _problematic_names = _expected_names - _valid_names
    if _problematic_names:
        print(f"  Problematic/invalid exception names: {', '.join(_problematic_names)}")

if not _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE:
    print("CRITICAL WARNING: No valid OpenAI-specific exception types could be identified for retries.")
    if isinstance(APIError, type) and issubclass(APIError, BaseException):
        _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE = (APIError,)
        print("INFO: Falling back to retrying on generic openai.APIError only.")
    else:
        _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE = None 
        print("CRITICAL ERROR: openai.APIError itself is not a valid exception type. Specific API error retries are disabled.")
# --- End of pre-filtering ---

# Image size and resizing parameters
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # API limit: 20MB
TARGET_RESIZE_BYTES = 10 * 1024 * 1024 # Softer target for resizing: 10MB
MAX_RESIZE_ATTEMPTS = 4 # Increased attempts for more aggressive resizing if needed
MIN_DIMENSION_AFTER_RESIZE = 50 # Minimum width/height to avoid over-shrinking


def encode_image_bytes(image_bytes):
    """Encodes image bytes to a base64 string."""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error base64 encoding image bytes: {e}") # Keep for now, or make verbose
        return None

def process_markdown_file(
    markdown_file_path,
    output_base_dir,
    openai_client,
    deployment_name,
    max_tokens_completion,
    verbose, 
    max_figures
):
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    current_tokens_used = 0 
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print

    try:
        if verbose:
            print(f"Worker {worker_id}: Starting to process {markdown_file_path}")

        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            input_markdown_content = f.read()

        markdown_dir = Path(markdown_file_path).parent
        paper_folder_level_dir = markdown_dir.parent 
        unique_paper_identifier = paper_folder_level_dir.name
        openreview_id_for_link = unique_paper_identifier.split('_')[0]
        
        if verbose:
            print(f"Worker {worker_id}: MD dir: {markdown_dir}, Paper root: {paper_folder_level_dir}, ID: {unique_paper_identifier}")

        paper_content_for_prompt = f"\nPaper link: https://openreview.net/forum?id={openreview_id_for_link}\n" + input_markdown_content
        # --- Truncate inputs to avoid context length errors ---
        paper_max_tokens = CONTEXT_LENGTH_LIMIT - TOKEN_BUFFER
        paper_content_for_prompt = truncate_text(paper_content_for_prompt, paper_max_tokens)


        # Prompts list of dict
        prompts_list = [
            {
                "name": "main",
                "prompt": MainPrompt,
            },
            {
                "name": "compliance",
                "prompt": CompliancePrompt,
            },
            {
                "name": "contribution",
                "prompt": ContributionPrompt,
            },
            {
                "name": "soundness",
                "prompt": SoundnessPrompt,
            },
            {
                "name": "presentation",
                "prompt": PresentationPrompt,
            }
        ]

        # Main loop over prompts (5 prompts per paper)
        json_response = None
        for prompt_dict in prompts_list:

            current_prompt = prompt_dict["prompt"].get_prompt(paper_content_for_prompt)
            content_list = [{"type": "input_text", "text": current_prompt}]

            added_figures_count = 0
            if (max_figures is None or max_figures > 0) and Image is not None:
                md_img_patterns = [
                    r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", 
                    r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>",   
                    r"<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>" 
                ]
                valid_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
                all_matches = []
                for pattern_str in md_img_patterns:
                    for match in re.finditer(pattern_str, input_markdown_content, flags=re.IGNORECASE | re.DOTALL):
                        all_matches.append({'match': match, 'start_pos': match.start()})
                all_matches.sort(key=lambda x: x['start_pos'])

                if verbose and all_matches: print(f"Worker {worker_id}: Found {len(all_matches)} potential image references in MD.")
                queued_image_abs_paths = set()

                for item in all_matches:
                    if max_figures is not None and added_figures_count >= max_figures:
                        if verbose: _print_method(f"Worker {worker_id}: Reached max_figures ({max_figures}) for API.")
                        break 
                    
                    raw_path = item['match'].group(1)
                    try: decoded_path_str = urllib.parse.unquote(raw_path)
                    except Exception as e_dec:
                        if verbose: _print_method(f"Worker {worker_id}: Could not decode path '{raw_path}': {e_dec}. Skipping.")
                        continue
                    
                    if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']:
                        if verbose: _print_method(f"Worker {worker_id}: Skipping web URL: {decoded_path_str}")
                        continue

                    found_abs_path = None
                    candidate_path_obj = (markdown_dir / decoded_path_str).resolve()
                    if candidate_path_obj.is_file() and candidate_path_obj.suffix.lower() in valid_image_extensions:
                        found_abs_path = candidate_path_obj
                    elif not candidate_path_obj.suffix: 
                        for ext_ in valid_image_extensions:
                            path_with_ext = (markdown_dir / (decoded_path_str + ext_)).resolve()
                            if path_with_ext.is_file(): found_abs_path = path_with_ext; break
                    
                    if not found_abs_path and not Path(decoded_path_str).is_absolute():
                        candidate_obj_paper_root = (paper_folder_level_dir / decoded_path_str).resolve()
                        if candidate_obj_paper_root.is_file() and candidate_obj_paper_root.suffix.lower() in valid_image_extensions:
                            found_abs_path = candidate_obj_paper_root
                        elif not candidate_obj_paper_root.suffix: 
                            for ext_ in valid_image_extensions:
                                path_with_ext = (paper_folder_level_dir / (decoded_path_str + ext_)).resolve()
                                if path_with_ext.is_file(): found_abs_path = path_with_ext; break
                    
                    if found_abs_path:
                        if str(found_abs_path) in queued_image_abs_paths:
                            if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' already processed. Skipping.")
                            continue
                        
                        image_data_for_api_bytes = None
                        try:
                            initial_file_size = found_abs_path.stat().st_size
                            if verbose: _print_method(f"Worker {worker_id}: Found local image '{found_abs_path}' (Initial size: {initial_file_size / (1024*1024):.2f}MB).")

                            if initial_file_size <= MAX_IMAGE_SIZE_BYTES:
                                with open(found_abs_path, "rb") as f_img: image_data_for_api_bytes = f_img.read()
                                if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' is within API size limit. Using original.")
                            else: # Image is too large, needs resizing
                                if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' ({initial_file_size / (1024*1024):.2f}MB) > {MAX_IMAGE_SIZE_BYTES/(1024*1024):.0f}MB. Attempting resize.")
                                
                                pil_img_original = Image.open(found_abs_path)
                                original_format = pil_img_original.format or Path(found_abs_path).suffix[1:].upper()
                                if not original_format or original_format.lower() not in ['jpeg', 'png', 'webp', 'gif']:
                                    original_format = 'PNG' if original_format and original_format.lower() == 'png' else 'JPEG'
                                
                                if pil_img_original.is_animated and original_format.upper() == 'GIF':
                                    if verbose: _print_method(f"Worker {worker_id}: Animated GIF '{found_abs_path}'. Using first frame.")
                                    pil_img_original.seek(0)

                                current_pil_img_to_resize = pil_img_original
                                current_data_size_for_scaling = initial_file_size

                                for attempt in range(MAX_RESIZE_ATTEMPTS):
                                    if current_data_size_for_scaling <= MAX_IMAGE_SIZE_BYTES: # Should have been caught by initial check or previous iteration
                                        # This means the *previous* resize was successful. We need to get those bytes.
                                        # This case is handled if image_data_for_api_bytes was set.
                                        # For safety, if we are here and it's small, re-save current_pil_img_to_resize
                                        img_byte_buffer_final_check = io.BytesIO()
                                        save_params = {'format': original_format}
                                        if original_format.upper() == 'JPEG': save_params['quality'] = 85
                                        elif original_format.upper() == 'WEBP': save_params['quality'] = 80
                                        current_pil_img_to_resize.save(img_byte_buffer_final_check, **save_params)
                                        if img_byte_buffer_final_check.tell() <= MAX_IMAGE_SIZE_BYTES:
                                            image_data_for_api_bytes = img_byte_buffer_final_check.getvalue()
                                        break # Exit resize loop

                                    # Calculate scale factor to aim for TARGET_RESIZE_BYTES from current_data_size_for_scaling
                                    scale_factor = math.sqrt(TARGET_RESIZE_BYTES / current_data_size_for_scaling) if current_data_size_for_scaling > 0 else 0.5
                                    scale_factor = max(0.1, min(scale_factor, 0.95)) # Clamp factor

                                    new_width = int(current_pil_img_to_resize.width * scale_factor)
                                    new_height = int(current_pil_img_to_resize.height * scale_factor)

                                    if new_width < MIN_DIMENSION_AFTER_RESIZE or new_height < MIN_DIMENSION_AFTER_RESIZE:
                                        if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' dimensions ({new_width}x{new_height}) too small after scaling by {scale_factor:.2f}. Stopping resize.")
                                        break 

                                    if verbose: _print_method(f"Worker {worker_id}: Resize attempt {attempt + 1}/{MAX_RESIZE_ATTEMPTS} for '{found_abs_path}'. From est. {current_data_size_for_scaling/(1024*1024):.2f}MB. Scale: {scale_factor:.2f} -> {new_width}x{new_height}.")
                                    
                                    resized_pil_img = current_pil_img_to_resize.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                    img_byte_buffer = io.BytesIO()
                                    save_params = {'format': original_format}
                                    if original_format.upper() == 'JPEG': save_params['quality'] = 85
                                    elif original_format.upper() == 'WEBP': save_params['quality'] = 80
                                    
                                    resized_pil_img.save(img_byte_buffer, **save_params)
                                    resized_bytes_in_buffer = img_byte_buffer.tell()
                                    current_data_size_for_scaling = resized_bytes_in_buffer # Update for next iteration

                                    if verbose: _print_method(f"Worker {worker_id}: Resized '{found_abs_path}' (attempt {attempt+1}), new actual size: {resized_bytes_in_buffer / (1024*1024):.2f}MB.")

                                    if resized_bytes_in_buffer <= MAX_IMAGE_SIZE_BYTES:
                                        image_data_for_api_bytes = img_byte_buffer.getvalue()
                                        if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' successfully resized to fit API limit.")
                                        break 
                                    else:
                                        current_pil_img_to_resize = resized_pil_img # Use for next attempt
                                
                                if image_data_for_api_bytes is None: # All resize attempts failed
                                    if verbose: _print_method(f"Worker {worker_id}: Skipping image '{found_abs_path}' after failing to resize it within limits after {MAX_RESIZE_ATTEMPTS} attempts.")
                                    continue
                        
                        except OSError as e_stat:
                            if verbose: _print_method(f"Worker {worker_id}: OSError accessing/processing image '{found_abs_path}': {e_stat}. Skipping.")
                            continue
                        except Exception as e_img_proc: # Catch other Pillow errors
                            if verbose: _print_method(f"Worker {worker_id}: Error processing image '{found_abs_path}' with Pillow: {e_img_proc}. Skipping.")
                            continue


                        if image_data_for_api_bytes:
                            base64_image = encode_image_bytes(image_data_for_api_bytes)
                            if base64_image:
                                image_ext = Path(found_abs_path).suffix.lower()
                                mime_type = f"image/{image_ext[1:]}" 
                                if image_ext == ".jpg": mime_type = "image/jpeg" 
                                image_id_text = '/'.join(Path(found_abs_path).parts[-2:])
                                content_list.append({"type": "input_text", "text": f"ID for the next image: {image_id_text}"})
                                if verbose: print(f"Worker {worker_id}: Adding image to API: {image_id_text} (Orig: {found_abs_path}, Final API size: {len(image_data_for_api_bytes) / (1024*1024):.2f}MB)")
                                content_list.append({"type": "input_image", "image_url": f"data:{mime_type};base64,{base64_image}"})
                                added_figures_count += 1
                                queued_image_abs_paths.add(str(found_abs_path))
                            elif verbose: _print_method(f"Worker {worker_id}: Failed to base64 encode image: {found_abs_path}")
                        elif verbose and initial_file_size > MAX_IMAGE_SIZE_BYTES : # Only log if it was supposed to be resized but failed
                            _print_method(f"Worker {worker_id}: No image data obtained for '{found_abs_path}' after processing (e.g. resize failed).")
                    elif verbose:
                        _print_method(f"Worker {worker_id}: Could not find local file for MD path: '{raw_path}' (decoded: '{decoded_path_str}')")
                
                if verbose:
                    if not added_figures_count and all_matches: _print_method(f"Worker {worker_id}: Found MD image refs, but none added to API (check logs).")
                    elif not all_matches: _print_method(f"Worker {worker_id}: No image references found in MD.")
            elif Image is None and (max_figures is None or max_figures > 0):
                if verbose: _print_method(f"Worker {worker_id}: Pillow library not available. Skipping image processing.")

            if verbose: print(f"Worker {worker_id}: Preparing to send request to Azure OpenAI for {markdown_file_path}...")
            
            # Completion Arguments for Azure API Call
            completion_args = {
                "model": deployment_name,
                "input": [{"role": "user", "content": content_list}],
                "timeout": 600.0,
            }
            
            if max_tokens_completion is not None: completion_args["max_tokens"] = max_tokens_completion
                
            response_obj = None; last_exception = None
            for attempt in range(MAX_RETRIES):
                try:
                    if verbose: print(f"Worker {worker_id}: Attempt {attempt + 1}/{MAX_RETRIES} to call API for {markdown_file_path}")
                    response_obj = openai_client.responses.create(**completion_args)
                    if verbose: print(f"Worker {worker_id}: API call successful on attempt {attempt + 1} for {markdown_file_path}")
                    break 
                except Exception as e:
                    last_exception = e 
                    if _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE and isinstance(e, _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE):
                        wait_time = INITIAL_BACKOFF_SECONDS * (2 ** attempt)
                        _print_method(f"Worker {worker_id}: API attempt {attempt + 1} failed (retryable: {type(e).__name__}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        _print_method(f"Worker {worker_id}: API attempt {attempt + 1} failed (non-retryable: {type(e).__name__} - {e}).")
                        break 
            
            if response_obj is None: 
                err_msg = f"All API call attempts failed for {markdown_file_path}."
                if last_exception: err_msg += f" Last error: {type(last_exception).__name__} - {last_exception}"
                _print_method(f"Worker {worker_id}: {err_msg}")
                raise last_exception if last_exception else Exception(err_msg)

            if response_obj.usage:
                current_tokens_used = response_obj.usage.total_tokens or 0
                if verbose: print(f"Worker {worker_id}: API call for {markdown_file_path} successful. Tokens: {current_tokens_used} (P: {response_obj.usage.prompt_tokens}, C: {response_obj.usage.completion_tokens}).")
            elif verbose: print(f"Worker {worker_id}: API call for {markdown_file_path} successful. Token usage not available.")

            review_json_content = response_obj.output[0].content
            try:
                review_json_content_parsed = json.loads(review_json_content)
            except json.JSONDecodeError as e:
                if verbose: print("Initial JSON parse failed:", e); print("Attempting to sanitize the JSON...")

                cleaned_json = _sanitize_json_string(review_json_content)

                try:
                    review_json_content_parsed = json.loads(cleaned_json)
                except json.JSONDecodeError as e2:
                    if verbose: print("Sanitized JSON also failed:", e2); print(f"\n\nRAW JSON FROM LLM:\n{review_json_content}\n")
                    
                    # in case of json parsing error, store error in the json so that we can check problematic jsons later
                    review_json_content_parsed = {"error": "Error serializing json", "error_in_prompt": prompt_dict["name"]}

            # First prompt is main prompt so we cretea the json respnse
            # For other prompts we just appent to the checklists
            if prompt_dict["name"] == "main":
                json_response = review_json_content_parsed
                json_response["checklists"] = []
            else:
                json_response["checklists"].append(review_json_content_parsed)
        
        # Commented the html generation part
        # parsed_review = ChecklistReview.model_validate_json(review_json_content)
        # Generate the rich HTML output from the parsed object
        # final_html_content = generate_html_from_review(parsed_review)
        
        # Save the raw JSON
        output_json_file_path = Path(output_base_dir) / unique_paper_identifier / f"{unique_paper_identifier}_checklist.json"
        output_json_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(json_response, f, ensure_ascii=False, indent=2)

        paper_output_dir_path = Path(output_base_dir) / unique_paper_identifier
        paper_output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Commented the html generation part
        # output_file_path = paper_output_dir_path / f"{unique_paper_identifier}_checklist.html"
        # with open(output_file_path, 'w', encoding='utf-8') as f:
        #     f.write(final_html_content)
            
        return f"Successfully processed {markdown_file_path}", current_tokens_used

    except FileNotFoundError:
        message = f"Worker {worker_id}: ERROR - MD file not found: {markdown_file_path}"
        _print_method(message); return message, 0
    except Exception as e: 
        message = f"Worker {worker_id}: ERROR processing {markdown_file_path}: {type(e).__name__} - {e}"
        if _print_method == print: _print_method(message, flush=True)
        else: _print_method(message)
        import traceback; traceback.print_exc() 
        return message, 0


def _sanitize_json_string(json_str):
    # Remove trailing commas before object/array close
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)

    # Escape single backslashes (e.g., \gamma -> \\gamma)
    json_str = json_str.replace('\\', '\\\\')

    # Optionally: remove leading markdown/code block markers
    json_str = json_str.strip().strip("```json").strip("```")

    return json_str

def print_debug_timer_report(duration_sec, total_tokens, cli_args, num_processed_successfully, num_files_attempted):
    _print_method = tqdm.write if not cli_args.verbose and num_files_attempted > 0 else print
    _print_method("\n--- Debug Timer Report ---")
    _print_method(f"Total files attempted: {num_files_attempted}")
    _print_method(f"Successfully processed: {num_processed_successfully}")
    _print_method(f"Total wall-clock time: {duration_sec:.2f} seconds")
    _print_method(f"Total Azure tokens consumed: {total_tokens}")
    if num_processed_successfully > 0 and total_tokens > 0 and duration_sec > 0:
        TARGET_TPM = 1000000.0; duration_min = duration_sec / 60.0
        w_effective = cli_args.max_workers if cli_args.max_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        _print_method(f"Effective max_workers: {w_effective}")
        current_tpm_observed = total_tokens / duration_min
        _print_method(f"Observed TPM: {current_tpm_observed:.2f}")
        if current_tpm_observed > 0 and w_effective > 0:
            sugg_w_float = (TARGET_TPM / current_tpm_observed) * w_effective
            sugg_w_int = math.ceil(sugg_w_float)
            _print_method(f"To reach ~{TARGET_TPM:.0f} TPM, suggested max_workers: {sugg_w_int} (Calc: ({TARGET_TPM:.0f}/{current_tpm_observed:.2f})*{w_effective}={sugg_w_float:.2f})")
        else: _print_method("Cannot suggest max_workers (current TPM or effective workers is zero).")
    else: _print_method("Not enough data for max_workers suggestion.")
    _print_method("--------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Markdown files with Azure OpenAI.")
    parser.add_argument("--input_dir", type=str, default=".", help="Base input directory.")
    parser.add_argument("--output_dir", type=str, default="./html_outputs/", help="Output directory for HTMLs.")
    parser.add_argument("--deployment_name", type=str, required=True, help="Azure OpenAI deployment name.")
    parser.add_argument("--max_tokens_completion", type=int, default=None, help="Max tokens for completion.")
    parser.add_argument("--file_pattern", type=str, default="*/structured_paper_output/paper.md", help="Glob pattern for markdown files.")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads.")
    parser.add_argument("--max_figures", type=int, default=10, help="Max figures to process per paper (0 for none).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug_timer", action="store_true", help="Enable performance timer and TPM suggestion.")
    args = parser.parse_args()

    if Image is None: print("INFO: Pillow (PIL) not installed. Image resizing will be skipped.")
    if not all([KEY, API_VERSION, ENDPOINT, args.deployment_name]):
        print("Error: Missing Azure credentials or deployment name."); exit(1)
    try: azure_openai_client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY, api_version=API_VERSION)
    except Exception as e: print(f"Error initializing AzureOpenAI client: {e}"); exit(1)
    
    search_pattern = args.file_pattern if os.path.isabs(args.file_pattern) else os.path.join(args.input_dir, args.file_pattern)
    if args.verbose: print(f"Searching for MD files: {os.path.abspath(search_pattern)}")
    markdown_files_to_process = glob.glob(search_pattern)

    if not markdown_files_to_process: print(f"No MD files found: {search_pattern}")
    elif args.verbose:
        print(f"Found {len(markdown_files_to_process)} MD files:")
        for i, fp in enumerate(markdown_files_to_process): print(f"  {i+1}. {os.path.abspath(fp)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processed_count, error_count, total_tokens_for_debug = 0, 0, 0
    overall_start_time = time.time() if args.debug_timer else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(process_markdown_file, md, args.output_dir, azure_openai_client, 
                            args.deployment_name, args.max_tokens_completion, args.verbose, args.max_figures): md
            for md in markdown_files_to_process
        }
        iterable_futures = concurrent.futures.as_completed(future_to_file)
        if markdown_files_to_process:
            iterable_futures = tqdm(iterable_futures, total=len(markdown_files_to_process), 
                                    desc="Processing files", disable=args.verbose)
        for future in iterable_futures:
            file_path, tokens_this_file = future_to_file[future], 0
            try:
                result_message, tokens_this_file = future.result()
                if "Successfully processed" in result_message: processed_count += 1
                else: error_count += 1
                if args.verbose: print(f"Main: Result for {os.path.abspath(file_path)}: {result_message} (Tokens: {tokens_this_file})")
            except Exception as exc:
                error_count += 1
                _pm = tqdm.write if not args.verbose and markdown_files_to_process else print
                _pm(f"Main: Unhandled exception for {os.path.abspath(file_path)}: {type(exc).__name__} - {exc}")
            if args.debug_timer: total_tokens_for_debug += tokens_this_file
    
    if args.debug_timer:
        print_debug_timer_report(time.time() - overall_start_time, total_tokens_for_debug, 
                                 args, processed_count, len(markdown_files_to_process))
    
    _pm_final = tqdm.write if not args.verbose and markdown_files_to_process else print
    _pm_final(f"\n--- Processing Complete ---")
    _pm_final(f"Total files found/attempted: {len(markdown_files_to_process)}")
    _pm_final(f"Successfully processed: {processed_count}")
    _pm_final(f"Failed attempts/errors: {error_count}")
    _pm_final(f"---------------------------\n")