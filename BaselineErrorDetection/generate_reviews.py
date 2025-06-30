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

# Import the new prompts and Pydantic model
from review_prompts import NeurIPSReview, ReviewerPrompts


# --- Constants and Configuration ---
CONTEXT_LENGTH_LIMIT = 160000  # The context limit of the model being used
TOKEN_BUFFER = 10000 # Buffer for system prompts, response format, etc.
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # API limit: 20MB
TARGET_RESIZE_BYTES = 10 * 1024 * 1024 # Softer target for resizing
MAX_RESIZE_ATTEMPTS = 4
MIN_DIMENSION_AFTER_RESIZE = 50

# --- Helper Functions ---

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

def encode_image_bytes(image_bytes):
    """Encodes image bytes to a base64 string."""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error base64 encoding image bytes: {e}")
        return None

def _sanitize_json_string(json_str):
    """A simple function to clean common JSON errors from LLM output."""
    # Remove trailing commas before object/array close
    json_str = re.sub(r',\s*(?=[}\]])', '', json_str)
    # Remove code block markers
    json_str = json_str.strip().strip("```json").strip("```")
    return json_str

# --- Environment & API Configuration ---
load_dotenv()

KEY = os.environ.get("KEY")
API_VERSION = os.environ.get("API_VERSION")
ENDPOINT = os.environ.get("ENDPOINT")

# --- Exception Handling for Retries ---
_RETRYABLE_EXCEPTIONS = (APIError, RateLimitError, APIConnectionError, InternalServerError)


def process_markdown_file(
    markdown_file_path,
    output_base_dir,
    openai_client,
    deployment_name,
    max_tokens_completion,
    verbose,
    max_figures,
    original_data_dir # New argument for original data path
):
    """
    Processes a single markdown paper, generating one comprehensive review.
    """
    worker_id = concurrent.futures.thread.get_ident() if hasattr(concurrent.futures.thread, 'get_ident') else os.getpid()
    total_tokens_for_paper = 0
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print

    try:
        if verbose:
            print(f"Worker {worker_id}: Starting to process {markdown_file_path}")

        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            input_markdown_content = f.read()

        # --- Intelligent Path Resolution ---
        markdown_dir = Path(markdown_file_path).parent
        # From flawed structure: .../{status}/{paper_id}/flawed_papers/{file}.md
        # paper_folder_level_dir is the directory named with the paper_id
        paper_folder_level_dir = markdown_dir.parent
        unique_paper_identifier = paper_folder_level_dir.name
        
        # status_dir is the parent of the paper_id dir, e.g., 'accepted' or 'rejected'
        status_dir = paper_folder_level_dir.parent
        status = status_dir.name

        # Construct the base path to where the original paper's content (and figures) are stored.
        # e.g., {original_data_dir}/{status}/{paper_id}/structured_paper_output/
        original_paper_content_dir = Path(original_data_dir) / status / unique_paper_identifier / "structured_paper_output"
        
        if verbose:
            print(f"Worker {worker_id}: Paper ID: {unique_paper_identifier}, Status: {status}")
            print(f"Worker {worker_id}: Constructed path for original figures: {original_paper_content_dir}")


        openreview_id_for_link = unique_paper_identifier.split('_')[0]
        paper_content_for_prompt = f"Paper link: https://openreview.net/forum?id={openreview_id_for_link}\n\n" + input_markdown_content
        paper_max_tokens = CONTEXT_LENGTH_LIMIT - TOKEN_BUFFER
        paper_content_for_prompt = truncate_text(paper_content_for_prompt, paper_max_tokens)

        # --- Image Processing Logic (Now uses original_paper_content_dir) ---
        content_list_images = []
        added_figures_count = 0
        if (max_figures is None or max_figures > 0) and Image is not None:
            md_img_patterns = [
                r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", 
                r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>",
            ]
            valid_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
            
            all_matches = []
            for pattern_str in md_img_patterns:
                for match in re.finditer(pattern_str, input_markdown_content, flags=re.IGNORECASE | re.DOTALL):
                    all_matches.append({'match': match, 'start_pos': match.start()})
            all_matches.sort(key=lambda x: x['start_pos'])

            if verbose and all_matches:
                _print_method(f"Worker {worker_id}: Found {len(all_matches)} potential image references in MD.")
            
            queued_image_abs_paths = set()

            for item in all_matches:
                if max_figures is not None and added_figures_count >= max_figures:
                    if verbose: _print_method(f"Worker {worker_id}: Reached max_figures ({max_figures}) for API.")
                    break 
                
                raw_path = item['match'].group(1)
                try:
                    decoded_path_str = urllib.parse.unquote(raw_path)
                except Exception as e_dec:
                    if verbose: _print_method(f"Worker {worker_id}: Could not decode path '{raw_path}': {e_dec}. Skipping.")
                    continue
                
                if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']:
                    if verbose: _print_method(f"Worker {worker_id}: Skipping web URL: {decoded_path_str}")
                    continue

                # Resolve the image path against the *original* paper's directory
                found_abs_path = None
                candidate_path_obj = (original_paper_content_dir / decoded_path_str).resolve()
                if candidate_path_obj.is_file() and candidate_path_obj.suffix.lower() in valid_image_extensions:
                    found_abs_path = candidate_path_obj
                elif not candidate_path_obj.suffix: 
                    for ext_ in valid_image_extensions:
                        path_with_ext = (original_paper_content_dir / (decoded_path_str + ext_)).resolve()
                        if path_with_ext.is_file():
                            found_abs_path = path_with_ext
                            break
                
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
                                if current_data_size_for_scaling <= MAX_IMAGE_SIZE_BYTES:
                                    img_byte_buffer_final_check = io.BytesIO()
                                    save_params = {'format': original_format}
                                    if original_format.upper() == 'JPEG': save_params['quality'] = 85
                                    elif original_format.upper() == 'WEBP': save_params['quality'] = 80
                                    current_pil_img_to_resize.save(img_byte_buffer_final_check, **save_params)
                                    if img_byte_buffer_final_check.tell() <= MAX_IMAGE_SIZE_BYTES:
                                        image_data_for_api_bytes = img_byte_buffer_final_check.getvalue()
                                    break

                                scale_factor = math.sqrt(TARGET_RESIZE_BYTES / current_data_size_for_scaling) if current_data_size_for_scaling > 0 else 0.5
                                scale_factor = max(0.1, min(scale_factor, 0.95))

                                new_width = int(current_pil_img_to_resize.width * scale_factor)
                                new_height = int(current_pil_img_to_resize.height * scale_factor)

                                if new_width < MIN_DIMENSION_AFTER_RESIZE or new_height < MIN_DIMENSION_AFTER_RESIZE:
                                    if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' dimensions too small. Stopping resize.")
                                    break 

                                if verbose: _print_method(f"Worker {worker_id}: Resize attempt {attempt + 1}/{MAX_RESIZE_ATTEMPTS} for '{found_abs_path}'. Scale: {scale_factor:.2f} -> {new_width}x{new_height}.")
                                
                                resized_pil_img = current_pil_img_to_resize.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                img_byte_buffer = io.BytesIO()
                                save_params = {'format': original_format}
                                if original_format.upper() == 'JPEG': save_params['quality'] = 85
                                elif original_format.upper() == 'WEBP': save_params['quality'] = 80
                                
                                resized_pil_img.save(img_byte_buffer, **save_params)
                                resized_bytes_in_buffer = img_byte_buffer.tell()
                                current_data_size_for_scaling = resized_bytes_in_buffer

                                if verbose: _print_method(f"Worker {worker_id}: Resized '{found_abs_path}' (attempt {attempt+1}), new size: {resized_bytes_in_buffer / (1024*1024):.2f}MB.")

                                if resized_bytes_in_buffer <= MAX_IMAGE_SIZE_BYTES:
                                    image_data_for_api_bytes = img_byte_buffer.getvalue()
                                    if verbose: _print_method(f"Worker {worker_id}: Image '{found_abs_path}' successfully resized.")
                                    break 
                                else:
                                    current_pil_img_to_resize = resized_pil_img
                            
                            if image_data_for_api_bytes is None:
                                if verbose: _print_method(f"Worker {worker_id}: Skipping image '{found_abs_path}' after failing to resize.")
                                continue
                    
                    except OSError as e_stat:
                        if verbose: _print_method(f"Worker {worker_id}: OSError processing image '{found_abs_path}': {e_stat}. Skipping.")
                        continue
                    except Exception as e_img_proc:
                        if verbose: _print_method(f"Worker {worker_id}: Error with Pillow for image '{found_abs_path}': {e_img_proc}. Skipping.")
                        continue

                    if image_data_for_api_bytes:
                        base64_image = encode_image_bytes(image_data_for_api_bytes)
                        if base64_image:
                            image_ext = found_abs_path.suffix.lower()
                            mime_type = f"image/{image_ext[1:]}" 
                            if image_ext == ".jpg": mime_type = "image/jpeg" 
                            
                            content_list_images.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                            added_figures_count += 1
                            queued_image_abs_paths.add(str(found_abs_path))
                            if verbose: _print_method(f"Worker {worker_id}: Successfully added image: {found_abs_path}")
                elif verbose:
                    _print_method(f"Worker {worker_id}: Could not find local file for MD path: '{raw_path}' (decoded: '{decoded_path_str}')")
        
        # --- Generate One Comprehensive Review ---
        if verbose:
            print(f"Worker {worker_id}: Generating comprehensive review...")

        review_schema = NeurIPSReview.model_json_schema()
        final_prompt = ReviewerPrompts.get_review_prompt(
            paper_content=paper_content_for_prompt,
            json_schema=review_schema
        )

        content_list = [{"type": "text", "text": final_prompt}] + content_list_images

        completion_args = {
            "model": deployment_name,
            "messages": [{"role": "user", "content": content_list}],
            "response_format": {"type": "json_object"},
            "timeout": 600.0,
        }
        if max_tokens_completion is not None:
            completion_args["max_completion_tokens"] = max_tokens_completion # Corrected parameter name

        response_obj = None
        last_exception = None
        for attempt in range(MAX_RETRIES):
            try:
                if verbose: print(f"Worker {worker_id}: Attempt {attempt + 1}/{MAX_RETRIES} to call API for review")
                response_obj = openai_client.chat.completions.create(**completion_args)
                if verbose: print(f"Worker {worker_id}: API call successful for review")
                break
            except Exception as e:
                last_exception = e
                if isinstance(e, _RETRYABLE_EXCEPTIONS):
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
            total_tokens_for_paper = response_obj.usage.total_tokens or 0
            if verbose: print(f"Worker {worker_id}: Tokens for review: {total_tokens_for_paper}")

        review_json_content = response_obj.choices[0].message.content

        try:
            parsed_review = NeurIPSReview.model_validate_json(review_json_content)
            final_json_output = parsed_review.model_dump()
        except Exception as e:
            _print_method(f"Worker {worker_id}: CRITICAL - Pydantic validation failed. Error: {e}")
            _print_method(f"Worker {worker_id}: Attempting to save raw, sanitized JSON as fallback.")
            try:
                final_json_output = json.loads(_sanitize_json_string(review_json_content))
                final_json_output["__pydantic_validation_error"] = str(e)
            except json.JSONDecodeError as json_e:
                _print_method(f"Worker {worker_id}: CRITICAL - Fallback JSON parsing also failed. Error: {json_e}")
                final_json_output = {
                    "error": "Failed to parse or validate JSON from LLM.",
                    "raw_content": review_json_content,
                }

        output_json_file_path = Path(output_base_dir) / unique_paper_identifier / f"{unique_paper_identifier}_{Path(markdown_file_path).stem}_review.json"
        output_json_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, ensure_ascii=False, indent=2)

        if verbose:
            _print_method(f"Worker {worker_id}: Successfully saved comprehensive review to {output_json_file_path}")

        return f"Successfully processed {markdown_file_path}", total_tokens_for_paper

    except FileNotFoundError:
        message = f"Worker {worker_id}: ERROR - MD file not found: {markdown_file_path}"
        _print_method(message)
        return message, 0
    except Exception as e:
        message = f"Worker {worker_id}: FATAL ERROR processing {markdown_file_path}: {type(e).__name__} - {e}"
        _print_method(message)
        import traceback
        traceback.print_exc()
        return message, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NeurIPS-style reviews for Markdown papers using Azure OpenAI.")
    parser.add_argument("--input_dir", type=str, default=".", help="Base input directory for the flawed papers.")
    parser.add_argument("--output_dir", type=str, default="./json_reviews/", help="Output directory for review JSON files.")
    # New required argument for the original data path
    parser.add_argument("--original_data_dir", type=str, required=True, help="Base directory of the original papers containing the 'figures' folders (e.g., './data/content/parsed_arxiv/ICLR2024_latest').")
    parser.add_argument("--deployment_name", type=str, required=True, help="Azure OpenAI deployment name.")
    parser.add_argument("--max_tokens_completion", type=int, default=4096, help="Max tokens for completion.")
    parser.add_argument("--file_pattern", type=str, default="**/*.md", help="Glob pattern for markdown files within the input_dir.")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads.")
    parser.add_argument("--max_figures", type=int, default=10, help="Max figures to process per paper (0 for none).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    if not all([KEY, API_VERSION, ENDPOINT, args.deployment_name]):
        print("Error: Missing Azure credentials or deployment name in .env file or environment variables."); exit(1)
    try:
        azure_openai_client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY, api_version=API_VERSION)
    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {e}"); exit(1)

    search_pattern = os.path.join(args.input_dir, args.file_pattern)
    markdown_files = glob.glob(search_pattern, recursive=True)

    if not markdown_files:
        print(f"No markdown files found matching pattern: {search_pattern}")
        exit(0)
    
    print(f"Found {len(markdown_files)} papers to review.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processed_count, error_count = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_markdown_file, md, args.output_dir, azure_openai_client,
                args.deployment_name, args.max_tokens_completion, args.verbose, 
                args.max_figures, args.original_data_dir # Pass the new argument
            ): md for md in markdown_files
        }
        
        progress = tqdm(concurrent.futures.as_completed(future_to_file), total=len(markdown_files), desc="Reviewing Papers")
        for future in progress:
            try:
                result_message, _ = future.result()
                if "Successfully processed" in result_message:
                    processed_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                error_count += 1
                tqdm.write(f"A paper review process generated an unhandled exception: {exc}")

    print("\n--- Processing Complete ---")
    print(f"Total papers found: {len(markdown_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Errored: {error_count}")
    print("---------------------------\n")
