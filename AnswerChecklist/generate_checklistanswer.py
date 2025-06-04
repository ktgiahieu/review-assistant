import os
import glob
import base64
import argparse
import re # Keep re if ChecklistPrompt or other parts might use it, otherwise can remove
import concurrent.futures
import time # Added for sleep in retry logic
import math # Added for math.ceil in debug timer
import urllib.parse # Added for URL parsing and unquoting
from pathlib import Path # Added for path operations
import io # Added for in-memory byte streams for resized images

try:
    from PIL import Image # Added for image resizing
    # Pillow's LANCZOS is now Resampling.LANCZOS
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image # For older Pillow versions
except ImportError:
    print("ERROR: Pillow library not found. Please install it (pip install Pillow) to use image resizing features.")
    Image = None # Set to None so we can check its availability

from tqdm import tqdm # Added for progress bar
from openai import AzureOpenAI, APIError, Timeout, RateLimitError, APIConnectionError, InternalServerError # Added specific exceptions
from checklist_prompt import ChecklistPrompt # Assuming this contains ChecklistPrompt.generate_prompt_for_paper()
from dotenv import load_dotenv

# For HTML parsing in validation
from bs4 import BeautifulSoup, NavigableString


# Load environment variables
load_dotenv()

KEY = os.environ.get("KEY")
API_VERSION = os.environ.get("API_VERSION")
LOCATION = os.environ.get("LOCATION") # Not directly used in client init, but good practice
ENDPOINT = os.environ.get("ENDPOINT")
# DEPLOYMENT_NAME is now an arg

# --- Pre-filter OpenAI Exception classes for robust retry handling ---
_POTENTIAL_OPENAI_RETRY_EXCEPTIONS = [
    APIError, Timeout, RateLimitError, APIConnectionError, InternalServerError
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
MAX_RESIZE_ATTEMPTS = 4 
MIN_DIMENSION_AFTER_RESIZE = 50 

# HTML Checklist Validation Parameters
MAX_HTML_GENERATION_RETRIES = 2 # Number of times to retry HTML generation if validation fails
MIN_VALID_ITEM_PERCENTAGE_THRESHOLD = 0.75 # e.g., 75% of expected items must be found

def get_max_items_per_category_for_validation():
    """
    Calculates the maximum number of questions for each category for validation.
    This is similar to the one in checklist_parser_to_csv but used internally here.
    """
    max_items = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    # Check if ChecklistPrompt and its lists are available (imported from checklist_prompt.py)
    if hasattr(ChecklistPrompt, 'compliance_list') and \
       hasattr(ChecklistPrompt, 'contribution_list') and \
       hasattr(ChecklistPrompt, 'soundness_list') and \
       hasattr(ChecklistPrompt, 'presentation_list'):
        
        question_marker = "[ ]" # Each question starts with this in the template lists
        try:
            max_items['A'] = ChecklistPrompt.compliance_list.count(question_marker)
            max_items['B'] = ChecklistPrompt.contribution_list.count(question_marker)
            max_items['C'] = ChecklistPrompt.soundness_list.count(question_marker)
            max_items['D'] = ChecklistPrompt.presentation_list.count(question_marker)
        except Exception as e:
            print(f"Warning: Error counting max items from ChecklistPrompt lists: {e}. Max item counts for validation might be zero.")
            return {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    else:
        print("Warning: ChecklistPrompt lists not found. Max item counts for validation will be zero.")
    return max_items

MAX_ITEMS_PER_CATEGORY_VALIDATION = get_max_items_per_category_for_validation()


def validate_generated_checklist_html(html_content, verbose=False):
    """
    Validates the generated HTML checklist content.
    Checks if a reasonable number of checklist items are present for each category.
    """
    _print_method = tqdm.write if not verbose else print
    if not html_content:
        if verbose: _print_method("Validation Error: HTML content is empty.")
        return False

    soup = BeautifulSoup(html_content, 'html.parser')
    parsed_item_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    
    section_map = {
        "A. Compliance": "A", "B. Contribution": "B",
        "C. Soundness": "C", "D. Presentation": "D"
    }
    # Regex to find any checklist item marker: [+], [-], or [ ] (space)
    validation_pattern = re.compile(r"\[\s*([+ -])\s*\]\s*([ABCD]\d+)\s+.*?(?=\s*\[\s*[+ -]\s*\]\s*[ABCD]\d+|\Z)", re.DOTALL)

    all_h1_tags = soup.find_all('h1')
    current_category_char = None

    for h1_tag in all_h1_tags:
        h1_text = h1_tag.get_text(strip=True)
        is_section_header = False
        for section_title_key, category_val in section_map.items():
            if section_title_key in h1_text:
                current_category_char = category_val
                is_section_header = True
                break
        if not is_section_header: current_category_char = None; continue

        if current_category_char:
            for sibling in h1_tag.next_siblings:
                if sibling.name == 'h1': break 
                text_content = ""
                if isinstance(sibling, NavigableString): text_content = str(sibling).strip()
                elif sibling.name == 'p': text_content = sibling.get_text(strip=True)
                
                if text_content:
                    for _ in re.finditer(validation_pattern, text_content): # We just need to count matches
                        parsed_item_counts[current_category_char] += 1
    
    is_valid = True
    total_items_found_overall = 0
    for category, count in parsed_item_counts.items():
        total_items_found_overall += count
        max_expected = MAX_ITEMS_PER_CATEGORY_VALIDATION.get(category, 0)
        if max_expected > 0: # Only validate if we expect items
            if count == 0:
                if verbose: _print_method(f"Validation Error: Category {category} has 0 items found. Expected ~{max_expected}.")
                is_valid = False
            elif count < max_expected * MIN_VALID_ITEM_PERCENTAGE_THRESHOLD:
                if verbose: _print_method(f"Validation Error: Category {category} has only {count} items. Expected at least {int(max_expected * MIN_VALID_ITEM_PERCENTAGE_THRESHOLD)} (which is {MIN_VALID_ITEM_PERCENTAGE_THRESHOLD*100}% of {max_expected}).")
                is_valid = False
    
    if total_items_found_overall == 0 and any(MAX_ITEMS_PER_CATEGORY_VALIDATION.values()):
        if verbose: _print_method("Validation Error: No checklist items found at all across all categories.")
        is_valid = False # If we expect items but find none at all, it's invalid.

    if is_valid and verbose:
        _print_method(f"HTML Checklist Validation Passed. Parsed item counts: {parsed_item_counts}")
    return is_valid


def encode_image_bytes(image_bytes):
    """Encodes image bytes to a base64 string."""
    try:
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print(f"Error base64 encoding image bytes: {e}") 
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
    # Network/API level retries
    API_CALL_MAX_RETRIES = 3 # Renamed from MAX_RETRIES to distinguish
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
        current_prompt = ChecklistPrompt.generate_prompt_for_paper(paper_content_for_prompt)
        content_list = [{"type": "text", "text": current_prompt}]

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
                except Exception: continue # Skip if decode fails
                if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']: continue

                found_abs_path = None
                # ... (image path resolution logic - kept concise for brevity, assume it's the same as before) ...
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
                    if str(found_abs_path) in queued_image_abs_paths: continue
                    image_data_for_api_bytes = None
                    try:
                        initial_file_size = found_abs_path.stat().st_size
                        if initial_file_size <= MAX_IMAGE_SIZE_BYTES:
                            with open(found_abs_path, "rb") as f_img: image_data_for_api_bytes = f_img.read()
                        else: # Image is too large, needs resizing
                            pil_img_original = Image.open(found_abs_path)
                            original_format = pil_img_original.format or Path(found_abs_path).suffix[1:].upper()
                            if not original_format or original_format.lower() not in ['jpeg', 'png', 'webp', 'gif']:
                                original_format = 'PNG' if original_format and original_format.lower() == 'png' else 'JPEG'
                            if pil_img_original.is_animated and original_format.upper() == 'GIF': pil_img_original.seek(0)
                            current_pil_img_to_resize = pil_img_original
                            current_data_size_for_scaling = initial_file_size
                            for _ in range(MAX_RESIZE_ATTEMPTS): # Renamed attempt to _
                                if current_data_size_for_scaling <= MAX_IMAGE_SIZE_BYTES: break # Should be caught earlier or by previous successful resize
                                scale_factor = math.sqrt(TARGET_RESIZE_BYTES / current_data_size_for_scaling) if current_data_size_for_scaling > 0 else 0.5
                                scale_factor = max(0.1, min(scale_factor, 0.95))
                                new_width = int(current_pil_img_to_resize.width * scale_factor)
                                new_height = int(current_pil_img_to_resize.height * scale_factor)
                                if new_width < MIN_DIMENSION_AFTER_RESIZE or new_height < MIN_DIMENSION_AFTER_RESIZE: break 
                                resized_pil_img = current_pil_img_to_resize.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                img_byte_buffer = io.BytesIO()
                                save_params = {'format': original_format}
                                if original_format.upper() == 'JPEG': save_params['quality'] = 85
                                elif original_format.upper() == 'WEBP': save_params['quality'] = 80
                                resized_pil_img.save(img_byte_buffer, **save_params)
                                resized_bytes_in_buffer = img_byte_buffer.tell()
                                current_data_size_for_scaling = resized_bytes_in_buffer
                                if resized_bytes_in_buffer <= MAX_IMAGE_SIZE_BYTES:
                                    image_data_for_api_bytes = img_byte_buffer.getvalue(); break
                                else: current_pil_img_to_resize = resized_pil_img
                            if image_data_for_api_bytes is None: continue # Skip if still too large
                    except Exception: continue # Skip image on any processing error

                    if image_data_for_api_bytes:
                        base64_image = encode_image_bytes(image_data_for_api_bytes)
                        if base64_image:
                            image_ext = Path(found_abs_path).suffix.lower()
                            mime_type = f"image/{image_ext[1:]}" 
                            if image_ext == ".jpg": mime_type = "image/jpeg" 
                            image_id_text = '/'.join(Path(found_abs_path).parts[-2:])
                            content_list.append({"type": "text", "text": f"ID for the next image: {image_id_text}"})
                            content_list.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                            added_figures_count += 1
                            queued_image_abs_paths.add(str(found_abs_path))
        elif Image is None and (max_figures is None or max_figures > 0):
            if verbose: _print_method(f"Worker {worker_id}: Pillow library not available. Skipping image processing.")

        if verbose: print(f"Worker {worker_id}: Preparing to send request to Azure OpenAI for {markdown_file_path}...")
        completion_args = {"model": deployment_name, "messages": [{"role": "user", "content": content_list}]}
        if max_tokens_completion is not None: completion_args["max_tokens"] = max_tokens_completion
            
        response_obj = None; last_exception = None
        review_content = None

        # Loop for HTML generation retries (if validation fails)
        for html_gen_attempt in range(MAX_HTML_GENERATION_RETRIES + 1): # +1 to allow initial attempt
            # Loop for API call retries (network errors, etc.)
            for api_call_attempt in range(API_CALL_MAX_RETRIES):
                try:
                    if verbose: print(f"Worker {worker_id}: API Call Attempt {api_call_attempt + 1}/{API_CALL_MAX_RETRIES} (HTML Gen Attempt {html_gen_attempt +1}/{MAX_HTML_GENERATION_RETRIES+1}) for {markdown_file_path}")
                    response_obj = openai_client.chat.completions.create(**completion_args)
                    if verbose: print(f"Worker {worker_id}: API call successful on attempt {api_call_attempt + 1} for {markdown_file_path}")
                    last_exception = None # Reset last exception on success
                    break # Successful API call, exit this inner retry loop
                except Exception as e:
                    last_exception = e 
                    if _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE and isinstance(e, _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE):
                        wait_time = INITIAL_BACKOFF_SECONDS * (2 ** api_call_attempt)
                        _print_method(f"Worker {worker_id}: API attempt {api_call_attempt + 1} failed (retryable: {type(e).__name__}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        _print_method(f"Worker {worker_id}: API attempt {api_call_attempt + 1} failed (non-retryable: {type(e).__name__} - {e}).")
                        break # Exit inner retry loop for non-retryable API error
            
            if last_exception: # If API call loop failed
                _print_method(f"Worker {worker_id}: All API call attempts failed for {markdown_file_path}. Last error: {type(last_exception).__name__} - {last_exception}")
                raise last_exception # Propagate the error, stop HTML gen retries

            if response_obj:
                review_content = response_obj.choices[0].message.content
                if validate_generated_checklist_html(review_content, verbose):
                    if verbose: _print_method(f"Worker {worker_id}: Generated HTML checklist validated successfully for {markdown_file_path} on HTML gen attempt {html_gen_attempt+1}.")
                    break # HTML is valid, exit HTML generation retry loop
                else:
                    if verbose: _print_method(f"Worker {worker_id}: Generated HTML checklist FAILED validation for {markdown_file_path} on HTML gen attempt {html_gen_attempt+1}. Retrying HTML generation if attempts left.")
                    review_content = None # Invalidate content for next attempt
                    if html_gen_attempt < MAX_HTML_GENERATION_RETRIES:
                        time.sleep(INITIAL_BACKOFF_SECONDS) # Small delay before retrying HTML generation
                    # else: loop will end, and review_content will be None or the last invalid one
            else: # Should not happen if last_exception was not raised
                _print_method(f"Worker {worker_id}: No response object from API after retries for {markdown_file_path}. This state should not be reached if an exception occurred.")
                raise Exception(f"API call failed to produce a response for {markdown_file_path} without raising a specific exception.")


        if review_content is None: # All HTML generation attempts failed validation or API call failed
            err_msg = f"Failed to generate a valid HTML checklist for {markdown_file_path} after {MAX_HTML_GENERATION_RETRIES + 1} attempts."
            _print_method(f"Worker {worker_id}: {err_msg}")
            # If last_exception is from API call, re-raise it. Otherwise, raise a generic one.
            if last_exception and isinstance(last_exception, tuple(_VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE if _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE else [])):
                 raise last_exception
            raise Exception(err_msg)


        if response_obj and response_obj.usage: # Ensure response_obj exists before accessing usage
            current_tokens_used = response_obj.usage.total_tokens or 0
            if verbose: print(f"Worker {worker_id}: API call for {markdown_file_path} successful. Tokens: {current_tokens_used} (P: {response_obj.usage.prompt_tokens}, C: {response_obj.usage.completion_tokens}).")
        elif verbose: print(f"Worker {worker_id}: API call for {markdown_file_path} successful. Token usage not available (or response_obj was None).")

        paper_output_dir_path = Path(output_base_dir) / unique_paper_identifier
        paper_output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = paper_output_dir_path / f"{unique_paper_identifier}_checklist.html"
        with open(output_file_path, 'w', encoding='utf-8') as f: f.write(review_content) # review_content is now validated
        if verbose: print(f"Worker {worker_id}: Successfully wrote review to {output_file_path}")
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
                else: error_count += 1 # Assume if not success, it's an error reported by the function
                if args.verbose: print(f"Main: Result for {os.path.abspath(file_path)}: {result_message} (Tokens: {tokens_this_file})")
            except Exception as exc: # Catch exceptions raised from process_markdown_file
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
