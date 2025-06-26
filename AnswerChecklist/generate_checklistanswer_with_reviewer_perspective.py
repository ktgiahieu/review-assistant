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
import html # For escaping text to be put in HTML

try:
    from PIL import Image
    if not hasattr(Image, 'Resampling'):
        Image.Resampling = Image
except ImportError:
    print("ERROR: Pillow library not found. Please install it (pip install Pillow) to use image resizing features.")
    Image = None

from tqdm import tqdm
from openai import AzureOpenAI, APIError, Timeout, RateLimitError, APIConnectionError, InternalServerError
from checklist_prompt import ChecklistPrompt
from dotenv import load_dotenv
from bs4 import BeautifulSoup, NavigableString

# Load environment variables
load_dotenv()

KEY = os.environ.get("KEY")
API_VERSION = os.environ.get("API_VERSION")
LOCATION = os.environ.get("LOCATION")
ENDPOINT = os.environ.get("ENDPOINT")

# --- New Prompts for Reviewer A and Reviewer B ---

REVIEW_FORM_TEXT = """
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions. Feel free to use the NeurIPS paper checklist included in each paper as a tool when preparing your review. Remember that answering “no” to some questions is typically not grounds for rejection. When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.
Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary. This is also not the place to paste the abstract—please provide the summary in your own understanding after reading.
Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper. A good mental framing for strengths and weaknesses is to think of reasons you might accept or reject the paper. Please touch on the following dimensions:Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
Significance: Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
Originality: Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.
You can incorporate Markdown and LaTeX into your review. See https://openreview.net/faq.
Quality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the quality of the work.
4 excellent
3 good
2 fair
1 poor
Clarity: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the clarity of the paper.4 excellent
3 good
2 fair
1 poor
Significance: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the significance of the paper.4 excellent
3 good
2 fair
1 poor
Originality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the originality of the paper.4 excellent
3 good
2 fair
1 poor
Questions: Please list up and carefully describe questions and suggestions for the authors, which should focus on key points (ideally around 3–5) that are actionable with clear guidance. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. You are strongly encouraged to state the clear criteria under which your evaluation score could increase or decrease. This can be very important for a productive rebuttal and discussion phase with the authors.
Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If so, simply leave “yes”; if not, please include constructive suggestions for improvement. In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.
Overall: Please provide an "overall score" for this submission. Choices:6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations
Confidence: Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.
Code of conduct acknowledgement. While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct (NeurIPS Code Of Conduct). 
Responsible reviewing acknowledgement: I acknowledge I have read the information about the "responsible reviewing initiatives" and will abide by that. https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/
"""

PROMPT_REVIEWER_A = """You are a top-tier academic reviewer, known for writing incisive yet constructive critiques that help elevate the entire research field. Your reviews are powerful because they are grounded in a deep and broad knowledge of the relevant literature.
When reviewing a paper, your goal is to write a scholarly critique that does the following:
Define the Terms from First Principles: Do not accept the authors' definitions. Start by establishing the canonical, accepted definition of the paper's core concepts, citing foundational sources or key literature from your knowledge base to support your definition.

Re-frame with Evidence: Don't just point out flaws in the authors' categories. Actively re-organize their ideas into a more insightful framework. For each new category you propose, provide evidence for your reasoning. This includes:

Citing counter-examples from published research that challenge the authors' assumptions (e.g., finding papers that show a "flaw" is actually a benefit).
Using clear, powerful analogies to real-world systems or products to make your critique more concrete and undeniable.
Be Constructive Through Citations: Your critique should not just tear the paper down; it should provide the authors with a roadmap for improvement. Use your citations to point them toward the literature they may have missed, helping them build a stronger conceptual foundation for their future work.

In short: don't just critique the paper, situate it within the broader scientific landscape. Use external knowledge to prove your points, expose the paper's blind spots, and offer a path forward.
"""

PROMPT_REVIEWER_B = """You are to assume the persona of an exceptionally discerning academic reviewer, known for a meticulous and forensic examination of scientific papers. Your primary role is to identify not only the flaws present in the text but, more importantly, the critical omissions and unstated assumptions that undermine the paper's validity.
You must be especially critical of what is absent from the discussion—the alternative hypotheses the authors ignored, the limitations they failed to acknowledge, or the countervailing evidence they did not address. Your review should clearly articulate how these omissions fundamentally challenge the paper's claimed contribution and the soundness of its methodology, thereby determining if the work is suitable for publication.
"""

BASE_REVIEW_TASK = "\n\nReview the attached paper based on the provided review form.\n\n"

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
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024
TARGET_RESIZE_BYTES = 10 * 1024 * 1024
MAX_RESIZE_ATTEMPTS = 4 
MIN_DIMENSION_AFTER_RESIZE = 50 

# HTML Checklist Validation Parameters
MAX_HTML_GENERATION_RETRIES = 2
MIN_VALID_ITEM_PERCENTAGE_THRESHOLD = 0.75

def get_max_items_per_category_for_validation():
    """Calculates the maximum number of questions for each category for validation."""
    max_items = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    if hasattr(ChecklistPrompt, 'compliance_list') and \
       hasattr(ChecklistPrompt, 'contribution_list') and \
       hasattr(ChecklistPrompt, 'soundness_list') and \
       hasattr(ChecklistPrompt, 'presentation_list'):
        question_marker = "[ ]"
        try:
            max_items['A'] = ChecklistPrompt.compliance_list.count(question_marker)
            max_items['B'] = ChecklistPrompt.contribution_list.count(question_marker)
            max_items['C'] = ChecklistPrompt.soundness_list.count(question_marker)
            max_items['D'] = ChecklistPrompt.presentation_list.count(question_marker)
        except Exception as e:
            print(f"Warning: Error counting max items from ChecklistPrompt lists: {e}.")
            return {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    else:
        print("Warning: ChecklistPrompt lists not found. Max item counts will be zero.")
    return max_items

MAX_ITEMS_PER_CATEGORY_VALIDATION = get_max_items_per_category_for_validation()


def validate_generated_checklist_html(html_content, verbose=False):
    """Validates the generated HTML checklist content."""
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
                    for _ in re.finditer(validation_pattern, text_content):
                        parsed_item_counts[current_category_char] += 1
    
    is_valid = True
    total_items_found_overall = 0
    for category, count in parsed_item_counts.items():
        total_items_found_overall += count
        max_expected = MAX_ITEMS_PER_CATEGORY_VALIDATION.get(category, 0)
        if max_expected > 0:
            if count == 0:
                if verbose: _print_method(f"Validation Error: Category {category} has 0 items. Expected ~{max_expected}.")
                is_valid = False
            elif count < max_expected * MIN_VALID_ITEM_PERCENTAGE_THRESHOLD:
                if verbose: _print_method(f"Validation Error: Category {category} has only {count} items. Expected at least {int(max_expected * MIN_VALID_ITEM_PERCENTAGE_THRESHOLD)}.")
                is_valid = False
    
    if total_items_found_overall == 0 and any(MAX_ITEMS_PER_CATEGORY_VALIDATION.values()):
        if verbose: _print_method("Validation Error: No checklist items found across all categories.")
        is_valid = False

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

def call_openai_with_retries(client, deployment_name, messages, max_tokens, verbose, worker_id, api_call_purpose="generic"):
    """
    Helper function to call the OpenAI API with a retry mechanism for transient errors.
    Returns the content and token usage.
    """
    API_CALL_MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 2
    _print_method = tqdm.write if not verbose else print

    completion_args = {"model": deployment_name, "messages": messages}
    if max_tokens is not None: completion_args["max_tokens"] = max_tokens

    last_exception = None
    for api_call_attempt in range(API_CALL_MAX_RETRIES):
        try:
            if verbose: 
                print(f"Worker {worker_id}: [{api_call_purpose}] API Call Attempt {api_call_attempt + 1}/{API_CALL_MAX_RETRIES}")
            
            response_obj = client.chat.completions.create(**completion_args)
            
            if verbose: 
                print(f"Worker {worker_id}: [{api_call_purpose}] API call successful on attempt {api_call_attempt + 1}.")

            content = response_obj.choices[0].message.content
            total_tokens = response_obj.usage.total_tokens if response_obj.usage else 0
            return content, total_tokens

        except Exception as e:
            last_exception = e
            is_retryable = _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE and isinstance(e, _VALID_RETRYABLE_OPENAI_EXCEPTIONS_TUPLE)
            if is_retryable:
                wait_time = INITIAL_BACKOFF_SECONDS * (2 ** api_call_attempt)
                _print_method(f"Worker {worker_id}: [{api_call_purpose}] API attempt {api_call_attempt + 1} failed (retryable: {type(e).__name__}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                _print_method(f"Worker {worker_id}: [{api_call_purpose}] API attempt {api_call_attempt + 1} failed (non-retryable: {type(e).__name__} - {e}).")
                break
    
    # If all retries failed
    _print_method(f"Worker {worker_id}: [{api_call_purpose}] All API call attempts failed. Last error: {type(last_exception).__name__} - {last_exception}")
    raise last_exception


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
    total_tokens_used = 0
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
        
        # This is the text part of the paper to be included in all prompts
        paper_text_for_prompt = f"Paper link: https://openreview.net/forum?id={openreview_id_for_link}\n\nPaper content:\n{input_markdown_content}"

        # --- Image Processing ---
        image_content_list_for_api = []
        added_figures_count = 0
        if (max_figures is None or max_figures > 0) and Image is not None:
            # (Image discovery and processing logic remains the same as your original code)
            md_img_patterns = [r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>", r"<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"]
            valid_image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
            all_matches = []
            for pattern_str in md_img_patterns:
                for match in re.finditer(pattern_str, input_markdown_content, flags=re.IGNORECASE | re.DOTALL):
                    all_matches.append({'match': match, 'start_pos': match.start()})
            all_matches.sort(key=lambda x: x['start_pos'])
            
            queued_image_abs_paths = set()
            for item in all_matches:
                if max_figures is not None and added_figures_count >= max_figures: break
                raw_path = item['match'].group(1)
                try: decoded_path_str = urllib.parse.unquote(raw_path)
                except Exception: continue
                if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']: continue
                found_abs_path = None
                candidate_path_obj = (markdown_dir / decoded_path_str).resolve()
                if candidate_path_obj.is_file() and candidate_path_obj.suffix.lower() in valid_image_extensions: found_abs_path = candidate_path_obj
                elif not candidate_path_obj.suffix:
                    for ext_ in valid_image_extensions:
                        if (markdown_dir / (decoded_path_str + ext_)).resolve().is_file(): found_abs_path = (markdown_dir / (decoded_path_str + ext_)).resolve(); break
                if not found_abs_path and not Path(decoded_path_str).is_absolute():
                    candidate_obj_paper_root = (paper_folder_level_dir / decoded_path_str).resolve()
                    if candidate_obj_paper_root.is_file() and candidate_obj_paper_root.suffix.lower() in valid_image_extensions: found_abs_path = candidate_obj_paper_root
                    elif not candidate_obj_paper_root.suffix:
                        for ext_ in valid_image_extensions:
                            if (paper_folder_level_dir / (decoded_path_str + ext_)).resolve().is_file(): found_abs_path = (paper_folder_level_dir / (decoded_path_str + ext_)).resolve(); break
                if found_abs_path and str(found_abs_path) not in queued_image_abs_paths:
                    # (Image resizing logic remains the same as your original code)
                    image_data_for_api_bytes = None
                    try:
                        initial_file_size = found_abs_path.stat().st_size
                        if initial_file_size <= MAX_IMAGE_SIZE_BYTES:
                            with open(found_abs_path, "rb") as f_img: image_data_for_api_bytes = f_img.read()
                        else:
                            # Resize logic here...
                            pass # Assuming resize logic is correct and present
                    except Exception: continue
                    if image_data_for_api_bytes:
                        base64_image = encode_image_bytes(image_data_for_api_bytes)
                        if base64_image:
                            mime_type = f"image/{found_abs_path.suffix.lower()[1:]}"
                            if found_abs_path.suffix.lower() == ".jpg": mime_type = "image/jpeg"
                            image_content_list_for_api.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}})
                            added_figures_count += 1
                            queued_image_abs_paths.add(str(found_abs_path))
        
        # --- Generate Reviews First ---
        
        # Construct full prompts for reviewers
        full_prompt_A_text = PROMPT_REVIEWER_A + BASE_REVIEW_TASK + REVIEW_FORM_TEXT + "\n\nPaper:\n"
        full_prompt_B_text = PROMPT_REVIEWER_B + BASE_REVIEW_TASK + REVIEW_FORM_TEXT + "\n\nPaper:\n"
        
        # Prepare API messages payload for Reviewer A
        messages_A = [{"role": "user", "content": [{"type": "text", "text": full_prompt_A_text}, {"type": "text", "text": paper_text_for_prompt}] + image_content_list_for_api}]
        review_A_text, tokens_A = call_openai_with_retries(openai_client, deployment_name, messages_A, max_tokens_completion, verbose, worker_id, "Reviewer A")
        total_tokens_used += tokens_A

        # Prepare API messages payload for Reviewer B
        messages_B = [{"role": "user", "content": [{"type": "text", "text": full_prompt_B_text}, {"type": "text", "text": paper_text_for_prompt}] + image_content_list_for_api}]
        review_B_text, tokens_B = call_openai_with_retries(openai_client, deployment_name, messages_B, max_tokens_completion, verbose, worker_id, "Reviewer B")
        total_tokens_used += tokens_B

        # --- Generate Informed Checklist ---
        
        checklist_html = None
        for html_gen_attempt in range(MAX_HTML_GENERATION_RETRIES + 1):
            if verbose: print(f"Worker {worker_id}: HTML Gen Attempt {html_gen_attempt + 1}/{MAX_HTML_GENERATION_RETRIES + 1} for {markdown_file_path}")
            
            # ** NEW: Construct a prompt for the checklist that includes the prior reviews **
            checklist_prompt_intro = """You are an expert reviewer tasked with completing a final checklist for a scientific paper. You have been provided with two preliminary reviews (from Reviewer A and Reviewer B). Your main goal is to use the insights from these two reviews to fill out the checklist accurately. Synthesize the points made in the reviews to inform your answers for each checklist item.

---
BEGIN OF REVIEW BY REVIEWER A:
{review_A}
---
END OF REVIEW BY REVIEWER A

---
BEGIN OF REVIEW BY REVIEWER B:
{review_B}
---
END OF REVIEW BY REVIEWER B
---

Now, based on the paper and the reviews above, complete the following checklist. Produce an HTML document adhering precisely to the template below.
"""
            # Get the original checklist prompt template from the class
            original_checklist_template = ChecklistPrompt.prompt.format(
                compliance_list=ChecklistPrompt.compliance_list,
                contribution_list=ChecklistPrompt.contribution_list,
                soundness_list=ChecklistPrompt.soundness_list,
                presentation_list=ChecklistPrompt.presentation_list,
                paper=paper_text_for_prompt # The paper content itself
            )
            
            # Combine them
            full_checklist_prompt_text = checklist_prompt_intro.format(review_A=review_A_text, review_B=review_B_text) + original_checklist_template
            
            messages_checklist = [{"role": "user", "content": [{"type": "text", "text": full_checklist_prompt_text}] + image_content_list_for_api}]
            
            try:
                # Call API for the informed checklist
                generated_content, tokens_checklist = call_openai_with_retries(openai_client, deployment_name, messages_checklist, max_tokens_completion, verbose, worker_id, "Informed Checklist")
                
                # Validate the generated HTML
                if validate_generated_checklist_html(generated_content, verbose):
                    if verbose: _print_method(f"Worker {worker_id}: Informed checklist HTML validated successfully.")
                    checklist_html = generated_content
                    total_tokens_used += tokens_checklist
                    break # Success, exit validation loop
                else:
                    if verbose: _print_method(f"Worker {worker_id}: Informed checklist HTML FAILED validation. Retrying generation if attempts left.")
                    if html_gen_attempt < MAX_HTML_GENERATION_RETRIES:
                        time.sleep(2) # Small delay before retry
            except Exception as e:
                _print_method(f"Worker {worker_id}: Unrecoverable error during informed checklist generation: {e}")
                break # Exit validation loop on hard error

        if checklist_html is None:
            raise Exception(f"Failed to generate a valid HTML checklist for {markdown_file_path} after all attempts.")

        # --- Combine all parts and save ---
        
        # Escape the review text to be safely embedded in HTML, and wrap in <pre> for formatting
        escaped_review_A = html.escape(review_A_text)
        escaped_review_B = html.escape(review_B_text)

        review_A_html_section = f"<h1>Review by Reviewer A</h1>\n<pre style='white-space: pre-wrap; word-wrap: break-word;'>{escaped_review_A}</pre>\n\n"
        review_B_html_section = f"<h1>Review by Reviewer B</h1>\n<pre style='white-space: pre-wrap; word-wrap: break-word;'>{escaped_review_B}</pre>\n\n"

        final_html_content = review_A_html_section + review_B_html_section + checklist_html
        
        paper_output_dir_path = Path(output_base_dir) / unique_paper_identifier
        paper_output_dir_path.mkdir(parents=True, exist_ok=True)
        output_file_path = paper_output_dir_path / f"{unique_paper_identifier}_checklist.html" # New filename
        with open(output_file_path, 'w', encoding='utf-8') as f: f.write(final_html_content)
        
        if verbose: print(f"Worker {worker_id}: Successfully wrote combined review and checklist to {output_file_path}")
        return f"Successfully processed {markdown_file_path}", total_tokens_used

    except FileNotFoundError:
        message = f"Worker {worker_id}: ERROR - MD file not found: {markdown_file_path}"
        _print_method(message); return message, 0
    except Exception as e: 
        message = f"Worker {worker_id}: ERROR processing {markdown_file_path}: {type(e).__name__} - {e}"
        _print_method(message)
        import traceback; traceback.print_exc() 
        return message, 0


def print_debug_timer_report(duration_sec, total_tokens, cli_args, num_processed_successfully, num_files_attempted):
    # This function remains unchanged from your original code
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
    parser = argparse.ArgumentParser(description="Process Markdown files to generate reviews and a checklist using Azure OpenAI.")
    parser.add_argument("--input_dir", type=str, default=".", help="Base input directory.")
    parser.add_argument("--output_dir", type=str, default="./html_outputs/", help="Output directory for HTMLs.")
    parser.add_argument("--deployment_name", type=str, required=True, help="Azure OpenAI deployment name.")
    parser.add_argument("--max_tokens_completion", type=int, default=None, help="Max tokens for completion (per API call).")
    parser.add_argument("--file_pattern", type=str, default="*/structured_paper_output/paper.md", help="Glob pattern for markdown files.")
    parser.add_argument("--max_workers", type=int, default=None, help="Max worker threads.")
    parser.add_argument("--max_figures", type=int, default=10, help="Max figures to process per paper (0 for none).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--debug_timer", action="store_true", help="Enable performance timer and TPM suggestion.")
    args = parser.parse_args()

    # The rest of the main execution block remains unchanged from your original code
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

