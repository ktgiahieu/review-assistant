import os
import re
import csv
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString
import argparse
from dotenv import load_dotenv
from tqdm import tqdm 

load_dotenv()
# Attempt to import ChecklistPrompt.
try:
    from checklist_prompt import ChecklistPrompt
    CHECKLIST_PROMPT_AVAILABLE = True
except ImportError:
    ChecklistPrompt = None
    CHECKLIST_PROMPT_AVAILABLE = False
    print("WARNING: checklist_prompt.py not found. Advanced parsing for incomplete checklists will be limited.")

# Attempt to import OpenReview client
try:
    import openreview
    OPENREVIEW_CLIENT_AVAILABLE = True
except ImportError:
    openreview = None
    OPENREVIEW_CLIENT_AVAILABLE = False
    print("WARNING: openreview-py library not found. Human review score fetching will be disabled. Install with 'pip install openreview-py'")


score_chars_dict = {
    "+": 1,
    "-": -1,
    "?": 0,
    "!": -99
}


def get_max_items_per_category():
    """
    Calculates the maximum number of questions for each category
    by parsing the lists from ChecklistPrompt.
    """
    max_items = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    if not CHECKLIST_PROMPT_AVAILABLE or ChecklistPrompt is None: 
        return max_items 

    question_marker = "[ ]" 
    
    try:
        max_items['A'] = ChecklistPrompt.compliance_list.count(question_marker)
        max_items['B'] = ChecklistPrompt.contribution_list.count(question_marker)
        max_items['C'] = ChecklistPrompt.soundness_list.count(question_marker)
        max_items['D'] = ChecklistPrompt.presentation_list.count(question_marker)
    except AttributeError:
        print("WARNING: Could not access question lists from ChecklistPrompt. Max item counts will be zero.")
        return {'A': 0, 'B': 0, 'C': 0, 'D': 0} 
        
    return max_items

MAX_ITEMS_PER_CATEGORY = get_max_items_per_category()

def extract_score_value(value_from_content):
    """
    Extracts a numerical score from various possible formats in review content.
    Tries to get a float, falls back to string if not purely numerical.
    """
    if isinstance(value_from_content, (int, float)):
        return float(value_from_content)
    
    text_val = ""
    if isinstance(value_from_content, str):
        text_val = value_from_content
    elif isinstance(value_from_content, dict) and 'value' in value_from_content:
        text_val = str(value_from_content['value']) # Ensure it's a string to process
    
    if not text_val:
        return "N/A"

    # Try to extract leading number (e.g., "8: Excellent" -> "8", "3 good" -> "3")
    match_with_colon = re.match(r"^\s*(-?\d+(\.\d+)?)\s*:", text_val)
    if match_with_colon:
        try: return float(match_with_colon.group(1))
        except ValueError: pass # Fall through if conversion fails

    match_leading_number = re.match(r"^\s*(-?\d+(\.\d+)?)", text_val)
    if match_leading_number:
        try: return float(match_leading_number.group(1))
        except ValueError: pass # Fall through

    # If it's just a number string by itself
    try:
        return float(text_val.strip())
    except ValueError:
        return text_val.strip() # Return original stripped string if not a number

def get_human_review_scores(client, openreview_id, verbose=False):
    """
    Fetches human review scores and ethics flag from OpenReview for a given paper ID.
    Collects scores from all identified official reviews.
    """
    _print_method = tqdm.write if not verbose else print
    # Initialize with empty lists to collect multiple scores
    all_review_data = {
        'human_rating': [],
        'human_soundness': [],
        'human_contribution': [],
        'human_presentation': [],
        'ethics_flag': [] # Will store 0 or 1 for each review, then take max
    }
    if not client or not OPENREVIEW_CLIENT_AVAILABLE:
        # Return structure with empty lists if client not available
        return {k: ("N/A" if k != 'ethics_flag' else 0) for k in all_review_data.keys()}


    try:
        notes = client.get_notes(forum=openreview_id)
        # print(f"Debug: Fetched {len(notes)} notes for forum {openreview_id}") # Optional debug
        
        score_field_map = {
            'rating': 'human_rating', 'recommendation': 'human_rating',
            'soundness': 'human_soundness', 'technical_soundness': 'human_soundness',
            'contribution': 'human_contribution',
            'presentation': 'human_presentation', 'clarity': 'human_presentation'
        }
        
        reviews_processed_count = 0
        for note in notes:
            is_official_review = False
            invitation_str = ""
            if note.invitations and len(note.invitations) > 0:
                invitation_str = note.invitations[0]
            # Some older notes might have invitation in signatures (less common for api2)
            elif note.signatures and 'invitation' in note.signatures[0]: 
                invitation_str = note.signatures[0]

            if invitation_str and ('official_review' in invitation_str.lower() or '/review' in invitation_str.lower()):
                is_official_review = True
            
            # Fallback: if content has typical review fields
            if not is_official_review and isinstance(note.content, dict) and \
               any(key.lower().replace('_score', '').replace(' ', '_') in score_field_map for key in note.content.keys()):
                is_official_review = True 

            if is_official_review and isinstance(note.content, dict):
                reviews_processed_count += 1
                if verbose: _print_method(f"Debug: Processing review {note.id} for {openreview_id}. Content keys: {list(note.content.keys())}")
                
                current_review_scores_found = False
                for key, value_from_content in note.content.items():
                    normalized_key = key.lower().replace('_score', '').replace(' ', '_')
                    if normalized_key in score_field_map:
                        score_val = extract_score_value(value_from_content)
                        all_review_data[score_field_map[normalized_key]].append(score_val)
                        current_review_scores_found = True
                
                # Ethics flag processing
                ethics_flag_value = 0 # Default to 0 (not flagged)
                if 'flag_for_ethics_review' in note.content:
                    flag_content = note.content['flag_for_ethics_review']
                    # Expecting {'value': ['text']}
                    if isinstance(flag_content, dict) and 'value' in flag_content and isinstance(flag_content['value'], list) and flag_content['value']:
                        first_flag_item = str(flag_content['value'][0]).strip().lower()
                        if first_flag_item != "no ethics review needed." and first_flag_item != "no ethics review needed": # Handle with or without period
                            ethics_flag_value = 1 # Flagged
                    elif isinstance(flag_content, list) and flag_content: # If value is directly a list
                        first_flag_item = str(flag_content[0]).strip().lower()
                        if first_flag_item != "no ethics review needed." and first_flag_item != "no ethics review needed":
                             ethics_flag_value = 1
                all_review_data['ethics_flag'].append(ethics_flag_value)

                if verbose and current_review_scores_found:
                     _print_method(f"Debug: Extracted from review {note.id} for {openreview_id}: {[ (k,v) for k,v in all_review_data.items() if v]}")
        
        if reviews_processed_count == 0 and verbose:
            _print_method(f"Info: No official reviews found with score fields for {openreview_id} among {len(notes)} notes.")

    except openreview.OpenReviewException as e:
        _print_method(f"OpenReview API error for {openreview_id}: {e}")
    except Exception as e:
        _print_method(f"Unexpected error fetching human review scores for {openreview_id}: {e}")
        import traceback
        traceback.print_exc()


    # Process collected lists for final output structure
    final_output_scores = {}
    for key, score_list in all_review_data.items():
        if key == 'ethics_flag':
            final_output_scores[key] = max(score_list) if score_list else 0 # Max of flags (if any review flagged it)
        else:
            # Filter out "N/A" strings if other valid scores exist, otherwise keep "N/A" if that's all
            valid_scores = [s for s in score_list if s != "N/A"]
            if valid_scores:
                final_output_scores[key] = valid_scores
            elif score_list: # List is not empty but contains only "N/A"s
                 final_output_scores[key] = ["N/A"] # Represent as list with N/A
            else: # Empty list
                 final_output_scores[key] = ["N/A"]


    # For CSV, we'll join lists later. Function now returns dict of lists.
    return final_output_scores


def parse_html_checklist(html_filepath, verbose=False): 
    _print_method = tqdm.write if not verbose else print
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        _print_method(f"Error reading HTML file {html_filepath}: {e}")
        return None

    soup = BeautifulSoup(content, 'html.parser')
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else "N/A"
    
    initial_scores = {'A': [], 'B': [], 'C': [], 'D': []}
    section_map = {"A. Compliance": "A", "B. Contribution": "B", "C. Soundness": "C", "D. Presentation": "D"}
    question_pattern_pass1 = re.compile(r"\[\s*([+\-!?])\s*\]\s*([ABCD]\d+)\s+.*?(?=\s*\[\s*[+\-!?]\s*\]\s*[ABCD]\d+|\Z)", re.DOTALL)
    question_pattern_pass2 = re.compile(r"\[\s*([+ ])\s*\]\s*([ABCD]\d+)\s+.*?(?=\s*\[\s*[+ ]\s*\]\s*[ABCD]\d+|\Z)", re.DOTALL)

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
                if isinstance(sibling, NavigableString):
                    text_content = str(sibling).strip()
                elif sibling.name == 'p': 
                    text_content = sibling.get_text(strip=True)
                
                if text_content:
                    for match in re.finditer(question_pattern_pass1, text_content):
                        score_char = match.group(1)
                        initial_scores[current_category_char].append(score_chars_dict[score_char])
    
    final_scores = {k: list(v) for k, v in initial_scores.items()}

    if CHECKLIST_PROMPT_AVAILABLE and any(MAX_ITEMS_PER_CATEGORY.values()):
        for category_char in ['A', 'B', 'C', 'D']:
            scores_pass1 = initial_scores[category_char]
            count_plus_pass1 = scores_pass1.count(1)
            count_minus_pass1 = scores_pass1.count(-1)
            max_expected = MAX_ITEMS_PER_CATEGORY.get(category_char, 0)
            needs_second_pass = (count_minus_pass1 == 0 and count_plus_pass1 > 0 and count_plus_pass1 < max_expected and max_expected > 0)

            if needs_second_pass:
                if verbose: _print_method(f"INFO: Category {category_char} in {html_filepath.name} meets criteria for second pass parsing.")
                scores_pass2 = []
                current_category_char_for_pass2 = None
                for h1_tag_p2 in all_h1_tags:
                    h1_text_p2 = h1_tag_p2.get_text(strip=True)
                    is_target_section = False
                    for section_title_key, category_val_p2 in section_map.items():
                        if section_title_key in h1_text_p2 and category_val_p2 == category_char:
                            current_category_char_for_pass2 = category_val_p2
                            is_target_section = True; break
                    if not is_target_section: current_category_char_for_pass2 = None; continue
                    if current_category_char_for_pass2 == category_char:
                        for sibling_p2 in h1_tag_p2.next_siblings:
                            if sibling_p2.name == 'h1': break
                            text_content_p2 = ""
                            if isinstance(sibling_p2, NavigableString):
                                text_content_p2 = str(sibling_p2).strip()
                            elif sibling_p2.name == 'p':
                                text_content_p2 = sibling_p2.get_text(strip=True)
                            
                            if text_content_p2:
                                for match_p2 in re.finditer(question_pattern_pass2, text_content_p2):
                                    scores_pass2.append(1 if match_p2.group(1) == '+' else 0)
                        break 
                if len(scores_pass2) >= len(scores_pass1) and len(scores_pass2) <= max_expected:
                    if verbose: _print_method(f"INFO: Category {category_char} in {html_filepath.name} updated with second pass. Old count: {len(scores_pass1)}, New count: {len(scores_pass2)}")
                    final_scores[category_char] = scores_pass2
    return {'title': title, 'scores': final_scores}

def write_data_to_csv(data_list, output_csv_filepath):
    if not data_list: print("No data to write to CSV."); return
    fieldnames = [
        'openreview_id', 'arxiv_id', 'title', 'venue', 'decision',
        'compliance_score', 'contribution_score',
        'soundness_score', 'presentation_score',
        'human_rating', 'human_soundness', 'human_contribution', 'human_presentation',
        'ethics_flag' # New field
    ]
    try:
        with open(output_csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in data_list:
                row = {
                    'openreview_id': item['openreview_id'],
                    'arxiv_id': item['arxiv_id'],
                    'title': item['title'],
                    'venue': item['venue'],
                    'decision': item['decision'],
                    'compliance_score': item['scores']['A'], # Sum of LLM checklist scores
                    'contribution_score': item['scores']['B'],
                    'soundness_score': item['scores']['C'],
                    'presentation_score': item['scores']['D'],
                    'human_rating': item.get('human_rating', []),
                    'human_soundness': item.get('human_soundness', []),
                    'human_contribution': item.get('human_contribution', []),
                    'human_presentation': item.get('human_presentation', []),
                    'ethics_flag': item.get('ethics_flag', 0) # Default to 0 if not found
                }
                writer.writerow(row)
        print(f"Data successfully written to {output_csv_filepath}")
    except IOError as e:
        print(f"Error writing CSV file {output_csv_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        import traceback
        traceback.print_exc()


def process_data_directory(base_input_dir_str, output_csv_str, or_client, verbose_flag): 
    _print_method = tqdm.write if not verbose_flag else print

    base_input_dir = Path(base_input_dir_str)
    output_csv_file = Path(output_csv_str)
    folders_missing_checklist = []
    parsed_data_for_csv = []

    if not base_input_dir.is_dir():
        _print_method(f"Error: Input directory '{base_input_dir}' not found."); return

    for venue_folder in base_input_dir.iterdir():
        if not venue_folder.is_dir(): continue
        venue_name = venue_folder.name
        for decision_name in ['accepted', 'rejected']:
            decision_path = venue_folder / decision_name
            if not decision_path.is_dir(): continue
            
            paper_id_folders_to_process = [pidf for pidf in decision_path.iterdir() if pidf.is_dir()]
            
            iterable_paper_folders = paper_id_folders_to_process
            if paper_id_folders_to_process:
                 iterable_paper_folders = tqdm(
                    paper_id_folders_to_process,
                    desc=f"Processing {venue_name}/{decision_name}",
                    disable=verbose_flag 
                )

            for paper_id_folder in iterable_paper_folders:
                paper_id_folder_name = paper_id_folder.name
                checklist_html_filename = f"{paper_id_folder_name}_checklist.html"
                checklist_html_filepath = paper_id_folder / checklist_html_filename

                if not checklist_html_filepath.exists():
                    folders_missing_checklist.append(paper_id_folder_name)
                else:
                    try:
                        parts = paper_id_folder_name.split('_', 1)
                        openreview_id = parts[0]
                        arxiv_id = parts[1].replace('_', '.') if len(parts) > 1 else "N/A" 
                    except Exception as e:
                        _print_method(f"Error parsing OpenReview/ArXiv ID from folder '{paper_id_folder_name}': {e}")
                        openreview_id = paper_id_folder_name; arxiv_id = "N/A"
                    
                    html_data = parse_html_checklist(checklist_html_filepath, verbose_flag)
                    # Fetch human scores. get_human_review_scores now returns a dict with lists of scores.
                    human_scores_data = get_human_review_scores(or_client, openreview_id, verbose_flag) if or_client else {}
                    
                    if html_data:
                        checklist_scores_found = any(s_list for s_list in html_data['scores'].values())
                        # Check if any human score list is non-empty and doesn't just contain "N/A"
                        human_scores_actually_found = any(
                            isinstance(score_list, list) and any(s != "N/A" for s in score_list)
                            for key, score_list in human_scores_data.items() if key != 'ethics_flag'
                        )


                        if checklist_scores_found or human_scores_actually_found: 
                             entry = {
                                'openreview_id': openreview_id,
                                'arxiv_id': arxiv_id,
                                'title': html_data['title'],
                                'venue': venue_name,
                                'decision': decision_name,
                                'scores': html_data['scores'], # LLM scores
                             }
                             # human_scores_data already has the right keys for update
                             entry.update(human_scores_data) 
                             parsed_data_for_csv.append(entry)
                        else:
                            _print_method(f"Warning: No checklist scores nor valid human scores found for {checklist_html_filepath.name} (OR ID: {openreview_id}). Skipping entry.")
                    else:
                        _print_method(f"Warning: Failed to parse checklist data from {checklist_html_filepath.name}")
    
    _print_method("\n--- Folders Missing Checklist HTML ---")
    if folders_missing_checklist:
        for folder_id in folders_missing_checklist: pass #_print_method(folder_id)
    else:
        _print_method("No folders found missing their checklist.html file.")
    _print_method("-------------------------------------\n")
    write_data_to_csv(parsed_data_for_csv, output_csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse paper checklist HTML files and generate a CSV summary.")
    parser.add_argument("--input_dir", type=str, default="./data/", help="Base input directory.")
    parser.add_argument("--output_csv", type=str, default="checklist_summary.csv", help="Path to the output CSV file.")
    parser.add_argument("--or_username", type=str, default=os.environ.get('OPENREVIEW_USERNAME'), help="OpenReview Username (or set OPENREVIEW_USERNAME env var).")
    parser.add_argument("--or_password", type=str, default=os.environ.get('OPENREVIEW_PASSWORD'), help="OpenReview Password (or set OPENREVIEW_PASSWORD env var).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output (disables tqdm progress bars).")

    args = parser.parse_args()

    if not CHECKLIST_PROMPT_AVAILABLE:
        print("Proceeding without checklist_prompt.py. Max item counts for score validation will not be available.")
    
    openreview_client = None
    if OPENREVIEW_CLIENT_AVAILABLE:
        if args.or_username and args.or_password:
            try:
                openreview_client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net',
                    username=args.or_username,
                    password=args.or_password
                )
                print(f"Successfully connected to OpenReview API as {args.or_username}.")
            except Exception as e:
                print(f"Failed to connect to OpenReview API: {e}. Human review scores will not be fetched.")
                openreview_client = None 
        else:
            print("OpenReview username or password not provided (either via args or env vars). Human review scores will not be fetched.")
    
    process_data_directory(args.input_dir, args.output_csv, openreview_client, args.verbose)
