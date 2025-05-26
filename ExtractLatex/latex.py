#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import subprocess # For running external commands (pdflatex, bibtex, pandoc)
import tempfile # For creating temporary files
from pathlib import Path
import argparse
import sys
import shutil # For copying files and checking for executables (shutil.which)
import urllib.parse # For decoding URL-encoded paths in Markdown
import requests # For Semantic Scholar & OpenAlex APIs
import time      # For politely pausing between API calls
import io        # For PDF processing
import xml.etree.ElementTree as ET # For Elsevier API
import json # For Springer API

# Attempt to import PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

# Attempt to import BeautifulSoup
try:
    from bs4 import BeautifulSoup, Comment
except ImportError:
    BeautifulSoup = None
    Comment = None # Ensure Comment is also None if bs4 is not available

# Attempt to import Pillow (PIL)
try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# Attempt to import pdf2image
try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


# List of common venue style file name patterns (regex)
VENUE_STYLE_PATTERNS = [
    r"neurips_?\d{4}",
    r"icml_?\d{4}",
    r"iclr_?\d{4}(_conference)?",
    r"aistats_?\d{4}",
    r"collas_?\d{4}(_conference)?",
    r"cvpr_?\d{4}?",
    r"iccv_?\d{4}?",
    r"eccv_?\d{4}?",
    r"acl_?\d{4}?",
    r"emnlp_?\d{4}?",
    r"naacl_?\d{4}?",
    r"siggraph_?\d{4}?",
    r"chi_?\d{4}?",
    r"aaai_?\d{2,4}",
    r"ijcai_?\d{2,4}",
    r"uai_?\d{4}?",
    r"IEEEtran",
    r"ieeeconf",
    r"acmart"
]

PROCESSED_BBL_MARKER = "% L2M_PROCESSED_BBL_V1_DO_NOT_EDIT_MANUALLY_BELOW_THIS_LINE"

class LatexToMarkdownConverter:
    def __init__(self, folder_path_str, verbose=True, template_path=None,
                 openalex_email="your-email@example.com", elsevier_api_key=None, springer_api_key=None,
                 poppler_path=None): 
        self.folder_path = Path(folder_path_str).resolve()
        self.main_tex_path = None
        self.original_main_tex_content = ""
        self.verbose = verbose
        self.template_path = template_path
        self.final_output_folder_path = None
        self.openalex_email = openalex_email
        self.elsevier_api_key = elsevier_api_key
        self.springer_api_key = springer_api_key
        self.PdfReader = PdfReader
        self.BeautifulSoup = BeautifulSoup
        self.Comment = Comment
        self.PILImage = PILImage 
        self.convert_from_path = convert_from_path 
        self.poppler_path = poppler_path           

        if self.PdfReader is None and self.verbose:
            self._log("PyPDF2 library not found. PDF parsing for abstracts will be disabled. Install with: pip install PyPDF2", "warn")
        if self.BeautifulSoup is None and self.verbose:
            self._log("BeautifulSoup4 library not found. HTML parsing for abstracts will be disabled. Install with: pip install beautifulsoup4", "warn")
        if self.PILImage is None and self.verbose:
            self._log("Pillow (PIL) library not found. Image conversion to PNG will be disabled. Install with: pip install Pillow", "warn")
        if self.convert_from_path is None and self.verbose:
            self._log("pdf2image library not found. PDF to PNG conversion using this library will be disabled. Install with: pip install pdf2image", "warn")
        if self.poppler_path and self.verbose:
            self._log(f"Using Poppler path: {self.poppler_path}", "info")
        elif self.convert_from_path and self.verbose: 
             self._log("Poppler path not specified. pdf2image will try to find Poppler in system PATH. If PDF conversion fails, try providing --poppler-path.", "info")


        if self.openalex_email == "your-email@example.com" and self.verbose:
            self._log("OpenAlex email is set to default. For responsible API use, please provide your email via --openalex-email.", "warn")
        if not self.elsevier_api_key and self.verbose:
            self._log("Elsevier API key not provided. Abstract fetching from linkinghub.elsevier.com via API will be disabled.", "info")
        if not self.springer_api_key and self.verbose:
            self._log("Springer API key not provided. Abstract fetching from link.springer.com via API will be disabled.", "info")


    def _log(self, message, level="info"):
        if level == "error": print(f"[-] Error: {message}", file=sys.stderr)
        elif level == "warn": print(f"[!] Warning: {message}", file=sys.stderr)
        elif self.verbose:
            if level == "info": print(f"[*] {message}")
            elif level == "success": print(f"[+] {message}")
            elif level == "debug": print(f"    [*] {message}")

    def find_main_tex_file(self):
        self._log("Finding main .tex file...")
        for path in self.folder_path.rglob("*.tex"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
                if r'\begin{document}' in content:
                    self.main_tex_path, self.original_main_tex_content = path, content
                    self._log(f"Main .tex file found: {self.main_tex_path}", "success"); return True
            except Exception as e: self._log(f"Could not read file {path} due to {e}", "warn")
        self._log(f"No main .tex file with \\begin{{document}} found in '{self.folder_path}'.", "error"); return False

    def _get_project_sty_basenames(self):
        sty_basenames = []
        for sty_path in self.folder_path.rglob("*.sty"):
            if sty_path.stem not in ["article", "report", "book", "amsmath", "graphicx", "geometry", "hyperref", "inputenc", "fontenc", "babel", "xcolor", "listings", "fancyhdr", "enumitem", "parskip", "setspace", "tocbibind", "titling", "titlesec", "etoolbox", "iftex", "xparse", "expl3", "l3keys2e", "natbib", "biblatex", "microtype", "amsfonts", "amssymb", "amsthm", "mathtools", "soul", "url", "booktabs", "float", "caption", "subcaption", "multirow", "multicol", "threeparttable", "xspace", "textcomp", "makecell", "tcolorbox", "wasysym", "colortbl", "algorithmicx", "algorithm", "algpseudocode", "marvosym", "ulem", "trimspaces", "environ", "keyval", "graphics", "trig", "ifvtex"]:
                sty_basenames.append(sty_path.stem)
        return sty_basenames

    def _comment_out_style_packages(self, tex_content, mode="venue_only"):
        modified_content = tex_content
        initial_content = tex_content
        if mode == "venue_only":
            for style_pattern_re in VENUE_STYLE_PATTERNS:
                pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{(" + style_pattern_re + r"),*[^\}]*\}[^\n]*)$"
                modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE | re.IGNORECASE)
        elif mode == "all_project":
            styles_to_comment_out = self._get_project_sty_basenames()
            if not styles_to_comment_out:
                self._log("No project-specific .sty files to comment out for 'all_project' mode.", "debug")
                return tex_content
            for sty_basename in styles_to_comment_out:
                pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{" + re.escape(sty_basename) + r",*[^\}]*\}[^\n]*)$"
                modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE)

        if modified_content != initial_content: self._log(f"Commented out \\usepackage commands (mode: {mode}).", "debug")
        else: self._log(f"No \\usepackage commands were modified (mode: {mode}).", "debug")
        return modified_content

    def _generate_bbl_content(self):
        """
        Generates or retrieves BBL content.
        If a .bbl file exists and is marked as processed, it's used from cache.
        Otherwise, it generates/reads the raw .bbl, processes it fully (abstracts, etc.),
        writes the processed version (with marker) back to the .bbl file, and returns it.
        Falls back to using any .bbl file in the folder if the specific one isn't found/generated.
        """
        if not self.main_tex_path:
            self._log("Cannot generate .bbl: Main .tex not identified.", "error")
            return None

        main_file_stem = self.main_tex_path.stem
        specific_bbl_path = self.folder_path / f"{main_file_stem}.bbl"
        raw_bbl_content_to_process = None

        # 1. Check existing specific .bbl file
        if specific_bbl_path.exists():
            self._log(f"Found existing specific .bbl file: {specific_bbl_path}", "debug")
            try:
                content = specific_bbl_path.read_text(encoding="utf-8", errors="ignore")
                if content.startswith(PROCESSED_BBL_MARKER + "\n"):
                    self._log("Using cached fully processed specific .bbl file.", "success")
                    return content.split(PROCESSED_BBL_MARKER + "\n", 1)[1]
                else:
                    self._log("Existing specific .bbl found but not marked as processed. Will use its raw content.", "info")
                    raw_bbl_content_to_process = content
            except Exception as e:
                self._log(f"Error reading existing specific .bbl file '{specific_bbl_path}': {e}. Will attempt generation/fallback.", "warn")
                raw_bbl_content_to_process = None

        # 2. If no usable specific BBL yet, try to generate it
        if raw_bbl_content_to_process is None:
            self._log(f"Attempting to generate specific .bbl file: {specific_bbl_path.name}...", "info")
            original_cwd = Path.cwd()
            os.chdir(self.folder_path)
            bibtex_run_successful = False
            try:
                commands = [
                    ["pdflatex", "-interaction=nonstopmode", "-draftmode", self.main_tex_path.name],
                    ["bibtex", main_file_stem]
                ]
                for i, cmd_args in enumerate(commands):
                    self._log(f"Running BBL generation command: {' '.join(cmd_args)}", "debug")
                    process = subprocess.run(cmd_args, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                    if process.returncode != 0:
                        self._log(f"Error running {' '.join(cmd_args)}. STDERR: {process.stderr[:500]}", "error")
                        if i == 0 and "pdflatex" in cmd_args[0]:
                            self._log("Initial pdflatex run failed for BBL generation.", "error")
                        bibtex_run_successful = False # Mark as failed
                        
                        # Remove badly generated bbl file
                        if specific_bbl_path.exists():
                            specific_bbl_path.unlink()
                        
                        break 
                    bibtex_run_successful = True # Mark as successful if all commands pass
                
                if bibtex_run_successful and specific_bbl_path.exists():
                    self._log(f"Specific .bbl file generated successfully: {specific_bbl_path}", "debug")
                    raw_bbl_content_to_process = specific_bbl_path.read_text(encoding="utf-8", errors="ignore")
                elif bibtex_run_successful: # Commands succeeded but file not created
                    self._log(f"BibTeX ran but did not create '{specific_bbl_path.name}'.", "warn")
                # If bibtex_run_successful is False, it means a command failed.
                    
            except Exception as e:
                self._log(f"Exception during .bbl generation: {e}", "error")

            finally:
                os.chdir(original_cwd)

        # 3. If still no BBL, try any alternative .bbl file in the project folder
        if raw_bbl_content_to_process is None:
            self._log(f"Specific .bbl file '{specific_bbl_path.name}' not found or generated. Searching for any .bbl file in '{self.folder_path}'...", "info")
            bbl_files_in_folder = list(self.folder_path.glob("*.bbl"))
            if bbl_files_in_folder:
                alternative_bbl_path = bbl_files_in_folder[0]
                self._log(f"Found alternative .bbl file: '{alternative_bbl_path.name}'. Using its raw content.", "info")
                try:
                    # We use the raw content of the alternative. It will be processed and saved to specific_bbl_path.
                    # We do not check for PROCESSED_BBL_MARKER in the alternative here, to keep logic simpler
                    # and ensure the canonical `specific_bbl_path` gets the processed, marked version.
                    raw_bbl_content_to_process = alternative_bbl_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    self._log(f"Error reading alternative .bbl file '{alternative_bbl_path.name}': {e}", "warn")
            else:
                self._log("No .bbl files (specific or alternative) found in the project folder.", "warn")

        # 4. Process and Cache (if raw content was obtained)
        if raw_bbl_content_to_process is not None:
            self._log("Starting full BBL processing (abstracts, keys) for caching...", "info")
            processed_bbl_string = self._fully_process_bbl(raw_bbl_content_to_process)
            
            try:
                with open(specific_bbl_path, "w", encoding="utf-8") as f:
                    f.write(PROCESSED_BBL_MARKER + "\n")
                    f.write(processed_bbl_string)
                self._log(f"Successfully wrote processed BBL content to '{specific_bbl_path}' (cached).", "success")
            except Exception as e_write:
                self._log(f"Error writing processed BBL to cache file '{specific_bbl_path}': {e_write}", "warn")
            
            return processed_bbl_string # Return the content without the marker for inlining
        
        self._log("Failed to obtain BBL content through any method.", "error")
        return None


    def _latex_escape_abstract(self, text: str) -> str:
        if not text: return ""
        text = text.replace('\\', r'\textbackslash{}')
        text = text.replace('{', r'\{'); text = text.replace('}', r'\}')
        text = text.replace('&', r'\&'); text = text.replace('%', r'\%')
        text = text.replace('$', r'\$'); text = text.replace('#', r'\#')
        text = text.replace('_', r'\_'); text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _deinvert_abstract_openalex(self, inverted_index):
        if not inverted_index: return None
        max_pos = -1
        for positions in inverted_index.values():
            if positions: max_pos = max(max_pos, max(positions))
        if max_pos == -1: return ""
        length = max_pos + 1; abstract_list = [""] * length
        for word, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < length: abstract_list[pos] = word
        return " ".join(abstract_list).strip()

    def _extract_text_from_pdf_content(self, pdf_content):
        if not self.PdfReader: return None
        try:
            reader = self.PdfReader(io.BytesIO(pdf_content)); text = ""
            for page in reader.pages: text += page.extract_text() or ""
            return text
        except Exception as e: self._log(f"PDF text extraction error: {e}", "warn"); return None

    def _find_abstract_in_pdf_text(self, text, max_chars_abstract=2500):
        if not text: return None
        text_lower = text.lower()
        abstract_match = re.search(r'\b(a\s*b\s*s\s*t\s*r\s*a\s*c\s*t)\b\.?\s*\n?', text_lower, re.IGNORECASE)
        if abstract_match:
            start_index = abstract_match.end()
            end_patterns = [r'\n\s*(keywords|key words|index terms)\b', r'\n\s*(1|I)\.\s*(Introduction|INTRODUCTION)\b', r'\n\s*(Introduction|INTRODUCTION)\b', r'\n\s*\n\s*\n']
            end_index = len(text)
            text_after_abstract_keyword = text[start_index:]
            for pattern in end_patterns:
                match = re.search(pattern, text_after_abstract_keyword, re.IGNORECASE)
                if match and match.start() < (end_index - start_index) : end_index = start_index + match.start()
            potential_abstract = text[start_index:min(end_index, start_index + max_chars_abstract)].strip()
            potential_abstract = re.sub(r'\s+', ' ', potential_abstract)
            if len(potential_abstract) > 50: self._log("PDF Abstract: Found abstract.", "debug"); return potential_abstract
        return None

    def _get_elsevier_abstract_from_linking_url(self, linking_hub_url: str, api_key: str) -> str | None:
        pii_match = re.search(r'/pii/([^/?]+)', linking_hub_url)
        if not pii_match:
            self._log(f"Elsevier API: Could not extract PII from URL: {linking_hub_url}", "warn")
            return "❌ Error: Could not extract PII from the provided URL."
        pii = pii_match.group(1)
        self._log(f"Elsevier API: Extracted PII: {pii} from URL: {linking_hub_url}", "debug")
        api_url = f"https://api.elsevier.com/content/article/pii/{pii}"
        headers = {'X-ELS-APIKey': api_key, 'Accept': 'application/xml'}
        self._log(f"Elsevier API: Constructed API URL: {api_url}", "debug")

        max_retries = 3
        retry_delay_seconds = 2 # Wait 2 seconds between retries
        server_error_codes = [500, 502, 503, 504] # Common server-side errors to retry

        for attempt in range(max_retries):
            try:
                response = requests.get(api_url, headers=headers, timeout=30) 
                response.raise_for_status() # Raises HTTPError for 4xx/5xx status codes
                
                # If successful, proceed to parse XML
                self._log(f"Elsevier API: Request successful (Attempt {attempt + 1}/{max_retries}).", "debug")
                try:
                    root = ET.fromstring(response.content)
                    namespaces = {'svapi': 'http://www.elsevier.com/xml/svapi/article/dtd', 'dc': 'http://purl.org/dc/elements/1.1/'}
                    coredata = root.find('svapi:coredata', namespaces)
                    if coredata is not None:
                        desc = coredata.find('dc:description', namespaces)
                        if desc is not None and desc.text:
                            self._log("Elsevier API: Abstract extracted.", "success")
                            return ' '.join(desc.text.strip().split())
                    self._log("Elsevier API: Abstract not found in XML.", "debug")
                    return "❌ Abstract not found in XML." # Return if abstract not found, no need to retry this
                except Exception as e_xml: 
                    self._log(f"Elsevier API: XML parsing error: {e_xml}. Resp: {response.text[:200]}...", "warn")
                    return f"❌ XML parsing error: {e_xml}. Resp: {response.text[:200]}..." # XML error, no retry

            except requests.exceptions.HTTPError as http_err:
                self._log(f"Elsevier API: HTTP error (Attempt {attempt + 1}/{max_retries}): {http_err} - Status: {http_err.response.status_code}", "warn")
                if http_err.response.status_code in server_error_codes:
                    if attempt < max_retries - 1:
                        self._log(f"Elsevier API: Retrying in {retry_delay_seconds}s...", "info")
                        time.sleep(retry_delay_seconds)
                        continue # Go to next attempt
                    else: # Last attempt failed
                        self._log("Elsevier API: Max retries reached for server error.", "error")
                        return f"❌ HTTP error after retries: {http_err} - Status: {http_err.response.status_code}"
                else: # Client error (4xx) or other HTTPError not in server_error_codes
                    return f"❌ HTTP error: {http_err} - Status: {http_err.response.status_code} - Resp: {http_err.response.text[:200]}..."
            
            except requests.exceptions.RequestException as req_err: # Catches other network errors like timeouts, connection errors
                self._log(f"Elsevier API: Request error (Attempt {attempt + 1}/{max_retries}): {req_err}", "warn")
                if attempt < max_retries - 1:
                    self._log(f"Elsevier API: Retrying in {retry_delay_seconds}s...", "info")
                    time.sleep(retry_delay_seconds)
                    continue # Go to next attempt
                else: # Last attempt failed
                    self._log("Elsevier API: Max retries reached for request exception.", "error")
                    return f"❌ Request error after retries: {req_err}"
            
            # Should not be reached if successful return or exception occurs and is handled
            break # Break if no exception was raised (should have returned if successful)
        
        # If loop finishes without returning (e.g., unexpected flow, though unlikely with current logic)
        return "❌ Elsevier API: Failed after all retries or due to unexpected issue."


    def _get_springer_abstract_from_url(self, springer_url: str, api_key: str) -> str | None:
        doi_match = re.search(r'/(?:chapter|article|book)/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)', springer_url, re.IGNORECASE)
        if not doi_match:
            self._log(f"Springer API: Could not extract DOI from URL: {springer_url}", "warn")
            return "❌ Error: Could not extract DOI from the provided URL."
        doi = doi_match.group(1)
        self._log(f"Springer API: Extracted DOI: {doi} from URL: {springer_url}", "debug")
        api_url = "https://api.springernature.com/meta/v2/json"
        params = {'q': f'doi:{doi}', 'api_key': api_key}
        self._log(f"Springer API: Constructed API URL: {api_url} with params: {params}", "debug")
        try:
            response = requests.get(api_url, params=params, timeout=20)
            response.raise_for_status()
            self._log("Springer API: API request successful.", "debug")
        except requests.exceptions.HTTPError as http_err:
            self._log(f"Springer API: HTTP error: {http_err} - Status: {response.status_code} - Resp: {response.text[:200]}...", "warn")
            return f"❌ HTTP error: {http_err} - Status: {response.status_code} - Resp: {response.text[:200]}..."
        except requests.exceptions.RequestException as err:
            self._log(f"Springer API: Request error: {err}", "warn")
            return f"❌ Request error: {err}"
        try:
            data = response.json()
            if data.get("records") and len(data["records"]) > 0:
                first_record = data["records"][0]
                if "abstract" in first_record and first_record["abstract"]:
                    self._log("Springer API: Abstract extracted.", "success"); return first_record["abstract"].strip()
            total_results = data.get("result", [{}])[0].get("total", "unknown")
            if str(total_results) == "0":
                self._log(f"Springer API: No records found for DOI: {doi}. API returned 0 results.", "debug")
                return f"❌ No records found for DOI: {doi}. The API returned 0 results."
            self._log("Springer API: Abstract not found in API response.", "debug")
            return "❌ Abstract field not found or empty in API response."
        except Exception as e:
            self._log(f"Springer API: JSON parsing error: {e}. Resp: {response.text[:200]}...", "warn")
            return f"❌ JSON parsing error: {e}. Resp: {response.text[:200]}..."

    def _get_arxiv_abstract_from_page(self, arxiv_url: str) -> str | None:
        if not self.BeautifulSoup:
            self._log("arXiv Page Fetch: BeautifulSoup library not available. Cannot parse HTML.", "warn")
            return None
        original_url_for_logging = arxiv_url
        if "arxiv.org/pdf/" in arxiv_url:
            arxiv_url = arxiv_url.replace("/pdf/", "/abs/").replace(".pdf", "")
            self._log(f"arXiv Page Fetch: Converted PDF URL '{original_url_for_logging}' to abstract URL '{arxiv_url}'.", "debug")
        elif not arxiv_url.startswith("https://arxiv.org/abs/"):
            self._log(f"arXiv Page Fetch: URL '{original_url_for_logging}' is not a standard arXiv abstract or PDF URL. Proceeding cautiously.", "warn")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            response = requests.get(arxiv_url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self._log(f"arXiv Page Fetch: Error fetching URL {arxiv_url}: {e}", "warn")
            return None
        try:
            soup = self.BeautifulSoup(response.content, 'html.parser')
            meta_abstract = soup.find('meta', attrs={'name': 'citation_abstract'})
            if meta_abstract and meta_abstract.get('content'):
                self._log(f"arXiv Page Fetch: Found abstract in 'citation_abstract' meta tag for {arxiv_url}.", "debug")
                return meta_abstract.get('content').strip()
            meta_og_description = soup.find('meta', property='og:description')
            if meta_og_description and meta_og_description.get('content'):
                self._log(f"arXiv Page Fetch: Found abstract in 'og:description' meta tag for {arxiv_url}.", "debug")
                return meta_og_description.get('content').strip()
            blockquote_abstract = soup.find('blockquote', class_='abstract')
            if blockquote_abstract:
                abstract_text_content = blockquote_abstract.get_text(separator=' ', strip=True)
                cleaned_abstract = re.sub(r'^\s*Abstract:?\s*', '', abstract_text_content, flags=re.IGNORECASE).strip()
                if cleaned_abstract:
                    self._log(f"arXiv Page Fetch: Found abstract in blockquote.abstract for {arxiv_url}.", "debug")
                    return cleaned_abstract
            self._log(f"arXiv Page Fetch: Could not find abstract on page: {arxiv_url} using known meta tags or blockquote.", "debug")
            return None
        except Exception as e:
            self._log(f"arXiv Page Fetch: Error parsing HTML content from {arxiv_url}: {e}", "warn")
            return None

    def _fetch_and_parse_html_for_abstract(self, url: str):
        if not self.BeautifulSoup or not self.Comment:
            self._log("HTML Parsing: BeautifulSoup or Comment not available.", "warn"); return None
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            if not url.startswith("file:///"):
                self._log(f"HTML Parsing: Invalid URL scheme: {url}", "debug"); return None

        self._log(f"HTML Parsing: Attempting to fetch from: {url}", "debug")
        effective_url = url
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; LatexToMarkdownConverter/1.3; +http://example.com/bot)'}
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            effective_url = response.url
            if url.startswith("file:///"): effective_url = url
            self._log(f"HTML Parsing: Effective URL after redirects: {effective_url}", "debug")
            response.raise_for_status()

            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                self._log(f"HTML Parsing: Content from {effective_url} not HTML (type: {content_type}).", "debug")
                return None

            soup = self.BeautifulSoup(response.content, 'html.parser')
            abstract_text_candidate = None 

            if not abstract_text_candidate:
                if self.springer_api_key and "link.springer.com/" in effective_url:
                    api_abstract = self._get_springer_abstract_from_url(effective_url, self.springer_api_key)
                    if api_abstract and not api_abstract.startswith("❌"):
                        self._log(f"Springer API: Successfully fetched abstract from {effective_url} after HTML parse failed.", "success")
                        return api_abstract 
                    elif api_abstract: self._log(f"Springer API: Attempt after HTML parse fail for {effective_url} also failed: {api_abstract}", "warn")
                    else: self._log(f"Springer API: Attempt after HTML parse fail for {effective_url} returned None.", "warn")

            if not abstract_text_candidate:
                self._log(f"HTML Abstract: Trying MHTML-like comment-based extraction for {effective_url}", "debug")
                abstract_start_comment = soup.find(string=lambda text: isinstance(text, self.Comment) and "Abstract." in text and "<!-- Abstract." in text and not "/ Abstract." in text)
                if abstract_start_comment:
                    self._log(f"HTML Abstract: Found '' comment in {effective_url}", "debug")
                    possible_outer_divs = soup.find_all('div', class_='columns is-centered has-text-centered')
                    for outer_div in possible_outer_divs:
                        column_div = outer_div.find('div', class_='column is-four-fifths')
                        if column_div:
                            h1_abstract = column_div.find('h1', class_='title is-3', string=re.compile(r'^\s*Abstract\s*$', re.I))
                            content_div = column_div.find('div', class_='content has-text-justified')
                            if h1_abstract and content_div and h1_abstract.find_next_sibling('div', class_='content has-text-justified') == content_div:
                                texts = []
                                for element in content_div.find_all(['p', 'ul'], recursive=False):
                                    if element.name == 'p': texts.append(element.get_text(separator=' ', strip=True))
                                    elif element.name == 'ul':
                                        for li_item in element.find_all('li', recursive=False): texts.append("- " + li_item.get_text(separator=' ', strip=True))
                                extracted_abstract = "\n".join(texts)
                                if extracted_abstract.strip():
                                    self._log(f"HTML Abstract: Extracted using MHTML-like comment structure from {effective_url}.", "success")
                                    return extracted_abstract 
                else:
                    self._log(f"HTML Abstract: MHTML-like '' comment not found in {effective_url}", "debug")

            if not abstract_text_candidate: 
                abstract_heading = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'strong', 'div'], string=re.compile(r'^\s*Abstract\s*$', re.I))
                if abstract_heading:
                    self._log(f"HTML Abstract: Found heading '{abstract_heading.get_text(strip=True)}'. Looking for content.", "debug")
                    acl_style_element = None
                    if "acl-abstract" in " ".join(abstract_heading.get('class', [])): acl_style_element = abstract_heading
                    elif abstract_heading.parent and "acl-abstract" in " ".join(abstract_heading.parent.get('class', [])): acl_style_element = abstract_heading.parent
                    elif abstract_heading.parent and abstract_heading.parent.parent and "acl-abstract" in " ".join(abstract_heading.parent.parent.get('class',[])): acl_style_element = abstract_heading.parent.parent
                    if acl_style_element:
                        abs_span = acl_style_element.find('span') 
                        if abs_span:
                            text = abs_span.get_text(separator=' ', strip=True)
                            if len(text) > 70: self._log("HTML Abstract: Extracted from ACL-style span.", "debug"); return text 
                        abs_p = acl_style_element.find('p') 
                        if abs_p:
                            text = abs_p.get_text(separator=' ', strip=True)
                            if len(text) > 70: self._log("HTML Abstract: Extracted from ACL-style p.", "debug"); return text 
                    next_content_node = abstract_heading.find_next_sibling()
                    for _ in range(3): 
                        if not next_content_node: break
                        if next_content_node.name in ['div', 'section'] and \
                            ((hasattr(next_content_node, 'attrs') and any(c in " ".join(next_content_node.attrs.get('class',[])) for c in ['content', 'abstract-content', 'entry-content', 'abstract__content'])) or \
                                not hasattr(next_content_node, 'attrs') or not next_content_node.attrs.get('class', None)):
                            collected_texts = []
                            for child_el in next_content_node.find_all(['p', 'ul', 'div'], recursive=True): 
                                if child_el.find_parent(['table', 'figure', 'figcaption', 'nav', 'header', 'footer']): continue
                                text = child_el.get_text(separator=' ', strip=True)
                                if text: collected_texts.append(text)
                                if sum(len(t) for t in collected_texts) > 2500 : break 
                            final_abstract_text = " ".join(collected_texts).strip()
                            if len(final_abstract_text) > 70:
                                self._log("HTML Abstract: Extracted from div/section after 'Abstract' heading.", "debug"); return final_abstract_text 
                            break 
                        next_content_node = next_content_node.find_next_sibling() if hasattr(next_content_node, 'find_next_sibling') else None
                    collected_general_texts = []
                    current_sib = abstract_heading.find_next_sibling()
                    while current_sib:
                        if current_sib.name in ['h1','h2','h3','h4','h5'] or \
                            (hasattr(current_sib, 'attrs') and any(val in current_sib.attrs.get(attr, '').lower() for attr in ['id', 'class'] for val in ['reference', 'citation', 'biblio'])):
                            break 
                        text = current_sib.get_text(separator=' ', strip=True)
                        if text: collected_general_texts.append(text)
                        if sum(len(t) for t in collected_general_texts) > 2500: break
                        current_sib = current_sib.find_next_sibling()
                    final_general_text = " ".join(collected_general_texts).strip()
                    if len(final_general_text) > 70:
                        self._log("HTML Abstract: Extracted from general siblings after 'Abstract' heading.", "debug"); return final_general_text 

            if not abstract_text_candidate: 
                self._log(f"HTML Abstract: No explicit 'Abstract' found for {effective_url}. Trying 'Introduction' section.", "debug")
                introduction_heading_element = None
                for heading_tag_name in ['h1', 'h2', 'h3']:
                    headings = soup.find_all(heading_tag_name, string=re.compile(r'^\s*Introduction\s*$', re.I))
                    if headings:
                        article_body = soup.find('article', class_='markdown-body')
                        if article_body:
                            intro_in_article = article_body.find(heading_tag_name, string=re.compile(r'^\s*Introduction\s*$', re.I))
                            if intro_in_article: introduction_heading_element = intro_in_article; break 
                        if not introduction_heading_element: introduction_heading_element = headings[0]; break
                if introduction_heading_element:
                    self._log(f"HTML Abstract: Found '{introduction_heading_element.name}' heading for 'Introduction'. Collecting content.", "debug")
                    start_node = introduction_heading_element.parent if introduction_heading_element.parent.name == 'div' and 'markdown-heading' in introduction_heading_element.parent.get('class', []) else introduction_heading_element
                    collected_texts, char_count, max_chars_for_intro = [], 0, 2500
                    for sibling in start_node.find_next_siblings():
                        if char_count >= max_chars_for_intro: break
                        if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] or (sibling.name == 'div' and "markdown-heading" in sibling.get('class', [])): break
                        current_text = ""
                        if sibling.name == 'p': current_text = sibling.get_text(separator=' ', strip=True)
                        elif sibling.name == 'ul': current_text = "\n".join([f"- {li.get_text(separator=' ', strip=True)}" for li in sibling.find_all('li', recursive=False)])
                        if current_text:
                            if char_count + len(current_text) > max_chars_for_intro: current_text = current_text[:max_chars_for_intro - char_count] + "..."
                            collected_texts.append(current_text); char_count += len(current_text)
                    if collected_texts:
                        introduction_abstract = "\n\n".join(collected_texts).strip()
                        if len(introduction_abstract) > 70:
                            self._log(f"HTML Abstract: Extracted from 'Introduction' section for {effective_url}.", "success"); return introduction_abstract 
                else: self._log(f"HTML Abstract: 'Introduction' heading not found for {effective_url}.", "debug")

            if not abstract_text_candidate: 
                self._log("HTML Abstract: Trying general structural selectors as fallback.", "debug")
                general_selectors = ['div.abstract', 'section.abstract', 'article.abstract', 'div#abstract', 'section#abstract', 'div[class*="abstract-text"]', 'div[itemprop="description"]', 'div[class*="entry-summary"]', 'div.summary', 'article p']
                for selector in general_selectors:
                    elements = soup.select(selector)
                    if elements:
                        candidate_texts = []
                        for i, el in enumerate(elements):
                            text = el.get_text(separator=' ', strip=True)
                            if text: candidate_texts.append(text)
                            if selector == 'article p' and i >= 4: break 
                        full_text = " ".join(candidate_texts).strip()
                        if len(full_text) > 150: self._log(f"HTML Abstract: Found using general selector '{selector}'.", "debug"); return full_text 

            if not abstract_text_candidate: 
                self._log(f"HTML Abstract: Structural search failed for {effective_url}. Trying meta tags (last resort).", "debug")
                for meta_attr_spec in [{'name': 'description'}, {'property': 'og:description'}, {'name': 'twitter:description'}]:
                    meta_tag = soup.find('meta', attrs=meta_attr_spec)
                    if meta_tag and meta_tag.get('content'):
                        abs_text = meta_tag['content'].strip()
                        if len(abs_text.split(',')) > 5 and len(abs_text) < 300 : self._log(f"HTML Abstract: Meta tag {meta_attr_spec} looks like author list, skipping.", "debug"); continue
                        if len(abs_text) > 70: self._log(f"HTML Abstract: Found in meta tag {meta_attr_spec} (last resort).", "debug"); return abs_text 

            self._log(f"HTML Parsing: No abstract found in {effective_url} via any heuristic after all attempts.", "debug")
            return None 

        except requests.exceptions.HTTPError as http_err:
            log_url_info = f"original URL: {url}" + (f", effective (error) URL: {effective_url}" if url != effective_url else "")
            self._log(f"HTML Parsing: HTTP error {http_err.response.status_code} for {log_url_info}: {http_err}", "debug")
            self._log(f"HTML Parsing: Checking API fallback conditions. Status: {http_err.response.status_code}, Effective URL: {effective_url}", "debug")

            if http_err.response.status_code == 403: 
                if self.elsevier_api_key and "sciencedirect.com/science/article/pii/" in effective_url:
                    self._log(f"HTML Parsing: Detected 403 on ScienceDirect PII link ({effective_url}). Attempting Elsevier API.", "info")
                    api_abstract = self._get_elsevier_abstract_from_linking_url(effective_url, self.elsevier_api_key)
                    if api_abstract and not api_abstract.startswith("❌"):
                        self._log(f"Elsevier API: Successfully fetched abstract via PII from {effective_url} after 403 on HTML.", "success")
                        return api_abstract
                    elif api_abstract: self._log(f"Elsevier API: Attempt after 403 on {effective_url} failed: {api_abstract}", "warn")
                    else: self._log(f"Elsevier API: Attempt after 403 on {effective_url} returned None.", "warn")

                elif self.springer_api_key and "link.springer.com/" in effective_url:
                    self._log(f"HTML Parsing: Detected 403 on Springer link ({effective_url}). Attempting Springer API.", "info")
                    api_abstract = self._get_springer_abstract_from_url(effective_url, self.springer_api_key)
                    if api_abstract and not api_abstract.startswith("❌"):
                        self._log(f"Springer API: Successfully fetched abstract from {effective_url} after 403 on HTML.", "success")
                        return api_abstract
                    elif api_abstract: self._log(f"Springer API: Attempt after 403 on {effective_url} failed: {api_abstract}", "warn")
                    else: self._log(f"Springer API: Attempt after 403 on {effective_url} returned None.", "warn")
                else:
                    self._log(f"HTML Parsing: 403 error, but no specific API fallback for URL pattern: {effective_url}", "debug")
            else:
                self._log(f"HTML Parsing: HTTP error was not 403, or no API key available for this publisher. No API fallback.", "debug")
            return None
        except requests.exceptions.RequestException as e:
            log_url_info = f"original URL: {url}" + (f", effective URL during attempt: {effective_url}" if url != effective_url and effective_url != url else "")
            self._log(f"HTML Parsing: Request error for {log_url_info}: {e}", "warn")
            return None
        return None


    def _clean_title_for_search(self, title_str: str, attempt=1) -> str:
        if not title_str: return ""
        cleaned = title_str
        cleaned = re.sub(r'\\(?:emph|textbf|textit|texttt|textsc|mathrm|mathsf|mathcal|mathbf|bm)\s*\{(.*?)\}', r'\1', cleaned)
        cleaned = cleaned.replace(r"\'e", "e").replace(r'\"u', 'ue').replace(r'\`a', 'a')
        cleaned = re.sub(r'\{([A-Za-z\d\-:]+)\}', r'\1', cleaned)
        cleaned = re.sub(r'\\url\{[^\}]+\}', '', cleaned); cleaned = re.sub(r'\\href\{[^\}]+\}\{[^\}]+\}', '', cleaned)
        cleaned = cleaned.replace(r'\&', '&').replace(r'\%', '%').replace(r'\$', '$').replace(r'\_', '_')
        cleaned = cleaned.replace('\n', ' ').strip()
        cleaned = re.sub(r'\{\\natexlab\{[^}]*\}\}', '', cleaned); cleaned = re.sub(r'\{\\noop\{[^}]*\}\}', '', cleaned)
        cleaned = re.sub(r'^In\s*:\s*(?=[A-Z])', '', cleaned).strip(); cleaned = re.sub(r'^In\s+(?=[A-Z])', '', cleaned).strip()
        cleaned = re.sub(r'[:?!;]+', ' ', cleaned)
        cleaned = re.sub(r',', '', cleaned)
        if attempt == 1:
            cleaned = re.sub(r'[\s\(]+\d{4}[a-z]?\)?\s*$', '', cleaned).strip()
            cleaned = re.sub(r'\s+et\s+al\.?\s*$', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.rstrip('.')
        cleaned = cleaned.lower()
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r"^[^\w\s]+", "", cleaned); cleaned = re.sub(r"[^\w\s]+$", "", cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned.strip()


    def _fetch_abstract_from_openalex(self, title_query: str, authors_str: str = ""):
        if not title_query: return None
        for attempt in range(1, 3):
            cleaned_title = self._clean_title_for_search(title_query, attempt=attempt)
            if not cleaned_title: continue
            self._log(f"OpenAlex(A{attempt}): Searching:'{cleaned_title}'", "debug")
            base_url = "https://api.openalex.org/works"; params = {"filter": f"title.search:{re.escape(cleaned_title)}", "mailto": self.openalex_email}
            headers = {"User-Agent": f"LatexToMarkdownConverter/1.3 (mailto:{self.openalex_email})"}
            try:
                response = requests.get(base_url, params=params, headers=headers, timeout=20)
                response.raise_for_status(); data = response.json(); results = data.get("results", [])
                if results:
                    for i, work in enumerate(results):
                        if i >= 2: break
                        if work.get("abstract"): self._log("OpenAlex: Direct abstract.", "success"); time.sleep(0.2); return work["abstract"]
                        if work.get("abstract_inverted_index"):
                            deinverted = self._deinvert_abstract_openalex(work["abstract_inverted_index"])
                            if deinverted: self._log("OpenAlex: Deinverted abstract.", "success"); time.sleep(0.2); return deinverted
                        landing_page_url = work.get("primary_location", {}).get("landing_page_url")

                        if self.elsevier_api_key and landing_page_url and "linkinghub.elsevier.com/retrieve/pii/" in landing_page_url:
                            elsevier_abs = self._get_elsevier_abstract_from_linking_url(landing_page_url, self.elsevier_api_key)
                            if elsevier_abs and not elsevier_abs.startswith("❌"):
                                self._log("OpenAlex (via Elsevier API): Abstract found.", "success"); time.sleep(0.2); return elsevier_abs
                        elif self.springer_api_key and landing_page_url and "link.springer.com/" in landing_page_url:
                            springer_abs = self._get_springer_abstract_from_url(landing_page_url, self.springer_api_key)
                            if springer_abs and not springer_abs.startswith("❌"):
                                self._log("OpenAlex (via Springer API): Abstract found.", "success"); time.sleep(0.2); return springer_abs

                        if self.BeautifulSoup and landing_page_url and not landing_page_url.lower().endswith(".pdf"):
                            html_abs = self._fetch_and_parse_html_for_abstract(landing_page_url)
                            if html_abs: self._log("OpenAlex: HTML abstract from landing page.", "success"); time.sleep(0.2); return html_abs

                        if self.PdfReader:
                            pdf_url = work.get("primary_location", {}).get("pdf_url") or (landing_page_url if landing_page_url and landing_page_url.lower().endswith(".pdf") else None)
                            if not pdf_url:
                                for loc in work.get("locations", []):
                                    if loc.get("is_oa") and (loc.get("pdf_url") or loc.get("landing_page_url","").lower().endswith(".pdf")):
                                        pdf_url = loc.get("pdf_url") or loc.get("landing_page_url"); break
                            if pdf_url:
                                try:
                                    pdf_resp = requests.get(pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}, allow_redirects=True)
                                    pdf_resp.raise_for_status()
                                    if 'application/pdf' in pdf_resp.headers.get('Content-Type','').lower():
                                        pdf_text = self._extract_text_from_pdf_content(pdf_resp.content)
                                        if pdf_text:
                                            pdf_abs_candidate = self._find_abstract_in_pdf_text(pdf_text)
                                            if pdf_abs_candidate: self._log("OpenAlex: PDF abstract from primary/OA location.", "success"); time.sleep(0.2); return pdf_abs_candidate
                                except Exception as e_pdf: self._log(f"OpenAlex: PDF error for {pdf_url}: {e_pdf}", "warn")
                    self._log(f"OpenAlex(A{attempt}): Found papers for '{cleaned_title[:60]}' but no abstract.", "debug"); return None
            except Exception as e: self._log(f"OpenAlex(A{attempt}): API/Req error for '{cleaned_title[:60]}': {e}", "warn")
            time.sleep(0.3)
        return None

    def _fetch_abstract_from_semantic_scholar(self, title: str, authors_str: str = ""):
        if not title: return None
        for attempt in range(1, 6):
            cleaned_title = self._clean_title_for_search(title, attempt=attempt)
            if not cleaned_title: continue
            self._log(f"S2(A{attempt}): Searching:'{cleaned_title}'", "debug")
            try:
                search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote_plus(cleaned_title)}&fields=title,abstract,url&limit=2"
                headers = {'User-Agent': 'LatexToMarkdownConverter/1.3'}
                response = requests.get(search_url, headers=headers, timeout=15)
                response.raise_for_status(); data = response.json()
                if data.get("data"):
                    for paper_data in data["data"]:
                        if paper_data.get("abstract"):
                            self._log(f"S2(A{attempt}): Found abstract for '{cleaned_title}'.", "success"); time.sleep(0.3); return paper_data["abstract"]
                        s2_url = paper_data.get("url")
                        if self.elsevier_api_key and s2_url and "linkinghub.elsevier.com/retrieve/pii/" in s2_url:
                            self._log(f"S2: No direct abstract, found Elsevier URL: {s2_url}. Trying API.", "debug")
                            elsevier_abs = self._get_elsevier_abstract_from_linking_url(s2_url, self.elsevier_api_key)
                            if elsevier_abs and not elsevier_abs.startswith("❌"):
                                self._log("S2 (via Elsevier API): Abstract found.", "success"); time.sleep(0.3); return elsevier_abs
                        elif self.springer_api_key and s2_url and "link.springer.com/" in s2_url:
                            self._log(f"S2: No direct abstract, found Springer URL: {s2_url}. Trying API.", "debug")
                            springer_abs = self._get_springer_abstract_from_url(s2_url, self.springer_api_key)
                            if springer_abs and not springer_abs.startswith("❌"):
                                self._log("S2 (via Springer API): Abstract found.", "success"); time.sleep(0.3); return springer_abs
                        elif self.BeautifulSoup and s2_url and not s2_url.lower().endswith(".pdf"):
                                self._log(f"S2: No direct abstract, found URL: {s2_url}. Trying HTML parse.", "debug")
                                html_abs = self._fetch_and_parse_html_for_abstract(s2_url)
                                if html_abs: self._log("S2 (via HTML parse): Abstract found.", "success"); time.sleep(0.3); return html_abs
                    self._log(f"S2(A{attempt}): Found papers for '{cleaned_title}' but no abstract (or via fallback URLs).", "debug"); return None
            except Exception as e: self._log(f"S2(A{attempt}): API error for '{cleaned_title}': {e}", "debug")
            time.sleep(0.3)
        return None

    def _extract_bibitem_components(self, bibitem_text_chunk: str) -> dict:
        key_match = re.match(r"\\bibitem(?:\[[^\]]*\])?\{([^\}]+)\}", bibitem_text_chunk)
        key = key_match.group(1) if key_match else ""
        content_after_key = bibitem_text_chunk[key_match.end():].strip() if key_match else bibitem_text_chunk
        author_parts = content_after_key.split(r'\newblock', 1)
        authors_str = author_parts[0].strip().rstrip('.,;')
        text_after_authors = author_parts[1].strip() if len(author_parts) > 1 else ""
        title_raw, details_after_title_str, is_url_title_flag, url_if_title_val = "", "", False, None
        url_patterns = [r"^(?P<url>\\url\{([^}]+)\})(?P<rest>.*)", r"^(?P<url>https?://[^\s]+)(?P<rest>.*)"]
        for pat_str in url_patterns:
            url_match = re.match(pat_str, text_after_authors, re.DOTALL)
            if url_match:
                potential_url_block, url_content = url_match.group("url").strip(), url_match.group(2) if pat_str.startswith(r"^(?P<url>\\url") else url_match.group("url").strip()
                rest_content = url_match.group("rest").strip()
                if not rest_content or re.match(r"^[,\s]*\d{4}\.?$", rest_content) or rest_content.startswith(r"\newblock") or len(rest_content) < 15:
                    title_raw = potential_url_block
                    if rest_content and not rest_content.startswith(r"\newblock"): title_raw += " " + rest_content; details_after_title_str = ""
                    else: details_after_title_str = rest_content
                    is_url_title_flag, url_if_title_val = True, url_content.strip(); break
        if not is_url_title_flag:
            title_block_parts = text_after_authors.split(r'\newblock', 1)
            current_title_candidate_block = title_block_parts[0].strip()
            further_details_block = ("\\newblock " + title_block_parts[1].strip()) if len(title_block_parts) > 1 else ""
            title_end_delimiters = [
                r"In\s+(?:Proc\.?(?:eedings)?|Workshop|Conference|Journal|Symposium)\b",
                r"\b(?:[A-Z][a-z]+)\s+(?:Press|Publishers|Verlag)\b",
                r"Ph\.?D\.?\s+thesis\b",
                r"Master(?:'s)?\s+thesis\b",
                r"arXiv preprint arXiv:",
                r"[,;\s]\(?(?P<year>\d{4})\)?(?=\W|$|\s*\\newblock|\s*notes\b)",
                r"\.\s+\d{4}\.",
                r"Article\s+No\.",
                r"vol\.\s*\d+", r"pp\.\s*\d+", r"no\.\s*\d+",
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
            ]
            min_delimiter_idx = len(current_title_candidate_block)
            for pat in title_end_delimiters:
                match = re.search(pat, current_title_candidate_block, re.IGNORECASE)
                if match and match.start() < min_delimiter_idx:
                    if 'year' in match.groupdict() and len(current_title_candidate_block[:match.start()].strip()) < 15 and not re.search(r'[,.;:]$', current_title_candidate_block[:match.start()].strip()): pass
                    else: min_delimiter_idx = match.start()
            title_raw = current_title_candidate_block[:min_delimiter_idx].strip().rstrip('.,;/')


            details_after_title_str = (current_title_candidate_block[min_delimiter_idx:] + " " + further_details_block).strip()

        title_cleaned_for_api = self._clean_title_for_search(title_raw) if not is_url_title_flag else url_if_title_val
        return {"key": key, "authors": authors_str, "title_raw": title_raw, "title_cleaned": title_cleaned_for_api, "is_url_title": is_url_title_flag, "url_if_title": url_if_title_val, "details_after_title": details_after_title_str.strip()}

    def _fully_process_bbl(self, raw_bbl_content: str) -> str:
        self._log("Starting full BBL processing (abstracts, keys)...", "info")
        if not raw_bbl_content: return ""

        bbl_preamble = ""
        begin_thebibliography_match = re.search(r"\\begin{thebibliography}\{[^}]*\}", raw_bbl_content)
        
        if begin_thebibliography_match:
            bbl_preamble = raw_bbl_content[:begin_thebibliography_match.start()]
            # Corrected patterns: \\\\ for literal \, \\{ for literal { etc.
            external_cleanup_patterns_str = [
                r"\\providecommand\\{\\natexlab\\}\\[1\\]\\{#1\\}\\s*",
                r"\\providecommand\\{\\url\\}\\[1\\]\\{\\texttt\\{#1\\}\\}\\s*",
                r"\\providecommand\\{\\doi\\}\\[1\\]\\{doi:\s*#1\\}\\s*",
                r"\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*",
                r"\\expandafter\\ifx\\csname\s*doi\\endcsname\\relax[\s\S]*?\\fi\s*",
            ]
            for p_str in external_cleanup_patterns_str:
                try:
                    bbl_preamble = re.sub(p_str, "", bbl_preamble, flags=re.DOTALL)
                except re.error as e_re:
                    self._log(f"Regex error cleaning external BBL preamble with pattern '{p_str}': {e_re}", "warn")

            bbl_preamble = bbl_preamble.strip()

        bibliography_env_start_for_pandoc = r"\begin{thebibliography}{}" # Use empty arg for Pandoc
        
        items_text_start_offset = 0
        if begin_thebibliography_match:
            items_text_start_offset = begin_thebibliography_match.end()
        
        end_thebibliography_match = re.search(r"\\end{thebibliography}", raw_bbl_content)
        items_text_end_offset = len(raw_bbl_content)
        if end_thebibliography_match:
            items_text_end_offset = end_thebibliography_match.start()

        bbl_items_text = raw_bbl_content[items_text_start_offset:items_text_end_offset].strip()

        if bbl_items_text:
            # Corrected patterns for internal cleanup
            internal_cleanup_patterns = [
                re.compile(r"^\s*\\providecommand\\{\\natexlab\\}\\[1\\]\\{#1\\}\\s*", flags=re.MULTILINE | re.DOTALL),
                re.compile(r"^\s*\\providecommand\\{\\url\\}\\[1\\]\\{\\texttt\\{#1\\}\\}\\s*", flags=re.MULTILINE | re.DOTALL),
                re.compile(r"^\s*\\providecommand\\{\\doi\\}\\[1\\]\\{doi:\s*#1\\}\\s*", flags=re.MULTILINE | re.DOTALL),
                re.compile(r"^\s*\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.MULTILINE | re.DOTALL),
                re.compile(r"^\s*\\expandafter\\ifx\\csname\s*doi\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.MULTILINE | re.DOTALL),
            ]
            
            cleaned_any_internal_preamble = False
            original_bbl_items_text_len_for_log = len(bbl_items_text) 
            
            while True:
                text_before_cleaning_pass = bbl_items_text
                for pattern_obj in internal_cleanup_patterns:
                    try:
                        bbl_items_text = pattern_obj.sub("", bbl_items_text).strip()
                    except re.error as e_re_int:
                         self._log(f"Regex error cleaning internal BBL preamble with pattern '{pattern_obj.pattern}': {e_re_int}", "warn")
                if bbl_items_text == text_before_cleaning_pass: 
                    break 
                cleaned_any_internal_preamble = True 
            
            if cleaned_any_internal_preamble:
                 self._log(f"Cleaned internal BBL preamble. Remaining item text starts with: '{bbl_items_text[:100]}...'", "debug")
            elif original_bbl_items_text_len_for_log > 0 and not bbl_items_text.startswith(r"\bibitem") and bbl_items_text: 
                 self._log(f"BBL item text starts with non-bibitem content after internal preamble cleaning attempt: '{bbl_items_text[:100]}...'", "debug")


        processed_bibitems_parts = []
        if bbl_items_text: 
            split_bibitems = re.split(r'(\\bibitem)', bbl_items_text)
            
            if split_bibitems and split_bibitems[0].strip() and not split_bibitems[0].startswith("\\bibitem"):
                self._log(f"Discarding unexpected leading text in bbl_items_text: '{split_bibitems[0].strip()[:100]}...'", "warn")
                split_bibitems.pop(0)
            elif split_bibitems and not split_bibitems[0].strip() and len(split_bibitems) > 1 : 
                 split_bibitems.pop(0)

            k = 0
            while k < len(split_bibitems):
                if split_bibitems[k] == r'\bibitem':
                    if k + 1 < len(split_bibitems):
                        item_chunk = r'\bibitem' + split_bibitems[k+1].strip()
                        k += 2
                    else: 
                        self._log("Warning: Dangling \\bibitem in BBL processing.", "warn")
                        processed_bibitems_parts.append(r'\bibitem')
                        k += 1
                        continue
                else: 
                    self._log(f"Warning: Unexpected chunk in BBL processing, expected \\bibitem: '{split_bibitems[k][:50]}...'", "warn")
                    k += 1
                    continue
                
                components = self._extract_bibitem_components(item_chunk)
                reconstructed_item_parts = [f"\\bibitem{{{components['key']}}}", components['authors']]
                if components['authors'] and components['title_raw']: reconstructed_item_parts.append(r"\newblock")
                if components['title_raw']: reconstructed_item_parts.append(components['title_raw'])
                if components['details_after_title'].strip(): reconstructed_item_parts.append(components['details_after_title'])
                current_item_reconstructed = " ".join(p.strip() for p in reconstructed_item_parts if p.strip())
                current_item_reconstructed = re.sub(r'\s*\\newblock\s*', r' \\newblock ', current_item_reconstructed).strip()
                current_item_reconstructed = re.sub(r'\s+', ' ', current_item_reconstructed)
                
                abstract_text = None 
                if 'arxiv' in item_chunk.lower():
                    self._log(f"Bibitem {components['key']}: 'arxiv' keyword found. Attempting extract abstract from Arxiv.", "debug")
                    arxiv_id_local = None 
                    specific_match = re.search(r'arxiv[:\s]*(\d{4}\.\d{4,5}(?:v\d+)?)', item_chunk, re.IGNORECASE)
                    if specific_match: arxiv_id_local = specific_match.group(1)
                    else:
                        general_match = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', item_chunk)
                        if general_match: arxiv_id_local = general_match.group(1)
                    if arxiv_id_local and re.fullmatch(r'\d{4}\.\d{4,5}(v\d+)?', arxiv_id_local):
                        self._log(f"Bibitem {components['key']}: Extracted valid arXiv ID: {arxiv_id_local}", "debug")
                        arxiv_abs_url = f"https://arxiv.org/abs/{arxiv_id_local}"
                        self._log(f"Bibitem {components['key']}: Attempting to fetch abstract from arXiv: {arxiv_abs_url}", "debug")
                        abstract_text = self._get_arxiv_abstract_from_page(arxiv_abs_url)
                        if abstract_text: self._log(f"Bibitem {components['key']}: Abstract successfully extracted from arXiv {arxiv_abs_url}.", "success")
                    elif 'arxiv' in item_chunk.lower():
                        self._log(f"Bibitem {components['key']}: 'arxiv' keyword found but could not extract a valid arXiv ID from chunk: '{item_chunk[:100]}...'", "debug")
                if not abstract_text:
                    title_for_api = components["title_cleaned"]
                    if components["is_url_title"] and components["url_if_title"]:
                        current_url_to_check = components["url_if_title"]
                        if "linkinghub.elsevier.com/retrieve/pii/" in current_url_to_check and self.elsevier_api_key:
                            api_abs = self._get_elsevier_abstract_from_linking_url(current_url_to_check, self.elsevier_api_key)
                            if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                        elif "link.springer.com/" in current_url_to_check and self.springer_api_key:
                            api_abs = self._get_springer_abstract_from_url(current_url_to_check, self.springer_api_key)
                            if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                        if not abstract_text and self.BeautifulSoup and not current_url_to_check.lower().endswith(".pdf"):
                            if not (("arxiv.org/abs/" in current_url_to_check or "arxiv.org/html/" in current_url_to_check) and 'arxiv' in item_chunk.lower()):
                                abstract_text = self._fetch_and_parse_html_for_abstract(current_url_to_check)
                    if not abstract_text and title_for_api:
                        abstract_text = self._fetch_abstract_from_openalex(title_for_api, components["authors"])
                        if not abstract_text:
                            abstract_text = self._fetch_abstract_from_semantic_scholar(title_for_api, components["authors"])
                    if not abstract_text:
                        latex_url_match = re.search(r"\\url\{([^}]+)\}", item_chunk)
                        http_url_match = re.search(r"\b(https?://[^\s\"'<>()\[\]{},;]+[^\s\"'<>()\[\]{},;\.])", item_chunk)
                        potential_urls = []
                        if latex_url_match: potential_urls.append(latex_url_match.group(1).strip())
                        if http_url_match:
                            url_candidate = http_url_match.group(1).strip()
                            if not (latex_url_match and latex_url_match.group(1).strip() == url_candidate):
                                preceding_char_idx = http_url_match.start(1) - 1
                                if preceding_char_idx < 0 or item_chunk[preceding_char_idx] not in ['{']: potential_urls.append(url_candidate)
                        for p_url in potential_urls:
                            if components["is_url_title"] and components["url_if_title"] == p_url: continue
                            if ("arxiv.org/pdf/" in p_url or "arxiv.org/abs/" in p_url) and 'arxiv' in item_chunk.lower(): continue
                            if "linkinghub.elsevier.com/retrieve/pii/" in p_url and self.elsevier_api_key:
                                api_abs = self._get_elsevier_abstract_from_linking_url(p_url, self.elsevier_api_key)
                                if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                            elif "link.springer.com/" in p_url and self.springer_api_key:
                                api_abs = self._get_springer_abstract_from_url(p_url, self.springer_api_key)
                                if api_abs and not api_abs.startswith("❌"): abstract_text = api_abs
                            if not abstract_text and self.BeautifulSoup and not p_url.lower().endswith(".pdf"):
                                 if not (("arxiv.org/abs/" in p_url or "arxiv.org/html/" in p_url) and 'arxiv' in item_chunk.lower()):
                                    html_abstract = self._fetch_and_parse_html_for_abstract(p_url)
                                    if html_abstract: abstract_text = html_abstract; self._log(f"Fallback: Abstract from HTML URL '{p_url}'.", "success")
                            if not abstract_text and self.PdfReader and (p_url.lower().endswith(".pdf") or "arxiv.org/pdf/" in p_url):
                                try:
                                    pdf_resp = requests.get(p_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/pdf, */*'}, allow_redirects=True)
                                    pdf_resp.raise_for_status()
                                    if 'application/pdf' in pdf_resp.headers.get('Content-Type','').lower():
                                        pdf_txt = self._extract_text_from_pdf_content(pdf_resp.content)
                                        if pdf_txt:
                                            pdf_abs_candidate = self._find_abstract_in_pdf_text(pdf_txt)
                                            if pdf_abs_candidate: abstract_text = pdf_abs_candidate; self._log(f"Fallback: Abstract from PDF URL '{p_url}'.", "success")
                                except Exception as fb_pdf_err: self._log(f"Fallback: PDF error for {p_url}: {fb_pdf_err}", "warn")
                            if abstract_text: break
                if abstract_text:
                    escaped_abstract = self._latex_escape_abstract(abstract_text)
                    current_item_reconstructed += f" \\newblock \\textbf{{Abstract:}} {escaped_abstract}"
                escaped_key_for_display = components['key'].replace('_', r'\_')
                current_item_reconstructed += f" \\newblock (@{escaped_key_for_display})"
                
                processed_bibitems_parts.append(current_item_reconstructed)
        
        final_bbl_items_joined = "\n\n".join(p for p in processed_bibitems_parts if p)

        final_bbl_string = ""
        if bbl_preamble: 
            final_bbl_string += bbl_preamble + "\n"
        
        if final_bbl_items_joined or begin_thebibliography_match: # Ensure env is added if it was originally there
            final_bbl_string += bibliography_env_start_for_pandoc + "\n"
            if final_bbl_items_joined:
                final_bbl_string += final_bbl_items_joined + "\n"
            final_bbl_string += r"\end{thebibliography}"
        elif not begin_thebibliography_match and final_bbl_items_joined : 
            self._log("Warning: Bibitems found but no \\begin{thebibliography} was detected. Outputting items directly.", "warn")
            final_bbl_string += final_bbl_items_joined 
        
        return final_bbl_string.strip()

    def _simple_inline_processed_bbl(self, tex_content: str, processed_bbl_string: str) -> str:
        self._log("Inlining fully processed BBL content into TeX...", "debug")
        references_heading_tex = "\n\n\\section*{References}\n\n"
        # Only add references heading if there's actual bbl content to inline
        content_to_inline = (references_heading_tex + processed_bbl_string) if processed_bbl_string.strip() else ""

        modified_tex_content = re.sub(r"^\s*\\bibliographystyle\{[^{}]*\}\s*$", "", tex_content, flags=re.MULTILINE)
        
        bibliography_command_pattern = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"
        input_references_bbl_command_pattern = r"\\input{[^\}]*.bbl}"
        if re.search(bibliography_command_pattern, tex_content, flags=re.MULTILINE):
            modified_tex_content = re.sub(input_references_bbl_command_pattern, "", modified_tex_content, flags=re.MULTILINE)
        elif re.search(input_references_bbl_command_pattern, tex_content, flags=re.MULTILINE):
            modified_tex_content = re.sub(input_references_bbl_command_pattern, r"\\bibliography{references}", modified_tex_content, flags=re.MULTILINE)
        else:
            # add before
            #\clearpage
            #\appendix
            #if possible
            clearpage_appendix_command_pattern = r"\\clearpage\n\\appendix"
            if re.search(clearpage_appendix_command_pattern, tex_content, flags=re.MULTILINE):
                # Add r"\\bibliography{references}" before it
                modified_tex_content = re.sub(clearpage_appendix_command_pattern, r"\\bibliography{references}\n\\clearpage\n\\appendix", modified_tex_content, flags=re.MULTILINE)
            else:
                end_document_command_pattern = r"\\end\{document\}"
                if re.search(end_document_command_pattern, tex_content, flags=re.MULTILINE):
                    modified_tex_content = re.sub(end_document_command_pattern, r"\\bibliography{references}\n\n\\end{document}", modified_tex_content, flags=re.MULTILINE)
                else:
                    raise Exception('Cannot add references...')

        if content_to_inline: # Only try to inline if there's content
            if re.search(bibliography_command_pattern, modified_tex_content, flags=re.MULTILINE):
                modified_tex_content = re.sub(bibliography_command_pattern, lambda m: content_to_inline, modified_tex_content, count=1, flags=re.MULTILINE)
            else:
                end_document_match = re.search(r"\\end\{document\}", modified_tex_content)
                if end_document_match:
                    insertion_point = end_document_match.start()
                    modified_tex_content = modified_tex_content[:insertion_point] + content_to_inline + "\n" + modified_tex_content[insertion_point:]
                else: # Append if \end{document} not found (less ideal)
                    modified_tex_content += "\n" + content_to_inline
        else: # No BBL content, just remove the bibliography command
            modified_tex_content = re.sub(bibliography_command_pattern, "", modified_tex_content, flags=re.MULTILINE)


        modified_tex_content = re.sub(r"^\s*\\nobibliography\{[^{}]*(?:,[^{}]*)*\}\s*$", "", modified_tex_content, flags=re.MULTILINE)
        return modified_tex_content

    def _find_and_copy_figures_from_markdown(self, markdown_content, output_figure_folder_path):
        """
        Finds image paths in markdown, copies them to output_figure_folder_path (which should be a 'figures' subdirectory),
        and returns a list of dictionaries with details about each copied figure.
        Note: This function flattens the directory structure of copied images into output_figure_folder_path.
              e.g., "figures/myimg.pdf" or "another_subfolder/myimg.pdf" from source becomes "myimg.pdf" in output_figure_folder_path.
        """
        copied_figure_details = []
        if not markdown_content:
            self._log("No markdown content to find figures from.", "debug")
            return copied_figure_details

        patterns = [
            r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)", # Standard Markdown: ![alt](path "title")
            r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>",   # HTML <img>: <img src="path">
            r"<figure>.*?<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>.*?</figure>" # HTML <figure><embed src="path"></figure>
        ]
        img_paths = []
        for p in patterns:
            for match in re.finditer(p, markdown_content, flags=re.IGNORECASE | re.DOTALL): # Added re.DOTALL for multiline figure/embed
                img_paths.append(match.group(1))

        if not img_paths:
            self._log("No image paths found in markdown content.", "debug")
            return copied_figure_details

        self._log(f"Found {len(img_paths)} potential image paths in markdown. Target figure folder: {output_figure_folder_path}", "debug")
        output_figure_folder_path.mkdir(parents=True, exist_ok=True)

        common_ext = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.svg', '.gif', '.bmp', '.tiff']
        copied_count = 0
        copied_source_abs_paths_set = set()

        for raw_path in img_paths:
            try:
                decoded_path_str = urllib.parse.unquote(raw_path)
            except Exception as e_dec:
                self._log(f"Could not decode path '{raw_path}': {e_dec}. Skipping.", "warn")
                continue

            if urllib.parse.urlparse(decoded_path_str).scheme in ['http', 'https']:
                self._log(f"Skipping web URL: {decoded_path_str}", "debug")
                continue

            src_abs_candidate = (self.folder_path / decoded_path_str).resolve()
            found_abs_path = None

            if src_abs_candidate.is_file():
                found_abs_path = src_abs_candidate
            elif not src_abs_candidate.suffix:
                for ext_ in common_ext:
                    path_with_ext = (self.folder_path / (decoded_path_str + ext_)).resolve()
                    if path_with_ext.is_file():
                        found_abs_path = path_with_ext
                        self._log(f"Found '{decoded_path_str}' as '{path_with_ext.name}' by adding extension.", "debug")
                        break
            
            if not found_abs_path and self.main_tex_path:
                src_abs_candidate_rel_main = (self.main_tex_path.parent / decoded_path_str).resolve()
                if src_abs_candidate_rel_main.is_file():
                    found_abs_path = src_abs_candidate_rel_main
                elif not src_abs_candidate_rel_main.suffix:
                    for ext_ in common_ext:
                        path_with_ext = (self.main_tex_path.parent / (decoded_path_str + ext_)).resolve()
                        if path_with_ext.is_file():
                            found_abs_path = path_with_ext
                            self._log(f"Found '{decoded_path_str}' as '{path_with_ext.name}' (relative to main .tex) by adding extension.", "debug")
                            break

            if found_abs_path:
                dest_filename = found_abs_path.name 
                dest_abs_path = (output_figure_folder_path / dest_filename).resolve()

                if found_abs_path in copied_source_abs_paths_set and dest_abs_path.exists():
                    self._log(f"Figure source '{found_abs_path.name}' (from MD path '{raw_path}') already copied to '{dest_abs_path}'. Adding reference.", "debug")
                else: 
                    try:
                        shutil.copy2(str(found_abs_path), str(dest_abs_path))
                        copied_source_abs_paths_set.add(found_abs_path) 
                        copied_count +=1
                        self._log(f"Copied '{found_abs_path.name}' (from MD path '{raw_path}') to '{dest_abs_path}'.", "debug")
                    except Exception as e_copy:
                        self._log(f"Figure copy error for '{found_abs_path}' to '{dest_abs_path}': {e_copy}", "warn")
                        continue 

                details = {
                    "raw_markdown_path": raw_path,
                    "original_filename_with_ext": found_abs_path.name,
                    "source_abs_path": str(found_abs_path),
                    "copied_dest_abs_path": str(dest_abs_path), 
                    "original_ext": found_abs_path.suffix.lower()
                }
                copied_figure_details.append(details)
            else:
                self._log(f"Could not find figure source for markdown path: '{raw_path}' (decoded: '{decoded_path_str}')", "warn")

        if copied_count > 0:
            self._log(f"Copied {copied_count} unique figure files to '{output_figure_folder_path}'.", "success")
        elif img_paths:
            self._log("Found image paths in markdown, but no new files were copied (e.g., all URLs, files not found, or already copied).", "info")
        return copied_figure_details

    def _convert_and_update_figure_paths_in_markdown(self, markdown_content, figure_details_list, output_base_path):
        """
        Converts specified figures to PNG and updates their paths in the markdown content.
        Figures are expected to be in a 'figures' subdirectory of output_base_path.
        Markdown paths will be updated to './figures/filename.ext'.
        Also converts <embed> tags within <figure> to <img> tags.
        """
        updated_markdown_content = markdown_content # Start with the original content

        if figure_details_list: # Process figure paths and conversions only if there are details
            NO_CONVERSION_EXTENSIONS = ['.jpg', '.jpeg', '.png']

            for fig_info in figure_details_list:
                original_md_path_in_doc = fig_info["raw_markdown_path"]
                
                copied_file_abs_path = Path(fig_info["copied_dest_abs_path"]) 
                original_ext = fig_info["original_ext"]
                original_filename_stem = copied_file_abs_path.stem

                target_filename_in_figures_subdir = copied_file_abs_path.name
                conversion_done = False 

                # --- PDF Conversion using pdf2image (Primary for PDFs) ---
                if self.convert_from_path and original_ext == '.pdf':
                    target_png_filename_stem = f"{original_filename_stem}.png"
                    target_png_abs_path_in_figures_subdir = copied_file_abs_path.with_name(target_png_filename_stem)
                    
                    if target_png_abs_path_in_figures_subdir.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                        self._log(f"Converted PNG '{target_png_abs_path_in_figures_subdir.name}' (from PDF) already exists. Using it.", "debug")
                        conversion_done = True
                    else:
                        try:
                            self._log(f"Attempting PDF to PNG conversion for '{copied_file_abs_path.name}' using pdf2image...", "debug")
                            images = self.convert_from_path(copied_file_abs_path, 
                                                            poppler_path=self.poppler_path, 
                                                            first_page=1, last_page=1, fmt='png')
                            if images:
                                images[0].save(target_png_abs_path_in_figures_subdir, "PNG")
                                self._log(f"Successfully converted PDF '{copied_file_abs_path.name}' to '{target_png_abs_path_in_figures_subdir.name}' using pdf2image.", "success")
                                conversion_done = True
                            else:
                                self._log(f"pdf2image returned no images for '{copied_file_abs_path.name}'.", "warn")
                        except Exception as e_pdf2img_generic: 
                            self._log(f"Error during pdf2image conversion for '{copied_file_abs_path.name}': {e_pdf2img_generic}", "warn")
                    
                    if conversion_done:
                        target_filename_in_figures_subdir = target_png_filename_stem
                        if copied_file_abs_path.exists() and copied_file_abs_path.suffix == '.pdf' and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                            try:
                                copied_file_abs_path.unlink()
                                self._log(f"Deleted original PDF '{copied_file_abs_path.name}' from figures subdir after pdf2image conversion.", "debug")
                            except Exception as e_del:
                                self._log(f"Failed to delete original PDF '{copied_file_abs_path.name}' from figures subdir: {e_del}", "warn")

                # --- Pillow Conversion (Fallback for PDFs if pdf2image failed, or for other types) ---
                if not conversion_done and original_ext not in NO_CONVERSION_EXTENSIONS:
                    target_png_filename_stem = f"{original_filename_stem}.png"
                    target_png_abs_path_in_figures_subdir = copied_file_abs_path.with_name(target_png_filename_stem)
                    
                    pillow_conversion_attempted = False
                    if target_png_abs_path_in_figures_subdir.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                        self._log(f"Converted PNG '{target_png_abs_path_in_figures_subdir.name}' (Pillow fallback) already exists. Using it.", "debug")
                        conversion_done = True 
                    elif self.PILImage:
                        pillow_conversion_attempted = True
                        self._log(f"Attempting conversion to PNG for '{copied_file_abs_path.name}' (in figures subdir) using Pillow...", "debug")
                        try:
                            if original_ext in ['.pdf', '.eps'] and not shutil.which("gs"):
                                self._log(f"Ghostscript (gs) command not found. Pillow may fail to convert '{copied_file_abs_path.name}'. Install Ghostscript for PDF/EPS conversion.", "warn")
                            if original_ext == '.svg':
                                self._log(f"Pillow does not directly support SVG conversion. Skipping Pillow conversion for '{copied_file_abs_path.name}'.", "warn")
                            else:
                                img = self.PILImage.open(copied_file_abs_path)
                                if img.mode == 'P' and 'transparency' in img.info: img = img.convert("RGBA")
                                elif img.mode not in ['RGB', 'RGBA', 'L', 'LA']: img = img.convert("RGBA")
                                img.save(target_png_abs_path_in_figures_subdir, "PNG")
                                self._log(f"Successfully converted '{copied_file_abs_path.name}' to '{target_png_abs_path_in_figures_subdir.name}' (in figures subdir) using Pillow.", "debug")
                                conversion_done = True
                        except FileNotFoundError: 
                            self._log(f"Pillow conversion failed for '{copied_file_abs_path.name}': A dependent component (like Ghostscript for PDF/EPS) might be missing or not in PATH.", "warn")
                        except Exception as e_conv:
                            self._log(f"Error converting '{copied_file_abs_path.name}' to PNG using Pillow: {e_conv}", "warn")
                    elif not pillow_conversion_attempted: 
                        self._log("Pillow (PIL) library not found. Cannot convert images using Pillow.", "warn")

                    if conversion_done:
                        target_filename_in_figures_subdir = target_png_filename_stem 
                        if copied_file_abs_path.exists() and copied_file_abs_path != target_png_abs_path_in_figures_subdir:
                            try:
                                copied_file_abs_path.unlink()
                                self._log(f"Deleted original file '{copied_file_abs_path.name}' from figures subdir after Pillow conversion.", "debug")
                            except Exception as e_del:
                                self._log(f"Failed to delete original file '{copied_file_abs_path.name}' from figures subdir: {e_del}", "warn")
                    elif pillow_conversion_attempted: 
                         self._log(f"Pillow conversion skipped or failed for '{copied_file_abs_path.name}'. Markdown will link to original in figures subdir.", "debug")


                # --- Update Markdown Path ---
                new_md_path_for_doc = f"./figures/{target_filename_in_figures_subdir}"
                escaped_original_md_path = re.escape(original_md_path_in_doc)
                
                md_link_pattern = rf"(!\[(?:[^\]]*)\]\()({escaped_original_md_path})(\))"
                updated_markdown_content = re.sub(md_link_pattern, rf"\1{new_md_path_for_doc}\3", updated_markdown_content)
                
                img_src_pattern = rf'(<img\s+[^>]*?src\s*=\s*["\'])({escaped_original_md_path})(["\'][^>]*?>)'
                updated_markdown_content = re.sub(img_src_pattern, rf"\1{new_md_path_for_doc}\3", updated_markdown_content)

                figure_embed_pattern = rf'(<figure>.*?<embed\s+[^>]*?src\s*=\s*["\'])({escaped_original_md_path})(["\'][^>]*?>.*?</figure>)'
                updated_markdown_content = re.sub(figure_embed_pattern, rf"\1{new_md_path_for_doc}\3", updated_markdown_content, flags=re.DOTALL)

                if original_md_path_in_doc != new_md_path_for_doc and original_md_path_in_doc in markdown_content : 
                     self._log(f"Updated Markdown path for '{original_md_path_in_doc}' to '{new_md_path_for_doc}'.", "debug")
                elif original_md_path_in_doc == new_md_path_for_doc:
                     self._log(f"Markdown path '{original_md_path_in_doc}' effectively unchanged to '{new_md_path_for_doc}'.", "debug")
        
        # After all path updates, convert any remaining <embed> to <img> within <figure>
        final_markdown_content = re.sub(r"(<figure>.*?)<embed(\s+[^>]*?)>(.*?</figure>)", 
                                        r"\1<img\2 />\3", 
                                        updated_markdown_content, flags=re.DOTALL | re.IGNORECASE)
        if final_markdown_content != updated_markdown_content: # Check if any embed->img replacement actually happened
            self._log("Converted standalone <embed> to <img> tags in <figure> elements.", "debug")
        
        return final_markdown_content

    def _resolve_include_path(self, filename: str, current_dir: Path) -> Path | None:
        """
        Resolves the path of an included TeX file.
        Searches relative to current_dir, then project_folder.
        Tries with .tex extension and without.
        """
        # Normalize filename (e.g., remove leading/trailing whitespace)
        filename = filename.strip()
        
        # Paths to try for resolution
        potential_filenames = []
        if filename.endswith(".tex"):
            potential_filenames.append(filename)
            potential_filenames.append(filename[:-4]) # Try without .tex
        else:
            potential_filenames.append(filename)
            potential_filenames.append(filename + ".tex") # Try with .tex

        # Search locations: current file's directory, then project root
        search_dirs = [current_dir, self.folder_path]
        
        for fname_variant in potential_filenames:
            for search_dir in search_dirs:
                try_path = (search_dir / fname_variant).resolve()
                if try_path.is_file():
                    return try_path
        return None

    def _input_replacer(self, match: re.Match, current_dir: Path, visited_files: set) -> str:
        """
        Replacement function for re.sub to handle \input and \include.
        Recursively expands the content of the included file.
        """
        command = match.group(1)  # 'input' or 'include'
        filename_group = match.group(2) # filename from {filename}

        resolved_path = self._resolve_include_path(filename_group, current_dir)

        if resolved_path and resolved_path not in visited_files:
            self._log(f"Expanding {command}: {resolved_path.name} (from {current_dir})", "debug")
            visited_files.add(resolved_path)
            try:
                with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f_inc:
                    included_content = f_inc.read()
                # Recursively expand includes within this newly included content
                expanded_included_content = self._recursively_expand_tex_includes(
                    resolved_path, 
                    included_content, 
                    visited_files
                )
                return f"\n% --- Start content from {resolved_path.name} ---\n{expanded_included_content}\n% --- End content from {resolved_path.name} ---\n"
            except Exception as e_inc:
                self._log(f"Could not read included file {resolved_path}: {e_inc}", "warn")
                return match.group(0) # Return original command if file not found/readable
        elif resolved_path in visited_files:
            self._log(f"Skipping already visited file during expansion: {resolved_path.name}", "debug")
            return f"% Skipped re-inclusion of {resolved_path.name}\n" # Comment out to avoid loops
        else:
            self._log(f"Could not resolve include path for '{filename_group}' from dir '{current_dir}'. Command: {match.group(0)}", "warn")
            return match.group(0) # Return original command if file not found

    def _recursively_expand_tex_includes(self, current_file_path: Path, current_content: str, visited_files: set) -> str:
        """
        Recursively replaces \input{file} and \include{file} commands with the content
        of the specified files.
        """
        include_pattern = re.compile(r"\\(input|include)\s*\{([^}]+)\}")
        current_dir = current_file_path.parent
        
        # Iteratively apply substitution until no more changes are made,
        # as an included file might itself include other files.
        # The visited_files set prevents infinite loops.
        while True:
            new_content = include_pattern.sub(
                lambda m: self._input_replacer(m, current_dir, visited_files),
                current_content
            )
            if new_content == current_content: # No more substitutions made in this pass
                break
            current_content = new_content
            
        return current_content

    def _preprocess_latex_table_environments(self, initial_main_tex_content: str) -> str:
        """
        Expands includes and then converts specific non-standard LaTeX table environments 
        to standard ones to aid tools like Pandoc.
        """
        self._log("Preprocessing LaTeX: Expanding includes and then converting table environments...", "debug")
        
        # 1. Recursively expand all \input and \include commands from the main .tex file
        visited_files = {self.main_tex_path.resolve()} 
        expanded_content = self._recursively_expand_tex_includes(
            current_file_path=self.main_tex_path, 
            current_content=initial_main_tex_content, 
            visited_files=visited_files
        )
        self._log(f"LaTeX content expanded to ~{len(expanded_content)//1024} KB before table conversion.", "debug")

        # 2. Apply table conversions to the fully expanded content
        initial_table_content_for_log = expanded_content 
        processed_table_content = expanded_content 
        
        processed_table_content = re.sub(r'\\cr', r'\\\\', processed_table_content) 
        processed_table_content = re.sub(r'\\centering', '', processed_table_content) 
        processed_table_content = re.sub(r'\\vspace{[^}]*}', '', processed_table_content)
        processed_table_content = re.sub(r'\\setlength{[^}]*}{[^}]*}', '', processed_table_content)
        processed_table_content = re.sub(r'\\small', '', processed_table_content)
        
        def remove_pipes_from_multicolumn_spec(match):
            # group 1: \multicolumn{num_cols}
            # group 2: the part of the alignment spec (e.g., 'c|r', 'c', '|c|')
            # group 3: the content {actual content}
            multicolumn_command = match.group(1)
            align_spec_with_braces = match.group(2) # e.g. "{c|}" or "{c}"
            content = match.group(3)
            
            # Remove pipes from the alignment spec inside the braces
            cleaned_align_spec_no_braces = align_spec_with_braces[1:-1].replace("|", "")
            cleaned_align_spec_with_braces = "{" + cleaned_align_spec_no_braces + "}"
            return f"{multicolumn_command}{cleaned_align_spec_with_braces}{content}"

        processed_table_content = re.sub(
            r"(\\multicolumn\s*\{[^}]*\})(\{[^}|]*?\|[^}]*?\})(\{[^}]*\})", # Matches if | is present
            remove_pipes_from_multicolumn_spec,
            processed_table_content
        )
        
        def remove_pipes_from_tabular_spec(match):
            tabular_start = match.group(1) 
            optional_align = match.group(2) if match.group(2) else "" 
            column_specs_with_braces = match.group(3) 
            
            cleaned_specs_no_braces = column_specs_with_braces[1:-1].replace("|", "")
            cleaned_specs_with_braces = "{" + cleaned_specs_no_braces + "}"
            return f"{tabular_start}{optional_align}{cleaned_specs_with_braces}"

        processed_table_content = re.sub(
            r"(\\begin{tabular})(\[[^\]]*\])?({[^}]*})", 
            remove_pipes_from_tabular_spec,
            processed_table_content
        )
        
        processed_table_content = re.sub(
            r"\\scalebox{[^}]*}{\s*(\\begin{tabular}[\s\S]*?\\end{tabular})\s*}",
            r"\1",  
            processed_table_content
        )
        processed_table_content = re.sub(
            r"\\scalebox{[^}]*}\[[^\]]*\]{\s*(\\begin{tabular}[\s\S]*?\\end{tabular})\s*}",
            r"\1",
            processed_table_content
        )

        processed_table_content = re.sub(
            r"\\begin{wraptable}(?:\[[^\]]*\])?\s*\{[^}]*\}\s*\{[^}]*\}",
            r"\\begin{table}[htb]",
            processed_table_content
        )
        processed_table_content = re.sub(
            r"\\end{wraptable}",
            r"\\end{table}",
            processed_table_content
        )

        processed_table_content = re.sub(
            r"\\begin{table\*}(?:\[[^\]]*\])?",
            r"\\begin{table}[htb]",
            processed_table_content
        )
        processed_table_content = re.sub(
            r"\\end{table\*}",
            r"\\end{table}",
            processed_table_content
        )
            
        processed_table_content = re.sub(r"\\begin{tablenotes}\s*(?:\[.*?\])?([\s\S]*?)\\end{tablenotes}", r"\n% tablenotes content:\n\1\n", processed_table_content) 
        
        processed_table_content = re.sub(r"\\begin{threeparttable}(?:\[[^\]]*\])?", "% threeparttable start removed\n", processed_table_content)
        processed_table_content = re.sub(r"\\end{threeparttable}", "% threeparttable end removed\n", processed_table_content)
        
        def replace_content_inside_shortstack(match):
            # Get the content inside \shortstack{...}
            content = match.group(1)
            # Replace '\\' (possibly followed by whitespace) with a single space.
            # The regex r"\\\\\s*" matches two literal backslashes '\\'
            # followed by zero or more whitespace characters (\s*).
            processed_content = re.sub(r"\\\\\s*", " ", content)
            return processed_content

        # Regex to find \shortstack{content}.
        # - \\shortstack\{ matches the literal "\shortstack{".
        # - (.*?) captures any characters non-greedily (group 1) until the first closing brace.
        # - \} matches the literal "}".
        print(re.findall(r"\\shortstack\{((?:[^{}]*|\{[^{}]*\})*)\}", processed_table_content))
        processed_table_content = re.sub(r"\\shortstack\{((?:[^{}]*|\{[^{}]*\})*)\}", replace_content_inside_shortstack, processed_table_content)
        
        if processed_table_content != initial_table_content_for_log:
            self._log("Applied LaTeX table preprocessing to expanded content.", "info")
        else:
            self._log("No changes made during LaTeX table preprocessing of expanded content.", "debug")
        print(processed_table_content)
        return processed_table_content

    # --- Checklist Preprocessing Methods ---
    def _add_paragraph_spacing_to_checklist_block(self, text_block):
        lines = text_block.splitlines()
        result_lines = []
        if not lines:
            return ""

        for i, line in enumerate(lines):
            current_stripped = line.strip()
            
            if result_lines: 
                prev_line_in_result_stripped = result_lines[-1].strip()
                
                current_is_qajg_transformed = current_stripped.startswith(r"{\bf Question:") or \
                                              current_stripped.startswith(r"{\bf Answer:") or \
                                              current_stripped.startswith(r"{\bf Justification:") or \
                                              current_stripped.startswith(r"{\bf Guidelines:")
                
                prev_was_main_item = prev_line_in_result_stripped.startswith(r"\item") and \
                                     not prev_line_in_result_stripped.startswith(r"{\bf")
                prev_was_qaj_transformed = prev_line_in_result_stripped.startswith(r"{\bf Question:") or \
                                           prev_line_in_result_stripped.startswith(r"{\bf Answer:") or \
                                           prev_line_in_result_stripped.startswith(r"{\bf Justification:")

                if current_is_qajg_transformed and prev_line_in_result_stripped: 
                    if prev_was_main_item or prev_was_qaj_transformed:
                        if not (prev_line_in_result_stripped.startswith(r"{\bf Guidelines:") and \
                                current_stripped.startswith(r"\begin{itemize}")):
                            result_lines.append("")
            
            result_lines.append(line) 

        final_lines = []
        if result_lines:
            start_idx = 0
            while start_idx < len(result_lines) and not result_lines[start_idx].strip():
                start_idx += 1
            
            for j in range(start_idx, len(result_lines)):
                if not (result_lines[j].strip() == "" and final_lines and not final_lines[-1].strip()):
                    final_lines.append(result_lines[j])
            
            while final_lines and not final_lines[-1].strip():
                final_lines.pop()
                
        return "\n".join(final_lines)

    def _validate_checklist_major_item_structure(self, major_item_lines_list):
        if not major_item_lines_list or not re.match(r"^\s*\\item(?!\s*\[)", major_item_lines_list[0].strip()):
            return False

        cursor = 1 

        def _get_next_significant_line_content(lines, current_cursor):
            idx = current_cursor
            while idx < len(lines) and not lines[idx].strip(): 
                idx += 1
            if idx < len(lines):
                return lines[idx].strip(), idx + 1 
            return None, idx 

        expected_sequence_items = [
            r"\item\[\] Question:",
            r"\item\[\] Answer:",
            r"\item\[\] Justification:",
            r"\item\[\] Guidelines:",
        ]

        for i, pattern_start in enumerate(expected_sequence_items):
            line_content, cursor = _get_next_significant_line_content(major_item_lines_list, cursor)
            if not (line_content and line_content.startswith(pattern_start)):
                return False
            
            if pattern_start == r"\item\[\] Guidelines:":
                next_struct_line_content, _ = _get_next_significant_line_content(major_item_lines_list, cursor)
                if not (next_struct_line_content and next_struct_line_content.startswith(r"\begin{itemize}")):
                    return False
        
        return True

    def _transform_checklist_enumerate_block_content(self, content):
        transformed_content = content
        transformed_content = re.sub(
            r"^(\s*)\\item\[\]\s*(Question|Answer|Justification):\s*(.*)$",
            r"\1{\\bf \2:} \3", transformed_content, flags=re.MULTILINE)
        transformed_content = re.sub(
            r"^(\s*)\\item\[\]\s*(Guidelines):\s*$",
            r"\1{\\bf \2:}", transformed_content, flags=re.MULTILINE)
        
        spaced_content = self._add_paragraph_spacing_to_checklist_block(transformed_content)
        return spaced_content

    def _preprocess_checklist_enumerations(self, latex_full_text: str) -> str:
        self._log("Preprocessing LaTeX checklist 'enumerate' environments...", "debug")
        initial_content = latex_full_text

        def replacement_validator_transformer(matchobj):
            begin_env = matchobj.group(1)
            content = matchobj.group(2)
            end_env = matchobj.group(3)
            
            content_lines = content.splitlines()
            
            main_item_indices = [
                i for i, line in enumerate(content_lines) 
                if re.match(r"^\s*\\item(?!\s*\[)", line.strip())
            ]

            if not main_item_indices:
                return matchobj.group(0) 

            major_item_line_blocks = []
            for k in range(len(main_item_indices)):
                start_idx = main_item_indices[k]
                end_idx = main_item_indices[k+1] if k + 1 < len(main_item_indices) else len(content_lines)
                major_item_line_blocks.append(content_lines[start_idx:end_idx])

            if not major_item_line_blocks:
                 return matchobj.group(0)

            all_sub_blocks_are_valid = True
            for item_block_lines_list in major_item_line_blocks:
                if not self._validate_checklist_major_item_structure(item_block_lines_list):
                    all_sub_blocks_are_valid = False
                    break
            
            if all_sub_blocks_are_valid:
                self._log(f"Found a valid QAJG checklist block. Applying transformations.", "debug")
                processed_content = self._transform_checklist_enumerate_block_content(content)
                return begin_env + processed_content + end_env
            else:
                self._log(f"Enumerate block did not match strict QAJG structure. Leaving unchanged.", "debug")
                return matchobj.group(0)

        fixed_text = re.sub(
            r"(\\begin\{enumerate\})([\s\S]*?)(\\end\{enumerate\})",
            replacement_validator_transformer,
            latex_full_text
        )
        
        if fixed_text != initial_content:
            self._log("Applied checklist enumeration preprocessing.", "info")
        else:
            self._log("No changes made during checklist enumeration preprocessing.", "debug")
        return fixed_text
    # --- End Checklist Preprocessing Methods ---

    def _run_pandoc_conversion(self, tex_content_for_pandoc, pandoc_timeout=None):
        final_md_path = self.final_output_folder_path / "paper.md"; tmp_tex_path_obj = None
        pandoc_local_out = "_pandoc_temp_paper.md"; original_cwd = Path.cwd()
        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tex", encoding="utf-8", dir=self.folder_path) as tmp_f:
                tmp_f.write(tex_content_for_pandoc)
                tmp_tex_path_obj = Path(tmp_f.name)

            cmd = ["pandoc", str(tmp_tex_path_obj.name), "-f", "latex", "-t", "gfm-tex_math_dollars", "--strip-comments", "--wrap=none", "-o", pandoc_local_out]
            if self.template_path and Path(self.template_path).exists():
                 cmd.extend(["--template", self.template_path])
            else:
                self._log(f"Pandoc template '{self.template_path}' not found or not specified. Using Pandoc default.", "info")

            os.chdir(self.folder_path)
            self._log(f"Running Pandoc in '{self.folder_path}': {' '.join(cmd)}", "debug")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore', timeout=pandoc_timeout)
            
            created_md_path_local = self.folder_path / pandoc_local_out

            if proc.returncode == 0 and created_md_path_local.exists():
                self._log("Pandoc conversion step successful.", "debug")
                self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
                shutil.move(str(created_md_path_local), str(final_md_path))
                
                if final_md_path.exists():
                    self._log(f"Markdown file generated at: {final_md_path}", "debug")

                    with open(final_md_path, "r", encoding="utf-8", errors="ignore") as md_f:
                        markdown_content_after_pandoc = md_f.read()

                    output_figures_subdir = self.final_output_folder_path / "figures"
                    output_figures_subdir.mkdir(parents=True, exist_ok=True)

                    copied_figure_details = self._find_and_copy_figures_from_markdown(
                        markdown_content_after_pandoc,
                        output_figures_subdir
                    )

                    updated_markdown_content = self._convert_and_update_figure_paths_in_markdown(
                        markdown_content_after_pandoc,
                        copied_figure_details, 
                        self.final_output_folder_path
                    )
                    with open(final_md_path, "w", encoding="utf-8") as md_f:
                        md_f.write(updated_markdown_content)
                    
                    if copied_figure_details:
                        self._log("Markdown figure paths updated and relevant images processed into 'figures' subdirectory.", "success")
                    else:
                        self._log("No local figures found in Markdown to process for copy/conversion, but checked for embed->img conversion.", "debug")
                    return True
                else:
                    self._log(f"Pandoc seemed to succeed but final markdown file '{final_md_path}' not found after move.", "error")
                    return False
            else:
                self._log(f"Pandoc conversion failed. Return code: {proc.returncode}", "error")
                self._log(f"Pandoc STDOUT: {proc.stdout[:500]}...", "debug" if proc.stdout else "debug")
                self._log(f"Pandoc STDERR: {proc.stderr[:1000]}...", "error" if proc.stderr else "debug")
                if created_md_path_local.exists():
                    try: created_md_path_local.unlink()
                    except Exception as e_del_tmp: self._log(f"Could not delete temporary pandoc output '{created_md_path_local}': {e_del_tmp}", "warn")
                return False
        except subprocess.TimeoutExpired:
            self._log(f"Pandoc command timed out after {pandoc_timeout} seconds.", "error")
            return False
        except Exception as e:
            self._log(f"An exception occurred during Pandoc conversion: {e}", "error")
            return False
        finally:
            if Path.cwd() != original_cwd:
                os.chdir(original_cwd)
            if tmp_tex_path_obj and tmp_tex_path_obj.exists():
                try: tmp_tex_path_obj.unlink()
                except Exception as e_unlink: self._log(f"Could not delete temporary TeX file '{tmp_tex_path_obj}': {e_unlink}", "warn")
            pandoc_temp_output_in_src = self.folder_path / pandoc_local_out
            if pandoc_temp_output_in_src.exists():
                try: pandoc_temp_output_in_src.unlink()
                except Exception as e_del_tmp2: self._log(f"Could not delete temp pandoc output '{pandoc_temp_output_in_src}': {e_del_tmp2}", "warn")

    def convert_to_markdown(self, output_folder_path_str):
        if not self.find_main_tex_file(): return False
        self.final_output_folder_path = Path(output_folder_path_str).resolve()
        try: self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
        except Exception as e: self._log(f"Output dir error: {e}", "error"); return False

        # _generate_bbl_content now returns the fully processed BBL string (from cache or new processing)
        processed_bbl_with_abstracts_and_keys = self._generate_bbl_content()
        if processed_bbl_with_abstracts_and_keys is None:
            self._log("Failed to generate or process BBL content. Cannot proceed.", "error")
            return False

        problematic_macros = r"""\providecommand{\linebreakand}{\par\noindent\ignorespaces} \providecommand{\email}[1]{\texttt{#1}} \providecommand{\IEEEauthorblockN}[1]{#1\par} \providecommand{\IEEEauthorblockA}[1]{#1\par} \providecommand{\and}{\par\noindent\ignorespaces} \providecommand{\And}{\par\noindent\ignorespaces} \providecommand{\AND}{\par\noindent\ignorespaces} \providecommand{\IEEEoverridecommandlockouts}{} \providecommand{\CLASSINPUTinnersidemargin}{} \providecommand{\CLASSINPUToutersidemargin}{} \providecommand{\CLASSINPUTtoptextmargin}{} \providecommand{\CLASSINPUTbottomtextmargin}{} \providecommand{\CLASSOPTIONcompsoc}{} \providecommand{\CLASSOPTIONconference}{} \providecommand{\@toptitlebar}{} \providecommand{\@bottomtitlebar}{} \providecommand{\@thanks}{} \providecommand{\@notice}{} \providecommand{\@noticestring}{} \providecommand{\acksection}{} \newenvironment{ack}{\par\textbf{Acknowledgments}\par}{\par} \providecommand{\answerYes}[1]{[Yes] ##1} \providecommand{\answerNo}[1]{[No] ##1} \providecommand{\answerNA}[1]{[NA] ##1} \providecommand{\answerTODO}[1]{[TODO] ##1} \providecommand{\justificationTODO}[1]{[TODO] ##1} \providecommand{\textasciitilde}{~} \providecommand{\textasciicircum}{^} \providecommand{\textbackslash}{\symbol{92}}"""

        pandoc_attempts_config = [
            {"mode": "venue_only", "desc": "Venue-specific styles commented", "timeout": 30},
            {"mode": "original", "desc": "Original TeX content (no script style commenting)", "timeout": 30},
            {"mode": "all_project", "desc": "All project-specific styles commented", "timeout": 45}
        ]

        initial_main_tex_content_for_processing = self.original_main_tex_content
        # Step 1: Expand includes and preprocess table environments
        expanded_and_table_processed_tex = self._preprocess_latex_table_environments(initial_main_tex_content_for_processing)
        # Step 2: Preprocess checklist enumerations on the already expanded and table-processed content
        fully_preprocessed_tex = self._preprocess_checklist_enumerations(expanded_and_table_processed_tex)


        for i, attempt_config in enumerate(pandoc_attempts_config):
            self._log(f"Pandoc Conversion Attempt {i+1}/{len(pandoc_attempts_config)} ({attempt_config['desc']})...", "info")

            current_tex_base = fully_preprocessed_tex # Use the fully preprocessed content

            if attempt_config["mode"] == "venue_only":
                style_modified_tex = self._comment_out_style_packages(current_tex_base, mode="venue_only")
            elif attempt_config["mode"] == "all_project":
                style_modified_tex = self._comment_out_style_packages(current_tex_base, mode="all_project")
            else: 
                style_modified_tex = current_tex_base

            tex_with_final_bib = self._simple_inline_processed_bbl(style_modified_tex, processed_bbl_with_abstracts_and_keys)
            final_tex_for_pandoc = problematic_macros + "\n" + tex_with_final_bib

            if self._run_pandoc_conversion(final_tex_for_pandoc, pandoc_timeout=attempt_config['timeout']):
                self._log(f"Pandoc conversion successful on attempt {i+1}.", "success")
                return True

            if i < len(pandoc_attempts_config) - 1:
                self._log(f"Pandoc attempt {i+1} failed. Trying next strategy.", "warn")
            else:
                self._log("All Pandoc conversion attempts failed.", "error")

        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts LaTeX to Markdown with advanced abstract fetching and table processing.")
    parser.add_argument("project_folder", help="Path to LaTeX project folder.")
    parser.add_argument("-o", "--output_folder", default=None, help="Output folder. Default: '[project_name]_output'.")
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=True, help="Suppress info messages.")
    parser.add_argument("--template", default="template.md", help="Pandoc Markdown template. Default: 'template.md'.")
    parser.add_argument("--openalex-email", default=os.environ.get("OPENALEX_EMAIL", "your-email@example.com"), help="Your email for OpenAlex API. Can also be set via OPENALEX_EMAIL environment variable.")
    parser.add_argument("--elsevier-api-key", default=os.environ.get("ELSEVIER_API_KEY"), help="Your Elsevier API key. Can also be set via ELSEVIER_API_KEY environment variable.")
    parser.add_argument("--springer-api-key", default=os.environ.get("SPRINGER_API_KEY"), help="Your Springer Nature API key. Can also be set via SPRINGER_API_KEY environment variable.")
    parser.add_argument("--poppler-path", default=os.environ.get("POPPLER_PATH"), help="Path to Poppler binaries directory for pdf2image. Can also be set via POPPLER_PATH environment variable.")


    args = parser.parse_args()
    project_path = Path(args.project_folder).resolve()
    if not project_path.is_dir(): print(f"Error: Project folder '{args.project_folder}' not found.", file=sys.stderr); sys.exit(1)
    output_folder_path = Path(args.output_folder).resolve() if args.output_folder else project_path.parent / f"{project_path.name}_output"

    template_fpath_resolved = Path(args.template)
    final_template_path_str = None

    if template_fpath_resolved.is_file():
        final_template_path_str = str(template_fpath_resolved.resolve())
    else:
        script_dir_template = (Path(__file__).parent / args.template).resolve()
        if script_dir_template.is_file():
            final_template_path_str = str(script_dir_template)
            print(f"[*] Info: Template '{args.template}' not found directly, using template from script directory: {final_template_path_str}")
        else:
            cwd_template = (Path.cwd() / args.template).resolve()
            if cwd_template.is_file():
                final_template_path_str = str(cwd_template)
                print(f"[*] Info: Template '{args.template}' not found directly or in script dir, using template from CWD: {final_template_path_str}")
            else:
                print(f"[!] Warning: Pandoc template '{args.template}' not found in standard locations. Pandoc will use its default.", file=sys.stderr)

    converter = LatexToMarkdownConverter(
        str(project_path),
        verbose=args.verbose,
        template_path=final_template_path_str, 
        openalex_email=args.openalex_email,
        elsevier_api_key=args.elsevier_api_key,
        springer_api_key=args.springer_api_key,
        poppler_path=args.poppler_path # Pass Poppler path
    )
    converter._log(f"Processing LaTeX project in: '{project_path}'", "info")

    if converter.convert_to_markdown(str(output_folder_path)):
        converter._log(f"Conversion successful. Output: '{output_folder_path / 'paper.md'}'", "success")
    else: converter._log("Conversion failed.", "error")

    converter._log("Dependencies: Pandoc, LaTeX (pdflatex, bibtex), requests.", "info")
    if converter.PdfReader is None: converter._log("Optional for PDF abstracts: PyPDF2 (pip install PyPDF2)", "info")
    if converter.BeautifulSoup is None: converter._log("Optional for HTML abstracts: beautifulsoup4 (pip install beautifulsoup4)", "info")
    if converter.PILImage is None: converter._log("Optional for image conversion: Pillow (pip install Pillow)", "info")
    if converter.convert_from_path is None: converter._log("Optional for PDF to PNG: pdf2image (pip install pdf2image) and Poppler.", "info")

