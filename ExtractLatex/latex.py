#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import subprocess # For running external commands (pdflatex, bibtex, pandoc)
import tempfile # For creating temporary files
from pathlib import Path
import argparse
import sys
import shutil # For copying files
import urllib.parse # For decoding URL-encoded paths in Markdown
import requests # For Semantic Scholar API
import time     # For politely pausing between API calls

# List of common venue style file name patterns (regex)
# Users can extend this list.
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
    # Add more generic conference style names if known
    # r"conference", # Potentially too broad, use with caution
    # r"proceeding",
    r"IEEEtran", 
    r"ieeeconf",
    r"acmart" 
]


class LatexToMarkdownConverter:
    def __init__(self, folder_path_str, verbose=True, template_path=None): 
        self.folder_path = Path(folder_path_str).resolve()
        self.main_tex_path = None
        self.original_main_tex_content = ""
        self.verbose = verbose
        self.template_path = template_path
        self.final_output_folder_path = None # Will be set in convert_to_markdown

    def _log(self, message, level="info"):
        """Helper function for conditional printing."""
        if level == "error":
            print(f"[-] Error: {message}", file=sys.stderr)
        elif level == "warn":
            print(f"[!] Warning: {message}", file=sys.stderr)
        elif self.verbose:
            if level == "info":
                print(f"[*] {message}")
            elif level == "success":
                print(f"[+] {message}")
            elif level == "debug": 
                print(f"    [*] {message}")


    def find_main_tex_file(self):
        """Finds and stores the main .tex file containing \\begin{document}."""
        self._log("Finding main .tex file...")
        for path in self.folder_path.rglob("*.tex"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if r'\begin{document}' in content:
                        self.main_tex_path = path
                        self.original_main_tex_content = content
                        self._log(f"Main .tex file found: {self.main_tex_path}", "success")
                        return True
            except Exception as e:
                self._log(f"Could not read file {path} due to {e}", "warn")
                continue
        self._log(f"No main .tex file with \\begin{document} found in '{self.folder_path}'.", "error")
        return False

    def _get_project_sty_basenames(self):
        """Finds all .sty files in the project folder and returns their basenames."""
        self._log("Searching for all custom .sty files in project folder (for fallback)...", "debug")
        sty_basenames = []
        for sty_path in self.folder_path.rglob("*.sty"):
            if sty_path.stem not in ["article", "report", "book", "amsmath", "graphicx", "geometry", "hyperref", "inputenc", "fontenc", "babel", "xcolor", "listings", "fancyhdr", "enumitem", "parskip", "setspace", "tocbibind", "titling", "titlesec", "etoolbox", "iftex", "xparse", "expl3", "l3keys2e", "natbib", "biblatex", "microtype", "amsfonts", "amssymb", "amsthm", "mathtools", "soul", "url", "booktabs", "float", "caption", "subcaption", "multirow", "multicol", "threeparttable", "xspace", "textcomp", "makecell", "tcolorbox", "wasysym", "colortbl", "algorithmicx", "algorithm", "algpseudocode", "marvosym", "ulem", "trimspaces", "environ", "keyval", "graphics", "trig", "ifvtex"]:
                sty_basenames.append(sty_path.stem)
        if sty_basenames:
            self._log(f"Found potentially project-specific .sty files (basenames): {', '.join(sty_basenames)}", "debug")
        else:
            self._log("No unique project-specific .sty files found in the project folder.", "debug")
        return sty_basenames

    def _comment_out_style_packages(self, tex_content, mode="venue_only"):
        """
        Comments out \\usepackage commands.
        mode="venue_only": Uses VENUE_STYLE_PATTERNS.
        mode="all_project": Uses all .sty files found in the project folder (excluding common ones).
        """
        self._log(f"Commenting out \\usepackage commands (mode: {mode})...")
        modified_content = tex_content
        initial_content = tex_content 
        
        if mode == "venue_only":
            self._log(f"Targeting venue-specific styles based on patterns.", "debug")
            for style_pattern_re in VENUE_STYLE_PATTERNS:
                pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{(" + style_pattern_re + r")\}[^\n]*)$"
                modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE | re.IGNORECASE)

        elif mode == "all_project":
            styles_to_comment_out = self._get_project_sty_basenames() 
            if not styles_to_comment_out:
                self._log("No project-specific .sty files to comment out for 'all_project' mode.", "info")
                return tex_content
            self._log(f"Targeting all project-specific styles: {', '.join(styles_to_comment_out)}", "debug")
            for sty_basename in styles_to_comment_out:
                pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{" + re.escape(sty_basename) + r"\}[^\n]*)$"
                modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE)
        else:
            self._log(f"Unknown mode '{mode}' for _comment_out_style_packages.", "warn")
            return tex_content
        
        if modified_content != initial_content:
            self._log(f"Successfully commented out relevant \\usepackage commands (mode: {mode}).", "success")
        else:
            self._log(f"No \\usepackage commands were modified (mode: {mode}).", "info")
        return modified_content

    def _generate_bbl_content(self):
        if not self.main_tex_path: 
            self._log("Cannot generate .bbl file: Main .tex file not identified.", "error")
            return None

        main_file_stem = self.main_tex_path.stem
        specific_bbl_path = self.folder_path / f"{main_file_stem}.bbl"

        if specific_bbl_path.exists() and specific_bbl_path.is_file():
            self._log(f"Found specific .bbl file: {specific_bbl_path}. Using its content.", "info")
            try:
                with open(specific_bbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                self._log(f"Error reading specific .bbl file '{specific_bbl_path}': {e}", "warn")
        
        self._log(f"Specific .bbl file '{specific_bbl_path.name}' not found or unreadable. Searching for any .bbl file in project root.", "debug")
        bbl_files_in_project = list(self.folder_path.glob("*.bbl"))
        if bbl_files_in_project:
            fallback_bbl_path = bbl_files_in_project[0] 
            self._log(f"Found fallback .bbl file: {fallback_bbl_path}. Using its content.", "warn")
            try:
                with open(fallback_bbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                self._log(f"Error reading fallback .bbl file '{fallback_bbl_path}': {e}", "warn")
                self._log(f"Proceeding to attempt .bbl regeneration.", "info")
        else:
            self._log(f"No existing .bbl files found in '{self.folder_path}'. Attempting to generate.", "info")

        self._log("Generating .bbl file content via pdflatex and bibtex...")
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "-draftmode", self.main_tex_path.name],
            ["bibtex", main_file_stem],
        ]
        original_cwd = Path.cwd()
        log_file_path = self.folder_path / f"{main_file_stem}.log" 
        self._log(f"Changing CWD to: {self.folder_path}", "debug")
        os.chdir(self.folder_path) 
        try:
            for i, cmd_args in enumerate(commands):
                command_str = " ".join(str(arg) for arg in cmd_args)
                self._log(f"Running command: {command_str} (in CWD: {Path.cwd()})", "debug")
                if cmd_args[0] == "pdflatex" and log_file_path.exists():
                    try: log_file_path.unlink()
                    except Exception as e_unlink: self._log(f"Could not delete log file {log_file_path}: {e_unlink}", "warn")
                elif cmd_args[0] == "bibtex":
                    blg_file_path = self.folder_path / f"{main_file_stem}.blg"
                    if blg_file_path.exists():
                        try: blg_file_path.unlink()
                        except Exception as e_unlink: self._log(f"Could not delete .blg file {blg_file_path}: {e_unlink}", "warn")
                process = subprocess.run(cmd_args, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                if process.returncode != 0:
                    self._log(f"Error running command: {command_str}", "error")
                    self._log(f"--- STDOUT --- (Max 1000 chars)\n{process.stdout[:1000]}", "error")
                    self._log(f"--- STDERR --- (Max 1000 chars)\n{process.stderr[:1000]}", "error")
                    log_to_check = log_file_path if cmd_args[0] == "pdflatex" else (self.folder_path / f"{main_file_stem}.blg")
                    self._log(f"Suggestion: Check the full log file for errors: {log_to_check.resolve()}", "warn")
                    if cmd_args[0] == "bibtex" and ("I found no \\bibdata command" in process.stdout or "I found no \\bibstyle command" in process.stdout):
                         self._log("BibTeX specific error: No \\bibdata or \\bibstyle command. Ensure original .tex calls \\bibliography and \\bibliographystyle.", "error")
                    elif cmd_args[0] == "pdflatex" and i == 0:
                        self._log("Critical: Initial pdflatex run failed. Cannot generate .aux for BibTeX.", "error")
                        self._log("Common causes: Missing LaTeX packages (.sty files) or class files not found by TeX Live.", "error")
                        self._log("On Colab/Linux, try installing more TeX Live components, e.g.:\n        !sudo apt-get install texlive-latex-extra texlive-publishers texlive-science texlive-fonts-extra", "error")
                        return None 
                else:
                    self._log(f"Command '{command_str}' executed successfully.", "success")
            
            generated_bbl_path = Path(f"{main_file_stem}.bbl") 
            if generated_bbl_path.exists():
                self._log(f".bbl file generated: {generated_bbl_path.resolve()}", "success")
                with open(generated_bbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            else:
                self._log(f".bbl file was not generated at '{generated_bbl_path.resolve()}'. Check LaTeX/BibTeX logs.", "error")
                aux_file = Path(f"{main_file_stem}.aux")
                if not aux_file.exists():
                    self._log(f"Auxiliary file '{aux_file.name}' not found. Initial pdflatex run likely failed critically.", "error")
                else:
                    with open(aux_file, 'r', encoding='utf-8', errors='ignore') as af:
                        aux_content = af.read()
                        if r'\bibdata' not in aux_content:
                             self._log(r"\bibdata command missing in .aux file. Check \bibliography{...} in original .tex.", "error")
                        if not (re.search(r'\\citation\{', aux_content) or r'\nocite{*}' in aux_content):
                             self._log(r"No \citation or \nocite{*} found in .aux file. Bibliography might be empty.", "warn")
                return None 
        except FileNotFoundError as e:
            self._log(f"FileNotFoundError: {e}. Ensure LaTeX (pdflatex, bibtex) is installed and in your system PATH.", "error")
            return None
        except Exception as e:
            self._log(f"An unexpected error occurred during .bbl generation: {e}", "error")
            return None
        finally:
            self._log(f"Changing CWD back to: {original_cwd}", "debug")
            os.chdir(original_cwd)

    def _latex_escape_abstract(self, text: str) -> str:
        """Escapes special LaTeX characters in the abstract text."""
        if not text: return ""
        # Order matters for backslash
        text = text.replace('\\', r'\textbackslash{}')
        text = text.replace('{', r'\{')
        text = text.replace('}', r'\}')
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('_', r'\_')
        text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('^', r'\textasciicircum{}')
        # Handle newlines: LaTeX typically ignores single newlines in text.
        # Convert double newlines to \par for paragraph breaks.
        # text = re.sub(r'\n\s*\n', '\\par ', text) 
        text = text.replace('\n', ' ') # Convert single newlines to space
        return text

    def _fetch_abstract_from_semantic_scholar(self, title: str, authors_str: str = ""):
        """
        Fetches abstract from Semantic Scholar API based on title and optionally authors.
        Returns the abstract string or None if not found or an error occurs.
        """
        if not title:
            return None
        
        try:
            query_title = re.sub(r'\s+', ' ', title).strip()
            # Basic cleaning of title for search (remove common LaTeX formatting)
            clean_title_for_search = re.sub(r'\{.*?\}', '', query_title) 
            clean_title_for_search = re.sub(r'\\[a-zA-Z]+', '', clean_title_for_search)
            clean_title_for_search = clean_title_for_search.strip()

            if not clean_title_for_search:
                self._log(f"Skipping Semantic Scholar search due to empty title after cleaning: '{title}'", "debug")
                return None

            search_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote_plus(clean_title_for_search)}&fields=title,authors,abstract&limit=3"
            
            self._log(f"Querying Semantic Scholar: {search_url}", "debug")
            
            headers = {'User-Agent': 'LatexToMarkdownConverter/1.0 (Python Script)'} 
            response = requests.get(search_url, headers=headers, timeout=15) # 15s timeout
            response.raise_for_status() 
            
            data = response.json()
            
            if data.get("total", 0) > 0 and data.get("data"):
                for paper_data in data["data"]:
                    s2_title = paper_data.get("title", "")
                    s2_title_clean = re.sub(r'[^a-z0-9]', '', s2_title.lower())
                    query_title_clean_cmp = re.sub(r'[^a-z0-9]', '', clean_title_for_search.lower())

                    if query_title_clean_cmp in s2_title_clean or s2_title_clean in query_title_clean_cmp or \
                       (len(query_title_clean_cmp) > 10 and len(s2_title_clean) > 10 and \
                        (query_title_clean_cmp[:10] == s2_title_clean[:10] or query_title_clean_cmp[-10:] == s2_title_clean[-10:])): # Basic similarity
                        if paper_data.get("abstract"):
                            self._log(f"Found abstract for '{title}' (Matched S2 title: '{s2_title}')", "debug")
                            time.sleep(0.5) # Polite delay
                            return paper_data["abstract"]
                        else:
                            self._log(f"Paper '{title}' (S2: '{s2_title}') found but no abstract.", "debug")
                self._log(f"No suitable match with abstract found for '{title}' among S2 results.", "debug")
            else:
                self._log(f"No papers found on Semantic Scholar for title: '{clean_title_for_search}'", "debug")
                
        except requests.exceptions.RequestException as e:
            self._log(f"Semantic Scholar API request failed for '{title}': {e}", "warn")
        except Exception as e:
            self._log(f"Error processing Semantic Scholar response for '{title}': {e}", "warn")
            
        time.sleep(0.5) # Polite delay even on failure
        return None

    def _inline_bibliography(self, tex_content, bbl_content_original):
        if not bbl_content_original:
            self._log("BBL content is empty. Bibliography will not be inlined.", "warn")
            return tex_content
        self._log("Inlining .bbl content into .tex content (with abstract fetching)...")

        bbl_content = bbl_content_original # Work on a copy

        # Extract and clean preamble (before \begin{thebibliography})
        bbl_preamble = ""
        bibliography_env_start = r"\begin{thebibliography}{}" # Default
        
        begin_env_match = re.search(r"(\\begin\{thebibliography\}\{[^}]*\})", bbl_content)
        if begin_env_match:
            bibliography_env_start = begin_env_match.group(1)
            preamble_end_index = begin_env_match.start()
            bbl_preamble = bbl_content[:preamble_end_index]
            
            # Clean the extracted preamble
            bbl_cleanup_patterns = [
                re.compile(r"\\providecommand\{\\natexlab\}\[1\]\{#1\}\s*", flags=re.DOTALL),
                re.compile(r"\\providecommand\{\\url\}\[1\]\{\\texttt\{#1\}\}\s*", flags=re.DOTALL),
                re.compile(r"\\providecommand\{\\doi\}\[1\]\{doi:\s*#1\}\s*", flags=re.DOTALL),
                re.compile(r"\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.DOTALL),
            ]
            for pattern_re in bbl_cleanup_patterns:
                bbl_preamble = pattern_re.sub("", bbl_preamble)
            bbl_preamble = bbl_preamble.strip()
        else:
            self._log("Could not find '\\begin{thebibliography}{...}' in .bbl content. Proceeding with raw content for items.", "warn")
            bbl_content = re.sub(r"\\providecommand\{\\natexlab\}\[1\]\{#1\}\s*", "", bbl_content, flags=re.DOTALL)

        items_text_start_index = bbl_content.find(bibliography_env_start) + len(bibliography_env_start) if begin_env_match else 0
        items_text_end_index = bbl_content.rfind(r"\end{thebibliography}") if begin_env_match else len(bbl_content)
        
        bbl_items_text = bbl_content[items_text_start_index:items_text_end_index].strip()

        if not bbl_items_text:
            self._log("No bibitem content found to process after isolating.", "warn")
            final_bbl_content_for_inline = bbl_content_original
        else:
            bibitem_starts = [match.start() for match in re.finditer(r'\\bibitem', bbl_items_text)]
            processed_bibitems_parts = []

            if not bibitem_starts:
                self._log("No \\bibitem entries found within isolated BBL items text.", "warn")
                processed_bibitems_parts.append(bbl_items_text) 
            else:
                if bibitem_starts[0] > 0:
                    leading_text = bbl_items_text[:bibitem_starts[0]].strip()
                    if leading_text: processed_bibitems_parts.append(leading_text)

                for i in range(len(bibitem_starts)):
                    start_idx = bibitem_starts[i]
                    end_idx = bibitem_starts[i+1] if (i + 1) < len(bibitem_starts) else len(bbl_items_text)
                    
                    item_full_text = bbl_items_text[start_idx:end_idx].strip()
                    if not item_full_text: continue

                    bibitem_prefix_match = re.match(r"(\\bibitem(?:\[[^\]]*\])?\{[^\}]+\})", item_full_text)
                    if not bibitem_prefix_match:
                        processed_bibitems_parts.append(item_full_text) 
                        continue
                    
                    bibitem_prefix = bibitem_prefix_match.group(1)
                    details_str = item_full_text[len(bibitem_prefix):].strip()
                    
                    parts = details_str.split(r'\newblock')
                    authors_str = parts[0].strip().rstrip('.') if parts else ""
                    title_str = parts[1].strip().rstrip('.') if len(parts) > 1 else ""
                    
                    current_item_reconstructed_parts = [bibitem_prefix]
                    if authors_str: current_item_reconstructed_parts.append(authors_str + ".")
                    
                    if len(parts) > 1: 
                        current_item_reconstructed_parts.append(r"\newblock " + title_str + ".")
                        for j in range(2, len(parts)):
                            current_item_reconstructed_parts.append(r"\newblock " + parts[j].strip())
                    
                    current_item_reconstructed = " ".join(current_item_reconstructed_parts)

                    abstract_text = None
                    if title_str:
                        self._log(f"Attempting to fetch abstract for title: '{title_str[:70]}...'", "debug")
                        abstract_text = self._fetch_abstract_from_semantic_scholar(title_str, authors_str)

                    if abstract_text:
                        escaped_abstract = self._latex_escape_abstract(abstract_text)
                        current_item_reconstructed += f" \\newblock \\textbf{{Abstract:}} {escaped_abstract}"
                        self._log(f"Abstract added for: '{title_str[:70]}...'", "debug")
                    
                    processed_bibitems_parts.append(current_item_reconstructed)
            
            final_bbl_items_joined = "\n\n".join(p.strip() for p in processed_bibitems_parts if p.strip())
            final_bbl_content_for_inline = (bbl_preamble + "\n" if bbl_preamble else "") + \
                                   bibliography_env_start + "\n" + \
                                   final_bbl_items_joined + "\n" + \
                                   r"\end{thebibliography}"
        
        references_heading_tex = "\n\n\\section*{References}\n\n" 
        content_to_inline = references_heading_tex + final_bbl_content_for_inline

        modified_tex_content = re.sub(r"^\s*\\bibliographystyle\{[^{}]*\}\s*$", "", tex_content, flags=re.MULTILINE)
        bibliography_command_pattern = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"
        
        if re.search(bibliography_command_pattern, modified_tex_content, flags=re.MULTILINE):
            modified_tex_content = re.sub(bibliography_command_pattern, lambda m: content_to_inline, modified_tex_content, count=1, flags=re.MULTILINE)
            self._log("Replaced \\bibliography command with inlined content (with abstracts).", "success")
        else:
            self._log("\\bibliography{...} command not found. Appending inlined content before \\end{document}.", "warn")
            end_document_match = re.search(r"\\end\{document\}", modified_tex_content)
            if end_document_match:
                insertion_point = end_document_match.start()
                modified_tex_content = modified_tex_content[:insertion_point] + content_to_inline + "\n" + modified_tex_content[insertion_point:]
            else:
                self._log("\\end{document} not found. Appending inlined content to the end.", "warn")
                modified_tex_content += "\n" + content_to_inline
        
        modified_tex_content = re.sub(bibliography_command_pattern, "", modified_tex_content, flags=re.MULTILINE)
        modified_tex_content = re.sub(r"^\s*\\nobibliography\{[^{}]*(?:,[^{}]*)*\}\s*$", "", modified_tex_content, flags=re.MULTILINE)
        
        return modified_tex_content

    def _find_and_copy_figures_from_markdown(self, markdown_content, output_figure_folder_path):
        if not markdown_content:
            self._log("Cannot find figures: Markdown content is empty.", "warn")
            return
        self._log(f"Searching for and copying figures (from Markdown) to: {output_figure_folder_path}")
        markdown_image_pattern = r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)"
        html_img_pattern = r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"
        html_embed_pattern = r"<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"
        html_source_pattern = r"<source\s+[^>]*?srcset\s*=\s*[\"']([^\"'\s]+)(?:\s+\S+)?[\"'][^>]*?>"
        image_paths_in_md = []
        for pattern in [markdown_image_pattern, html_img_pattern, html_embed_pattern, html_source_pattern]:
            for match in re.finditer(pattern, markdown_content, flags=re.IGNORECASE):
                image_paths_in_md.append(match.group(1))
        if not image_paths_in_md:
            self._log("No image references found in the Markdown content.", "info")
            return
        self._log(f"Found {len(image_paths_in_md)} potential image references in Markdown.", "debug")
        common_image_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.tikz', '.svg'] 
        figures_copied_count = 0
        copied_files_set = set() 
        for image_path_in_md_raw in image_paths_in_md:
            try:
                decoded_image_path_str = urllib.parse.unquote(image_path_in_md_raw)
            except Exception as e_decode:
                self._log(f"Could not URL-decode image path '{image_path_in_md_raw}': {e_decode}", "warn")
                decoded_image_path_str = image_path_in_md_raw 
            if urllib.parse.urlparse(decoded_image_path_str).scheme in ['http', 'https']:
                self._log(f"Skipping web URL: {decoded_image_path_str}", "debug")
                continue
            potential_src_image_path = (self.folder_path / decoded_image_path_str).resolve()
            found_image_file_abs = None
            if potential_src_image_path.exists() and potential_src_image_path.is_file():
                found_image_file_abs = potential_src_image_path
            else:
                if not Path(decoded_image_path_str).suffix:
                    for ext in common_image_extensions:
                        path_with_ext_abs = (self.folder_path / (decoded_image_path_str + ext)).resolve()
                        if path_with_ext_abs.exists() and path_with_ext_abs.is_file():
                            found_image_file_abs = path_with_ext_abs
                            break
            if found_image_file_abs:
                if found_image_file_abs in copied_files_set:
                    self._log(f"Skipping already copied file: {found_image_file_abs.name}", "debug")
                    continue
                try:
                    output_figure_folder_path.mkdir(parents=True, exist_ok=True)
                    relative_dest_path_component = Path(image_path_in_md_raw).name 
                    if Path(image_path_in_md_raw).parent != Path('.'): 
                         relative_dest_path_component = Path(image_path_in_md_raw) 

                    destination_path = (output_figure_folder_path / relative_dest_path_component).resolve()
                    destination_path.parent.mkdir(parents=True, exist_ok=True) 

                    shutil.copy2(str(found_image_file_abs), str(destination_path))
                    self._log(f"Copied figure: '{found_image_file_abs.name}' to '{destination_path}'", "success")
                    copied_files_set.add(found_image_file_abs)
                    figures_copied_count += 1
                except Exception as e_copy:
                    self._log(f"Could not copy figure '{found_image_file_abs}' to '{destination_path}': {e_copy}", "warn")
            else:
                self._log(f"Referenced image '{decoded_image_path_str}' (from Markdown) not found in project folder '{self.folder_path}'. Raw path from MD: '{image_path_in_md_raw}'", "warn")
        if figures_copied_count > 0:
            self._log(f"Copied {figures_copied_count} unique figure(s) based on Markdown content.", "success")
        else:
            self._log("No new figures found in Markdown to copy.", "info")

    def _run_pandoc_conversion(self, tex_content_for_pandoc, pandoc_timeout=None):
        self._log(f"Preparing to run Pandoc (timeout: {pandoc_timeout}s)...", "debug")
        final_markdown_file_path = self.final_output_folder_path / "paper.md" 
        tmp_tex_file_path_str = "" 
        pandoc_local_output_name = "_pandoc_temp_paper.md" 
        original_cwd = Path.cwd()

        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tex", encoding="utf-8") as tmp_tex_file:
                tmp_tex_file.write(tex_content_for_pandoc)
                tmp_tex_file_path_str = tmp_tex_file.name 
            self._log(f"Temporary processed .tex file created at: {Path(tmp_tex_file_path_str).resolve()}", "debug")
            
            cmd_pandoc = [
                "pandoc", str(Path(tmp_tex_file_path_str).resolve()),
                "-f", "latex", 
                "-t", "markdown+raw_tex", 
                "--strip-comments", 
                "--template", self.template_path, 
                "--wrap=none", 
                "-o", pandoc_local_output_name 
            ]
            
            self._log(f"Pandoc will output locally to: {pandoc_local_output_name} (relative to project folder)", "debug")
            self._log(f"Running Pandoc command: {' '.join(cmd_pandoc)}", "debug")
            os.chdir(self.folder_path)
            
            process_pandoc = subprocess.run(cmd_pandoc, capture_output=True, text=True, 
                                            check=False, encoding='utf-8', errors='ignore',
                                            timeout=pandoc_timeout) 
            
            pandoc_created_md_path = self.folder_path / pandoc_local_output_name

            if process_pandoc.returncode == 0:
                self._log(f"Pandoc successfully created local Markdown: {pandoc_created_md_path}", "success")
                try:
                    shutil.move(str(pandoc_created_md_path), str(final_markdown_file_path))
                    self._log(f"Successfully moved Markdown to final destination: {final_markdown_file_path}", "success")
                    
                    if final_markdown_file_path.exists():
                        post_process_success = self.post_process_markdown(final_markdown_file_path)
                        if not post_process_success:
                            self._log("Markdown post-processing failed. Output might contain unconverted tables.", "warn")
                        
                        with open(final_markdown_file_path, "r", encoding="utf-8", errors="ignore") as md_file:
                            markdown_content_for_figures = md_file.read()
                        self._find_and_copy_figures_from_markdown(markdown_content_for_figures, self.final_output_folder_path)
                    else:
                        self._log(f"Final markdown file '{final_markdown_file_path}' not found after supposed move.", "error")
                        return False 
                    return True 
                except Exception as e_move_or_copy:
                    self._log(f"Error moving Pandoc output, post-processing, or copying figures: {e_move_or_copy}", "error")
                    if pandoc_created_md_path.exists():
                        try: pandoc_created_md_path.unlink()
                        except: pass
                    return False 
            else:
                self._log(f"Pandoc failed with error code {process_pandoc.returncode}:", "error")
                self._log(f"--- PANDOC STDOUT ---\n{process_pandoc.stdout}", "error")
                self._log(f"--- PANDOC STDERR ---\n{process_pandoc.stderr}", "error")
                if pandoc_created_md_path.exists():
                    try: pandoc_created_md_path.unlink()
                    except: pass
                return False
        except subprocess.TimeoutExpired:
            self._log(f"Pandoc command timed out after {pandoc_timeout} seconds.", "error")
            pandoc_created_md_path_on_timeout = self.folder_path / pandoc_local_output_name
            if pandoc_created_md_path_on_timeout.exists():
                try: pandoc_created_md_path_on_timeout.unlink()
                except: pass
            return False 
        except FileNotFoundError:
            self._log("Pandoc command not found. Ensure pandoc is installed and in your system PATH.", "error")
            return False
        except Exception as e:
            self._log(f"An unexpected error occurred during Pandoc conversion: {e}", "error")
            return False
        finally:
            if Path.cwd() != original_cwd:
                os.chdir(original_cwd)
            if tmp_tex_file_path_str and Path(tmp_tex_file_path_str).exists():
                try: Path(tmp_tex_file_path_str).unlink()
                except Exception: pass
        return False

    # --- Start: Methods for Post-Processing Markdown (Table Conversion) ---
    def _process_data_rows_to_markdown(self, table_data_str: str) -> str | None:
        data_lines = [line for line in table_data_str.strip().split('\n') if line.strip()]
        if not data_lines:
            self._log("Table post-processing: No data lines found in table block.", "debug")
            return None
        header_line_str = data_lines[0]
        header_cells_raw = [cell.strip() for cell in header_line_str.split('&')]
        num_columns = len(header_cells_raw)
        if num_columns == 0 or (num_columns == 1 and not header_cells_raw[0]):
            self._log("Table post-processing: Invalid header.", "debug")
            return None
        markdown_table_rows = ["| " + " | ".join(header_cells_raw) + " |"]
        markdown_table_rows.append("| " + " | ".join(["---"] * num_columns) + " |")
        for i in range(1, len(data_lines)):
            row_str = data_lines[i].strip()
            if not row_str: continue
            row_cells_raw = [cell.strip() for cell in row_str.split('&')]
            if row_cells_raw and row_cells_raw[-1].endswith('\\\\'):
                row_cells_raw[-1] = row_cells_raw[-1][:-2].strip()
            final_row_cells = [''] * num_columns 
            for j in range(num_columns):
                if j < len(row_cells_raw): final_row_cells[j] = row_cells_raw[j]
            markdown_table_rows.append("| " + " | ".join(final_row_cells) + " |")
        return "\n".join(markdown_table_rows)

    def _convert_tabular_block_to_markdown(self, block_text: str) -> str:
        """
        Takes the full text of a `::: {.tabular} ... :::` block,
        parses it, and returns a Markdown table string.
        If conversion fails or the block is not a valid target,
        returns the original block_text.
        """
        lines = block_text.strip().split('\n')

        if len(lines) < 3 or not lines[0].strip().startswith("::: {.tabular") or lines[-1].strip() != ":::":
            self._log("Table post-processing: Block does not match expected ::: {.tabular} ... ::: structure.", "debug")
            return block_text

        potential_content_lines = lines[1:-1]
        if not potential_content_lines:
            self._log("Table post-processing: No content found between ::: {.tabular} markers.", "debug")
            return block_text

        first_content_line_raw = potential_content_lines[0]
        remaining_content_lines = potential_content_lines[1:]
        
        first_line_stripped = first_content_line_raw.strip()
        
        colspec_candidate = "" # For logging, not strictly used beyond parsing
        header_part_on_first_line = first_line_stripped # Default to whole line being header

        # Case 1: First line does NOT contain an ampersand - it's either all colspec or all header
        if '&' not in first_line_stripped:
            # Heuristic check if the entire line is a column specifier
            # A colspec usually isn't very long unless it has many columns or complex p{} m{} b{} entries.
            # It should only contain valid colspec characters.
            # It should contain at least one of 'l', 'c', 'r', or '|'.
            # It should not contain words of 3+ letters unless part of p{}, m{}, b{}.
            is_likely_colspec_line = True
            if len(first_line_stripped) > 40 and not ('{' in first_line_stripped and '}' in first_line_stripped) :
                 is_likely_colspec_line = False
            # Check for words that are not p{...}, m{...}, b{...}
            # Split by space, check individual words if they are not l,c,r,|,etc. or p{...}
            simple_words = re.findall(r'[a-zA-Z]{3,}', first_line_stripped)
            pmb_pattern = re.compile(r'(p|m|b)\{[^}]*\}')
            for word in simple_words:
                if not pmb_pattern.fullmatch(word): # if it's a word but not a p{arg} type
                    is_likely_colspec_line = False
                    break
            if not re.fullmatch(r'[lcrpmb\s\|\@\{\}\>\<\d\.\*\(\)\-\+\\\!\$\^\_\~\%\'\[\]\#\~]*', first_line_stripped, re.IGNORECASE):
                 is_likely_colspec_line = False 
            if not any(c in first_line_stripped.lower() for c in 'lcr|'):
                 is_likely_colspec_line = False

            if is_likely_colspec_line:
                self._log(f"Table post-processing: First line '{first_line_stripped}' (no '&') identified as full column specifier.", "debug")
                colspec_candidate = first_line_stripped
                header_part_on_first_line = "" # No header on this line
            else:
                self._log(f"Table post-processing: First line '{first_line_stripped}' (no '&') treated as header.", "debug")
                # header_part_on_first_line remains first_line_stripped
        
        # Case 2: First line CONTAINS an ampersand - it has header data, might have leading colspec
        else:
            tokens = re.split(r'(\s+)', first_line_stripped) # Split by space, keeping space delimiters
            current_colspec_prefix = ""
            colspec_ended_at_token_index = -1

            for idx, token_group_part in enumerate(tokens):
                token = token_group_part.strip()
                if not token: # It's a space delimiter
                    current_colspec_prefix += token_group_part
                    continue

                # Check if the current token is a valid colspec component
                # Valid: l, c, r, |, p{...}, m{...}, b{...}, @{...}, *{N}c, digits, single special chars
                # Invalid: most words, or anything with an ampersand itself
                is_token_colspec_like = bool(token and re.fullmatch(r'[lcrpmb\|\@\{\}\>\<\d\.\*\(\)\-\+\\\!\$\^\_\~\%\'\[\]\#\~]+', token, re.IGNORECASE))
                
                # If token contains characters not allowed in colspec, or is a word like 'Model' (more than 2-3 chars, not p{}), it's not colspec
                if re.search(r'[a-zA-Z]{3,}', token) and not re.fullmatch(r'(p|m|b)\{[^}]*\}', token, re.IGNORECASE):
                    is_token_colspec_like = False
                if '&' in token: # ampersand means it's definitely not part of colspec
                    is_token_colspec_like = False


                if is_token_colspec_like:
                    current_colspec_prefix += token_group_part
                else:
                    # This token is not part of the colspec. The colspec (if any) ended before this token.
                    colspec_ended_at_token_index = idx
                    break
            
            # After iterating tokens, evaluate current_colspec_prefix
            potential_colspec_str = current_colspec_prefix.strip()
            if potential_colspec_str and \
               re.fullmatch(r'[lcrpmb\s\|\@\{\}\>\<\d\.\*\(\)\-\+\\\!\$\^\_\~\%\'\[\]\#\~]*', potential_colspec_str, re.IGNORECASE) and \
               any(c in potential_colspec_str.lower() for c in 'lcr|'):
                
                colspec_candidate = potential_colspec_str
                # The header starts from where the colspec ended.
                # Reconstruct the part of the line that is header.
                if colspec_ended_at_token_index != -1:
                    # Join tokens from colspec_ended_at_token_index onwards
                    header_part_on_first_line = "".join(tokens[colspec_ended_at_token_index:]).strip()
                else: # All tokens looked like colspec, but line had '&'. This is ambiguous.
                      # Or, colspec_ended_at_token_index was not set because loop finished.
                      # If loop finished, means all tokens were colspec_like.
                      # If current_colspec_prefix is the whole line, but it has '&', then it's not a pure colspec.
                    if current_colspec_prefix == first_line_stripped: # Should not happen if '&' is in first_line_stripped and correctly breaks loop
                         self._log(f"Table post-processing: Ambiguous first line with '&': '{first_line_stripped}'. Treating as full header.", "debug")
                         header_part_on_first_line = first_line_stripped
                    else: # Fallback, something is off, treat as full header.
                         header_part_on_first_line = first_line_stripped


                self._log(f"Table post-processing: Split first line. Colspec: '{colspec_candidate}', Header part: '{header_part_on_first_line}'", "debug")
            else:
                # No valid leading colspec found, or prefix was not valid.
                self._log(f"Table post-processing: No valid leading colspec found on first line '{first_line_stripped}' (contains '&'). Treated as full header.", "debug")
                header_part_on_first_line = first_line_stripped


        actual_data_row_strings_list = []
        if header_part_on_first_line.strip(): 
            actual_data_row_strings_list.append(header_part_on_first_line)
        
        actual_data_row_strings_list.extend(remaining_content_lines)

        if not actual_data_row_strings_list or not actual_data_row_strings_list[0].strip(): # Check if first line is now empty
            self._log("Table post-processing: No data rows left or first data row is empty after parsing first line.", "debug")
            return block_text # or handle as empty table if appropriate

        rejoined_data_rows_str = "\n".join(s for s in actual_data_row_strings_list if s.strip()) # Ensure no empty lines are processed
        if not rejoined_data_rows_str:
             self._log("Table post-processing: All data rows are empty after processing.", "debug")
             return block_text

        markdown_table = self._process_data_rows_to_markdown(rejoined_data_rows_str)
        
        if markdown_table:
            self._log("Table post-processing: Successfully converted a tabular block to Markdown table.", "debug")
            return markdown_table
        else:
            self._log("Table post-processing: Failed to convert tabular data to Markdown table, returning original block.", "debug")
            return block_text

    def _convert_specific_tables_to_markdown_format(self, text: str) -> str:
        self._log("Table post-processing: Starting conversion of '::: {.tabular}' formats.", "debug")
        processed_text = text
        current_pos, table_blocks_found, table_blocks_converted = 0, 0, 0
        result_parts = [] 
        tabular_start_regex = re.compile(r"^::: \{\.tabular\}[^\n]*$", re.MULTILINE)
        block_end_regex = re.compile(r"^:::$", re.MULTILINE)
        while current_pos < len(processed_text):
            match_start = tabular_start_regex.search(processed_text, current_pos)
            if not match_start: 
                result_parts.append(processed_text[current_pos:])
                break
            table_blocks_found +=1
            result_parts.append(processed_text[current_pos:match_start.start()])
            
            # Determine search start for ":::"
            search_for_end_from = match_start.end()
            # Correctly advance past the newline after the ::: {.tabular...} line
            if search_for_end_from < len(processed_text) and processed_text[search_for_end_from] == '\n':
                search_for_end_from += 1
            
            match_end = block_end_regex.search(processed_text, search_for_end_from)

            if not match_end: 
                self._log(f"Table post-processing: Unmatched '::: {{.tabular}}' at pos {match_start.start()}. Appending rest of text.", "warn")
                result_parts.append(processed_text[match_start.start():])
                break 
            
            block_to_process = processed_text[match_start.start():match_end.end()]
            converted_content = self._convert_tabular_block_to_markdown(block_to_process)
            result_parts.append(converted_content)
            
            if converted_content != block_to_process:
                table_blocks_converted +=1
                if not converted_content.endswith('\n'): 
                    result_parts.append('\n') # Ensure table is followed by newline if conversion happened
            
            current_pos = match_end.end()
            # Consume the newline that might follow the original ':::'
            if current_pos < len(processed_text) and processed_text[current_pos] == '\n':
                # If we converted, the added newline (if any) handles spacing.
                # If we didn't convert, this newline was part of original structure.
                current_pos += 1 
        
        final_text = "".join(result_parts)
        self._log(f"Table post-processing: Found {table_blocks_found} '::: {{.tabular}}' blocks. Converted {table_blocks_converted}.", "info" if table_blocks_found > 0 else "debug")
        return final_text

    def post_process_markdown(self, markdown_file_path: Path) -> bool:
        self._log(f"Starting post-processing for Markdown file: {markdown_file_path}")
        if not markdown_file_path.exists():
            self._log(f"Markdown file not found for post-processing: {markdown_file_path}", "error")
            return False
        try:
            with open(markdown_file_path, "r", encoding="utf-8", errors="ignore") as f:
                original_md_content = f.read()
            processed_md_content = self._convert_specific_tables_to_markdown_format(original_md_content)
            if processed_md_content != original_md_content:
                with open(markdown_file_path, "w", encoding="utf-8") as f:
                    f.write(processed_md_content)
                self._log(f"Markdown file post-processed and updated: {markdown_file_path}", "success")
            else:
                self._log("No changes made during Markdown post-processing (table conversion).", "info")
            return True
        except Exception as e:
            self._log(f"Error during Markdown post-processing for {markdown_file_path}: {e}", "error")
            return False
    # --- End: Methods for Post-Processing Markdown ---

    def convert_to_markdown(self, output_folder_path_str):
        if not self.find_main_tex_file(): return False
        self._log("Starting LaTeX to Markdown conversion process...")
        self.final_output_folder_path = Path(output_folder_path_str).resolve()
        try:
            self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
            self._log(f"Final output directory confirmed: {self.final_output_folder_path}", "success")
        except Exception as e_mkdir:
            self._log(f"Error creating final output directory {self.final_output_folder_path}: {e_mkdir}", "error")
            return False

        bbl_file_content = self._generate_bbl_content() 
        if bbl_file_content is None:
            self._log("Failed to generate or find .bbl content. Stopping conversion process.", "error") 
            return False 
        
        problematic_macros_neutralization = r"""
\providecommand{\linebreakand}{\par\noindent\ignorespaces}
\providecommand{\email}[1]{\texttt{#1}}
\providecommand{\IEEEauthorblockN}[1]{#1\par}
\providecommand{\IEEEauthorblockA}[1]{#1\par}
\providecommand{\and}{\par\noindent\ignorespaces}
\providecommand{\And}{\par\noindent\ignorespaces} 
\providecommand{\AND}{\par\noindent\ignorespaces} 
\providecommand{\IEEEoverridecommandlockouts}{} 
\providecommand{\CLASSINPUTinnersidemargin}{}
\providecommand{\CLASSINPUToutersidemargin}{}
\providecommand{\CLASSINPUTtoptextmargin}{}
\providecommand{\CLASSINPUTbottomtextmargin}{}
\providecommand{\CLASSOPTIONcompsoc}{}
\providecommand{\CLASSOPTIONconference}{}
\providecommand{\@toptitlebar}{}
\providecommand{\@bottomtitlebar}{}
\providecommand{\@thanks}{}
\providecommand{\@notice}{}
\providecommand{\@noticestring}{}
\providecommand{\acksection}{}
\newenvironment{ack}{\par\textbf{Acknowledgments}\par}{\par}
\providecommand{\answerYes}[1]{[Yes] ##1}
\providecommand{\answerNo}[1]{[No] ##1}
\providecommand{\answerNA}[1]{[NA] ##1}
\providecommand{\answerTODO}[1]{[TODO] ##1}
\providecommand{\justificationTODO}[1]{[TODO] ##1}
\providecommand{\textasciitilde}{~}
\providecommand{\textasciicircum}{^}
\providecommand{\textbackslash}{\symbol{92}}
"""
        # Attempt 1
        self._log("Attempt 1: Processing with only venue-specific styles commented out.", "info")
        tex_content_attempt1 = self._comment_out_style_packages(self.original_main_tex_content, mode="venue_only")
        final_tex_for_pandoc_attempt1 = self._inline_bibliography(tex_content_attempt1, bbl_file_content)
        final_tex_for_pandoc_attempt1 = problematic_macros_neutralization + final_tex_for_pandoc_attempt1
        if self._run_pandoc_conversion(final_tex_for_pandoc_attempt1, pandoc_timeout=20): return True

        # Attempt 2
        self._log("Attempt 1 failed. Proceeding to Attempt 2: Using original TeX content.", "warn")
        final_tex_for_pandoc_attempt2 = self._inline_bibliography(self.original_main_tex_content, bbl_file_content)
        final_tex_for_pandoc_attempt2 = problematic_macros_neutralization + final_tex_for_pandoc_attempt2
        if self._run_pandoc_conversion(final_tex_for_pandoc_attempt2, pandoc_timeout=20): return True

        # Attempt 3
        self._log("Attempt 2 failed. Proceeding to Attempt 3: Commenting out all project-specific styles.", "warn")
        tex_content_attempt3 = self._comment_out_style_packages(self.original_main_tex_content, mode="all_project")
        final_tex_for_pandoc_attempt3 = self._inline_bibliography(tex_content_attempt3, bbl_file_content)
        final_tex_for_pandoc_attempt3 = problematic_macros_neutralization + final_tex_for_pandoc_attempt3 
        return self._run_pandoc_conversion(final_tex_for_pandoc_attempt3, pandoc_timeout=30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Converts a LaTeX project to Markdown. "
                    "Comments out custom styles, inlines bibliography (fetching abstracts from Semantic Scholar), "
                    "copies figures, and uses Pandoc. Output is 'paper.md'. Includes post-processing for specific table formats."
    )
    parser.add_argument("project_folder", help="Path to the LaTeX project folder.")
    parser.add_argument("-o", "--output_folder", default=None, help="Path for the output folder. Default: '[project_name]_output'.")
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose", default=True, help="Suppress informational messages.")
    parser.add_argument("--template", default="template.md", help="Path to Pandoc Markdown template. Default: 'template.md'.")
    
    args = parser.parse_args()
    project_path = Path(args.project_folder).resolve()
    if not project_path.is_dir():
        print(f"Error: Project folder '{args.project_folder}' not found.", file=sys.stderr)
        sys.exit(1)

    output_folder_path = Path(args.output_folder).resolve() if args.output_folder else project_path.parent / f"{project_path.name}_output"
    
    template_file_path_str = args.template
    template_p = Path(args.template)
    if not template_p.is_absolute() and not template_p.exists():
        script_dir_template = Path(__file__).parent.resolve() / args.template
        if script_dir_template.exists(): template_file_path_str = str(script_dir_template)
    elif template_p.exists():
        template_file_path_str = str(template_p.resolve())

    converter = LatexToMarkdownConverter(str(project_path), verbose=args.verbose, template_path=template_file_path_str)
    converter._log(f"Processing LaTeX project in: '{project_path}'", "info")
    converter._log(f"Markdown output will be saved to: '{output_folder_path}'", "info")
    if Path(converter.template_path).exists():
        converter._log(f"Using Pandoc template: '{converter.template_path}'", "info")
    else:
        converter._log(f"Pandoc template specified: '{converter.template_path}' (Pandoc will try to locate or use default).", "warn")

    success = converter.convert_to_markdown(str(output_folder_path))
    if success:
        converter._log(f"Conversion process finished successfully.", "info") 
        print(f"    Markdown output: '{output_folder_path / 'paper.md'}'") 
        print(f"    Figures copied to: '{output_folder_path}'")
    else:
        converter._log(f"Conversion process failed. Please check logs.", "error") 
    
    converter._log(f"Ensure Pandoc, LaTeX (pdflatex, bibtex), and Python 'requests' library are installed.", "info")

