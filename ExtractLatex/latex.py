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
    def __init__(self, folder_path_str, verbose=True): 
        self.folder_path = Path(folder_path_str).resolve()
        self.main_tex_path = None
        self.original_main_tex_content = ""
        self.verbose = verbose
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
            # Exclude very common LaTeX packages that might be locally copied but are standard
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
        initial_content = tex_content # To check if any changes were made
        
        if mode == "venue_only":
            self._log(f"Targeting venue-specific styles based on patterns.", "debug")
            for style_pattern_re in VENUE_STYLE_PATTERNS:
                # Regex to find \usepackage[optional_args]{package_name_matching_pattern}
                # It ensures not to comment out already commented lines.
                # It matches the whole line containing the \usepackage command.
                pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{(" + style_pattern_re + r")\}[^\n]*)$"
                modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE | re.IGNORECASE)

        elif mode == "all_project":
            styles_to_comment_out = self._get_project_sty_basenames() # Gets project-specific (non-common) .sty files
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

    def _inline_bibliography(self, tex_content, bbl_content):
        if not bbl_content:
            self._log("BBL content is empty. Bibliography will not be inlined.", "warn")
            return tex_content
        self._log("Inlining .bbl content into .tex content...")
        cleaned_bbl_content = bbl_content
        self._log("Cleaning .bbl content before inlining...", "debug")
        original_bbl_len = len(cleaned_bbl_content)
        bbl_cleanup_patterns = [
            re.compile(r"\\providecommand\{\\natexlab\}\[1\]\{#1\}\s*", flags=re.DOTALL),
            re.compile(r"\\providecommand\{\\url\}\[1\]\{\\texttt\{#1\}\}\s*", flags=re.DOTALL),
            re.compile(r"\\providecommand\{\\doi\}\[1\]\{doi:\s*#1\}\s*", flags=re.DOTALL),
            re.compile(r"\\expandafter\\ifx\\csname\s*urlstyle\\endcsname\\relax[\s\S]*?\\fi\s*", flags=re.DOTALL),
        ]
        for pattern_re in bbl_cleanup_patterns:
            cleaned_bbl_content = pattern_re.sub("", cleaned_bbl_content)
        cleaned_bbl_content = re.sub(r"(\\begin\{thebibliography\})\{[^}]*\}", r"\1{}", cleaned_bbl_content)
        cleaned_bbl_content = cleaned_bbl_content.strip()
        if len(cleaned_bbl_content) < original_bbl_len:
            self._log(".bbl content cleaned.", "success")
        else:
            self._log("No specific .bbl preamble patterns matched for cleaning (or argument to thebibliography already empty).", "debug")
        
        references_heading_tex = "\n\n\\section*{References}\n\n" 
        content_to_inline = references_heading_tex + cleaned_bbl_content

        modified_content = re.sub(r"^\s*\\bibliographystyle\{[^{}]*\}\s*$", "", tex_content, flags=re.MULTILINE)
        bibliography_command_pattern = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"
        if re.search(bibliography_command_pattern, modified_content, flags=re.MULTILINE):
            modified_content = re.sub(bibliography_command_pattern, lambda m: content_to_inline, modified_content, count=1, flags=re.MULTILINE)
            self._log("Replaced \\bibliography command with '\\section*{References}' and cleaned .bbl content.", "success")
        else:
            self._log("\\bibliography{...} command not found. Appending '\\section*{References}' and BBL before \\end{document}.", "warn")
            end_document_match = re.search(r"\\end\{document\}", modified_content)
            if end_document_match:
                insertion_point = end_document_match.start()
                modified_content = modified_content[:insertion_point] + content_to_inline + "\n" + modified_content[insertion_point:]
                self._log("Appended '\\section*{References}' and .bbl content before \\end{document}.", "success")
            else:
                self._log("\\end{document} not found. Appending '\\section*{References}' and .bbl content to the end.", "warn")
                modified_content += "\n" + content_to_inline
        modified_content = re.sub(bibliography_command_pattern, "", modified_content, flags=re.MULTILINE)
        modified_content = re.sub(r"^\s*\\nobibliography\{[^{}]*(?:,[^{}]*)*\}\s*$", "", modified_content, flags=re.MULTILINE)
        return modified_content

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
                    relative_dest_path = Path(decoded_image_path_str)
                    destination_path = (output_figure_folder_path / relative_dest_path).resolve()
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
        """
        Helper function to run the Pandoc conversion step.
        Returns True on success, False on failure or timeout.
        """
        self._log(f"Preparing to run Pandoc (timeout: {pandoc_timeout}s)...", "debug")
        
        # This is the final output path for the markdown file
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
                "-t", "markdown+raw_tex", # Use Pandoc's default markdown + allow raw TeX
                "--strip-comments", 
                "--wrap=none", 
                "-o", pandoc_local_output_name 
            ]
            
            self._log(f"Pandoc will output locally to: {pandoc_local_output_name} (relative to project folder)", "debug")
            self._log(f"Running Pandoc command: {' '.join(cmd_pandoc)}", "debug")
            
            self._log(f"Changing CWD for Pandoc to: {self.folder_path}", "debug")
            os.chdir(self.folder_path)
            
            process_pandoc = subprocess.run(cmd_pandoc, capture_output=True, text=True, 
                                            check=False, encoding='utf-8', errors='ignore',
                                            timeout=pandoc_timeout) # Apply timeout
            
            pandoc_created_md_path = self.folder_path / pandoc_local_output_name

            if process_pandoc.returncode == 0:
                self._log(f"Pandoc successfully created local Markdown: {pandoc_created_md_path}", "success")
                self._log(f"Moving '{pandoc_created_md_path}' to '{final_markdown_file_path}'", "debug")
                try:
                    shutil.move(str(pandoc_created_md_path), str(final_markdown_file_path))
                    self._log(f"Successfully moved Markdown to final destination: {final_markdown_file_path}", "success")
                    if final_markdown_file_path.exists():
                        with open(final_markdown_file_path, "r", encoding="utf-8", errors="ignore") as md_file:
                            markdown_content_for_figures = md_file.read()
                        self._find_and_copy_figures_from_markdown(markdown_content_for_figures, self.final_output_folder_path)
                    else:
                        self._log(f"Final markdown file '{final_markdown_file_path}' not found after supposed move. Cannot copy figures.", "error")
                    return True # Pandoc success
                except Exception as e_move_or_copy:
                    self._log(f"Error moving Pandoc output or copying figures: {e_move_or_copy}", "error")
                    if pandoc_created_md_path.exists():
                        try: pandoc_created_md_path.unlink()
                        except: pass
                    return False 
            else:
                self._log(f"Pandoc failed with error code {process_pandoc.returncode}:", "error")
                self._log(f"--- PANDOC STDOUT ---\n{process_pandoc.stdout}", "error")
                self._log(f"--- PANDOC STDERR ---\n{process_pandoc.stderr}", "error")
                try:
                    with open(tmp_tex_file_path_str, "r", encoding="utf-8") as f_err:
                        lines = f_err.readlines()
                    # Try to find line number from Pandoc error, e.g., "Error at "source" (line XXX,"
                    err_line_match = re.search(r"Error at \"source\" \(line (\d+),", process_pandoc.stderr)
                    if not err_line_match: # Fallback for other error formats
                        err_line_match = re.search(r"line (\d+), column \d+", process_pandoc.stderr)

                    if err_line_match:
                        err_line_no = int(err_line_match.group(1))
                        self._log(f"--- Snippet from temporary TeX file around line {err_line_no} ---", "error")
                        start = max(0, err_line_no - 3)
                        end = min(len(lines), err_line_no + 2)
                        for i in range(start, end):
                            self._log(f"{i+1:4d}: {lines[i].rstrip()}", "error")
                except Exception as e_snippet:
                    self._log(f"Could not print snippet of temp TeX file: {e_snippet}", "warn")
                if pandoc_created_md_path.exists():
                    try: pandoc_created_md_path.unlink()
                    except: pass
                return False
        except subprocess.TimeoutExpired:
            self._log(f"Pandoc command timed out after {pandoc_timeout} seconds.", "error")
            # Clean up local pandoc output if it exists due to timeout
            pandoc_created_md_path_on_timeout = self.folder_path / pandoc_local_output_name
            if pandoc_created_md_path_on_timeout.exists():
                try: pandoc_created_md_path_on_timeout.unlink()
                except: pass
            return False # Indicate timeout
        except FileNotFoundError:
            self._log("Pandoc command not found. Ensure pandoc is installed and in your system PATH.", "error")
            return False
        except Exception as e:
            self._log(f"An unexpected error occurred during Pandoc conversion: {e}", "error")
            return False
        finally:
            if Path.cwd() != original_cwd:
                self._log(f"Changing CWD back to: {original_cwd}", "debug")
                os.chdir(original_cwd)
            if tmp_tex_file_path_str and Path(tmp_tex_file_path_str).exists():
                try:
                    Path(tmp_tex_file_path_str).unlink()
                    self._log(f"Temporary .tex file {tmp_tex_file_path_str} deleted.", "debug")
                except Exception as e_del:
                    self._log(f"Could not delete temporary .tex file {tmp_tex_file_path_str}: {e_del}", "warn")
        # Should not reach here if logic is correct, returns happen within try block
        return False


    def convert_to_markdown(self, output_folder_path_str):
        if not self.find_main_tex_file():
            return False
        
        self._log("Starting LaTeX to Markdown conversion process...")
        
        # Store final output folder path at class level for _run_pandoc_conversion
        self.final_output_folder_path = Path(output_folder_path_str).resolve()
        self._log(f"Ensuring final output directory exists: {self.final_output_folder_path}", "debug")
        try:
            self.final_output_folder_path.mkdir(parents=True, exist_ok=True)
            if not self.final_output_folder_path.is_dir():
                self._log(f"Failed to create or access final output directory {self.final_output_folder_path}", "error")
                return False
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
% Add other \providecommand lines here if more issues arise
% e.g., \providecommand{\thanks}[1]{} % To ignore \thanks command
% e.g., \providecommand{\IEEEoverridecommandlockouts}{}
"""
        # --- Attempt 1: Comment out only venue-specific styles ---
        self._log("Attempt 1: Processing with only venue-specific styles commented out.", "info")
        tex_content_attempt1 = self._comment_out_style_packages(self.original_main_tex_content, mode="venue_only")
        final_tex_for_pandoc_attempt1 = self._inline_bibliography(tex_content_attempt1, bbl_file_content)
        final_tex_for_pandoc_attempt1 = problematic_macros_neutralization + final_tex_for_pandoc_attempt1
        self._log("Applied neutralizations for potentially problematic macros (Attempt 1).", "debug")
        pandoc_timeout_attempt1 = 20 
        success_attempt1 = self._run_pandoc_conversion(final_tex_for_pandoc_attempt1, pandoc_timeout=pandoc_timeout_attempt1)
        if success_attempt1:
            return True

        # --- Attempt 2: No style commenting (original TeX content) ---
        self._log("Attempt 1 failed or timed out. Proceeding to Attempt 2: Using original TeX content (no styles commented by script).", "warn")
        final_tex_for_pandoc_attempt2 = self._inline_bibliography(self.original_main_tex_content, bbl_file_content)
        final_tex_for_pandoc_attempt2 = problematic_macros_neutralization + final_tex_for_pandoc_attempt2
        self._log("Applied neutralizations for potentially problematic macros (Attempt 2).", "debug")
        pandoc_timeout_attempt2 = 20 
        success_attempt2 = self._run_pandoc_conversion(final_tex_for_pandoc_attempt2, pandoc_timeout=pandoc_timeout_attempt2)
        if success_attempt2:
            return True

        # --- Attempt 3: Comment out all project-specific styles (fallback) ---
        self._log("Attempt 2 failed or timed out. Proceeding to Attempt 3: Commenting out all project-specific styles.", "warn")
        tex_content_attempt3 = self._comment_out_style_packages(self.original_main_tex_content, mode="all_project")
        final_tex_for_pandoc_attempt3 = self._inline_bibliography(tex_content_attempt3, bbl_file_content)
        final_tex_for_pandoc_attempt3 = problematic_macros_neutralization + final_tex_for_pandoc_attempt3 
        self._log("Applied neutralizations for potentially problematic macros (Attempt 3).", "debug")
        pandoc_timeout_attempt3 = 30 
        success_attempt3 = self._run_pandoc_conversion(final_tex_for_pandoc_attempt3, pandoc_timeout=pandoc_timeout_attempt3)
        
        return success_attempt3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Converts a LaTeX project to Markdown. "
                    "Comments out custom styles, inlines bibliography, copies figures based on Markdown links, "
                    "and uses Pandoc for conversion. Output is 'paper.md'."
    )
    parser.add_argument(
        "project_folder", 
        help="Path to the LaTeX project folder containing the main .tex file."
    )
    parser.add_argument(
        "-o", "--output_folder", 
        help="Path for the output folder where 'paper.md' and figures will be saved. "
             "Default: A folder named '[project_folder_name]_output' in the project folder's parent directory.",
        default=None
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_false", 
        dest="verbose",
        default=True, 
        help="Suppress informational and success messages, only show warnings and errors."
    )
    
    args = parser.parse_args()

    project_path = Path(args.project_folder).resolve()
    if not project_path.is_dir():
        print(f"Error: Project folder '{args.project_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.output_folder is None:
        output_folder_path = project_path.parent / f"{project_path.name}_output"
    else:
        output_folder_path = Path(args.output_folder).resolve()

    converter = LatexToMarkdownConverter(str(project_path), verbose=args.verbose) 

    converter._log(f"Processing LaTeX project in: '{project_path}'", "info")
    converter._log(f"Markdown output ('paper.md') and figures will be saved to: '{output_folder_path}'", "info")
    
    success = converter.convert_to_markdown(str(output_folder_path))

    if success:
        converter._log(f"Conversion process finished successfully.", "info") 
        print(f"    Markdown output: '{output_folder_path / 'paper.md'}'") 
        print(f"    Figures copied to: '{output_folder_path}'")
    else:
        converter._log(f"Conversion process failed. Please check the logs above.", "error") 
    
    converter._log(f"Ensure Pandoc, LaTeX (pdflatex, bibtex) are installed and in your system PATH.", "info")
    converter._log(f"If pdflatex fails, you might need to install additional TeX Live packages in your environment (e.g., texlive-latex-extra, texlive-publishers, texlive-science).", "info")
    converter._log(f"If Pandoc fails with 'unexpected X' errors, try adding more '\\providecommand{{X}}{{...}}' lines to the 'problematic_macros_neutralization' string in the script.", "info")

