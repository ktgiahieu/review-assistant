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

class LatexToMarkdownConverter:
    def __init__(self, folder_path_str):
        self.folder_path = Path(folder_path_str).resolve()
        self.main_tex_path = None
        self.original_main_tex_content = ""
        # self.found_image_paths = set() # No longer needed at class level for this approach

    def find_main_tex_file(self):
        """Finds and stores the main .tex file containing \\begin{document}."""
        print("[*] Finding main .tex file...")
        for path in self.folder_path.rglob("*.tex"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    if r'\begin{document}' in content:
                        self.main_tex_path = path
                        self.original_main_tex_content = content
                        print(f"[+] Main .tex file found: {self.main_tex_path}")
                        return True
            except Exception as e:
                print(f"[!] Warning: Could not read file {path} due to {e}", file=sys.stderr)
                continue
        print(f"[-] Error: No main .tex file with \\begin{document} found in '{self.folder_path}'.", file=sys.stderr)
        return False

    def _get_project_sty_basenames(self):
        """Finds all .sty files in the project folder and returns their basenames."""
        print("[*] Searching for custom .sty files in project folder...")
        sty_basenames = []
        for sty_path in self.folder_path.rglob("*.sty"):
            sty_basenames.append(sty_path.stem)
        if sty_basenames:
            print(f"[+] Found custom .sty files (basenames): {', '.join(sty_basenames)}")
        else:
            print("[-] No custom .sty files found in the project folder.")
        return sty_basenames

    def _comment_out_custom_style_packages(self, tex_content):
        """Comments out \\usepackage commands for custom .sty files found in the project."""
        custom_sty_basenames = self._get_project_sty_basenames()
        if not custom_sty_basenames:
            return tex_content

        print("[*] Commenting out \\usepackage commands for custom styles...")
        modified_content = tex_content
        for sty_basename in custom_sty_basenames:
            pattern = r"^([^\%]*?)(\\usepackage(?:\[[^\]]*\])?\{" + re.escape(sty_basename) + r"\}[^\n]*)$"
            modified_content = re.sub(pattern, r"\1% \2", modified_content, flags=re.MULTILINE)
        
        if modified_content != tex_content:
            print("[+] Successfully commented out relevant \\usepackage commands.")
        else:
            print("[-] No \\usepackage commands for found custom styles were modified.")
        return modified_content

    def _generate_bbl_content(self):
        if not self.main_tex_path: # original_main_tex_content check removed as it's set with main_tex_path
            print("[-] Cannot generate .bbl file: Main .tex file not identified.", file=sys.stderr)
            return None

        main_file_stem = self.main_tex_path.stem
        # Check for existing .bbl file first (relative to project folder)
        existing_bbl_path = self.folder_path / f"{main_file_stem}.bbl"

        if existing_bbl_path.exists() and existing_bbl_path.is_file():
            print(f"[*] Found existing .bbl file: {existing_bbl_path}. Using its content.")
            try:
                with open(existing_bbl_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception as e:
                print(f"[!] Error reading existing .bbl file '{existing_bbl_path}': {e}", file=sys.stderr)
                print(f"    [*] Attempting to regenerate .bbl file...")
        else:
            print(f"[*] No existing .bbl file found at '{existing_bbl_path}'. Attempting to generate.")


        print("[*] Generating .bbl file content...")
        
        commands = [
            ["pdflatex", "-interaction=nonstopmode", "-draftmode", self.main_tex_path.name],
            ["bibtex", main_file_stem],
        ]

        original_cwd = Path.cwd()
        print(f"    [*] Changing CWD to: {self.folder_path}")
        os.chdir(self.folder_path) 

        try:
            for i, cmd_args in enumerate(commands):
                command_str = " ".join(str(arg) for arg in cmd_args)
                print(f"    [*] Running command: {command_str} (in CWD: {Path.cwd()})")
                process = subprocess.run(cmd_args, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
                
                if process.returncode != 0:
                    print(f"    [-] Error running command: {command_str}", file=sys.stderr)
                    print(f"        Stdout: {process.stdout[:500]}...", file=sys.stderr)
                    print(f"        Stderr: {process.stderr[:500]}...", file=sys.stderr)
                    if cmd_args[0] == "bibtex" and ("I found no \\bibdata command" in process.stdout or "I found no \\bibstyle command" in process.stdout):
                         print("    [-] BibTeX specific error: No \\bibdata or \\bibstyle command. Ensure original .tex calls \\bibliography and \\bibliographystyle.", file=sys.stderr)
                    elif cmd_args[0] == "pdflatex" and i == 0:
                        print("    [-] Critical: Initial pdflatex run failed. Cannot generate .aux for BibTeX.", file=sys.stderr)
                        return None
                else:
                    print(f"    [+] Command '{command_str}' executed.")
            
            bbl_file_path = Path(f"{main_file_stem}.bbl") 
            if bbl_file_path.exists():
                print(f"[+] .bbl file generated: {bbl_file_path.resolve()}")
                with open(bbl_file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            else:
                print(f"[-] .bbl file was not generated at '{bbl_file_path.resolve()}'. Check LaTeX/BibTeX logs.", file=sys.stderr)
                aux_file = Path(f"{main_file_stem}.aux")
                if not aux_file.exists():
                    print(f"    [-] Auxiliary file '{aux_file.name}' not found. Initial pdflatex run likely failed critically.", file=sys.stderr)
                else:
                    with open(aux_file, 'r', encoding='utf-8', errors='ignore') as af:
                        aux_content = af.read()
                        if r'\bibdata' not in aux_content:
                             print(r"    [-] \bibdata command missing in .aux file. Check \bibliography{...} in original .tex.", file=sys.stderr)
                        if not (re.search(r'\\citation\{', aux_content) or r'\nocite{*}' in aux_content):
                             print(r"    [-] No \citation or \nocite{*} found in .aux file. Bibliography might be empty.", file=sys.stderr)
                return None
        except FileNotFoundError as e:
            print(f"[-] FileNotFoundError: {e}. Ensure LaTeX (pdflatex, bibtex) is installed and in your system PATH.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"[-] An unexpected error occurred during .bbl generation: {e}", file=sys.stderr)
            return None
        finally:
            print(f"    [*] Changing CWD back to: {original_cwd}")
            os.chdir(original_cwd)

    def _inline_bibliography(self, tex_content, bbl_content):
        if not bbl_content:
            print("[!] Warning: BBL content is empty. Bibliography will not be inlined.")
            return tex_content

        print("[*] Inlining .bbl content into .tex content...")
        
        cleaned_bbl_content = bbl_content
        print("    [*] Cleaning .bbl content before inlining...")
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
            print("    [+] .bbl content cleaned.")
        else:
            print("    [*] No specific .bbl preamble patterns matched for cleaning (or argument to thebibliography already empty).")
        
        # Prepend a LaTeX section command for "References". Pandoc will convert this to a Markdown H1.
        references_heading_tex = "\n\n\\section*{References}\n\n"
        content_to_inline = references_heading_tex + cleaned_bbl_content

        modified_content = re.sub(r"^\s*\\bibliographystyle\{[^{}]*\}\s*$", "", tex_content, flags=re.MULTILINE)
        bibliography_command_pattern = r"^\s*\\bibliography\{[^{}]*(?:,[^{}]*)*\}\s*$"

        if re.search(bibliography_command_pattern, modified_content, flags=re.MULTILINE):
            modified_content = re.sub(bibliography_command_pattern, lambda m: content_to_inline, modified_content, count=1, flags=re.MULTILINE)
            print("[+] Replaced \\bibliography command with '\\section*{References}' and cleaned .bbl content.")
        else:
            print("[!] Warning: \\bibliography{...} command not found. Appending '\\section*{References}' and BBL before \\end{document}.")
            end_document_match = re.search(r"\\end\{document\}", modified_content)
            if end_document_match:
                insertion_point = end_document_match.start()
                modified_content = modified_content[:insertion_point] + content_to_inline + "\n" + modified_content[insertion_point:]
                print("[+] Appended '\\section*{References}' and .bbl content before \\end{document}.")
            else:
                print("[!] Warning: \\end{document} not found. Appending '\\section*{References}' and .bbl content to the end.", file=sys.stderr)
                modified_content += "\n" + content_to_inline
        
        modified_content = re.sub(bibliography_command_pattern, "", modified_content, flags=re.MULTILINE)
        modified_content = re.sub(r"^\s*\\nobibliography\{[^{}]*(?:,[^{}]*)*\}\s*$", "", modified_content, flags=re.MULTILINE)
        return modified_content

    def _find_and_copy_figures_from_markdown(self, markdown_content, output_figure_folder_path):
        """
        Finds image references in the Markdown content (standard Markdown and common HTML tags)
        and copies them from the original project folder to the specified output folder.
        """
        if not markdown_content:
            print("[!] Cannot find figures: Markdown content is empty.", file=sys.stderr)
            return

        print(f"[*] Searching for and copying figures (from Markdown) to: {output_figure_folder_path}")
        
        markdown_image_pattern = r"!\[[^\]]*\]\(([^)\s]+?)(?:\s+[\"'][^\"']*[\"'])?\)"
        html_img_pattern = r"<img\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"
        html_embed_pattern = r"<embed\s+[^>]*?src\s*=\s*[\"']([^\"']+)[\"'][^>]*?>"
        html_source_pattern = r"<source\s+[^>]*?srcset\s*=\s*[\"']([^\"'\s]+)(?:\s+\S+)?[\"'][^>]*?>"

        image_paths_in_md = []
        for pattern in [markdown_image_pattern, html_img_pattern, html_embed_pattern, html_source_pattern]:
            for match in re.finditer(pattern, markdown_content, flags=re.IGNORECASE):
                image_paths_in_md.append(match.group(1))
        
        if not image_paths_in_md:
            print("[-] No image references found in the Markdown content.")
            return

        print(f"    [*] Found {len(image_paths_in_md)} potential image references in Markdown.")
        
        common_image_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.eps', '.tikz', '.svg'] 
        figures_copied_count = 0
        copied_files_set = set() 

        for image_path_in_md_raw in image_paths_in_md:
            try:
                decoded_image_path_str = urllib.parse.unquote(image_path_in_md_raw)
            except Exception as e_decode:
                print(f"    [!] Warning: Could not URL-decode image path '{image_path_in_md_raw}': {e_decode}", file=sys.stderr)
                decoded_image_path_str = image_path_in_md_raw 
            
            if urllib.parse.urlparse(decoded_image_path_str).scheme in ['http', 'https']:
                print(f"    [*] Skipping web URL: {decoded_image_path_str}")
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
                    print(f"    [*] Skipping already copied file: {found_image_file_abs.name}")
                    continue

                try:
                    output_figure_folder_path.mkdir(parents=True, exist_ok=True)
                    relative_dest_path = Path(decoded_image_path_str)
                    destination_path = (output_figure_folder_path / relative_dest_path).resolve()
                    destination_path.parent.mkdir(parents=True, exist_ok=True) 
                    
                    shutil.copy2(str(found_image_file_abs), str(destination_path))
                    print(f"    [+] Copied figure: '{found_image_file_abs.name}' to '{destination_path}'")
                    copied_files_set.add(found_image_file_abs)
                    figures_copied_count += 1
                except Exception as e_copy:
                    print(f"    [!] Warning: Could not copy figure '{found_image_file_abs}' to '{destination_path}': {e_copy}", file=sys.stderr)
            else:
                print(f"    [!] Warning: Referenced image '{decoded_image_path_str}' (from Markdown) not found in project folder '{self.folder_path}'. Raw path from MD: '{image_path_in_md_raw}'", file=sys.stderr)

        if figures_copied_count > 0:
            print(f"[+] Copied {figures_copied_count} unique figure(s) based on Markdown content.")
        else:
            print("[-] No new figures found in Markdown to copy.")


    def convert_to_markdown(self, output_folder_path_str):
        """
        Orchestrates the full conversion process.
        """
        if not self.find_main_tex_file():
            return False
        
        print("[*] Starting LaTeX to Markdown conversion process...")
        tex_content_no_custom_styles = self._comment_out_custom_style_packages(self.original_main_tex_content)
        
        bbl_file_content = self._generate_bbl_content() # This now handles fallback to existing .bbl
        if bbl_file_content is None:
            print("[!] Warning: Failed to generate or find .bbl content. Proceeding without inlined bibliography.")
            final_tex_for_pandoc = tex_content_no_custom_styles
        else:
            final_tex_for_pandoc = self._inline_bibliography(tex_content_no_custom_styles, bbl_file_content)

        print("[*] Converting processed LaTeX to Markdown using Pandoc...")
        
        final_output_folder_path = Path(output_folder_path_str).resolve()
        final_markdown_file_path = final_output_folder_path / "paper.md" 
        
        print(f"    [*] Ensuring final output directory exists: {final_output_folder_path}")
        try:
            final_output_folder_path.mkdir(parents=True, exist_ok=True)
            if not final_output_folder_path.is_dir():
                print(f"[-] Error: Failed to create or access final output directory {final_output_folder_path}", file=sys.stderr)
                return False
            print(f"    [+] Final output directory confirmed: {final_output_folder_path}")
        except Exception as e_mkdir:
            print(f"[-] Error creating final output directory {final_output_folder_path}: {e_mkdir}", file=sys.stderr)
            return False
            
        tmp_tex_file_path_str = "" 
        pandoc_local_output_name = "_pandoc_temp_paper.md" 

        original_cwd = Path.cwd()

        try:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".tex", encoding="utf-8") as tmp_tex_file:
                tmp_tex_file.write(final_tex_for_pandoc)
                tmp_tex_file_path_str = tmp_tex_file.name 
            
            print(f"    [*] Temporary processed .tex file created at: {Path(tmp_tex_file_path_str).resolve()}")
            
            cmd_pandoc = [
                "pandoc", str(Path(tmp_tex_file_path_str).resolve()),
                "-f", "latex",
                "-t", "markdown_strict", 
                "--wrap=none", 
                "-o", pandoc_local_output_name 
            ]
            
            print(f"    [*] Pandoc will output locally to: {pandoc_local_output_name} (relative to project folder)")
            print(f"    [*] Running Pandoc command: {' '.join(cmd_pandoc)}")
            
            print(f"    [*] Changing CWD for Pandoc to: {self.folder_path}")
            os.chdir(self.folder_path)
            
            process_pandoc = subprocess.run(cmd_pandoc, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore')
            
            pandoc_created_md_path = self.folder_path / pandoc_local_output_name

            if process_pandoc.returncode == 0:
                print(f"[+] Pandoc successfully created local Markdown: {pandoc_created_md_path}")
                
                print(f"    [*] Moving '{pandoc_created_md_path}' to '{final_markdown_file_path}'")
                try:
                    shutil.move(str(pandoc_created_md_path), str(final_markdown_file_path))
                    print(f"[+] Successfully moved Markdown to final destination: {final_markdown_file_path}")
                    
                    if final_markdown_file_path.exists():
                        with open(final_markdown_file_path, "r", encoding="utf-8", errors="ignore") as md_file:
                            markdown_content_for_figures = md_file.read()
                        self._find_and_copy_figures_from_markdown(markdown_content_for_figures, final_output_folder_path)
                    else:
                        print(f"[!] Error: Final markdown file '{final_markdown_file_path}' not found after supposed move. Cannot copy figures.", file=sys.stderr)

                except Exception as e_move_or_copy:
                    print(f"[-] Error moving Pandoc output or copying figures: {e_move_or_copy}", file=sys.stderr)
                    if pandoc_created_md_path.exists():
                        try: pandoc_created_md_path.unlink()
                        except: pass
                    return False 
            else:
                print(f"[-] Pandoc failed with error code {process_pandoc.returncode}:", file=sys.stderr)
                print(f"    Pandoc Stderr: {process_pandoc.stderr}", file=sys.stderr)
                if pandoc_created_md_path.exists():
                    try: pandoc_created_md_path.unlink()
                    except: pass
                return False

        except FileNotFoundError:
            print("[-] Pandoc command not found. Ensure pandoc is installed and in your system PATH.", file=sys.stderr)
            return False
        except Exception as e:
            print(f"[-] An unexpected error occurred during Pandoc conversion: {e}", file=sys.stderr)
            return False
        finally:
            if Path.cwd() != original_cwd:
                print(f"    [*] Changing CWD back to: {original_cwd}")
                os.chdir(original_cwd)
            
            if tmp_tex_file_path_str and Path(tmp_tex_file_path_str).exists():
                try:
                    Path(tmp_tex_file_path_str).unlink()
                    print(f"    [*] Temporary .tex file {tmp_tex_file_path_str} deleted.")
                except Exception as e_del:
                    print(f"    [!] Warning: Could not delete temporary .tex file {tmp_tex_file_path_str}: {e_del}", file=sys.stderr)
        return True


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
    
    args = parser.parse_args()

    project_path = Path(args.project_folder).resolve()
    if not project_path.is_dir():
        print(f"Error: Project folder '{args.project_folder}' not found or is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.output_folder is None:
        output_folder_path = project_path.parent / f"{project_path.name}_output"
    else:
        output_folder_path = Path(args.output_folder).resolve()

    if project_path.name == "test_latex_project_pandoc": 
        if not project_path.exists():
            print(f"[*] Test project folder '{project_path.name}' not found. Creating dummy project at '{project_path}'...")
            project_path.mkdir(parents=True, exist_ok=True)
            
            (project_path / "images").mkdir(exist_ok=True)
            try:
                from PIL import Image as PILImage
                dummy_img = PILImage.new('RGB', (60, 30), color = 'red')
                dummy_img.save(project_path / "images" / "example_fig.png")
                print(f"    [+] Created dummy 'example_fig.png'")
            except ImportError:
                print("    [!] PIL/Pillow not installed. Creating an empty dummy png for testing copy.")
                with open(project_path / "images" / "example_fig.png", "w") as fimg:
                    fimg.write("dummy png content") 
            
            (project_path / "other_figs").mkdir(exist_ok=True)
            with open(project_path / "other_figs" / "another.jpeg", "w") as fimg:
                fimg.write("dummy jpeg content")
            print(f"    [+] Created dummy 'another.jpeg'")


            with open(project_path / "mycustomstyle.sty", "w", encoding="utf-8") as f:
                f.write(r"""
\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{mycustomstyle}[2024/05/19 My Custom Style]
\newcommand{\customstylecommand}{\textbf{Custom Style Text!}}
\RequirePackage{xcolor} 
\endinput
                """)

            with open(project_path / "main.tex", "w", encoding="utf-8") as f:
                f.write(r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{mycustomstyle} 
\usepackage{amsmath} 
\usepackage{graphicx} 

\title{Test Document for Pandoc Processing}
\author{Test Author}
\date{\today}
\graphicspath{{images/}{other_figs/}} 

\begin{document}
\maketitle

\begin{abstract}
This is the abstract. It mentions \cite{knuth1984}.
$p_\theta({\bm{y}}|{\bm{x}}) = \prod_{i=1}^{m} p_{\theta}(y_i|{\bm{x}}, {\bm{y}}_{<i})$
\end{abstract}

\section{Introduction}
This is the first section. \customstylecommand
It cites an important work by Knuth \cite{knuth1984}.
And another by Einstein \cite{einstein1905}.
Here is a reference to Figure \ref{fig:sample}.
$UCB(i)=w_i+C*\sqrt{2*\ln{\frac{N_i}{n_i}}}$

\begin{figure}[h!]
\centering
\includegraphics[width=0.5\textwidth]{example_fig} 
\caption{A sample figure from the images subfolder (no extension in command).}
\label{fig:sample}
\end{figure}

\section{Another Section}
This section includes an image from 'other_figs' folder.
\includegraphics[height=3cm]{another.jpeg} 

\section{Conclusion}
This is the conclusion.

\bibliographystyle{plain}
\bibliography{myrefs}

\end{document}
                """)
                
            with open(project_path / "myrefs.bib", "w", encoding="utf-8") as f:
                f.write(r"""
@article{knuth1984,
    author    = "Donald E. Knuth",
    title     = "Literate Programming",
    journal   = "The Computer Journal",
    volume    = "27",
    number    = "2",
    pages     = "97--111",
    year      = "1984"
}
@article{einstein1905,
    author    = "Albert Einstein",
    title     = "{Zur Elektrodynamik bewegter K{\"o}rper}",
    journal   = "Annalen der Physik",
    volume    = "322",
    number    = "10",
    pages     = "891--921",
    year      = "1905"
}
                """)
            print(f"[+] Dummy project created in '{project_path}'.")
            print(f"    To run with this dummy project: python your_script_name.py {project_path.name}")
        if args.output_folder is None: 
             output_folder_path = project_path.parent / f"{project_path.name}_output"


    print(f"[*] Processing LaTeX project in: '{project_path}'")
    print(f"[*] Markdown output ('paper.md') and figures will be saved to: '{output_folder_path}'")
    
    converter = LatexToMarkdownConverter(str(project_path))
    success = converter.convert_to_markdown(str(output_folder_path))

    if success:
        print(f"\n[*] Conversion process finished successfully.")
        print(f"    Markdown output: '{output_folder_path / 'paper.md'}'")
        print(f"    Figures copied to: '{output_folder_path}'")
    else:
        print(f"\n[!] Conversion process failed. Please check the logs above.", file=sys.stderr)
    
    print(f"[*] Ensure Pandoc, LaTeX (pdflatex, bibtex) are installed and in your system PATH.")

