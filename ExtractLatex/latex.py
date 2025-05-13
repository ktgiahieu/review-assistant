import re
import os
import json
import shutil
from pathlib import Path


class Latex:
    def __init__(self, folder_path):
        # Initialize paths and content variable
        self.folder_path = Path(folder_path)
        self.main_tex_path = None
        self.full_tex = ""
        self.parsed = {}

    def find_main_tex_file(self):
        """Finds and stores the main .tex file containing \\begin{document}."""

        print("[*] Finding main .tex file")
        for path in self.folder_path.rglob("*.tex"):
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    if r'\begin{document}' in f.read():
                        self.main_tex_path = path  # Store in instance variable
                        print("[+]")
                        return
            except Exception:
                continue
        raise FileNotFoundError("[-] No main .tex file with \\begin{document} found.")

    def read_latex(self):
        """Reads the main LaTeX file and resolves includes."""
        try:
            print(f"[*] Reading Latex file: {self.main_tex_path}")
            self.full_tex = self.read_tex_file(self.main_tex_path)
            print("[+]")
        except Exception as e:
            print(f"[-] Error reading LaTeX file: {e}")

    def read_tex_file(self, path):
        """Reads a LaTeX file recursively, resolving \\input and \\include."""
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except FileNotFoundError:
            return ""

        def replace_input_include(match):
            include_path = match.group(1)
            if not include_path.endswith(".tex"):
                include_path += ".tex"
            full_path = self.folder_path / include_path
            return self.read_tex_file(full_path)

        pattern = re.compile(r'\\(?:input|include)\{([^}]+)\}')
        content = re.sub(pattern, replace_input_include, content)
        return content

    def clean_latex(self):
        """Cleans LaTeX content by removing comments and preamble commands."""
        print("[*] Cleaning Latex")
        try:
            # Remove all comments
            tex = re.sub(r'(?<!\\)%.*', '', self.full_tex)

            # Remove preamble-related LaTeX commands
            tex = re.sub(r'\\documentclass.*?\n', '', tex)
            tex = re.sub(r'\\usepackage(?:\[[^\]]*\])?\{[^\}]*\}.*?\n?', '', tex)
            tex = re.sub(r'\\inputencoding\{.*?\}', '', tex)
            tex = re.sub(r'\\hypersetup\{.*?\}', '', tex)
            tex = re.sub(r'\\setlength\{.*?\}\{.*?\}', '', tex)
            tex = re.sub(r'\\geometry\{.*?\}', '', tex)
            tex = re.sub(r'\\pagestyle\{.*?\}', '', tex)
            tex = re.sub(r'\\thispagestyle\{.*?\}', '', tex)

            # Remove \clearpage commands
            tex = re.sub(r'\\clearpage', '', tex)

            # Remove \maketitle commands
            tex = re.sub(r'\\maketitle', '', tex)

            # Normalize blank lines
            tex = re.sub(r'\n\s*\n', '\n\n', tex)
            self.full_tex = tex.strip()
            print("[+]")
        except Exception as e:
            print(f"[-] Error cleaning LaTeX: {e}")

    def extract_title(self):
        """Extracts the title from LaTeX."""
        print("[*] Extracting Title")
        try:
            match = re.search(r'\\title\{(.+?)\}', self.full_tex, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception:
            print("[-] Error in extracting title")
            return ""

    def extract_abstract(self):
        """Extracts the abstract block."""
        print("[*] Extracting Abstract")
        try:
            match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', self.full_tex, re.DOTALL)
            return match.group(1).strip() if match else ""
        except Exception:
            print("[-] Error in extracting abstract")
            return ""

    def extract_sections(self):
        """Extracts section titles and their content."""
        print("[*] Extracting Sections")
        sections = []
        try:
            matches = re.findall(r'\\section\*?\{(.+?)\}(.*?)(?=(\\section|\Z))', self.full_tex, re.DOTALL)
            for sec_title, sec_body, _ in matches:
                clean_body = re.sub(r'\s+', ' ', sec_body).strip()
                sections.append({'title': sec_title.strip(), 'content': clean_body})
        except Exception:
            print("[-] Error in extracting sections")
            pass
        return sections

    def extract_all(self):
        """Extracts all parts into a structured dictionary."""
        self.parsed = {
            'title': self.extract_title(),
            'abstract': self.extract_abstract(),
            'sections': self.extract_sections()
        }

    def to_json(self, output_path):
        """Saves extracted data as a JSON file."""
        print("[*] Saving as json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.parsed, f, indent=4, ensure_ascii=False)
            print("[+]")
        except Exception as e:
            print(f"[-] Error saving JSON: {e}")

    def generate_and_save_html(self, output_path):
        """Generates HTML from parsed content and saves it to a file."""
        print("[*] Generating and saving HTML")
        try:
            html = "<html><head><meta charset='UTF-8'></head><body>"

            # Add title if available
            if self.parsed.get('title'):
                html += f"<h1>{self.parsed['title']}</h1>\n"

            # Add authors if available
            if self.parsed.get('authors'):
                html += f"<h3>{self.parsed['authors']}</h3>\n"

            # Add abstract if available
            if self.parsed.get('abstract'):
                html += f"<h2>Abstract</h2><p>{self.parsed['abstract']}</p>\n"

            # Add sections
            for sec in self.parsed.get('sections', []):
                html += f"<h2>{sec['title']}</h2><p>{sec['content']}</p>\n"

            html += "</body></html>"

            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
                
            # Find images path in html and copy them to output path also
            self.copy_images_to_output_path(html, output_path)

            print("[+]")

        except Exception as e:
            print(f"[-] Error generating or saving HTML: {e}")

    def copy_images_to_output_path(self, html, output_path):
        output_fig_path = os.path.dirname(output_path)
        # Find all image paths
        image_paths_indicated_in_html = re.findall(r"\\includegraphics\s*(?:\[.*?\])?\s*\{([^}]+?\.[a-zA-Z]+)\}", html)
        absolute_image_paths_source = [os.path.join(self.folder_path, x) for x in image_paths_indicated_in_html]
        absolute_image_paths_target = [os.path.join(output_fig_path, x) for x in image_paths_indicated_in_html]

        # Create neccessary folders for each image_path
        for image_path in absolute_image_paths_target:
            folder_path = os.path.dirname(image_path)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        # Copy these images to target
        for source, target in zip(absolute_image_paths_source, absolute_image_paths_target):
            shutil.copyfile(source, target)