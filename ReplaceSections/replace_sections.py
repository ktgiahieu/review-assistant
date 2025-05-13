import os
from bs4 import BeautifulSoup


class ReplaceSections:
    def __init__(
        self,
        input_dir,
        input_html_1,
        input_html_2,
        output_dir,
        output_html_1
    ):
        self.input_html1 = os.path.join(input_dir, input_html_1)
        self.input_html2 = os.path.join(input_dir, input_html_2)
        self.output_html1 = os.path.join(output_dir, output_html_1)
        self.soup1 = None
        self.soup2 = None

    def read_papers(self):
        with open(self.input_html1, 'r', encoding='utf-8') as f:
            self.soup1 = BeautifulSoup(f, 'html.parser')
        with open(self.input_html2, 'r', encoding='utf-8') as f:
            self.soup2 = BeautifulSoup(f, 'html.parser')

    def extract_sections(self, soup):
        """Extracts sections"""
        sections = {}
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for i, header in enumerate(headings):
            section_title = header.get_text(strip=True).lower()
            section_content = [header]
            next_node = header.find_next_sibling()
            while next_node and next_node.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                section_content.append(next_node)
                next_node = next_node.find_next_sibling()
            sections[section_title] = section_content
        return sections

    def find_common_section_in_both_papers(self):
        self.sections1 = self.extract_sections(self.soup1)
        self.sections2 = self.extract_sections(self.soup2)
        self.common_sections = {
            title: self.sections2[title]
            for title in self.sections1.keys()
            if title in self.sections2
        }

    def replace_sections(self):
        for title, new_content in self.common_sections.items():
            old_content = self.sections1[title]
            if not old_content:
                continue

            first_tag = old_content[0]
            parent = first_tag.parent

            # Find index of first_tag in parent contents
            index = list(parent.contents).index(first_tag)

            # Remove old section content
            for tag in old_content:
                tag.extract()

            # Insert new section content in correct order
            for i, tag in enumerate(new_content):
                parent.insert(index + i, tag)

    def save_updated_papers(self):
        with open(self.output_html1, 'w', encoding='utf-8') as f:
            f.write(str(self.soup1))
