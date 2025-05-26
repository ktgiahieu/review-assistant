# Extract Latex and convert everything to Markdown

For reference abstract extraction:

1. It will try to look for Arxiv ID if possible to get the abstract from arxiv.
2. If not, search on OpenAlex & Semantic scholar
3. If abstract is not directly available, follow the external DOI IDs, which can leads to sites like ScienceDirect or Springer
   1. Use Elsevier API for ScienceDirectID
   2. Use Meta API for Springer ID
4. If nothing works, scrap the HTML and look for "Abstract"/ "Introduction" to get the raw text

To extract latex from Arxiv downloaded folder and convert everything to Markdown:

```bash
python3 latex.py folder_with_latex_source -o output_folder
```

Prerequisites:

- pdflatex
- pandoc
- pdf2image
- Pillow
- reportlab
- elsapy
