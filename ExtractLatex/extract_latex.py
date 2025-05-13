from latex import Latex

LATEX_FOLDER = "arXiv-2404.12253v2"
OUTPUT_JSON_FILE = "output.json"
OUTPUT_HTML_FILE = "output.html"


latex = Latex(f"./{LATEX_FOLDER}")
latex.find_main_tex_file()
latex.read_latex()
latex.clean_latex()
latex.extract_all()
latex.to_json(OUTPUT_JSON_FILE)
latex.generate_and_save_html(OUTPUT_HTML_FILE)

