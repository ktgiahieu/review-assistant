from latex import Latex

LATEX_FOLDER = "arXiv-2404.12253v2"
OUPUT_JSON_FILE = "output.json"
OUPUT_HTML_FILE = "output.html"


latex = Latex(f"./{LATEX_FOLDER}")
latex.find_main_tex_file()
latex.read_latex()
latex.clean_latex()
latex.extract_all()
latex.to_json(OUPUT_JSON_FILE)
latex.generate_and_save_html(OUPUT_HTML_FILE)
