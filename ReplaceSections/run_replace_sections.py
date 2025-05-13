from replace_sections import ReplaceSections

INPUT_FOLDER = "./input"
INPUT_HTML_1 = "paper1.html"
INPUT_HTML_2 = "paper2.html"
OUTPUT_FOLDER = "./output"
OUTPUT_HTML_1 = "paper1.html"


replacer = ReplaceSections(
    input_dir=INPUT_FOLDER,
    input_html_1=INPUT_HTML_1,
    input_html_2=INPUT_HTML_2,
    output_dir=OUTPUT_FOLDER,
    output_html_1=OUTPUT_HTML_1,
)
replacer.read_papers()
replacer.find_common_section_in_both_papers()
replacer.replace_sections()
replacer.save_updated_papers()
