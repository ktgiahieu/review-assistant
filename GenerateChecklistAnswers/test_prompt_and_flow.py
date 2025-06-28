from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import json
from checklist_prompts import (
    MainPrompt,
    CompliancePrompt,
    ContributionPrompt,
    SoundnessPrompt,
    PresentationPrompt
)

load_dotenv()

KEY = os.getenv("KEY")
ENDPOINT = os.getenv("ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT = "Checklist-GPT-o3"


def generate_html_from_review(review):
    """Creates a user-friendly HTML report from the json review"""
    
    def get_answer_cell(answer):
        """Styles the answer cell based on the value."""
        color_map = {
            "Yes": "background-color: #e6ffed; color: #00611d;",
            "No": "background-color: #ffe6e6; color: #a30000;",
            "NA": "background-color: #f0f0f0; color: #555;",
            "UnKnown": "background-color: #fff8dc; color: #8b8000;",
        }
        style = color_map.get(answer, "")
        return f'<td style="{style}">{answer}</td>'

    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>Checklist Review: {review['paper_title']}</title>",
        """<style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 20px auto; padding: 0 15px; }
            h1, h2 { color: #111; border-bottom: 2px solid #eee; padding-bottom: 10px; }
            h1 { font-size: 2em; }
            h2 { font-size: 1.5em; margin-top: 40px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f8f8; font-weight: 600; }
            td:nth-child(1) { width: 5%; }
            td:nth-child(3) { width: 10%; text-align: center; font-weight: bold; }
            .abstract-section { margin-left: 20px; }
        </style>""",
        "</head>",
        "<body>",
        f"<h1>{review['paper_title']}</h1>",
        f"<p><strong>Paper Link:</strong> <a href='{review['paper_link']}'>{review['paper_link']}</a></p>",
        
        "<h2>Formatted Abstract</h2>",
        f"<div class='abstract-section'><p><strong>Background and Objectives:</strong> {review['formatted_abstract']['background_and_objectives']}</p>",
        f"<p><strong>Material and Methods:</strong> {review['formatted_abstract']['material_and_methods']}</p>",
        f"<p><strong>Results:</strong> {review['formatted_abstract']['results']}</p>",
        f"<p><strong>Conclusion:</strong> {review['formatted_abstract']['conclusion']}</p></div>"
    ]

    for section in review['checklists']:
        html_parts.append(f"<h2>{section['section_title']}</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>ID</th><th>Question</th><th>Answer</th><th>Reasoning</th></tr>")
        for item in section['items']:
            html_parts.append("<tr>")
            html_parts.append(f"<td>{item['question_id']}</td>")
            html_parts.append(f"<td>{item['question_text']}</td>")
            html_parts.append(get_answer_cell(item['answer']))
            html_parts.append(f"<td>{item['reasoning']}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table>")

    html_parts.extend(["</body>", "</html>"])
    return "\n".join(html_parts)


def load_paper():
    print("Loading paper")
    paper_path = "./data/ICLR2024_Sample/accepted/0akLDTFR9x_2310_20141v1/structured_paper_output/paper.md"
    with open(paper_path, "r", encoding="utf-8") as f:
        paper = f.read()
        return paper


def get_LLM_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    openai_client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=KEY, api_version=API_VERSION)
    completion_args = {
        "model": DEPLOYMENT,
        "messages": messages
    }
    llm_response = openai_client.chat.completions.create(**completion_args)
    response = llm_response.choices[0].message.content
    try:
        json_response = json.loads(response)
    except json.JSONDecodeError as e:
        print("Failed to parse main JSON response:", e)
        json_response = {"error": "Invalid JSON returned from main prompt"}
    return json_response


def get_responses(paper):
    main_prompt = MainPrompt.get_main_prompt(paper)
    checklist_prompts = [
        CompliancePrompt.get_compliance_prompt(paper),
        ContributionPrompt.get_contribution_prompt(paper),
        SoundnessPrompt.get_soundness_prompt(paper),
        PresentationPrompt.get_presentation_prompt(paper),
    ]

    print("Main prompt")
    json_response = get_LLM_response(main_prompt)
    json_response["checklists"] = []
    for i, prompt in enumerate(checklist_prompts):
        print(f"Checklist prompt {i+1}")
        json_response["checklists"].append(get_LLM_response(prompt))

    return json_response


def save_json(json_response):
    print("Saving json")
    with open("llm_responses.json", 'w', encoding='utf-8') as f:
        json.dump(json_response, f, ensure_ascii=False, indent=2)


def save_html(json_response):
    print("Saving html")
    html = generate_html_from_review(json_response)
    with open("llm_responses.html", "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    paper = load_paper()
    review = get_responses(paper)
    save_json(review)
    save_html(review)
    print("Responses saved to llm_responses.html")
