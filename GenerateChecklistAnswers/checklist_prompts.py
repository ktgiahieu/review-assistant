class MainPrompt:
    prompt = """You are an expert reviewer for a scientific conference. Your task is to analyze the provided conference paper and generate a detailed, structured review in JSON format.

The JSON output should follow this structure:
{
  "paper_title": "The verbatim title of the paper",
  "formatted_abstract": {
    "background_and_objectives": "Summary of background and objectives.",
    "material_and_methods": "Summary of materials and methods.",
    "results": "Summary of results.",
    "conclusion": "Summary of the conclusion."
  },
}

Rules:
1. "paper_title": Use the exact title of the provided paper.
2. "formatted_abstract: Write a summary of the provided conference paper. You may use the original abstract and the full paper content. Deliver your summary in these sections: "background_and_objectives", "material_and_methods", "results", and "conclusion".
3. JSON format: Your response MUST be a single, valid JSON object that conforms to the schema provided above. Do not include any html tags in the response. Do not include any invalid characters in the JSON respones. Do not include any text, markdown, or code formatting before or after the JSON object.


BEGIN OF PROVIDED CONFERENCE PAPER:
{paper}
END OF PROVIDED CONFERENCE PAPER
"""

    @staticmethod
    def get_prompt(paper):
        return MainPrompt.prompt.replace("{paper}", paper)


class CompliancePrompt:
    prompt = """Now lets complete a checklist for  "Compliance to ethics guidelines" based on the content of the provided conference paper.

For each checklist question, you must provide:
1.  `answer`: Choose one of "Yes", "No", or "NA".
    - "Yes": If the paper explicitly and satisfactorily addresses the question.
    - "No": If the paper fails to address the question or does so inadequately.
    - "Unknown": If the paper does not contain enough information to make a judgment.
    - "NA": If the question is not applicable to the paper's scope or content.
2.  `reasoning`: A concise, one-to-two-sentence explanation for your answer. Base your reasoning directly on the content of the paper, citing specific sections or findings where possible.

The JSON output should follow this structure:
{
    "section_title": "A. Compliance",
    "items": [
    {
        "question_id": "A1",
        "question_text": "Does the paper discuss potential positive societal impacts of the work?",
        "answer": "Yes",  # this is an exmaple answer
        "reasoning": "The paper discusses positive impacts in the 'Societal Impact' section on page 8."  # this is an example reasoning
    },
    // ... more items for this section
    ]
},

Rules:
1. JSON format: Your response MUST be a single, valid JSON object that conforms to the schema provided above. Do not include any html tags in the response. Do not include any invalid characters in the JSON respones. Do not include any text, markdown, or code formatting before or after the JSON object.


BEGIN OF COMPLIANCE LIST:
A. Compliance
{compliance_list}
END OF COMPLIANCE LIST


BEGIN OF PROVIDED CONFERENCE PAPER:
{paper}
END OF PROVIDED CONFERENCE PAPER
"""

    @staticmethod
    def get_prompt(paper):
        return CompliancePrompt.prompt.replace("{paper}", paper).replace("{compliance_list}", compliance_list)


class ContributionPrompt:
    prompt = """Now lets complete a checklist for  "Contribution" based on the content of the provided conference paper.

For each checklist question, you must provide:
1.  `answer`: Choose one of "Yes", "No", "UnKnown", or "NA".
    - "Yes": If the paper explicitly and satisfactorily addresses the question.
    - "No": If the paper fails to address the question or does so inadequately.
    - "Unknown": If the paper does not contain enough information to make a judgment.
    - "NA": If the question is not applicable to the paper's scope or content.
2.  `reasoning`: A concise, one-to-two-sentence explanation for your answer. Base your reasoning directly on the content of the paper, citing specific sections or findings where possible.

The JSON output should follow this structure:
{
    "section_title": "B. Contribution",
    "items": [
    {
        "question_id": "B1",
        "question_text": "Are the main claims clearly stated in the abstract?",
        "answer": "Yes", # this is an exmaple answer
        "reasoning": "The paper discusses main claims in the 'Abstract'."  # this is an example reasoning
    },
    // ... more items for this section
    ]
},

Rules:
1. JSON format: Your response MUST be a single, valid JSON object that conforms to the schema provided above. Do not include any html tags in the response. Do not include any invalid characters in the JSON respones. Do not include any text, markdown, or code formatting before or after the JSON object.


BEGIN OF CONTRIBUTION LIST:
B. Contribution
{contribution_list}
END OF CONTRIBUTION LIST


BEGIN OF PROVIDED CONFERENCE PAPER:
{paper}
END OF PROVIDED CONFERENCE PAPER
"""

    @staticmethod
    def get_prompt(paper):
        return ContributionPrompt.prompt.replace("{paper}", paper).replace("{contribution_list}", contribution_list)


class SoundnessPrompt:
    prompt = """Now lets complete a checklist for  "Soundness" based on the content of the provided conference paper.

For each checklist question, you must provide:
1.  `answer`: Choose one of "Yes", "No", "UnKnown", or "NA".
    - "Yes": If the paper explicitly and satisfactorily addresses the question.
    - "No": If the paper fails to address the question or does so inadequately.
    - "Unknown": If the paper does not contain enough information to make a judgment.
    - "NA": If the question is not applicable to the paper's scope or content.
2.  `reasoning`: A concise, one-to-two-sentence explanation for your answer. Base your reasoning directly on the content of the paper, citing specific sections or findings where possible.

The JSON output should follow this structure:
{
    "section_title": "C. Soundness",
    "items": [
    {
        "question_id": "C1",
        "question_text": "Are theorems, lemmas, or key theoretical results clearly stated?",
        "answer": "NA", # this is an exmaple answer
        "reasoning": "This is not a theoretical paper and it does not have any theorems."  # this is an example reasoning
    },
    // ... more items for this section
    ]
},

Rules:
1. JSON format: Your response MUST be a single, valid JSON object that conforms to the schema provided above. Do not include any html tags in the response. Do not include any invalid characters in the JSON respones. Do not include any text, markdown, or code formatting before or after the JSON object.


BEGIN OF SOUNDNESS LIST:
C. Soundness
{soundness_list}
END OF SOUNDNESS LIST


BEGIN OF PROVIDED CONFERENCE PAPER:
{paper}
END OF PROVIDED CONFERENCE PAPER
"""

    @staticmethod
    def get_prompt(paper):
        return SoundnessPrompt.prompt.replace("{paper}", paper).replace("{soundness_list}", soundness_list)


class PresentationPrompt:
    prompt = """Now lets complete a checklist for  "Presentation" based on the content of the provided conference paper.

   For each checklist question, you must provide:
1.  `answer`: Choose one of "Yes", "No", "UnKnown", or "NA".
    - "Yes": If the paper explicitly and satisfactorily addresses the question.
    - "No": If the paper fails to address the question or does so inadequately.
    - "Unknown": If the paper does not contain enough information to make a judgment.
    - "NA": If the question is not applicable to the paper's scope or content.
2.  `reasoning`: A concise, one-to-two-sentence explanation for your answer. Base your reasoning directly on the content of the paper, citing specific sections or findings where possible.

The JSON output should follow this structure:
{
    "section_title": "D. Presentation",
    "items": [
    {
        "question_id": "D1",
        "question_text": "Is the abstract a clear and concise summary of the paper’s content?",
        "answer": "Yes", # this is an exmaple answer
        "reasoning": "The abstract completely summarizes the paper content"  # this is an example reasoning
    },
    // ... more items for this section
    ]
},

Rules:
1. JSON format: Your response MUST be a single, valid JSON object that conforms to the schema provided above. Do not include any html tags in the response. Do not include any invalid characters in the JSON respones. Do not include any text, markdown, or code formatting before or after the JSON object.


BEGIN OF PRESENTATION LIST:
D. Presentation
{presentation_list}
END OF PRESENTATION LIST


BEGIN OF PROVIDED CONFERENCE PAPER:
{paper}
END OF PROVIDED CONFERENCE PAPER
"""

    @staticmethod
    def get_prompt(paper):
        return PresentationPrompt.prompt.replace("{paper}", paper).replace("{presentation_list}", presentation_list)


compliance_list = """[ ] A1 Does the paper discuss potential positive societal impacts of the work?
[ ] A2 Does the paper discuss potential negative societal impacts or risks of the work?
[ ] A3 If negative societal impacts or risks are identified, does the paper discuss potential mitigation strategies?
[ ] A4 Does the paper describe safeguards for responsible release of data or models with high misuse risk?
[ ] A5 Are the creators or original owners of all used assets (e.g., code, data, models) properly credited in the paper?
[ ] A6 Are the license and terms of use for all used assets explicitly mentioned and respected in the paper?
[ ] A7 For research involving human subjects or participants (including crowdsourcing), does the paper indicate that informed consent was obtained from all participants, and/or a clear explanation provided?
[ ] A8 For research involving human subjects or participants, does the paper detail compensation, and was it ensured to be fair (e.g., at least the minimum wage in the data collector's country/region)?
[ ] A9 For crowdsourcing experiments, does the paper include or provide access to the full text of instructions given to participants and relevant screenshots (if applicable)?
[ ] A10 Does the paper describe potential risks to study participants, if any?
[ ] A11 Does the paper describe whether any identified risks to study participants were disclosed to them prior to their participation?
[ ] A12 Does the paper describe whether Institutional Review Board (IRB) approvals (or an equivalent ethics review process based on institutional/national guidelines) were obtained for research with human subjects?
[ ] A13 Is the status of IRB approval (or equivalent ethics review) clearly stated in the paper?
[ ] A14 If IRB approval (or equivalent ethics review) was not required or obtained for research involving human subjects, is a clear justification provided in the paper?
[ ] A15 Does the research avoid using datasets that are known to be deprecated or have been taken down by their original creators (unless the use is specifically for audit or critical assessment and this is justified)?
[ ] A16 If new datasets were collected or existing datasets were significantly modified, were appropriate measures taken to assess and communicate their representativeness for the intended population and task, as indicated in the paper?
[ ] A17 Does the research avoid primarily aiming to develop or enhance the lethality of weapons systems?
[ ] A18 Has the research considered and addressed potential security vulnerabilities or risks of accidents if the developed system or technology were deployed in real-world scenarios?
[ ] A19 Has the research considered and addressed how the technology could be used to discriminate, exclude, or otherwise unfairly negatively impact individuals or groups, particularly those with protected characteristics?
[ ] A20 If bulk surveillance data was used, was its collection, processing, and analysis compliant with all relevant local and international laws and ethical guidelines regarding surveillance and privacy?
[ ] A21 Does the research avoid using surveillance data in ways that could predict protected characteristics or otherwise endanger individual well-being or rights?
[ ] A22 Has the research considered and addressed the risk of the technology being used to facilitate deceptive interactions (e.g., creating deepfakes for disinformation), harassment, or fraud?
[ ] A23 Has the research considered and addressed potential significant negative environmental impacts stemming from the research process or its potential applications?
[ ] A24 Does the research avoid building upon or facilitating any activities that are illegal under applicable laws?
[ ] A25 Does the research avoid applications that could foreseeably be used to deny people their fundamental human rights (e.g., freedom of speech, right to privacy, right to health, liberty, security)?
[ ] A26 Have potential biases in the data, algorithms, models, or interpretation of results been carefully considered, investigated, and transparently discussed in the paper, especially concerning fairness and impacts on different demographic groups?
[ ] A27 If data or models are released as part of the research, are licenses provided that clearly state the intended use, limitations, and any restrictions to prevent misuse?
[ ] A28 Were appropriate technical and organizational measures (e.g., anonymization, pseudonymization, encryption, secure storage, access controls) implemented to protect the privacy and security of data used or collected, especially for sensitive or personally identifiable information?
[ ] A29 If the research exposes a security vulnerability in an existing system or software, does the paper indicate that responsible disclosure procedures were followed in consultation with the owners or maintainers of that system/software?
[ ] A30 Has legal compliance with relevant local, national, and international laws and regulations (e.g., data protection acts like GDPR, copyright law, intellectual property rights) been ensured throughout the research process and in the paper's content?
[ ] A31 Does the paper accurately and honestly report all findings, including any limitations or negative results, without fabrication, falsification, or selective misrepresentation of data or outcomes?
[ ] A32 Are all methods, experimental setups, and results presented in a clear, unambiguous, and transparent manner?
[ ] A33 Are all external sources of information, ideas, data, code, or other materials appropriately and accurately cited and acknowledged in the paper?
[ ] A34 Does the paper avoid misrepresenting, unfairly criticizing, or mischaracterizing competing or related work by other researchers?
[ ] A35 Does the research avoid making unsubstantiated or exaggerated claims of diversity, universality, or general applicability for datasets, models, or findings?
[ ] A36 If the research involves elements that could intentionally cause harm (e.g., in security research like red-teaming), is the ethical justification for this approach clearly provided in the paper?
"""
contribution_list = """[ ] B1 Are the main claims clearly stated in the abstract?
[ ] B2 Are the main claims clearly stated in the introduction?
[ ] B3 Do the claims in the abstract accurately reflect the paper's contributions?
[ ] B4 Do the claims in the introduction accurately reflect the paper's contributions?
[ ] B5 Do the claims accurately reflect the scope and limitations of the work?
[ ] B6 Are the specific contributions clearly summarized?
[ ] B7 Is the novelty of the contributions clearly articulated relative to prior work?
[ ] B8 Is the significance of the contributions made clear?
[ ] B9 Do the claims match the presented theoretical results (if any)?
[ ] B10 Do the claims match the presented experimental results (if any)?
[ ] B11 Is the potential for generalization of results discussed appropriately in relation to the claims?
[ ] B12 Are aspirational goals (if mentioned) clearly distinguished from the paper's achieved results?
[ ] B13 Does the paper explicitly discuss the limitations of the work performed?
[ ] B14 Are the key assumptions underlying the work identified?
[ ] B15 Is the robustness of the results to violations of assumptions discussed (if applicable)?
[ ] B16 Is the scope of the claims (e.g., tested datasets, number of runs) acknowledged as a potential limitation?
[ ] B17 Are factors that might influence the performance of the proposed method discussed?
[ ] B18 Is the computational efficiency and scalability of the proposed method discussed?
[ ] B19 Are potential limitations related to fairness discussed (if applicable)?
[ ] B20 Are potential limitations related to privacy discussed (if applicable)?
[ ] B21 Does the paper identify any new or enhanced applications enabled by the research?
[ ] B22 Are both potential positive and negative societal impacts discussed?
[ ] B23 Are any significant uncertainties or limitations in the impact assessment acknowledged?
[ ] B24 Does the paper connect its contributions to broader technological or scientific trends?
[ ] B25 Does the paper identify specific stakeholders or entities likely to be affected by its outcomes?
[ ] B26 Are ethical considerations related to the research methodology or data use addressed?
[ ] B27 Does the paper discuss the potential for feedback loops or unintended consequences?
[ ] B28 Are suggestions provided for further work to improve or govern societal outcomes?
[ ] B29 Does the impact discussion go beyond vague generalities to mention tractable, neglected, or high-significance effects?
[ ] B30 Is there a plausible connection drawn between the research and its potential real-world implications?
[ ] B31 Is the broader impact statement meaningfully integrated into the paper, rather than treated as an afterthought?
[ ] B32 Are potential misuses or dual-use concerns of the research acknowledged?
[ ] B33 Does the research focus on a topic related to machine learning and artificial intelligence, including applications (vision, language, speech, audio, creative AI), deep learning, evaluation, general machine learning, infrastructure, ML for sciences, neuroscience and cognitive science, optimization, probabilistic methods, reinforcement learning, social and economic aspects of ML, or theory?
[ ] B34 If LLMs are used in the methodology, is their use clearly documented in the experimental setup or equivalent section?
[ ] B35 Does the submission avoid plagiarism?
[ ] B36 Are all citations accurate, correct, and verified (no fabricated or hallucinated references)?
[ ] B37 If citing contemporaneous work, does the paper cite and discuss it appropriately without omitting relevant related work?
[ ] B38 Is there no evidence of dual submission to another peer-reviewed venue with proceedings or a journal?
[ ] B39 If the paper builds on prior workshop papers, is it clear that prior versions did not appear in archival proceedings?
[ ] B40 Does the paper avoid thin slicing of contributions that would make it incremental relative to other concurrent submissions?
[ ] B41 Are all experiments, datasets, proofs, and algorithms described in sufficient detail for reproducibility?
[ ] B42 If supplementary material is included, is it clearly marked and uploaded as a single file or proper zipped archive?
[ ] B43 If a reproducibility statement is included, does it reference where reproducibility details can be found in the paper or supplementary materials?
"""
soundness_list = """[ ] C1 Are theorems, lemmas, or key theoretical results clearly stated?
[ ] C2 For each theoretical result, is the full set of assumptions clearly stated?
[ ] C3 Is a complete proof provided for each theoretical result, either in the main text or supplementary material?
[ ] C4 If proofs are deferred to an appendix or supplement, is an informative proof sketch included in the main paper?
[ ] C5 Are theorems, formulas, and proofs properly numbered for easy reference?
[ ] C6 Are theorems, formulas, and proofs consistently cross-referenced where they are used?
[ ] C7 Are theoretical tools or prior results used in proofs properly cited?
[ ] C8 Do the mathematical arguments appear correct and rigorous?
[ ] C9 Are theoretical claims demonstrated or validated empirically where appropriate?
[ ] C10 Is the experimental methodology clearly described?
[ ] C11 Is the motivation for using the chosen datasets clearly explained?
[ ] C12 Are the datasets used adequately described (including source, size, and key characteristics)?
[ ] C13 For novel datasets, are they included in a data appendix or supplemental material?
[ ] C14 Will all novel datasets be made publicly available under a research-appropriate license?
[ ] C15 Are datasets from the literature properly cited?
[ ] C16 Are datasets from the literature publicly available?
[ ] C17 For non-public datasets, is a detailed description provided along with a justification for their use?
[ ] C18 Are the data splits (e.g., training, validation, test) clearly specified?
[ ] C19 Are evaluation metrics clearly defined?
[ ] C20 Is the choice of evaluation metrics well motivated?
[ ] C21 Are baseline methods appropriate and relevant to the research question?
[ ] C22 Are implementation details of baseline methods either provided or properly referenced?
[ ] C23 Are all relevant hyperparameters reported?
[ ] C24 Is the procedure for hyperparameter selection clearly described (e.g., search space, selection criteria)?
[ ] C25 Does the paper state the number and range of values tried for each hyperparameter?
[ ] C26 Is the optimizer used clearly specified (e.g., Adam, SGD)?
[ ] C27 Are other training details provided (e.g., learning rate, batch size, number of epochs, initialization)?
[ ] C28 Are all training and test details needed to interpret the results (e.g., data splits, optimizer, hyperparameters) specified?
[ ] C29 If algorithms depend on randomness, is the method for setting random seeds clearly described?
[ ] C30 Is the computing infrastructure used (e.g., hardware, memory, OS, software libraries) clearly specified?
[ ] C31 Are the number of algorithm runs used to compute each reported result clearly stated?
[ ] C32 Are the main experimental results clearly presented?
[ ] C33 Are results compared fairly against the baselines?
[ ] C34 Are ablation studies conducted to isolate the impact of different components?
[ ] C35 Are error bars, confidence intervals, or other measures of variability reported for key results?
[ ] C36 Is the method for computing statistical significance or variability clearly described (e.g., number of trials, standard deviation)?
[ ] C37 Are the sources of variation (e.g., random seeds, data splits) clearly stated?
[ ] C38 Does the analysis include distributional information, not just summary statistics?
[ ] C39 Is statistical significance assessed using appropriate tests?
[ ] C40 Is the analysis of results insightful and supported by evidence?
[ ] C41 Does the paper disclose all resources (code, data, scripts) necessary to reproduce the main experimental results?
[ ] C42 Will the source code be made publicly available under a suitable license?
[ ] C43 Will any novel datasets introduced be made publicly available under a suitable license?
[ ] C44 Is code for data preprocessing included or referenced?
[ ] C45 Is source code for conducting and analyzing experiments included or referenced?
[ ] C46 Does the implementation code include comments referencing the paper sections they relate to?
[ ] C47 Are all new assets (e.g., datasets, models, code) introduced in the paper properly documented?
[ ] C48 Are all creators or owners of external assets (e.g., datasets, code) properly credited, including license and terms of use?
[ ] C49 Does the paper distinguish between speculation, hypotheses, and verified claims?
[ ] C50 Are pedagogical references provided to help less-experienced readers understand the background needed for replication?
"""
presentation_list = """[ ] D1 Is the abstract a clear and concise summary of the paper’s content?
[ ] D2 Does the abstract include enough context for readers to understand the topic?
[ ] D3 Does the introduction provide sufficient background and motivation?
[ ] D4 Is the problem setup understandable and accessible to the target audience?
[ ] D5 Is the paper organized in a logical and easy-to-follow manner?
[ ] D6 Are section and subsection headings clear and descriptive?
[ ] D7 Are paragraphs and sections well-structured and coherent?
[ ] D8 Is the narrative flow smooth across sections and transitions?
[ ] D9 Is the language clear, precise, and unambiguous?
[ ] D10 Is grammar, spelling, and punctuation correct throughout?
[ ] D11 Is the writing style suitable for a technical conference?
[ ] D12 Are sentences concise and free of unnecessary jargon?
[ ] D13 Is the paper accessible to readers not deeply familiar with the subfield?
[ ] D14 Are mathematical notations clearly defined upon first use?
[ ] D15 Are notations used consistently throughout the paper?
[ ] D16 Are acronyms and abbreviations defined at first use and used consistently?
[ ] D17 Are all figures and tables essential and well-integrated into the text?
[ ] D18 Are figure and table captions informative and self-contained?
[ ] D19 Are axes labeled clearly, with units where appropriate?
[ ] D20 Are font sizes in figures and tables readable?
[ ] D21 Are plots and diagrams visually clear and appropriate for the data?
[ ] D22 Are color, shape, or style distinctions clear in multi-series plots?
[ ] D23 Are legends provided when needed and easy to interpret?
[ ] D24 Are all figures and tables correctly referenced and discussed in the text?
[ ] D25 Are all references cited properly in the text?
[ ] D26 Are citations formatted consistently and according to guidelines?
[ ] D27 Is the reference list complete and free of formatting issues?
[ ] D28 Does the paper adhere to standard formatting guidelines (e.g., margins, font size, length)?
[ ] D29 Are all footnotes, equations, and lists properly formatted and placed?
[ ] D30 Are any appendices, figures, or tables referenced correctly from the main text?
[ ] D31 Are supplementary materials clearly labeled and easy to navigate?
"""
