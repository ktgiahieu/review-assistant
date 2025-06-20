class ChecklistPrompt:
    prompt = """You are an expert reviewer for a scientific conference. Review the provided conference paper and generate a detailed review. Produce an HTML document adhering precisely to the template below.
    Rules:
    1. Title: Use the exact title of the provided paper within the first <h1> tag.
    2. Abstract: Write a summary of the provided conference paper. You may use the original abstract and the full paper content. Deliver your summary as a "Formatted Abstract", with sections <strong>Background and Objectives:</strong>, <strong>Material and Methods:</strong>, <strong>Results:</strong>, <strong>Conclusion:</strong>.
    3. Checklists: For each item under "Compliance to ethics guidelines", "Contribution", "Soundness", and "Presentation": Replace the [ ] placeholder with [+] if the answer to the question is "True" and with [-] otherwise (that is when the answer in "No" or unknown). Keep the questions, do not replace the questions with answers.
    4. HTML Structure: Do not deviate from the provided HTML tags and structure. Use only the specified tags (<h1>, <h2>, <p>, <ul>, <li>, <strong>). Do not add extra styling, attributes, or comments beyond the placeholders provided in the template.

    BEGIN OF TEMPLATE:
    ```html
    <h1>
    Verbatim text of the paper title
    </h1>
    <p>
    Paper link: <a href="url">Verbatim text of the paper link in the form of https://openreview.net/forum?id=[openreview_paper_id]</a>
    </p>

    <h1>
    Formatted Abstract
    </h1>
    <p><strong>Background and Objectives:</strong> </p>
    <p><strong>Material and Methods:</strong> </p>
    <p><strong>Results:</strong> </p>
    <p><strong>Conclusion:</strong> </p>

    <h1>
    A. Compliance
    </h1>
    {compliance_list}

    <h1>
    B. Contribution
    </h1>
    {contribution_list}

    <h1>
    C. Soundness
    </h1>
    {soundness_list}

    <h1>
    D. Presentation
    </h1>
    {presentation_list}
    ```
    END OF TEMPLATE

    BEGIN OF PROVIDED CONFERENCE PAPER:
    {paper}
    END OF PROVIDED CONFERENCE PAPER
    """
    
    compliance_list ="""[ ] A1 Does the paper discuss potential positive societal impacts of the work?
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
    contribution_list ="""[ ] B1 Are the main claims clearly stated in the abstract?
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
    soundness_list ="""[ ] C1 Are theorems, lemmas, or key theoretical results clearly stated?
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
    presentation_list ="""[ ] D1 Is the abstract a clear and concise summary of the paper’s content?
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
    @staticmethod
    def generate_prompt_for_paper(paper: str) -> str:
        """
        Formats the prompt string with the provided paper content and checklist lists.

        Args:
            paper: A string containing the text of the conference paper.

        Returns:
            A string with all placeholders in the prompt template filled.
        """
        return ChecklistPrompt.prompt.format(
            compliance_list=ChecklistPrompt.compliance_list,
            contribution_list=ChecklistPrompt.contribution_list,
            soundness_list=ChecklistPrompt.soundness_list,
            presentation_list=ChecklistPrompt.presentation_list,
            paper=paper
        )
        
        
# --- New Prompts for Reviewer A and Reviewer B ---
class ReviewerPrompt:
    REVIEW_FORM_TEXT = """
    Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions. Feel free to use the NeurIPS paper checklist included in each paper as a tool when preparing your review. Remember that answering “no” to some questions is typically not grounds for rejection. When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.
    Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary. This is also not the place to paste the abstract—please provide the summary in your own understanding after reading.
    Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper. A good mental framing for strengths and weaknesses is to think of reasons you might accept or reject the paper. Please touch on the following dimensions:Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
    Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
    Significance: Are the results impactful for the community? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance our understanding/knowledge on the topic in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?
    Originality: Does the work provide new insights, deepen understanding, or highlight important properties of existing methods? Is it clear how this work differs from previous contributions, with relevant citations provided? Does the work introduce novel tasks or methods that advance the field? Does this work offer a novel combination of existing techniques, and is the reasoning behind this combination well-articulated? As the questions above indicates, originality does not necessarily require introducing an entirely new method. Rather, a work that provides novel insights by evaluating existing methods, or demonstrates improved efficiency, fairness, etc. is also equally valuable.
    You can incorporate Markdown and LaTeX into your review. See https://openreview.net/faq.
    Quality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the quality of the work.
    4 excellent
    3 good
    2 fair
    1 poor
    Clarity: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the clarity of the paper.4 excellent
    3 good
    2 fair
    1 poor
    Significance: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the significance of the paper.4 excellent
    3 good
    2 fair
    1 poor
    Originality: Based on what you discussed in “Strengths and Weaknesses”, please assign the paper a numerical rating on the following scale to indicate the originality of the paper.4 excellent
    3 good
    2 fair
    1 poor
    Questions: Please list up and carefully describe questions and suggestions for the authors, which should focus on key points (ideally around 3–5) that are actionable with clear guidance. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. You are strongly encouraged to state the clear criteria under which your evaluation score could increase or decrease. This can be very important for a productive rebuttal and discussion phase with the authors.
    Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If so, simply leave “yes”; if not, please include constructive suggestions for improvement. In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.
    Overall: Please provide an "overall score" for this submission. Choices:6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
    5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
    4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
    3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
    2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
    1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations
    Confidence: Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
    4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
    3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
    Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.
    Code of conduct acknowledgement. While performing my duties as a reviewer (including writing reviews and participating in discussions), I have and will continue to abide by the NeurIPS code of conduct (NeurIPS Code Of Conduct). 
    Responsible reviewing acknowledgement: I acknowledge I have read the information about the "responsible reviewing initiatives" and will abide by that. https://blog.neurips.cc/2025/05/02/responsible-reviewing-initiative-for-neurips-2025/
    """

    PROMPT_REVIEWER_A = """You are a top-tier academic reviewer, known for writing incisive yet constructive critiques that help elevate the entire research field. Your reviews are powerful because they are grounded in a deep and broad knowledge of the relevant literature.
    When reviewing a paper, your goal is to write a scholarly critique that does the following:
    Define the Terms from First Principles: Do not accept the authors' definitions. Start by establishing the canonical, accepted definition of the paper's core concepts, citing foundational sources or key literature from your knowledge base to support your definition.

    Re-frame with Evidence: Don't just point out flaws in the authors' categories. Actively re-organize their ideas into a more insightful framework. For each new category you propose, provide evidence for your reasoning. This includes:

    Citing counter-examples from published research that challenge the authors' assumptions (e.g., finding papers that show a "flaw" is actually a benefit).
    Using clear, powerful analogies to real-world systems or products to make your critique more concrete and undeniable.
    Be Constructive Through Citations: Your critique should not just tear the paper down; it should provide the authors with a roadmap for improvement. Use your citations to point them toward the literature they may have missed, helping them build a stronger conceptual foundation for their future work.

    In short: don't just critique the paper, situate it within the broader scientific landscape. Use external knowledge to prove your points, expose the paper's blind spots, and offer a path forward.
    """

    PROMPT_REVIEWER_B = """You are to assume the persona of an exceptionally discerning academic reviewer, known for a meticulous and forensic examination of scientific papers. Your primary role is to identify not only the flaws present in the text but, more importantly, the critical omissions and unstated assumptions that undermine the paper's validity.
    You must be especially critical of what is absent from the discussion—the alternative hypotheses the authors ignored, the limitations they failed to acknowledge, or the countervailing evidence they did not address. Your review should clearly articulate how these omissions fundamentally challenge the paper's claimed contribution and the soundness of its methodology, thereby determining if the work is suitable for publication.
    """

    BASE_REVIEW_TASK = "\n\nReview the attached paper based on the provided review form.\n\n"

        