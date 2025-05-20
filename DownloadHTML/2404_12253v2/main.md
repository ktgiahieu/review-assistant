---
lang: en
title: Toward Self-Improvement of LLMs via Imagination, Searching, and
  Criticizing
viewport: width=device-width, initial-scale=1, shrink-to-fit=no
---

1.  [[[1 ]{.ltx_tag .ltx_tag_ref}Introduction]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S1 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
2.  [[[2 ]{.ltx_tag .ltx_tag_ref}Related Work]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S2 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    1.  [[Search with LLM]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S2.SS0.SSS0.Px1 "In 2 Related Work ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    2.  [[LLM Self-improving]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S2.SS0.SSS0.Px2 "In 2 Related Work ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
3.  [[[3 ]{.ltx_tag .ltx_tag_ref}Preliminaries]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S3 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    1.  [[[3.1 ]{.ltx_tag .ltx_tag_ref}Problem Formulation]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S3.SS1 "In 3 Preliminaries ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    2.  [[[3.2 ]{.ltx_tag .ltx_tag_ref}Monte Carlo Tree
        Search]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S3.SS2 "In 3 Preliminaries ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
4.  [[[4 ]{.ltx_tag .ltx_tag_ref}[AlphaLLM]{.ltx_text
    .ltx_font_smallcaps}]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    1.  [[[4.1 ]{.ltx_tag .ltx_tag_ref}Overview]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS1 "In 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    2.  [[[4.2 ]{.ltx_tag .ltx_tag_ref}Data Synthesizing]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS2 "In 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    3.  [[[4.3 ]{.ltx_tag .ltx_tag_ref}$\eta$[Mcts]{.ltx_text
        .ltx_font_smallcaps}]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS3 "In 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        1.  [[[4.3.1 ]{.ltx_tag .ltx_tag_ref}Option-level
            MCTS]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS3.SSS1 "In 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        2.  [[[4.3.2 ]{.ltx_tag .ltx_tag_ref}Importance-Based Adaptive
            Branching]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS3.SSS2 "In 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        3.  [[[4.3.3 ]{.ltx_tag .ltx_tag_ref}State Merge]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS3.SSS3 "In 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        4.  [[[4.3.4 ]{.ltx_tag .ltx_tag_ref}Fast Rollout with
            Specialized LM]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS3.SSS4 "In 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    4.  [[[4.4 ]{.ltx_tag .ltx_tag_ref}Critic]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS4 "In 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        1.  [[Value Function]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS4.SSS0.Px1 "In 4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        2.  [[PRM]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS4.SSS0.Px2 "In 4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        3.  [[ORM]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS4.SSS0.Px3 "In 4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    5.  [[[4.5 ]{.ltx_tag .ltx_tag_ref}Policy
        Self-Improvement]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS5 "In 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        1.  [[Data generation]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS5.SSS0.Px1 "In 4.5 Policy Self-Improvement ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        2.  [[Policy finetuning]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S4.SS5.SSS0.Px2 "In 4.5 Policy Self-Improvement ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
5.  [[[5 ]{.ltx_tag .ltx_tag_ref}Experiments]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S5 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    1.  [[[5.1 ]{.ltx_tag .ltx_tag_ref}Experiment Setups]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S5.SS1 "In 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    2.  [[[5.2 ]{.ltx_tag .ltx_tag_ref}Results]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S5.SS2 "In 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    3.  [[[5.3 ]{.ltx_tag .ltx_tag_ref}Ablation Study]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S5.SS3 "In 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
6.  [[[6 ]{.ltx_tag .ltx_tag_ref}Conclusion]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#S6 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
7.  [[[A ]{.ltx_tag .ltx_tag_ref}Appendix]{.ltx_text
    .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1 "In Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    1.  [[[A.1 ]{.ltx_tag .ltx_tag_ref}Imagination, Searching,
        Criticizing and Learning Loop]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS1 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    2.  [[[A.2 ]{.ltx_tag .ltx_tag_ref}Option-level MCTS]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS2 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    3.  [[[A.3 ]{.ltx_tag .ltx_tag_ref}Importance-Based Adaptive
        Branching Under Uniform Distribution]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS3 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    4.  [[[A.4 ]{.ltx_tag .ltx_tag_ref}Importance-Based Adaptive
        Branching Under Gaussian Distribution]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS4 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    5.  [[[A.5 ]{.ltx_tag .ltx_tag_ref}Prompt Templates]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS5 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        1.  [[[A.5.1 ]{.ltx_tag .ltx_tag_ref}PRM]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS5.SSS1 "In A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        2.  [[[A.5.2 ]{.ltx_tag .ltx_tag_ref}ORM]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS5.SSS2 "In A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        3.  [[[A.5.3 ]{.ltx_tag .ltx_tag_ref}Policy
            Finetuning]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS5.SSS3 "In A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    6.  [[[A.6 ]{.ltx_tag .ltx_tag_ref}MCTS Details]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS6 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    7.  [[[A.7 ]{.ltx_tag .ltx_tag_ref}Additional Ablations]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS7 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
        1.  [[Fast-rollout model]{.ltx_text
            .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS7.SSS0.Px1 "In A.7 Additional Ablations ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    8.  [[[A.8 ]{.ltx_tag .ltx_tag_ref}Search Comparison]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS8 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    9.  [[[A.9 ]{.ltx_tag .ltx_tag_ref}Rollout Example]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS9 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    10. [[[A.10 ]{.ltx_tag .ltx_tag_ref}Critic Performance]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS10 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    11. [[[A.11 ]{.ltx_tag .ltx_tag_ref}Compute Resources]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS11 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
    12. [[[A.12 ]{.ltx_tag .ltx_tag_ref}Limitations and Future
        Work]{.ltx_text
        .ltx_ref_title}](https://arxiv.org/html/2404.12253v2#A1.SS12 "In Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}

::: {.ltx_page_main}
::: {.ltx_page_content}
# Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing {#toward-self-improvement-of-llms-via-imagination-searching-and-criticizing .ltx_title .ltx_title_document}

::: {.ltx_authors}
[ [Ye Tian^1,2^, Baolin Peng^1^[^1^[[^1^[footnotemark:
]{.ltx_note_type}[1]{.ltx_tag
.ltx_tag_note}]{.ltx_note_content}]{.ltx_note_outer}]{#footnotex1
.ltx_note .ltx_role_footnotemark}, Linfeng
Song^1^[^1^[[^1^[footnotemark: ]{.ltx_note_type}[1]{.ltx_tag
.ltx_tag_note}]{.ltx_note_content}]{.ltx_note_outer}]{#footnotex2
.ltx_note .ltx_role_footnotemark}, Lifeng Jin^1^, Dian Yu^1^, Lei
Han^2^\
[Haitao Mi^1†^, Dong Yu^1^\
^1^Tencent AI Lab, Bellevue, WA\
^2^Tencent Robotics X\
]{#id7.7.id7 .ltx_text
.ltx_font_bold}[{baolinpeng,lfsong,lifengjin,yudian,haitaomi,dyu}\@global.tencent.com]{#id8.8.id8
.ltx_text .ltx_font_typewriter}[\
]{#id9.9.id9 .ltx_text
.ltx_font_bold}[{yaptian,lxhan}\@tencent.com]{#id10.10.id10 .ltx_text
.ltx_font_typewriter}[\
\
]{#id11.11.id11 .ltx_text .ltx_font_bold} ]{.ltx_personname}[Equal
Contribution; †Corresponding Author]{.ltx_author_notes}]{.ltx_creator
.ltx_role_author}
:::

::: {.ltx_abstract}
###### Abstract {#abstract .ltx_title .ltx_title_abstract}

Despite the impressive capabilities of Large Language Models (LLMs) on
various tasks, they still struggle with scenarios that involves complex
reasoning and planning. Self-correction and self-learning emerge as
viable solutions, employing strategies that allow LLMs to refine their
outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs
in self-refining its response, particularly in complex reasoning and
planning task, remains dubious. In this paper, we introduce
[AlphaLLM]{#id12.id1.1 .ltx_text .ltx_font_smallcaps} for the
self-improvements of LLMs, which integrates Monte Carlo Tree Search
(MCTS) with LLMs to establish a self-improving loop, thereby enhancing
the capabilities of LLMs without additional annotations. Drawing
inspiration from the success of AlphaGo, [AlphaLLM]{#id12.id1.2
.ltx_text .ltx_font_smallcaps} addresses the unique challenges of
combining MCTS with LLM for self-improvement, including data scarcity,
the vastness search spaces of language tasks, and the subjective nature
of feedback in language tasks. [AlphaLLM]{#id12.id1.3 .ltx_text
.ltx_font_smallcaps} is comprised of prompt synthesis component, an
efficient MCTS approach tailored for language tasks, and a trio of
critic models for precise feedback. Our experimental results in
mathematical reasoning tasks demonstrate that [AlphaLLM]{#id12.id1.4
.ltx_text .ltx_font_smallcaps} significantly enhances the performance of
LLMs without additional annotations, showing the potential for
self-improvement in LLMs. The code is available at
<https://github.com/YeTianJHU/AlphaLLM>.
:::

::: {#S1 .section .ltx_section}
## [1 ]{.ltx_tag .ltx_tag_section}Introduction {#introduction .ltx_title .ltx_title_section}

::: {#S1.p1 .ltx_para}
LLMs, trained on trillions of tokens with billions of parameters have
shown unparalleled capabilities in a wide range of natural language
processing tasks (Touvron et al.,
[2023b](https://arxiv.org/html/2404.12253v2#bib.bib49){.ltx_ref}; Team
et al., [2023](https://arxiv.org/html/2404.12253v2#bib.bib47){.ltx_ref};
OpenAI,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib31){.ltx_ref}).
Nevertheless, they continue to face challenges in scenarios requiring
complex reasoning and strategic planning  (Valmeekam et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib51){.ltx_ref}; Stechly
et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib40){.ltx_ref}). While
advanced prompting approaches such as Chain, Tree, Graph-of-Thought (Wei
et al., [2022](https://arxiv.org/html/2404.12253v2#bib.bib55){.ltx_ref};
Yao et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib58){.ltx_ref}; Besta
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib4){.ltx_ref};
Ding et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib13){.ltx_ref}), it
remains essential to fine-tune LLMs using a substantial volume of
high-quality, supervised data to fundamentally improve the model
performance (Nye et al.,
[2021](https://arxiv.org/html/2404.12253v2#bib.bib30){.ltx_ref};
Lewkowycz et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib22){.ltx_ref}; Chung
et al., [2022](https://arxiv.org/html/2404.12253v2#bib.bib9){.ltx_ref}).
This methodology is inherently limited by the scope and quality of data
that humans can provide.
:::

::: {#S1.p2 .ltx_para}
Considering these challenges, the concept of self-correction and
self-learning have been proposed as promising solutions (Madaan et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib29){.ltx_ref};
Saunders et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib36){.ltx_ref}; Chen
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib6){.ltx_ref}).
Within these framework, LLMs typically operate by employing two main
strategies: 1) they continuously refine their responses based on the
feedback of their past responses, and 2) they extensively sample
responses then learn from preferences judged by itself as reward models
with PPO or DPO (Yuan et al.,
[2024a](https://arxiv.org/html/2404.12253v2#bib.bib60){.ltx_ref},
[b](https://arxiv.org/html/2404.12253v2#bib.bib61){.ltx_ref}; Chen
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib6){.ltx_ref}).
However, it remains a matter of ongoing research whether LLMs can
effectively critique their own outputs to either enhance response
quality or apply a scalar reward to indicate the quality of responses,
especially in contexts demanding intricate planning and
reasoning (Valmeekam et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib51){.ltx_ref}; Stechly
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib40){.ltx_ref};
Huang et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib21){.ltx_ref}; Hong
et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib20){.ltx_ref}). On the
other hand, advanced search algorithms such as MCTS, combined with
reinforcement learning, have enabled models to learn from self-play and
achieve human parity or even surpass human performance in complex tasks
such as the game of Go (Silver et al.,
[2016](https://arxiv.org/html/2404.12253v2#bib.bib38){.ltx_ref},
[2017](https://arxiv.org/html/2404.12253v2#bib.bib39){.ltx_ref}). This
naturally raises a question: is it viable to leverage the strengths of
MCTS alongside LLMs to inaugurate a novel paradigm of self-improving?
More precisely, could the assimilation of MCTS empower LLMs to more
effectively explore better responses, guided by strategic signals, and
subsequently optimize these responses to enhance overall performance?
:::

::: {#S1.p3 .ltx_para}
To answer this question, we begin with a systematic examination of
AlphaGo, identifying three critical aspects for its success:
([i]{#S1.p3.1.1 .ltx_text .ltx_font_italic}) The large volume of data,
including self-play data. ([ii]{#S1.p3.1.2 .ltx_text .ltx_font_italic})
The use of tree search, which facilitates the exploration of potential
moves through statistical sampling of the large search space.
([iii]{#S1.p3.1.3 .ltx_text .ltx_font_italic}) Accurate and unambiguous
environment feedback; the direct and accurate feedback (win or loss)
provided by the game of Go offers a clear and unequivocal learning
signal (Silver et al.,
[2017](https://arxiv.org/html/2404.12253v2#bib.bib39){.ltx_ref}). The
integration of MCTS with LLMs for self-improvement has several
challenges: ([i]{#S1.p3.1.4 .ltx_text .ltx_font_italic}) Limited Data:
High-quality annotated data for LLMs is generally scarce. Furthermore,
how to construct of synthetic data for LLMs training, similar to
AlphaGo's self-play data, remains unclear. ([ii]{#S1.p3.1.5 .ltx_text
.ltx_font_italic}) Search Efficiency: The vast number of potential token
combinations in natural language tasks results in an exponentially large
search space, posing a significant challenge to the efficiency of
MCTS (Ramamurthy et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib35){.ltx_ref}).
([iii]{#S1.p3.1.6 .ltx_text .ltx_font_italic}) Imperfect Feedback: In
contrast to the clear win/loss feedback in Go, feedback in natural
language tasks is often subjective and nuanced, without a
straightforward measure of success.
:::

![[Figure 1: ]{.ltx_tag
.ltx_tag_figure}Imagination-Searching-Criticizing self-improvement loop:
Imagination component synthesizes prompts as new learning examples, with
MCTS searching better trajectories guided by signals from critics for
policy improving.](x1.png){#S1.F1.g1 .ltx_graphics .ltx_centering
.ltx_img_landscape width="747" height="235"}

::: {#S1.p4 .ltx_para}
In this paper, we introduce [AlphaLLM]{#S1.p4.3.1 .ltx_text
.ltx_font_smallcaps}, an imagination-searching-criticizing framework
designed for the self-improvement of LLMs . [AlphaLLM]{#S1.p4.3.2
.ltx_text .ltx_font_smallcaps} consists of three key components, as
illustrated in Figure [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
First, an imagination component is designed to synthesize prompts,
alleviating the issues of data scarcity. Second, we propose
$\eta$[Mcts]{#S1.p4.3.3 .ltx_text .ltx_font_smallcaps} tailored for
efficient searching in language tasks. Particularly, it has been show
that planning at multiple levels of temporal abstraction is critical for
RL problems with a long horizon and large action space (Sutton et al.,
[1999b](https://arxiv.org/html/2404.12253v2#bib.bib44){.ltx_ref}; Peng
et al., [2017](https://arxiv.org/html/2404.12253v2#bib.bib33){.ltx_ref};
Luketina et al.,
[2019](https://arxiv.org/html/2404.12253v2#bib.bib27){.ltx_ref}). As
such, we propose formulating the text generation process as options over
a Markov Decision Process (MDP) problem, where each option represents
the generation of a collection of tokens for a specific subtask, similar
to the concept of chains in chain-of-thought prompting. This formulation
improves search efficiency by substantially reducing the search depth.
Additionally, we propose the use of state merge and adaptive branching
factors to further enhance search efficiency by balancing the trade-off
between search width and depth. Lastly, since accurate feedback is
crucial to the success of MCTS, we introduce a trio of critic models to
guide $\eta$[Mcts]{#S1.p4.3.4 .ltx_text .ltx_font_smallcaps}, including
a value function for estimating expected rewards, a process reward model
for assessing node correctness, and an outcome reward model for
evaluating the overall trajectory. For complex tasks with which LLMs
struggle assessing such as arithmetic computation and code execution, to
ensure the accuracy of feedback, we augment the critics with the
capacity to make dynamic decisions on which tools to use, when to use
them, and how to use them effectively. After $\eta$[Mcts]{#S1.p4.3.5
.ltx_text .ltx_font_smallcaps} stage, we collect the trajectory with the
largest reward from the critic models as the training examples to
improve LLMs.
:::

::: {#S1.p5 .ltx_para}
The experimental results on mathematical reasoning tasks demonstrate
that [AlphaLLM]{#S1.p5.1.1 .ltx_text .ltx_font_smallcaps} can
efficiently search for better responses and use them to improve LLMs'
performance, forming an effective self-improving loop. Notably, based on
Llama-2-70b and WizardMath-70B-V1.0, [AlphaLLM]{#S1.p5.1.2 .ltx_text
.ltx_font_smallcaps} can improve its performance from 57.8 to 92.0 on
GSM8K and from 20.7 to 51.0 on MATH, performing comparably to GPT-4.
:::
:::

::: {#S2 .section .ltx_section}
## [2 ]{.ltx_tag .ltx_tag_section}Related Work {#related-work .ltx_title .ltx_title_section}

::: {#S2.SS0.SSS0.Px1 .section .ltx_paragraph}
##### Search with LLM {#search-with-llm .ltx_title .ltx_title_paragraph}

::: {#S2.SS0.SSS0.Px1.p1 .ltx_para}
Effective search strategy has been shown crucial for tasks that involve
complex reasoning and planning, such as go (Silver et al.,
[2016](https://arxiv.org/html/2404.12253v2#bib.bib38){.ltx_ref}) and
math reasoning (Cobbe et al.,
[2021](https://arxiv.org/html/2404.12253v2#bib.bib11){.ltx_ref};
Hendrycks et al.,
[2021](https://arxiv.org/html/2404.12253v2#bib.bib19){.ltx_ref}). For
math reasoning tasks, various search methods have been studied. One
direction of research (Zhu et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib64){.ltx_ref}; Xie
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib56){.ltx_ref})
designed beam search with dynamic pruning, where beam items of low
quality are pruned. Another line of work (Yao et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib58){.ltx_ref}; Long,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib26){.ltx_ref}; Besta
et al., [2024](https://arxiv.org/html/2404.12253v2#bib.bib4){.ltx_ref};
Hao et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib18){.ltx_ref}; Feng
et al., [2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref})
maintains a tree or a graph that represents the current progress of
solving the input question where potential branches are iteratively
expanded. Both our approach and Feng et al.
([2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref}) are
based on the MCTS algorithm, while one main difference is how to define
a search step: Feng et al.
([2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref}) fix a
search step to be either a token or a sentence, while our approach is
more flexible on deciding steps. We have also carefully designed the
MCTS process, incorporating multiple critique signals to guide the
search more effectively and introducing adaptive search parameters for
improved state exploration. As the result, our approach achieves much
better performances.
:::
:::

::: {#S2.SS0.SSS0.Px2 .section .ltx_paragraph}
##### LLM Self-improving {#llm-self-improving .ltx_title .ltx_title_paragraph}

::: {#S2.SS0.SSS0.Px2.p1 .ltx_para}
Being a key to the success of scalable oversight (Bowman et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib5){.ltx_ref}),
self-improving for LLM aims to align the LLM to human preference and
values mainly using the supervision from the knowledge inside the LLM
(Zelikman et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib62){.ltx_ref},
[2024](https://arxiv.org/html/2404.12253v2#bib.bib63){.ltx_ref}). One
crucial part of self-improving is how to obtain reliable signal of
critique to distinguish between good responses from the LLM and bad
ones. Initial work (Bai et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib3){.ltx_ref}; Wang
et al., [2022](https://arxiv.org/html/2404.12253v2#bib.bib54){.ltx_ref})
first asks the LLM to generate input queries of diverse tasks and the
corresponding outputs. They then rely on hand-crafted heuristic rules to
filter out redundant or low-quality data pairs (e.g. the query is too
long or too short). Since it is non-trivial to compose effective
heuristic rule, later work (Sun et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib41){.ltx_ref}; Li
et al., [2023](https://arxiv.org/html/2404.12253v2#bib.bib23){.ltx_ref};
Guo et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib17){.ltx_ref})
proposes a few general principles or judging criteria and ask the LLM
itself to evaluate the quality its responses based on these guidance,
hoping that LLMs can automatically designate these principles into each
data point to better guide data filtering. However, this requires LLMs
to have strong abilities to apply these principles for each specific
case and make correct judgements. Different from previous work, we
propose to leverage the supervision from MCTS for LLM self-improvement:
taking the outputs of MCTS to continue train the LLM. This is because
the outputs from MCTS are usually in much better quality then standard
nucleus sampling, and the large gap ensure that the LLM can self
improve.
:::
:::
:::

::: {#S3 .section .ltx_section}
## [3 ]{.ltx_tag .ltx_tag_section}Preliminaries {#preliminaries .ltx_title .ltx_title_section}

::: {#S3.SS1 .section .ltx_subsection}
### [3.1 ]{.ltx_tag .ltx_tag_subsection}Problem Formulation {#problem-formulation .ltx_title .ltx_title_subsection}

::: {#S3.SS1.p1 .ltx_para}
In this paper, we consider a LLM characterized by probability
$p_{\theta}$ and denoted as policy $\pi_{\theta}$. It takes a sequence
${\mathbf{x}} = {\lbrack x_{1},\cdots,x_{n}\rbrack}$ as input, which is
typically referred as prompt, to generate the response
${\mathbf{y}} = {\lbrack y_{1},\cdots,y_{m}\rbrack}$. In the context of
LLMs, each $x_{i}$ and $y_{i}$ represents a token from a pre-defined
vocabulary. The policy $\pi_{\theta}$ operates in an autoregressive
manner, where each token is generated sequentially, relying solely on
the context provided by the previously generated tokens. The policy
therefore constitutes a Markov process in which the conditional
probability distribution
$p_{\theta}{(\left. {\mathbf{y}} \middle| {\mathbf{x}} \right.)}$ can be
decomposed and expressed with the chain rule as
${p_{\theta}{(\left. {\mathbf{y}} \middle| {\mathbf{x}} \right.)}} = {\prod_{i = 1}^{m}{p_{\theta}{(\left. y_{i} \middle| {{\mathbf{x}},{\mathbf{y}}_{< i}} \right.)}}}$.
:::

::: {#S3.SS1.p2 .ltx_para}
With this property, the text generation task can be formulated as an
Markov Decision Process (MDP) problem consisting of
$(\mathcal{S},\mathcal{A},T,R,\gamma)$  in which,
${\mathbf{s}}_{t} \in \mathcal{S}$ represents the context information of
current trajectory, *i.e.,* current status of the generation process,
*e.g.,* a partial response to a prompt; $a_{t} \in \mathcal{A}$ denotes
a single action or sampled token from the vocabulary, leading to a
transition to a new state ${\mathbf{s}}_{t + 1}$, by concatenating
${\mathbf{s}}_{t}$ and $a_{t}$; $r_{t} = {R{({\mathbf{s}}_{t},a_{t})}}$
manifest the evaluation of the generation to the prompt, reflecting the
desirability or preferences of each state-action pair.
:::

::: {#S3.SS1.p3 .ltx_para}
This MDP framework sets the stage for applying Reinforcement Learning
(RL) methods to optimize the policy $\pi_{\mathbf{θ}}$ aiming to
maximize the expected cumulative reward $R$. Base on these setups, we
describe the self-improving problem. Given a LLM $\pi_{\mathbf{θ}}$ and
an initial dataset $\mathcal{D}^{0}$, which consists of $N$
expert-generated prompt-response pairs
$\{{({\mathbf{x}}_{i}^{0},{\mathbf{y}}_{i}^{0})}\mid{i \in {\lbrack N\rbrack}}\}$,
the goal of self-improving is to iteratively refine $\pi_{\theta}$ to
maximize the reward. The refinement process includes learning from
synthesized prompts and corresponding responses. These responses are
obtained using an advanced search algorithm that navigates the space of
possible responses to maximize the expected reward. The detailed process
is described in Algorithm [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#algorithm1 "In A.1 Imagination, Searching, Criticizing and Learning Loop ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
in Appendix. The primary challenges in forming an effective
self-improving loop lie in synthesizing suitable prompts, efficiently
searching over a vast action space, and obtaining precise feedback,
which will be discussed in §[[4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4 "4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::

::: {#S3.SS2 .section .ltx_subsection}
### [3.2 ]{.ltx_tag .ltx_tag_subsection}Monte Carlo Tree Search {#monte-carlo-tree-search .ltx_title .ltx_title_subsection}

::: {#S3.SS2.p1 .ltx_para}
MCTS is a sampling-based search algorithm for policy optimization in
decision-making problems. It would iteratively build a search tree, by
repeating four phases: selection, expansion, evaluation, and
backpropagation. In the selection phase, it would recursively select the
children from the root node by Upper Confidence Bound (UCB)  (Auer
et al., [2002](https://arxiv.org/html/2404.12253v2#bib.bib2){.ltx_ref}),
${UCB{(i)}} = {w_{i} + {C \ast \sqrt{2 \ast {\ln\frac{N_{i}}{n_{i}}}}}}$,
where $n_{i}$ and $N_{i}$ are the visit counts for the node $i$ and its
parent respectively, $C$ represents a hyperparameter balancing
exploration and exploitation, and the $w_{i}$ is the average value of
all descendant nodes of $i$.
:::
:::
:::

::: {#S4 .section .ltx_section}
## [4 ]{.ltx_tag .ltx_tag_section}[AlphaLLM]{#S4.1.1 .ltx_text .ltx_font_smallcaps} {#alphallm .ltx_title .ltx_title_section}

::: {#S4.SS1 .section .ltx_subsection}
### [4.1 ]{.ltx_tag .ltx_tag_subsection}Overview {#overview .ltx_title .ltx_title_subsection}

::: {#S4.SS1.p1 .ltx_para}
The architecture of [AlphaLLM]{#S4.SS1.p1.1.1 .ltx_text
.ltx_font_smallcaps} is depicted in Figure [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref},
comprising three key components. Firstly, the imagination component is
tasked with synthesizing prompts as learning examples. Secondly, an
efficient search component, named $\eta$[Mcts]{#S4.SS1.p1.1.2 .ltx_text
.ltx_font_smallcaps}, is proposed to search high-quality trajectories
for optimizing the policy. Lastly, the search process is guided by
critics specifically designed to provide reliable signals.
:::
:::

::: {#S4.SS2 .section .ltx_subsection}
### [4.2 ]{.ltx_tag .ltx_tag_subsection}Data Synthesizing {#data-synthesizing .ltx_title .ltx_title_subsection}

::: {#S4.SS2.p1 .ltx_para}
Let
$\mathcal{D}^{0} = {\{{({\mathbf{x}}_{i},{\mathbf{y}}_{i})}\mid{i \in {\lbrack N\rbrack}}\}}$
denote the initial dataset consisting of $N$ expert-generated
prompt-response pairs. The data synthesizing process aims to expand this
dataset by generating a set of synthesized prompts
$\mathcal{D}^{1} = {\{{({\mathbf{x}}_{i}^{1},\cdots)}\mid{i \in {\lbrack N\rbrack}}\}}$.
The generation of each synthesized prompt ${\mathbf{x}}_{i}^{1}$ can be
mathematically described as a transformation $g$ applied to one or more
examples from $\mathcal{D}^{0}$,
${\mathbf{x}}_{i}^{1} = {g{({\mathbf{x}}_{i_{1}}^{0},\cdots,{\mathbf{x}}_{i_{m}}^{0},\pi^{0})}}$
where ${\mathbf{x}}_{i_{1}}^{0},\cdots,{\mathbf{x}}_{i_{m}}^{0}$ are
selected examples from $\mathcal{D}^{0}$. The transformation function
$g$ controls the synthesis process, which can be a learnable function,
manually defined heuristic rules, a strong LLM or the policy model
itself $\pi^{0}$ equipped with data synthesis instructions. The data
synthesizing process aims to enrich the diversity and complexity
presented for the training of the policy model. Among various
strategies, such as Self-instruct (Wang et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib54){.ltx_ref}),
Evol-instruct (Xu et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib57){.ltx_ref}), we opt
for a method akin to that described in Yu et al.
([2023](https://arxiv.org/html/2404.12253v2#bib.bib59){.ltx_ref}).
:::
:::

::: {#S4.SS3 .section .ltx_subsection}
### [4.3 ]{.ltx_tag .ltx_tag_subsection}$\eta$[Mcts]{#S4.SS3.2.1 .ltx_text .ltx_font_smallcaps} {#etamcts .ltx_title .ltx_title_subsection}

::: {#S4.SS3.SSS1 .section .ltx_subsubsection}
#### [4.3.1 ]{.ltx_tag .ltx_tag_subsubsection}Option-level MCTS {#option-level-mcts .ltx_title .ltx_title_subsubsection}

  [Search Node]{#S4.T1.9.10.1.1.1 .ltx_text .ltx_font_typewriter style="font-size:80%;"}   [Example]{#S4.T1.9.10.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:80%;"}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              [Termination]{#S4.T1.9.10.1.3.1 .ltx_text .ltx_font_typewriter style="font-size:80%;"}
  ---------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------
  [Token-level]{#S4.T1.1.1.2.1 .ltx_text style="font-size:80%;"}                           $y_{0}\rightarrow y_{1}\rightarrow y_{2}\rightarrow y_{3}\rightarrow y_{5}\rightarrow y_{6}\rightarrow y_{7}\rightarrow y_{8}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [token]{#S4.T1.1.1.3.1 .ltx_text style="font-size:80%;"}
  [Sentence-level]{#S4.T1.4.4.4.1 .ltx_text style="font-size:80%;"}                        $y_{0}y_{1}y_{2}$[ ]{#S4.T1.4.4.3.1 .ltx_text style="font-size:80%;"}[\\keys]{#S4.T1.4.4.3.2 .ltx_ERROR .undefined}[\\return]{#S4.T1.4.4.3.3 .ltx_ERROR .undefined}[ ]{#S4.T1.4.4.3.4 .ltx_text style="font-size:80%;"}$\rightarrow{y_{4}y_{5}y_{6}}$[ ]{#S4.T1.4.4.3.5 .ltx_text style="font-size:80%;"}[\\keys]{#S4.T1.4.4.3.6 .ltx_ERROR .undefined}[\\return]{#S4.T1.4.4.3.7 .ltx_ERROR .undefined}[ ]{#S4.T1.4.4.3.8 .ltx_text style="font-size:80%;"}$\rightarrow{y_{7}y_{8}y_{9}y_{10}}$                                                                                                                                                                                                                                                                                 [new line]{#S4.T1.4.4.5.1 .ltx_text style="font-size:80%;"}
  [Option-level]{#S4.T1.9.9.6.1 .ltx_text style="font-size:80%;"}                          $y_{0}$[ ]{#S4.T1.9.9.5.1 .ltx_text style="font-size:80%;"}$\rightarrow{y_{1}y_{2}}$[ ]{#S4.T1.9.9.5.2 .ltx_text style="font-size:80%;"}[\\keys]{#S4.T1.9.9.5.3 .ltx_ERROR .undefined}[\\return]{#S4.T1.9.9.5.4 .ltx_ERROR .undefined}[ ]{#S4.T1.9.9.5.5 .ltx_text style="font-size:80%;"}$\rightarrow{y_{4}y_{5}y_{6}}$[ ]{#S4.T1.9.9.5.6 .ltx_text style="font-size:80%;"}[\\keys]{#S4.T1.9.9.5.7 .ltx_ERROR .undefined}[\\return]{#S4.T1.9.9.5.8 .ltx_ERROR .undefined}[ ]{#S4.T1.9.9.5.9 .ltx_text style="font-size:80%;"}$y_{7}y_{8}y_{9}$[ ]{#S4.T1.9.9.5.10 .ltx_text style="font-size:80%;"}[\\keys]{#S4.T1.9.9.5.11 .ltx_ERROR .undefined}[\\return]{#S4.T1.9.9.5.12 .ltx_ERROR .undefined}[ ]{#S4.T1.9.9.5.13 .ltx_text style="font-size:80%;"}$\rightarrow y_{10}$   [termination function]{#S4.T1.9.9.7.1 .ltx_text style="font-size:80%;"}

[Table 1: ]{.ltx_tag .ltx_tag_table}Comparative illustration of
token-level, sentence-level, and option-level MCTS search nodes. $y$
denotes a token sampled from the policy model. The arrow $\rightarrow$
represents the transition from one search node to the subsequent node
within the search process.

::: {#S4.SS3.SSS1.p1 .ltx_para}
When applying MCTS to LLMs, it is natural to perform token-level search,
where each token is considered as an action (Liu et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib25){.ltx_ref}).
However, the substantial vocabulary size typical of LLMs presents a
significant challenge *i.e.,* conducting a deep search in such a vast
space becomes increasingly complex as the search space expands
exponentially. To mitigate this, some efforts proposed a sentence-level
search, treating each sentence or step as a search node (Feng et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref}). While
this method reduces the search space, it might compromise the
flexibility and effectiveness of applying MCTS to LLMs, which is
particularly true for tasks where subtle variations in token can
dramatically impact the outcome, or where a more comprehensive search
beyond a sentence is necessary.
:::

::: {#S4.SS3.SSS1.p2 .ltx_para}
Inspired by Sutton et al.
([1999a](https://arxiv.org/html/2404.12253v2#bib.bib43){.ltx_ref});
De Waard et al.
([2016](https://arxiv.org/html/2404.12253v2#bib.bib12){.ltx_ref}), we
use the term option as a search node and propose option-level MCTS where
each option represents a sequence of tokens, which can range from
multiple tokens to several sentences. A comparisons of different levels
search is listed in Table [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.T1 "Table 1 ‣ 4.3.1 Option-level MCTS ‣ 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
Mathematically, an option $o = {\langle\mathcal{I},\pi,\beta\rangle}$,
where $\mathcal{I} \subseteq \mathcal{S}$ is a set of initial states for
the option;
$\pi:{{\mathcal{S} \times \mathcal{A}}\rightarrow{\lbrack 0,1\rbrack}}$
is a policy to generate actions, which in our case is a LLM; and
$\beta:{\mathcal{S}^{+}\rightarrow{\lbrack 0,1\rbrack}}$ is the
termination function. Starting from a state $s_{t}$, we can choose all
the options for which $s_{t} \in \mathcal{I}$. Once an option is chosen,
the policy $\pi$ will generate actions for several steps until the
option terminates according to the termination function $\beta$. The
option-level MCTS consists of stages including selection, expansion,
simulation, and backpropagation. The option-level formulation offers
more flexibility compared to the sentence-level, as a new line can be
treated as a special case of the termination function, as demonstrated
in Table [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.T1 "Table 1 ‣ 4.3.1 Option-level MCTS ‣ 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
Additional detailed steps of the option-level MCTS can be found in
Appendix [[A.2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS2 "A.2 Option-level MCTS ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::

::: {#S4.SS3.SSS2 .section .ltx_subsubsection}
#### [4.3.2 ]{.ltx_tag .ltx_tag_subsubsection}Importance-Based Adaptive Branching {#importance-based-adaptive-branching .ltx_title .ltx_title_subsubsection}

::: {#S4.SS3.SSS2.p1 .ltx_para}
In previous works related to option/sentence level tree search  (Feng
et al., [2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref};
Yao et al.,
[2024](https://arxiv.org/html/2404.12253v2#bib.bib58){.ltx_ref}), it was
a common practice to assume that each node in the tree has the same
predefined width, [i.e.]{#S4.SS3.SSS2.p1.1.1 .ltx_text
.ltx_font_italic}, branching factor. This assumption was due to the fact
that unlike token-level MCTS with a limited action space, the sample
space at the option-level is exceedingly large, with an unlimited number
of token combinations. As a result, it was necessary to set a predefined
maximum width for each node. However, this predefined branching factor
is hard to set, as an improper choice can lead to a search tree that is
either too shallow or too thin, resulting in an inefficient exploration
of the search space.
:::

::: {#S4.SS3.SSS2.p2 .ltx_para}
To quantify the error induced by the branching factor limit, we defined
the branching error $E_{\phi}{(t)}$. For a node $t$ with a branching
factor of $m_{t}$, it aims to use the $m_{t}$ child options
${\mathbf{o}}_{t}^{i} \sim \mathcal{D}_{t}^{children}$ (where
$i \in {\{ 1,\ldots,m_{t}\}}$) to represent all possible options.
Consequently, for a legal option
${\mathbf{o}}_{t}^{j} \sim {\pi{({\mathbf{s}}_{t})}}$ from the option
space, we can calculate the minimal value difference between it and the
$m_{t}$ existing options, which captures the error associated with
representing other possible options using the $m_{t}$ available options.
It can be formulated as
${E_{\phi}{(t)}} = {{\mathbb{E}}_{{\mathbf{o}}_{t}^{j} \sim {\pi{({\mathbf{s}}_{t})}}}{\lbrack{\min_{{\mathbf{o}}_{t}^{i}}{|{{v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{j}\rbrack})}} - {v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}}}|}}\rbrack}}$,
where $v_{\phi}^{\pi}$ is the value function which will be detailed in
§[[4.4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS4 "4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
Here we define the importance of node ${\mathbf{s}}_{t}$ as
${{I{({\mathbf{s}}_{t})}} = {\max_{{\mathbf{o}}_{t}^{i}}{|{{v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}} - {v_{\phi}^{\pi}{({\mathbf{s}}_{t})}}}|}}}.$
For simplicity, we assume that the value of the children nodes are
uniformly distributed (a detailed analysis of the Gaussian distribution
can be found in Appendix [[A.4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS4 "A.4 Importance-Based Adaptive Branching Under Gaussian Distribution ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}).
Under this assumption, we show in Appendix [[A.3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS3 "A.3 Importance-Based Adaptive Branching Under Uniform Distribution ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
that ${{E_{\phi}{(t)}} \leq \frac{I{({\mathbf{s}}_{t})}}{m_{t} - 1}}.$
While $E_{\phi}$ is less than some $\epsilon$, we aim to use a smaller
total number of nodes for efficiency.
:::

::: {#S4.Thmtheorem1 .ltx_theorem .ltx_theorem_theorem}
###### [[Theorem 4.1]{#S4.Thmtheorem1.1.1.1 .ltx_text .ltx_font_bold}]{.ltx_tag .ltx_tag_theorem}[.]{#S4.Thmtheorem1.2.2 .ltx_text .ltx_font_bold} {#theorem-4.1. .ltx_title .ltx_runin .ltx_title_theorem}

::: {#S4.Thmtheorem1.p1 .ltx_para}
[The optimal branching factor $m_{t}$ in a tree search is set such that
$m_{t} - 1$ is proportional to the node importance
$I{(\mathbf{s}_{t})}$, under the condition
$\frac{I{(\mathbf{s}_{t})}}{m_{t} - 1} \leq \epsilon$.
]{#S4.Thmtheorem1.p1.4.4 .ltx_text .ltx_font_italic}Refer to Appendix
[[A.3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS3 "A.3 Importance-Based Adaptive Branching Under Uniform Distribution ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
for the detailed proof.
:::
:::

::: {#S4.SS3.SSS2.p3 .ltx_para}
A similar concept has also been proposed in  Taylor et al.
([2014](https://arxiv.org/html/2404.12253v2#bib.bib46){.ltx_ref});
Clouse
([1996](https://arxiv.org/html/2404.12253v2#bib.bib10){.ltx_ref}).
Intuitively, $I{({\mathbf{s}}_{t})}$ captures the maximum value
deviation from the current state. When this value is small, there is no
need to explore further on this node, as there will not be a significant
difference by rolling out on this node. Conversely, if the value is
large, it is worth trying different children. We set the number of
children allowed for a node $n{({\mathbf{s}}_{t})}$ (after extracting
$1$) to be linear with this importance, using a factor $\alpha$. In
practice, to avoid extreme cases of large variance of
$I{({\mathbf{s}}_{t})}$ in the early stage, we bound the number of
children by depth-dependent constants
$c_{\mathtt{m}\mathtt{i}\mathtt{n}}{(t)}$ and
$c_{\mathtt{m}\mathtt{a}\mathtt{x}}{(t)}$,
${{n{({\mathbf{s}}_{t})}} = {\max\left( {c_{\mathtt{m}\mathtt{i}\mathtt{n}}{(t)}},{\min\left( {{\lfloor{\alpha I{({\mathbf{s}}_{t})}}\rfloor} + 1},{c_{\mathtt{m}\mathtt{a}\mathtt{x}}{(t)}} \right)} \right)}}.$
:::
:::

::: {#S4.SS3.SSS3 .section .ltx_subsubsection}
#### [4.3.3 ]{.ltx_tag .ltx_tag_subsubsection}State Merge {#state-merge .ltx_title .ltx_title_subsubsection}

::: {#S4.SS3.SSS3.p1 .ltx_para}
With $n{({\mathbf{s}}_{t})}$ determined, another issue is that options
under the same node may be very similar, leading to many unnecessary
sub-trees. Since we cannot directly control the
${\mathbf{o}}_{t} \sim {\pi{({\mathbf{s}}_{t})}}$, one strategy to
mitigate this issue is to utilize the concept of move groups, as
discussed in  Van Eyck & Müller
([2012](https://arxiv.org/html/2404.12253v2#bib.bib52){.ltx_ref}). By
merging similar nodes into the same group, we can increase the diversity
among groups, thereby covering a larger problem space with limited
search rollouts and making the search process more efficient.
:::

::: {#S4.SS3.SSS3.p2 .ltx_para}
Here, we adapt the definition of node predicate $p_{vM}$ from  Abel
et al. ([2018](https://arxiv.org/html/2404.12253v2#bib.bib1){.ltx_ref})
and  Fu et al.
([2024](https://arxiv.org/html/2404.12253v2#bib.bib15){.ltx_ref}) to
represent whether two nodes are extremely similar. In practice, each
time we generate a new option from the policy, we use heuristic
functions as $p_{vM}$ to check its similarity with all existing groups.
The heuristic function can either be a faster rule-based measurement
(e.g., edit distance) or a model-based method (e.g., prompting a
language model). Based on this, we decide whether to merge this option
with a previous one or create a new group.
:::
:::

::: {#S4.SS3.SSS4 .section .ltx_subsubsection}
#### [4.3.4 ]{.ltx_tag .ltx_tag_subsubsection}Fast Rollout with Specialized LM {#fast-rollout-with-specialized-lm .ltx_title .ltx_title_subsubsection}

::: {#S4.SS3.SSS4.p1 .ltx_para}
The simulation operation which employs a rollout policy to project
future trajectories from a given state, is crucial for an effective
MCTS. This process significantly improves the efficiency of exploration
and exploitation, and enhances the accuracy of reward
estimation[^1^[[^1^[1]{.ltx_tag .ltx_tag_note}Typically, the closer the
simulation is to the termination state, the more accurate the reward
estimation becomes.]{.ltx_note_content}]{.ltx_note_outer}]{#footnote1
.ltx_note .ltx_role_footnote}. Estimations made at the end of
trajectories tend to have lower bias but higher variance; thus,
simulating multiple possible trajectories yields low-bias, low-variance
estimates, enabling a more informed and effective search process.
Ideally, $\pi_{\theta}$ would serve as the rollout policy, yet its
computational demands render it impractical for the rapid simulations
required by MCTS. To address this challenge, we propose the use of a
smaller, specialized LM as the fast rollout policy
$\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}$. Given a state
${\mathbf{s}}_{t}$, the fast rollout policy
$\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}$ efficiently continues
generation until it reaches a termination condition, denoted as
$\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}{({\mathbf{s}}_{t})}$.
:::
:::
:::

::: {#S4.SS4 .section .ltx_subsection}
### [4.4 ]{.ltx_tag .ltx_tag_subsection}Critic {#critic .ltx_title .ltx_title_subsection}

::: {#S4.SS4.p1 .ltx_para}
In [AlphaLLM]{#S4.SS4.p1.1.1 .ltx_text .ltx_font_smallcaps}, we design
three types of critic models to guide the search process.
:::

::: {#S4.SS4.SSS0.Px1 .section .ltx_paragraph}
##### Value Function {#value-function .ltx_title .ltx_title_paragraph}

::: {#S4.SS4.SSS0.Px1.p1 .ltx_para}
The value function, denoted as $v^{\pi}{({\mathbf{s}})}$, represents the
expected return starting from state $\mathbf{s}$ and following policy
$\pi$ thereafter, given by
${v^{\pi}{({\mathbf{s}})}} = {\mathbb{E}_{\tau \sim \pi}{\lbrack{\left. {R{(\tau)}} \middle| s_{0} \right. = {\mathbf{s}}}\rbrack}}$
where $R{(\tau)}$ represents the discounted return of trajectory $\tau$.
To train a parameterized value function
$v_{\phi}^{\pi}{({\mathbf{s}})}$, given the prompts
$\mathcal{D} = {\{{({\mathbf{x}}_{i},\cdots)}\mid{i \in {\lbrack N\rbrack}}\}}$,
for each prompt ${\mathbf{x}}_{i}$, we generate multiple trajectories
${\mathbf{τ}}_{i}^{j} = {\{{\mathbf{x}}_{i},{\mathbf{o}}_{i1}^{j},{\mathbf{o}}_{i2}^{j},\cdots,{\mathbf{o}}_{iT}^{j}\}}$
by following policy $\pi$ for $J$ times. A final reward $r_{i}^{j}$ is
assigned to indicate whether ${\mathbf{τ}}_{i}^{j}$ aligns with
${\mathbf{y}}_{i}$---for example, rewarding trajectories that contain
correct answers in mathematical tasks or closely follow instructions as
ground truth. We then construct a dataset
$\mathcal{D}_{\mathtt{v}\mathtt{a}\mathtt{l}\mathtt{u}\mathtt{e}} = {\{{({\mathbf{s}}_{it}^{j},v_{it}^{j})}\mid{{i \in {\lbrack N\rbrack}},{{t \in {\lbrack T\rbrack}},{j \in {\lbrack J\rbrack}}}}\}}$
where
${\mathbf{s}}_{it}^{j} = {\lbrack{{\mathbf{x}}_{i} \cdot {\mathbf{o}}_{< {it}}^{j}}\rbrack}$
and $v_{it}^{j} = r_{i}^{j}$. The value function $v_{\phi}^{\pi}$ is
optimized by minimizing the mean squared error:
$\mathcal{L}_{\phi} = {- {{\mathbb{E}}_{{({\mathbf{s}},v)} \sim \mathcal{D}_{\mathtt{v}\mathtt{a}\mathtt{l}\mathtt{u}\mathtt{e}}}{({{v_{\phi}^{\pi}{({\mathbf{s}})}} - v})}^{2}}}$.
Similar to  (Feng et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref}),
$v_{\phi}^{\pi}$ is a LLM with an MLP layer on top to output a scalar on
each token, using the scalar prediction at the last token of each state
as the value.
:::
:::

::: {#S4.SS4.SSS0.Px2 .section .ltx_paragraph}
##### PRM {#prm .ltx_title .ltx_title_paragraph}

::: {#S4.SS4.SSS0.Px2.p1 .ltx_para}
The value function often struggles with credit assignment
problem (Sutton,
[1984](https://arxiv.org/html/2404.12253v2#bib.bib45){.ltx_ref}) and its
learning could be inefficient due to delayed and sparse rewards (Sutton
& Barto,
[2018](https://arxiv.org/html/2404.12253v2#bib.bib42){.ltx_ref}).
Therefore, we propose to incorporate [PRM]{#S4.SS4.SSS0.Px2.p1.11.1
.ltx_text .ltx_font_typewriter} that introduces process
supervision (Lightman et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib24){.ltx_ref}) for
direct option assessment. [PRM]{#S4.SS4.SSS0.Px2.p1.11.2 .ltx_text
.ltx_font_typewriter} generates intrinsic rewards (Chentanez et al.,
[2004](https://arxiv.org/html/2404.12253v2#bib.bib7){.ltx_ref}) to
encourage explorations of advantageous options, effectively mitigating
issues of reward sparsity by providing immediate, action-specific
rewards. Given a state ${\mathbf{s}}_{t}$ and an option
${\mathbf{o}}_{t}$ at time $t$, the [PRM]{#S4.SS4.SSS0.Px2.p1.11.3
.ltx_text .ltx_font_typewriter} aims to predict the immediate reward
$r_{t}^{\text{PRM}}$ that results from taking option ${\mathbf{o}}_{t}$
in state ${\mathbf{s}}_{t}$. Formally, the
[PRM]{#S4.SS4.SSS0.Px2.p1.11.4 .ltx_text .ltx_font_typewriter} is a
function
${R{({\mathbf{s}}_{t},{\mathbf{o}}_{t})}}\rightarrow r_{t}^{\mathtt{P}\mathtt{R}\mathtt{M}}$.
While [PRM]{#S4.SS4.SSS0.Px2.p1.11.5 .ltx_text .ltx_font_typewriter}
ideally requires quality labels for each state  (Uesato et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib50){.ltx_ref}), due to
the high cost and time involved in obtaining these, MC estimation with
prefix sampling (Wang et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib53){.ltx_ref}) is used
as a proxy, which aligns with the objective of the value function.
Instead of adding a MLP layer on top of the policy model for outputting
a scalar reward (Ouyang et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib32){.ltx_ref}), we
formulate [PRM]{#S4.SS4.SSS0.Px2.p1.11.6 .ltx_text .ltx_font_typewriter}
as a text generation task to best leverage LLM's intrinsic knowledge for
assessing the quality of an option. We adapt the dataset constructed for
the value function as
$\mathcal{D}_{\mathtt{P}\mathtt{R}\mathtt{M}} = \left. \{{({\mathbf{s}}_{it},{\mathbf{o}}_{t},r_{t}^{\mathtt{P}\mathtt{R}\mathtt{M}})} \middle| {{i \in {\lbrack N\rbrack}},{t \in {\lbrack T\rbrack}}}\} \right.$
where $r_{t}^{\mathtt{P}\mathtt{R}\mathtt{M}}$ is the textual
description of the reward, *e.g.,* an option can be regarded as good if
$v_{it}$ is larger than certain threshold. To train
[PRM]{#S4.SS4.SSS0.Px2.p1.11.8 .ltx_text .ltx_font_typewriter}, we
initialize it from the policy model $\pi$ and use the following prompt
templates and typical language model loss. The prompt template is shown
in Appendix [[A.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS5 "A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::

::: {#S4.SS4.SSS0.Px3 .section .ltx_paragraph}
##### ORM {#orm .ltx_title .ltx_title_paragraph}

::: {#S4.SS4.SSS0.Px3.p1 .ltx_para}
In additional to the value function and [PRM]{#S4.SS4.SSS0.Px3.p1.5.1
.ltx_text .ltx_font_typewriter}, [ORM]{#S4.SS4.SSS0.Px3.p1.5.2 .ltx_text
.ltx_font_typewriter} is also used to guide MCTS.
[ORM]{#S4.SS4.SSS0.Px3.p1.5.3 .ltx_text .ltx_font_typewriter} is
designed to evaluate options sequences in their entirety, assessing the
extent to which the complete trajectory aligns with the desired end
goal (Uesato et al.,
[2022](https://arxiv.org/html/2404.12253v2#bib.bib50){.ltx_ref};
Lightman et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib24){.ltx_ref}; Wang
et al., [2023](https://arxiv.org/html/2404.12253v2#bib.bib53){.ltx_ref};
Feng et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib14){.ltx_ref}). The
outcome evaluation complements value function and
[PRM]{#S4.SS4.SSS0.Px3.p1.5.4 .ltx_text .ltx_font_typewriter} by
offering a comprehensive assessment of trajectories. Crucially,
[ORM]{#S4.SS4.SSS0.Px3.p1.5.5 .ltx_text .ltx_font_typewriter} plays a
vital role in the simulation stage of MCTS by providing more accurate
signals on the terminal state, which in turn facilitates a more balance
between exploration and exploitation strategies.
[ORM]{#S4.SS4.SSS0.Px3.p1.5.6 .ltx_text .ltx_font_typewriter} is
formulated as a text generation task, similar to
[PRM]{#S4.SS4.SSS0.Px3.p1.5.7 .ltx_text .ltx_font_typewriter}. We
leverage the same dataset for the value function training and construct
$\mathcal{D}_{\mathtt{O}\mathtt{R}\mathtt{M}} = \left. \{{({\mathbf{x}}_{i},{\mathbf{o}}_{1:T}^{i},r_{i}^{\mathtt{O}\mathtt{R}\mathtt{M}})} \middle| {i \in {\lbrack N\rbrack}}\} \right.$,
where each instance includes a initial state or prompt
${\mathbf{x}}_{i}$, a sequence of actions or options
${\mathbf{o}}_{1:T}^{i}$ taken from that state, and a textual reward
$r_{i}^{\mathtt{O}\mathtt{R}\mathtt{M}}$ indicating the sequence's
success or quality. Similarly, [ORM]{#S4.SS4.SSS0.Px3.p1.5.8 .ltx_text
.ltx_font_typewriter} is initialized from the policy model $\pi$ and the
following prompt templates and language model loss are used for
training. The prompt template is shown in Appendix [[A.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS5 "A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.\
:::

::: {#S4.SS4.SSS0.Px3.p2 .ltx_para}
The final score evaluation of a state $\mathbf{s}$ is a weighted sum of
the value function, [PRM]{#S4.SS4.SSS0.Px3.p2.9.1 .ltx_text
.ltx_font_typewriter}, and [ORM]{#S4.SS4.SSS0.Px3.p2.9.2 .ltx_text
.ltx_font_typewriter}:
${s{({\mathbf{s}})}} = {{{\beta_{\text{value}} \cdot v_{\phi}^{\pi}}{({\mathbf{s}})}} + {{\beta_{\text{PRM}} \cdot \text{PRM}}{({\mathbf{s}})}} + {{\beta_{\text{ORM}} \cdot {\mathbb{E}}_{\tau \sim {\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}{({\mathbf{s}})}}}}{\lbrack{\text{ORM}{(\tau)}}\rbrack}}}$,
where
$\tau \sim {\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}{({\mathbf{s}})}}$
represents trajectories starting from $\mathbf{s}$ under
$\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}$, and
$\beta_{\text{value}}$, $\beta_{\text{PRM}}$, $\beta_{\text{ORM}}$ are
hyperparameters. In practice, we found that the value function model has
better precision and calibration, while [PRM]{#S4.SS4.SSS0.Px3.p2.9.3
.ltx_text .ltx_font_typewriter} has superior recall (Appendix
[[A.10]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS10 "A.10 Critic Performance ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}).
Although [ORM]{#S4.SS4.SSS0.Px3.p2.9.4 .ltx_text .ltx_font_typewriter}
with fast rollouts provides low-bias, low-variance estimates, it still
inherits some bias from
$\pi^{\mathtt{f}\mathtt{a}\mathtt{s}\mathtt{t}}$. Thus, combining these
critics yields a stronger evaluation signal.
:::
:::
:::

::: {#S4.SS5 .section .ltx_subsection}
### [4.5 ]{.ltx_tag .ltx_tag_subsection}Policy Self-Improvement {#policy-self-improvement .ltx_title .ltx_title_subsection}

::: {#S4.SS5.p1 .ltx_para}
The policy improvement an iterative process with each iteration
containing two main steps: *data generation* and *policy finetuning*.
:::

::: {#S4.SS5.SSS0.Px1 .section .ltx_paragraph}
##### Data generation {#data-generation .ltx_title .ltx_title_paragraph}

::: {#S4.SS5.SSS0.Px1.p1 .ltx_para}
In this step, we assume to have the current policy $\pi_{\theta_{k}}$
and synthetic prompts
$\mathcal{D}_{k} = {\{{\mathbf{x}}_{1}^{k},\ldots\}}$ at the $k$-th
round, where each ${\mathbf{x}}_{1}^{k}$ represents a question. We
obtain the corresponding training data $\mathcal{D}_{k}$ for policy
$\pi_{\theta_{k}}$ by firstly performing
$\eta$[Mcts]{#S4.SS5.SSS0.Px1.p1.13.1 .ltx_text .ltx_font_smallcaps} on
$\mathcal{D}_{k}$ (§[[4.3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS3 "4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref})
and then sampling a trajectory ${\mathbf{y}}_{i}^{k}$ from the
corresponding tree for each question ${\mathbf{x}}_{i}^{k}$. Here we
choose the trajectory that yield the highest critic score on the leaf
node for each input question. Next, we filter out instances where the
corresponding trajectory is substandard forming
$\mathcal{D}_{k} = \left. \{{({\mathbf{x}}_{i}^{k},{\mathbf{y}}_{i}^{k})} \middle| {{f{({\mathbf{x}}_{i}^{k},{\mathbf{y}}_{i}^{k})}} > \gamma}\} \right.$
where $f$ represents a function for quality scoring, and $\gamma$
indicates a threshold. There can be several ways to implement the
function, and here we simply use the [ORM]{#S4.SS5.SSS0.Px1.p1.13.2
.ltx_text .ltx_font_typewriter} (§[[4.4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS4 "4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}).
:::
:::

::: {#S4.SS5.SSS0.Px2 .section .ltx_paragraph}
##### Policy finetuning {#policy-finetuning .ltx_title .ltx_title_paragraph}

::: {#S4.SS5.SSS0.Px2.p1 .ltx_para}
With the obtained training data $\mathcal{D}_{k}$, we organize the data
into the prompt templates shown in Appendix [[A.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS5 "A.5 Prompt Templates ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
Then the policy $\pi_{\theta_{k}}$ is finetuned using target-loss:
$\mathcal{L}_{\theta_{k}} = {{\mathbb{E}}_{{({\mathbf{x}}_{i}^{k},{\mathbf{y}}_{i}^{k})} \sim \mathcal{D}_{k}}\left\lbrack {{\log\pi_{\theta_{k}}}{(\left. {\mathbf{y}}_{i}^{k} \middle| {\mathbf{x}}_{i}^{k} \right.)}} \right\rbrack}$,
resulting in an updated policy $\pi_{\theta_{k + 1}}$. We leave other
training methods, such as DPO (Rafailov et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib34){.ltx_ref}) or PPO
(Schulman et al.,
[2017](https://arxiv.org/html/2404.12253v2#bib.bib37){.ltx_ref}) in
future work.
:::
:::
:::
:::

::: {#S5 .section .ltx_section}
## [5 ]{.ltx_tag .ltx_tag_section}Experiments {#experiments .ltx_title .ltx_title_section}

::: {#S5.SS1 .section .ltx_subsection}
### [5.1 ]{.ltx_tag .ltx_tag_subsection}Experiment Setups {#experiment-setups .ltx_title .ltx_title_subsection}

::: {#S5.SS1.p1 .ltx_para}
[AlphaLLM]{#S5.SS1.p1.1.1 .ltx_text .ltx_font_smallcaps} is generally
applicable to a wide spectrum tasks. As an early exploration, in this
paper, we conduct experiments on mathematical reasoning problems where
the learning signals are clear to define *i.e.,* , final answer is
correct or wrong. We choose to evaluate on two widely used datasets
GSM8K (Cobbe et al.,
[2021](https://arxiv.org/html/2404.12253v2#bib.bib11){.ltx_ref}) and
MATH (Hendrycks et al.,
[2021](https://arxiv.org/html/2404.12253v2#bib.bib19){.ltx_ref}). For
GSM8K, we utilize the whole test set while for MATH, due to computation
constraints, we utilize a subset following the same procedure
of Lightman et al.
([2023](https://arxiv.org/html/2404.12253v2#bib.bib24){.ltx_ref}). We
evaluate the performance of predicting answers correctly for policy
models. In addition, we calculate the average rollouts, represented by
the number of nodes in the tree, as a measure of computational
efficiency. We compare the performance of [AlphaLLM]{#S5.SS1.p1.1.3
.ltx_text .ltx_font_smallcaps} with a suite of proprietary model,
including OpenAI's GPT-4 and GPT-3.5, Anthropic's Claude-2, as well as
Google's PaLM-2 and the gemini model family. To ensure a fair and
consistent evaluation, we employ CoT as our primary prompting method.
Additionally, we conduct comparisons with strong open-source models,
including Llama-2-70b (Touvron et al.,
[2023a](https://arxiv.org/html/2404.12253v2#bib.bib48){.ltx_ref}) and
WizardMath-70B-V1.0 (Luo et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib28){.ltx_ref}).
:::

::: {#S5.SS1.p2 .ltx_para}
We select Llama-2-70b as the policy model for the GSM8K dataset and
WizardMath-70B-V1.0 for the MATH dataset. To construct the training
dataset for the value function, [PRM]{#S5.SS1.p2.3.1 .ltx_text
.ltx_font_typewriter} and [ORM]{#S5.SS1.p2.3.2 .ltx_text
.ltx_font_typewriter}, we generate 50 trajectories for each prompt and
construct the training target following Section [[4.4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS4 "4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
Both [PRM]{#S5.SS1.p2.3.3 .ltx_text .ltx_font_typewriter} and
[ORM]{#S5.SS1.p2.3.4 .ltx_text .ltx_font_typewriter} are initialized
using the weights from the policy model, while the value function uses a
smaller Llama-2-13b model, as we observed no performance gains from
increasing the value function model size. In the design of
[ORM]{#S5.SS1.p2.3.5 .ltx_text .ltx_font_typewriter}, tool usage is not
incorporated for GSM8K. However, for MATH, we enhance
[ORM]{#S5.SS1.p2.3.6 .ltx_text .ltx_font_typewriter} by incorporating
tools like python sympy to assess the quality of a trajectory, in a
manner similar to that described by Gou et al.
([2023](https://arxiv.org/html/2404.12253v2#bib.bib16){.ltx_ref}). The
training employ a learning rate of 1e-6 and are trained for one epoch.
For the fast rollout policy model, we opt for the Abel-002-7B
model (Chern et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib8){.ltx_ref}) for both
the GSM8K and MATH tasks for its high efficiency and superior
performance. For the MCTS parameters, they are configured at different
scales, as shown in Appendix [[A.6]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS6 "A.6 MCTS Details ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
We set $\beta_{\text{value}}$, $\beta_{\text{PRM}}$, and
$\beta_{\text{ORM}}$ all to 1.0.
:::

::: {#S5.SS1.p3 .ltx_para}
For policy self-improving (§[[4.5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS5 "4.5 Policy Self-Improvement ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}),
we train the policy model up to 3 epochs, setting batch size to 128,
learning rate to $5 \times 10^{- 6}$ and minimal learning rate to
$1 \times 10^{- 6}$. Linear warm-up and decay is used with warm-up
percent to be 10%. We perform early stopping based on a devset held out
from the training instances. For GSM8K experiments, we perform two
rounds of self-improving, synthesizing 6.4k and 7.9k prompts(Yu et al.,
[2023](https://arxiv.org/html/2404.12253v2#bib.bib59){.ltx_ref})
respectively to obtain the corresponding MCTS outputs for training. For
MATH experiments, we only perform one round of self-improving due to
limited computation resources, and 5.9k prompts are synthesized.
:::

::: {#S5.SS1.p4 .ltx_para}
The termination function for options can be either be learned or
rule-based. In practice, for the GSM8K dataset, the termination
condition occurs at the end of each line. This is based on the typical
structure of this dataset, where each line represents a distinct step or
point. For the MATH dataset, due to its complexity and the base model's
tendency to generate many [\\n\\n]{#S5.SS1.p4.1.1 .ltx_text
.ltx_font_typewriter} line breaks with some less meaningful content
between them, termination occurs at the end of a line if a formula
pattern is detected. During inference, if [\\n\\n]{#S5.SS1.p4.1.2
.ltx_text .ltx_font_typewriter} is encountered, we perform a rule-based
check for formula patterns. It terminates if a pattern is found or
continues generating until the next [\\n\\n]{#S5.SS1.p4.1.3 .ltx_text
.ltx_font_typewriter}.
:::
:::

::: {#S5.SS2 .section .ltx_subsection}
### [5.2 ]{.ltx_tag .ltx_tag_subsection}Results {#results .ltx_title .ltx_title_subsection}

  ----------------------------------------------------------------------------------- -------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------ -------------------------------------------------------------------------------- -------------------------------------------------------------------------------- --------------------------------------------------------------------------------- ----------------------------------------------------------------------------------- ----------------------------------------------------------------------------------
  [Model]{#S5.T2.20.21.1.1.1 .ltx_text style="font-size:90%;"}                        [Decoding]{#S5.T2.20.21.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [\#Annotation]{#S5.T2.20.21.1.3.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [RN]{#S5.T2.20.21.1.4.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [FA]{#S5.T2.20.21.1.5.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [SYN]{#S5.T2.20.21.1.6.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [GSM8K]{#S5.T2.20.21.1.7.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [MATH]{#S5.T2.20.21.1.8.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}
  [GPT-3.5 ]{#S5.T2.20.22.2.1.1 .ltx_text style="font-size:90%;"}                     [Sampling]{#S5.T2.20.22.2.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.22.2.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.22.2.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.22.2.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.22.2.6.1 .ltx_text style="font-size:90%;"}                          [80.8]{#S5.T2.20.22.2.7.1 .ltx_text style="font-size:90%;"}                         [35.5]{#S5.T2.20.22.2.8.1 .ltx_text style="font-size:90%;"}
  [GPT-4 ]{#S5.T2.20.23.3.1.1 .ltx_text style="font-size:90%;"}                       [Sampling]{#S5.T2.20.23.3.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.23.3.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.23.3.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.23.3.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.23.3.6.1 .ltx_text style="font-size:90%;"}                          [92.0]{#S5.T2.20.23.3.7.1 .ltx_text style="font-size:90%;"}                         [42.5]{#S5.T2.20.23.3.8.1 .ltx_text style="font-size:90%;"}
  [GPT-4 (PAL) ]{#S5.T2.20.24.4.1.1 .ltx_text style="font-size:90%;"}                 [Sampling]{#S5.T2.20.24.4.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.24.4.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.24.4.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.24.4.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.24.4.6.1 .ltx_text style="font-size:90%;"}                          [94.2]{#S5.T2.20.24.4.7.1 .ltx_text style="font-size:90%;"}                         [51.8]{#S5.T2.20.24.4.8.1 .ltx_text style="font-size:90%;"}
  [Gemini 1.0 Pro ]{#S5.T2.20.25.5.1.1 .ltx_text style="font-size:90%;"}              [Sampling]{#S5.T2.20.25.5.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.25.5.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.25.5.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.25.5.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.25.5.6.1 .ltx_text style="font-size:90%;"}                          [77.9]{#S5.T2.20.25.5.7.1 .ltx_text style="font-size:90%;"}                         [32.6]{#S5.T2.20.25.5.8.1 .ltx_text style="font-size:90%;"}
  [Gemini 1.0 Ultra ]{#S5.T2.20.26.6.1.1 .ltx_text style="font-size:90%;"}            [Sampling]{#S5.T2.20.26.6.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.26.6.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.26.6.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.26.6.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.26.6.6.1 .ltx_text style="font-size:90%;"}                          [88.9]{#S5.T2.20.26.6.7.1 .ltx_text style="font-size:90%;"}                         [53.2]{#S5.T2.20.26.6.8.1 .ltx_text style="font-size:90%;"}
  [Gemini 1.5 Pro ]{#S5.T2.20.27.7.1.1 .ltx_text style="font-size:90%;"}              [Sampling]{#S5.T2.20.27.7.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.27.7.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.27.7.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.27.7.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.27.7.6.1 .ltx_text style="font-size:90%;"}                          [92.5]{#S5.T2.20.27.7.7.1 .ltx_text style="font-size:90%;"}                         [58.5]{#S5.T2.20.27.7.8.1 .ltx_text style="font-size:90%;"}
  [Claude-2 ]{#S5.T2.20.28.8.1.1 .ltx_text style="font-size:90%;"}                    [Sampling]{#S5.T2.20.28.8.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.28.8.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.28.8.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.28.8.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.28.8.6.1 .ltx_text style="font-size:90%;"}                          [85.2]{#S5.T2.20.28.8.7.1 .ltx_text style="font-size:90%;"}                         [32.5]{#S5.T2.20.28.8.8.1 .ltx_text style="font-size:90%;"}
  [PaLM-2 540B ]{#S5.T2.20.29.9.1.1 .ltx_text style="font-size:90%;"}                 [Sampling]{#S5.T2.20.29.9.2.1 .ltx_text style="font-size:90%;"}                        [-]{#S5.T2.20.29.9.3.1 .ltx_text style="font-size:90%;"}                                   [-]{#S5.T2.20.29.9.4.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.29.9.5.1 .ltx_text style="font-size:90%;"}                         [-]{#S5.T2.20.29.9.6.1 .ltx_text style="font-size:90%;"}                          [80.7]{#S5.T2.20.29.9.7.1 .ltx_text style="font-size:90%;"}                         [34.3]{#S5.T2.20.29.9.8.1 .ltx_text style="font-size:90%;"}
  [Llama-2-70b]{#S5.T2.3.3.4.1 .ltx_text style="font-size:90%;"}                      [Greedy]{#S5.T2.3.3.5.1 .ltx_text style="font-size:90%;"}                              [0]{#S5.T2.3.3.6.1 .ltx_text style="font-size:90%;"}                                       $\times$                                                                         $\times$                                                                         $\times$                                                                          [57.8]{#S5.T2.3.3.7.1 .ltx_text style="font-size:90%;"}                             [-]{#S5.T2.3.3.8.1 .ltx_text style="font-size:90%;"}
  [Llama-2-70b SFT]{#S5.T2.6.6.4.1 .ltx_text style="font-size:90%;"}                  [Greedy]{#S5.T2.6.6.5.1 .ltx_text style="font-size:90%;"}                              [7.5k]{#S5.T2.6.6.6.1 .ltx_text style="font-size:90%;"}                                    $✓$                                                                              $✓$                                                                              $\times$                                                                          [69.3]{#S5.T2.6.6.7.1 .ltx_text style="font-size:90%;"}                             [-]{#S5.T2.6.6.8.1 .ltx_text style="font-size:90%;"}
  [WizardMath-70B-V1.0]{#S5.T2.9.9.4.1 .ltx_text style="font-size:90%;"}              [Greedy]{#S5.T2.9.9.5.1 .ltx_text style="font-size:90%;"}                              [96k]{#S5.T2.9.9.6.1 .ltx_text style="font-size:90%;"}                                     $✓$                                                                              $✓$                                                                              $\times$                                                                          [-]{#S5.T2.9.9.7.1 .ltx_text style="font-size:90%;"}                                [20.7]{#S5.T2.9.9.8.1 .ltx_text style="font-size:90%;"}
  [AlphaLLM]{#S5.T2.12.12.4.1 .ltx_text .ltx_font_smallcaps style="font-size:90%;"}   [Greedy]{#S5.T2.12.12.5.1 .ltx_text style="font-size:90%;"}                            [7.5k/7.5k]{#S5.T2.12.12.6.1 .ltx_text style="font-size:90%;"}                             $\times$                                                                         $✓$                                                                              $✓$                                                                               [73.7]{#S5.T2.12.12.7.1 .ltx_text style="font-size:90%;"}                           [23.6]{#S5.T2.12.12.8.1 .ltx_text style="font-size:90%;"}
  [AlphaLLM]{#S5.T2.16.16.5.1 .ltx_text .ltx_font_smallcaps style="font-size:90%;"}   $\eta$[Mcts]{#S5.T2.13.13.1.1 .ltx_text .ltx_font_smallcaps style="font-size:90%;"}    [7.5k/7.5k]{#S5.T2.16.16.6.1 .ltx_text style="font-size:90%;"}                             $\times$                                                                         $✓$                                                                              $\times$                                                                          [88.9]{#S5.T2.16.16.7.1 .ltx_text style="font-size:90%;"}                           [48.7]{#S5.T2.16.16.8.1 .ltx_text style="font-size:90%;"}
  [AlphaLLM]{#S5.T2.20.20.5.1 .ltx_text .ltx_font_smallcaps style="font-size:90%;"}   $\eta$[Mcts]{#S5.T2.17.17.1.1 .ltx_text .ltx_font_smallcaps style="font-size:90%;"}    [7.5k/7.5k]{#S5.T2.20.20.6.1 .ltx_text style="font-size:90%;"}                             $\times$                                                                         $✓$                                                                              $✓$                                                                               [92.0]{#S5.T2.20.20.7.1 .ltx_text style="font-size:90%;"}                           [51.0]{#S5.T2.20.20.8.1 .ltx_text style="font-size:90%;"}
  ----------------------------------------------------------------------------------- -------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------ -------------------------------------------------------------------------------- -------------------------------------------------------------------------------- --------------------------------------------------------------------------------- ----------------------------------------------------------------------------------- ----------------------------------------------------------------------------------

[Table 2: ]{.ltx_tag .ltx_tag_table}Comparison results of
[AlphaLLM]{#S5.T2.37.1 .ltx_text .ltx_font_smallcaps} on the GSM8K and
MATH datasets. [\#Annotation]{#S5.T2.38.2 .ltx_text
.ltx_font_typewriter} indicates the quantity of labeled data employed
for fine-tuning policy or training critic models. The annotation used
for training are noted as [RN]{#S5.T2.39.3 .ltx_text
.ltx_font_typewriter} for rationales and [FA]{#S5.T2.40.4 .ltx_text
.ltx_font_typewriter} for final answers. [SYN]{#S5.T2.41.5 .ltx_text
.ltx_font_typewriter} means models trained on synthetic prompts, where
trajectories were generated using $\eta$[Mcts]{#S5.T2.42.6 .ltx_text
.ltx_font_smallcaps}.

::: {#S5.SS2.p1 .ltx_para}
Table [[2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.T2 "Table 2 ‣ 5.2 Results ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
lists the performance comparisons of various methods on the GSM8K and
MATH datasets. Our findings reveal that [AlphaLLM]{#S5.SS2.p1.2.1
.ltx_text .ltx_font_smallcaps}, based on Llama-2-70B and
WizardMath-70B-V1.0, utilizes only final answer annotations and
continues to improve through training on responses from
$\eta$[Mcts]{#S5.SS2.p1.2.2 .ltx_text .ltx_font_smallcaps}. This
comparison underscores the efficacy and broad applicability of our
imagination-searching-criticizing self-improving framework. Moreover,
when our model is augmented with $\eta$[Mcts]{#S5.SS2.p1.2.3 .ltx_text
.ltx_font_smallcaps} decoding strategy, its performance markedly
improves, achieving scores of 88.9 and 48.7 on the GSM8K and MATH
datasets, respectively. Following two iterations of self-improvement
using synthetic prompts, [AlphaLLM]{#S5.SS2.p1.2.4 .ltx_text
.ltx_font_smallcaps} demonstrates performance comparable to that of
GPT-4. This suggests a viable approach to improving LLMs' capabilities
in complex problem-solving tasks in a self-improving fashion, leveraging
a minimal amount of labeled data. We also analyze the performance of
various search methods in Appendix [[A.8]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS8 "A.8 Search Comparison ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::

::: {#S5.SS3 .section .ltx_subsection}
### [5.3 ]{.ltx_tag .ltx_tag_subsection}Ablation Study {#ablation-study .ltx_title .ltx_title_subsection}

::: {.ltx_flex_figure .ltx_flex_table}
::: {.ltx_flex_cell .ltx_flex_size_2}
  [AB]{#S5.T3.30.30.31.1.1.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [PRM]{#S5.T3.30.30.31.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [FR]{#S5.T3.30.30.31.1.3.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}[-]{#S5.T3.30.30.31.1.3.2 .ltx_text style="font-size:90%;"}[ORM]{#S5.T3.30.30.31.1.3.3 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [SM]{#S5.T3.30.30.31.1.4.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [LG-\#Rollout]{#S5.T3.30.30.31.1.5.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [Acc]{#S5.T3.30.30.31.1.6.1 .ltx_text style="font-size:90%;"}
  ----------------------------------------------------------------------------------- ------------------------------------------------------------------------------------ -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------- ---------------------------------------------------------------
  $\times$                                                                            $\times$                                                                             $\times$                                                                                                                                                                                                                         $\times$                                                                            $\times$                                                                                      [79.5]{#S5.T3.5.5.5.6.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                 $\times$                                                                             $\times$                                                                                                                                                                                                                         $\times$                                                                            $\times$                                                                                      [84.9]{#S5.T3.10.10.10.6.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                 $✓$                                                                                  $\times$                                                                                                                                                                                                                         $\times$                                                                            $\times$                                                                                      [85.9]{#S5.T3.15.15.15.6.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                 $✓$                                                                                  $✓$                                                                                                                                                                                                                              $\times$                                                                            $\times$                                                                                      [86.5]{#S5.T3.20.20.20.6.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                 $✓$                                                                                  $✓$                                                                                                                                                                                                                              $✓$                                                                                 $\times$                                                                                      [87.0]{#S5.T3.25.25.25.6.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                 $✓$                                                                                  $✓$                                                                                                                                                                                                                              $✓$                                                                                 $✓$                                                                                           [88.9]{#S5.T3.30.30.30.6.1 .ltx_text style="font-size:90%;"}

\(a\) Ablation study on GSM8K
:::

::: {.ltx_flex_cell .ltx_flex_size_2}
  [TA]{#S5.T3.36.6.7.1.1.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}[-]{#S5.T3.36.6.7.1.1.2 .ltx_text style="font-size:90%;"}[ORM]{#S5.T3.36.6.7.1.1.3 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [Option]{#S5.T3.36.6.7.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [Acc]{#S5.T3.36.6.7.1.3.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [\#Rollout]{#S5.T3.36.6.7.1.4.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------
  $\times$                                                                                                                                                                                                                   $\times$                                                                              [38.8]{#S5.T3.32.2.2.3.1 .ltx_text style="font-size:90%;"}                         [201]{#S5.T3.32.2.2.4.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                                                                                                                                                        $\times$                                                                              [44.1]{#S5.T3.34.4.4.3.1 .ltx_text style="font-size:90%;"}                         [198]{#S5.T3.34.4.4.4.1 .ltx_text style="font-size:90%;"}
  $✓$                                                                                                                                                                                                                        $✓$                                                                                   [45.4]{#S5.T3.36.6.6.3.1 .ltx_text style="font-size:90%;"}                         [148]{#S5.T3.36.6.6.4.1 .ltx_text style="font-size:90%;"}

\(b\) Ablation study on MATH
:::
:::

[Table 3: ]{.ltx_tag .ltx_tag_table}[(a)]{#S5.T3.52.1 .ltx_text
.ltx_font_bold}: Ablation studies on the GSM8K test set of various
components of $\eta$[Mcts]{#S5.T3.53.2 .ltx_text .ltx_font_smallcaps},
including adaptive branching, [PRM]{#S5.T3.54.3 .ltx_text
.ltx_font_typewriter}, fast-rollout with [ORM]{#S5.T3.55.4 .ltx_text
.ltx_font_typewriter}, state merge, and large number of rollouts.
[(b)]{#S5.T3.56.5 .ltx_text .ltx_font_bold}: Ablation studies of the
impacts of tool-augmented [ORM]{#S5.T3.57.6 .ltx_text
.ltx_font_typewriter} and option-level formulation on MATH.

::: {#S5.SS3.p1 .ltx_para}
We assess the effectiveness of each component in
[AlphaLLM]{#S5.SS3.p1.1.1 .ltx_text .ltx_font_smallcaps} and report the
results on GSM8K in Table [[3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.T3 "Table 3 ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}(a).
Vanilla MCTS, configured with only the value function and a fixed number
of children per node, achieves an accuracy of 79.5%. This serves as a
reference point for evaluating the incremental benefits introduced by
each additional component. The use of adaptive branching increae the
accuracy to 84.9%. The addition of [PRM]{#S5.SS3.p1.1.2 .ltx_text
.ltx_font_typewriter} improves the accuracy modestly to 85.9%, showing
the effectivenss of process supervision for searching. A more
significant improvement is observed with the introduction of
[ORM]{#S5.SS3.p1.1.3 .ltx_text .ltx_font_typewriter} with fast rollout,
which boosts the accuracy to 86.5%. Integrating state merging results in
a further increase in accuracy, reaching 87.0%. Finally the combined of
increasing the number of rollouts with the other components yields the
best performance on this task.
:::

::: {#S5.SS3.p2 .ltx_para}
Table [[3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.T3 "Table 3 ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}(b)
presents the ablation study of option formulation and the tool-augmented
critic on the MATH dataset. Our proposed $\eta$[Mcts]{#S5.SS3.p2.1.1
.ltx_text .ltx_font_smallcaps} achieves an accuracy of 45.4 with 148
rollouts. When options are excluded, reverting to essentially
sentence-level MCTS, the performance decreases to 44.1 with a noticeable
increase in the number of rollouts to 198. This demonstrates that option
formulation introduces enhanced flexibility to MCTS, enabling better
performance with fewer search efforts. Furthermore, the most significant
decrease in performance is observed when only intrinsic knowledge is
utilized for [ORM]{#S5.SS3.p2.1.2 .ltx_text .ltx_font_typewriter}, which
drops to an accuracy of 38.8. This suggests that the absence of an
external tool critically impedes the [ORM]{#S5.SS3.p2.1.3 .ltx_text
.ltx_font_typewriter}'s capability to effectively assess challenging
math problems.
:::

![[Figure 2: ]{.ltx_tag .ltx_tag_figure}Empirical analysis on GSM8K of
different self-improving data collection methods and number of
iterations. Models are evaluated with greedy decoding,
$\eta$[Mcts]{#S5.F2.4.1 .ltx_text .ltx_font_smallcaps} with small
\#rollout and large
\#rollout.](model_self_improving_n_rounds_results_v2.png){#S5.F2.g1
.ltx_graphics .ltx_centering .ltx_img_landscape width="538"
height="278"}

::: {#S5.SS3.p3 .ltx_para}
Figure [[2]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.F2 "Figure 2 ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
depicts a comparative results on GSM8K of two rounds of self-improving
trained on trajectories collected using reranking and
$\eta$[Mcts]{#S5.SS3.p3.7.1 .ltx_text .ltx_font_smallcaps}. We report
the performance of greedy decoding, $\eta$[Mcts]{#S5.SS3.p3.7.2
.ltx_text .ltx_font_smallcaps} with a relatively small number of
rollouts (50-60), and $\eta$[Mcts]{#S5.SS3.p3.7.3 .ltx_text
.ltx_font_smallcaps} with a larger number of rollouts (200-300) for each
model. We observe that 1) Models trained on the trajectories from
reranking or $\eta$[Mcts]{#S5.SS3.p3.7.4 .ltx_text .ltx_font_smallcaps}
outperform the initial policy by a significant margin. In addition, the
performance can be iteratively improved with training suggesting that
self-improving has the potential to achieve continual performance gain.
2) While both reranking and $\eta$[Mcts]{#S5.SS3.p3.7.5 .ltx_text
.ltx_font_smallcaps} can generate high-quality trajectories for
self-improving , $\eta$[Mcts]{#S5.SS3.p3.7.6 .ltx_text
.ltx_font_smallcaps} is performant with high efficiency and better
accuracy. Models trained on trajectories generated by it not only exceed
the performance of those trained on reranked trajectories but also, when
decoded with $\eta$[Mcts]{#S5.SS3.p3.7.7 .ltx_text .ltx_font_smallcaps},
demonstrate on par performance with GPT-4, revealing that
[AlphaLLM]{#S5.SS3.p3.7.8 .ltx_text .ltx_font_smallcaps} is an effective
self-improving framework.
:::

::: {.ltx_flex_figure .ltx_flex_table}
::: {.ltx_flex_cell .ltx_flex_size_2}
  [Method]{#S5.T4.7.7.8.1.1.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}                                                                            [Threshold]{#S5.T4.7.7.8.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}   [Acc]{#S5.T4.7.7.8.1.3.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}
  ------------------------------------------------------------------------------------ ------------------------------------------------------------------------ --------------------------------------------------------------------------------------- ---------------------------------------------------------------------------------
                                                                                       [Edit distance]{#S5.T4.2.2.2.4.1 .ltx_text style="font-size:90%;"}       $20$                                                                                    $86.8$
                                                                                       [Edit distance]{#S5.T4.4.4.4.4.1 .ltx_text style="font-size:90%;"}       $50$                                                                                    $87.0$
                                                                                       [Cosine Similarity]{#S5.T4.6.6.6.4.1 .ltx_text style="font-size:90%;"}   $0.7$                                                                                   $86.3$
                                                                                       [Model-based]{#S5.T4.7.7.7.3.1 .ltx_text style="font-size:90%;"}         [N/A]{#S5.T4.7.7.7.4.1 .ltx_text style="font-size:90%;"}                                $86.7$

\(a\) Ablation on the choice of state merge functions.
:::

::: {.ltx_flex_cell .ltx_flex_size_2}
  ------------------------------------------------------------------------------------------ ----- ----------------------------------------------------------------------------------
  [\#Trajetory]{#S5.T4.13.6.7.1.1.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}         [Acc]{#S5.T4.13.6.7.1.2.1 .ltx_text .ltx_font_typewriter style="font-size:90%;"}
                                                                                             $1$   $85.9$
                                                                                             $4$   $86.5$
                                                                                             $8$   $86.7$
  ------------------------------------------------------------------------------------------ ----- ----------------------------------------------------------------------------------

\(b\) Ablation on the number of trajectories.
:::
:::

[Table 4: ]{.ltx_tag .ltx_tag_table}[(a)]{#S5.T4.19.1 .ltx_text
.ltx_font_bold}: Ablation studies on the choice of heuristic/model-based
functions in state merge on GSM8K with base Llama2-70b. The model used
in the model-based state merge is Llama-2-70b-chat. [(b)]{#S5.T4.20.2
.ltx_text .ltx_font_bold}: Ablation studies of the number of rollout
trajectories in fast-rollout estimation on GSM8K with base Llama2-70b.

::: {#S5.SS3.p4 .ltx_para}
We further analyze the impact of different hyperparameters and design
choices for each component. Table [[4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.T4 "Table 4 ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}(a)
shows that varying heuristic functions (with hyperparameters) for state
merge has limited impact on performance. Table [[4]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S5.T4 "Table 4 ‣ 5.3 Ablation Study ‣ 5 Experiments ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}(b)
shows that, as the number of fast-rollouts increases, there is a
corresponding improvement in performance. This is due to the reduction
in the variance of the estimates. We used $n = 4$ in our experiments for
better trade-off between performance and efficiency. Additional
ablations on the choice of fast-rollout models, are provided in Appendix
[[A.7]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.SS7 "A.7 Additional Ablations ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::
:::

::: {#S6 .section .ltx_section}
## [6 ]{.ltx_tag .ltx_tag_section}Conclusion {#conclusion .ltx_title .ltx_title_section}

::: {#S6.p1 .ltx_para}
In this paper, we introduce [AlphaLLM]{#S6.p1.1.1 .ltx_text
.ltx_font_smallcaps}, an imagination-searching-criticizing framework
designed for the self-improvement of LLMs without the necessity of
additional annotations. At the heart of it is the integration of MCTS
with LLMs. To tackle the inherent challenges associated with this
integration, including data scarcity, the vastness of search spaces, and
the subjective nature of feedback in language tasks, we introduce a data
synthesizer for strategic prompt synthesis, an optimized MCTS tailored
for efficient search in language tasks, and a trio of critic models to
provide precise feedback. Our experimental findings on mathematical
reasoning tasks reveal that [AlphaLLM]{#S6.p1.1.2 .ltx_text
.ltx_font_smallcaps} significantly boosts the performance of LLMs
without requiring extra data annotations. Moreover, when decoded with
$\eta$[Mcts]{#S6.p1.1.3 .ltx_text .ltx_font_smallcaps},
[AlphaLLM]{#S6.p1.1.4 .ltx_text .ltx_font_smallcaps} performs comparably
to GPT-4, highlighting the potential for self-improvement in LLMs.
:::

::: {.ltx_pagination .ltx_role_newpage}
:::
:::

::: {#bib .section .ltx_bibliography}
## References {#references .ltx_title .ltx_title_bibliography}

-   [[Abel et al. (2018)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    David Abel, Dilip Arumugam, Lucas Lehnert, and Michael Littman.
    ]{.ltx_bibblock} [State abstractions for lifelong reinforcement
    learning. ]{.ltx_bibblock} [In *International Conference on Machine
    Learning*, pp.  10--19. PMLR, 2018. ]{.ltx_bibblock}]{#bib.bib1}
-   [[Auer et al. (2002)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Peter Auer, Nicolo Cesa-Bianchi, and Paul Fischer. ]{.ltx_bibblock}
    [Finite-time analysis of the multiarmed bandit problem.
    ]{.ltx_bibblock} [*Machine learning*, 47:235--256, 2002.
    ]{.ltx_bibblock}]{#bib.bib2}
-   [[Bai et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson
    Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini,
    Cameron McKinnon, et al. ]{.ltx_bibblock} [Constitutional ai:
    Harmlessness from ai feedback. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2212.08073*, 2022. ]{.ltx_bibblock}]{#bib.bib3}
-   [[Besta et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal
    Podstawski, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert
    Niewiadomski, Piotr Nyczyk, et al. ]{.ltx_bibblock} [Graph of
    thoughts: Solving elaborate problems with large language models.
    ]{.ltx_bibblock} [In *Proceedings of the AAAI Conference on
    Artificial Intelligence*, pp.  17682--17690, 2024.
    ]{.ltx_bibblock}]{#bib.bib4}
-   [[Bowman et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Samuel R Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig
    Pettit, Scott Heiner, Kamilė Lukošiūtė, Amanda Askell, Andy Jones,
    Anna Chen, et al. ]{.ltx_bibblock} [Measuring progress on scalable
    oversight for large language models. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2211.03540*, 2022. ]{.ltx_bibblock}]{#bib.bib5}
-   [[Chen et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu.
    ]{.ltx_bibblock} [Self-play fine-tuning converts weak language
    models to strong language models. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2401.01335*, 2024. ]{.ltx_bibblock}]{#bib.bib6}
-   [[Chentanez et al. (2004)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Nuttapong Chentanez, Andrew Barto, and Satinder
    Singh. ]{.ltx_bibblock} [Intrinsically motivated reinforcement
    learning. ]{.ltx_bibblock} [*Advances in neural information
    processing systems*, 17, 2004. ]{.ltx_bibblock}]{#bib.bib7}
-   [[Chern et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Ethan Chern, Haoyang Zou, Xuefeng Li, Jiewen Hu, Kehua Feng, Junlong
    Li, and Pengfei Liu. ]{.ltx_bibblock} [Generative ai for math: Abel.
    ]{.ltx_bibblock} [<https://github.com/GAIR-NLP/abel>, 2023.
    ]{.ltx_bibblock}]{#bib.bib8}
-   [[Chung et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay,
    William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha
    Brahma, et al. ]{.ltx_bibblock} [Scaling instruction-finetuned
    language models. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2210.11416*, 2022. ]{.ltx_bibblock}]{#bib.bib9}
-   [[Clouse (1996)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Jeffery Allen Clouse. ]{.ltx_bibblock} [*On integrating apprentice
    learning and reinforcement learning*. ]{.ltx_bibblock} [University
    of Massachusetts Amherst, 1996. ]{.ltx_bibblock}]{#bib.bib10}
-   [[Cobbe et al. (2021)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo
    Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton,
    Reiichiro Nakano, et al. ]{.ltx_bibblock} [Training verifiers to
    solve math word problems. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2110.14168*, 2021. ]{.ltx_bibblock}]{#bib.bib11}
-   [[De Waard et al. (2016)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Maarten De Waard, Diederik M Roijers, and
    Sander CJ Bakkes. ]{.ltx_bibblock} [Monte carlo tree search with
    options for general video game playing. ]{.ltx_bibblock} [In *2016
    IEEE Conference on Computational Intelligence and Games (CIG)*, pp. 
    1--8. IEEE, 2016. ]{.ltx_bibblock}]{#bib.bib12}
-   [[Ding et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei
    Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang.
    ]{.ltx_bibblock} [Everything of thoughts: Defying the law of penrose
    triangle for thought generation. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2311.04254*, 2023. ]{.ltx_bibblock}]{#bib.bib13}
-   [[Feng et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Xidong Feng, Ziyu Wan, Muning Wen, Ying Wen, Weinan Zhang, and Jun
    Wang. ]{.ltx_bibblock} [Alphazero-like tree-search can guide large
    language model decoding and training. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2309.17179*, 2023. ]{.ltx_bibblock}]{#bib.bib14}
-   [[Fu et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Yangqing Fu, Ming Sun, Buqing Nie, and Yue Gao. ]{.ltx_bibblock}
    [Accelerating monte carlo tree search with probability tree state
    abstraction. ]{.ltx_bibblock} [*Advances in Neural Information
    Processing Systems*, 36, 2024. ]{.ltx_bibblock}]{#bib.bib15}
-   [[Gou et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Minlie Huang, Nan
    Duan, Weizhu Chen, et al. ]{.ltx_bibblock} [Tora: A tool-integrated
    reasoning agent for mathematical problem solving. ]{.ltx_bibblock}
    [*arXiv preprint arXiv:2309.17452*, 2023.
    ]{.ltx_bibblock}]{#bib.bib16}
-   [[Guo et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Hongyi Guo, Yuanshun Yao, Wei Shen, Jiaheng Wei, Xiaoying Zhang,
    Zhaoran Wang, and Yang Liu. ]{.ltx_bibblock} [Human-instruction-free
    llm self-alignment with limited samples. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2401.06785*, 2024. ]{.ltx_bibblock}]{#bib.bib17}
-   [[Hao et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and
    Zhiting Hu. ]{.ltx_bibblock} [Reasoning with language model is
    planning with world model. ]{.ltx_bibblock} [In *Proceedings of the
    2023 Conference on Empirical Methods in Natural Language
    Processing*, pp.  8154--8173, 2023. ]{.ltx_bibblock}]{#bib.bib18}
-   [[Hendrycks et al. (2021)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Dan Hendrycks, Collin Burns, Saurav Kadavath,
    Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
    Steinhardt. ]{.ltx_bibblock} [Measuring mathematical problem solving
    with the math dataset, 2021. ]{.ltx_bibblock}]{#bib.bib19}
-   [[Hong et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Ruixin Hong, Hongming Zhang, Xinyu Pang, Dong Yu, and Changshui
    Zhang. ]{.ltx_bibblock} [A closer look at the self-verification
    abilities of large language models in logical reasoning.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2311.07954*, 2023.
    ]{.ltx_bibblock}]{#bib.bib20}
-   [[Huang et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng,
    Adams Wei Yu, Xinying Song, and Denny Zhou. ]{.ltx_bibblock} [Large
    language models cannot self-correct reasoning yet. ]{.ltx_bibblock}
    [*arXiv preprint arXiv:2310.01798*, 2023.
    ]{.ltx_bibblock}]{#bib.bib21}
-   [[Lewkowycz et al. (2022)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Aitor Lewkowycz, Anders Andreassen, David Dohan,
    Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem
    Anil, Imanol Schlag, Theo Gutman-Solo, et al. ]{.ltx_bibblock}
    [Solving quantitative reasoning problems with language models.
    ]{.ltx_bibblock} [*Advances in Neural Information Processing
    Systems*, 35:3843--3857, 2022. ]{.ltx_bibblock}]{#bib.bib22}
-   [[Li et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer
    Levy, Jason Weston, and Mike Lewis. ]{.ltx_bibblock} [Self-alignment
    with instruction backtranslation. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2308.06259*, 2023. ]{.ltx_bibblock}]{#bib.bib23}
-   [[Lightman et al. (2023)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Hunter Lightman, Vineet Kosaraju, Yura Burda,
    Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman,
    Ilya Sutskever, and Karl Cobbe. ]{.ltx_bibblock} [Let's verify step
    by step. ]{.ltx_bibblock} [*arXiv preprint arXiv:2305.20050*, 2023.
    ]{.ltx_bibblock}]{#bib.bib24}
-   [[Liu et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Jiacheng Liu, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh
    Hajishirzi, and Asli Celikyilmaz. ]{.ltx_bibblock} [Making ppo even
    better: Value-guided monte-carlo tree search decoding.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2309.15028*, 2023.
    ]{.ltx_bibblock}]{#bib.bib25}
-   [[Long (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [ Jieyi
    Long. ]{.ltx_bibblock} [Large language model guided tree-of-thought.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2305.08291*, 2023.
    ]{.ltx_bibblock}]{#bib.bib26}
-   [[Luketina et al. (2019)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Jelena Luketina, Nantas Nardelli, Gregory
    Farquhar, Jakob N. Foerster, Jacob Andreas, Edward Grefenstette,
    Shimon Whiteson, and Tim Rocktäschel. ]{.ltx_bibblock} [A survey of
    reinforcement learning informed by natural language.
    ]{.ltx_bibblock} [*ArXiv*, abs/1906.03926, 2019. ]{.ltx_bibblock}
    [URL <https://api.semanticscholar.org/CorpusID:182952502>.
    ]{.ltx_bibblock}]{#bib.bib27}
-   [[Luo et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang
    Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang.
    ]{.ltx_bibblock} [Wizardmath: Empowering mathematical reasoning for
    large language models via reinforced evol-instruct. ]{.ltx_bibblock}
    [*arXiv preprint arXiv:2308.09583*, 2023.
    ]{.ltx_bibblock}]{#bib.bib28}
-   [[Madaan et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu
    Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye,
    Yiming Yang, et al. ]{.ltx_bibblock} [Self-refine: Iterative
    refinement with self-feedback. ]{.ltx_bibblock} [*Advances in Neural
    Information Processing Systems*, 36, 2024.
    ]{.ltx_bibblock}]{#bib.bib29}
-   [[Nye et al. (2021)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk
    Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor
    Lewkowycz, Maarten Bosma, David Luan, et al. ]{.ltx_bibblock} [Show
    your work: Scratchpads for intermediate computation with language
    models. ]{.ltx_bibblock} [*arXiv preprint arXiv:2112.00114*, 2021.
    ]{.ltx_bibblock}]{#bib.bib30}
-   [[OpenAI (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    R OpenAI. ]{.ltx_bibblock} [Gpt-4 technical report. ]{.ltx_bibblock}
    [*arXiv*, pp.  2303--08774, 2023. ]{.ltx_bibblock}]{#bib.bib31}
-   [[Ouyang et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll
    Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina
    Slama, Alex Ray, et al. ]{.ltx_bibblock} [Training language models
    to follow instructions with human feedback. ]{.ltx_bibblock}
    [*Advances in Neural Information Processing Systems*,
    35:27730--27744, 2022. ]{.ltx_bibblock}]{#bib.bib32}
-   [[Peng et al. (2017)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Baolin Peng, Xiujun Li, Lihong Li, Jianfeng Gao, Asli Celikyilmaz,
    Sungjin Lee, and Kam-Fai Wong. ]{.ltx_bibblock} [Composite
    task-completion dialogue policy learning via hierarchical deep
    reinforcement learning. ]{.ltx_bibblock} [In *Proceedings of the
    2017 Conference on Empirical Methods in Natural Language
    Processing*. Association for Computational Linguistics, 2017.
    ]{.ltx_bibblock}]{#bib.bib33}
-   [[Rafailov et al. (2023)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Rafael Rafailov, Archit Sharma, Eric Mitchell,
    Stefano Ermon, Christopher D Manning, and Chelsea Finn.
    ]{.ltx_bibblock} [Direct preference optimization: Your language
    model is secretly a reward model. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2305.18290*, 2023. ]{.ltx_bibblock}]{#bib.bib34}
-   [[Ramamurthy et al. (2022)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Rajkumar Ramamurthy, Prithviraj Ammanabrolu,
    Kianté Brantley, Jack Hessel, Rafet Sifa, Christian Bauckhage,
    Hannaneh Hajishirzi, and Yejin Choi. ]{.ltx_bibblock} [Is
    reinforcement learning (not) for natural language processing?:
    Benchmarks, baselines, and building blocks for natural language
    policy optimization. ]{.ltx_bibblock} [*ArXiv*, abs/2210.01241,
    2022. ]{.ltx_bibblock} [URL
    <https://api.semanticscholar.org/CorpusID:252693405>.
    ]{.ltx_bibblock}]{#bib.bib35}
-   [[Saunders et al. (2022)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ William Saunders, Catherine Yeh, Jeff Wu, Steven
    Bills, Long Ouyang, Jonathan Ward, and Jan Leike. ]{.ltx_bibblock}
    [Self-critiquing models for assisting human evaluators.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2206.05802*, 2022.
    ]{.ltx_bibblock}]{#bib.bib36}
-   [[Schulman et al. (2017)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ John Schulman, Filip Wolski, Prafulla Dhariwal,
    Alec Radford, and Oleg Klimov. ]{.ltx_bibblock} [Proximal policy
    optimization algorithms. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:1707.06347*, 2017. ]{.ltx_bibblock}]{#bib.bib37}
-   [[Silver et al. (2016)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent
    Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis
    Antonoglou, Veda Panneershelvam, Marc Lanctot, et al.
    ]{.ltx_bibblock} [Mastering the game of go with deep neural networks
    and tree search. ]{.ltx_bibblock} [*nature*, 529(7587):484--489,
    2016. ]{.ltx_bibblock}]{#bib.bib38}
-   [[Silver et al. (2017)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis
    Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre,
    Dharshan Kumaran, Thore Graepel, et al. ]{.ltx_bibblock} [Mastering
    chess and shogi by self-play with a general reinforcement learning
    algorithm. ]{.ltx_bibblock} [*arXiv preprint arXiv:1712.01815*,
    2017. ]{.ltx_bibblock}]{#bib.bib39}
-   [[Stechly et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Kaya Stechly, Karthik Valmeekam, and Subbarao Kambhampati.
    ]{.ltx_bibblock} [On the self-verification limitations of large
    language models on reasoning and planning tasks. ]{.ltx_bibblock}
    [*arXiv preprint arXiv:2402.08115*, 2024.
    ]{.ltx_bibblock}]{#bib.bib40}
-   [[Sun et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang
    Chen, David Cox, Yiming Yang, and Chuang Gan. ]{.ltx_bibblock}
    [Principle-driven self-alignment of language models from scratch
    with minimal human supervision. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2305.03047*, 2023. ]{.ltx_bibblock}]{#bib.bib41}
-   [[Sutton & Barto (2018)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Richard S Sutton and Andrew G Barto. ]{.ltx_bibblock}
    [*Reinforcement learning: An introduction*. ]{.ltx_bibblock} [MIT
    press, 2018. ]{.ltx_bibblock}]{#bib.bib42}
-   [[Sutton et al. (1999a)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Richard S. Sutton, Doina Precup, and Satinder Singh.
    ]{.ltx_bibblock} [Between mdps and semi-mdps: A framework for
    temporal abstraction in reinforcement learning. ]{.ltx_bibblock}
    [*Artificial Intelligence*, 112(1):181--211, 1999a. ]{.ltx_bibblock}
    [ISSN 0004-3702. ]{.ltx_bibblock} [doi:
    [https://doi.org/10.1016/S0004-3702(99)00052-1]{.ltx_ref .ltx_nolink
    .ltx_Url .ltx_ref_self}. ]{.ltx_bibblock} [URL
    <https://www.sciencedirect.com/science/article/pii/S0004370299000521>.
    ]{.ltx_bibblock}]{#bib.bib43}
-   [[Sutton et al. (1999b)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Richard S Sutton, Doina Precup, and Satinder Singh.
    ]{.ltx_bibblock} [Between mdps and semi-mdps: A framework for
    temporal abstraction in reinforcement learning. ]{.ltx_bibblock}
    [*Artificial intelligence*, 112(1-2):181--211, 1999b.
    ]{.ltx_bibblock}]{#bib.bib44}
-   [[Sutton (1984)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Richard Stuart Sutton. ]{.ltx_bibblock} [*Temporal credit assignment
    in reinforcement learning*. ]{.ltx_bibblock} [University of
    Massachusetts Amherst, 1984. ]{.ltx_bibblock}]{#bib.bib45}
-   [[Taylor et al. (2014)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Matthew E Taylor, Nicholas Carboni, Anestis Fachantidis, Ioannis
    Vlahavas, and Lisa Torrey. ]{.ltx_bibblock} [Reinforcement learning
    agents providing advice in complex video games. ]{.ltx_bibblock}
    [*Connection Science*, 26(1):45--63, 2014.
    ]{.ltx_bibblock}]{#bib.bib46}
-   [[Team et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu,
    Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
    Andrew M Dai, Anja Hauth, et al. ]{.ltx_bibblock} [Gemini: a family
    of highly capable multimodal models. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2312.11805*, 2023. ]{.ltx_bibblock}]{#bib.bib47}
-   [[Touvron et al. (2023a)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Hugo Touvron, Louis Martin, Kevin Stone, Peter
    Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
    Batra, Prajjwal Bhargava, Shruti Bhosale, et al. ]{.ltx_bibblock}
    [Llama 2: Open foundation and fine-tuned chat models.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2307.09288*, 2023a.
    ]{.ltx_bibblock}]{#bib.bib48}
-   [[Touvron et al. (2023b)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Hugo Touvron, Louis Martin, Kevin Stone, Peter
    Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya
    Batra, Prajjwal Bhargava, Shruti Bhosale, et al. ]{.ltx_bibblock}
    [Llama 2: Open foundation and fine-tuned chat models.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2307.09288*, 2023b.
    ]{.ltx_bibblock}]{#bib.bib49}
-   [[Uesato et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem}
    [ Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah
    Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina
    Higgins. ]{.ltx_bibblock} [Solving math word problems with
    process-and outcome-based feedback. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2211.14275*, 2022. ]{.ltx_bibblock}]{#bib.bib50}
-   [[Valmeekam et al. (2022)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Karthik Valmeekam, Alberto Olmo, Sarath
    Sreedharan, and Subbarao Kambhampati. ]{.ltx_bibblock} [Large
    language models still can't plan (a benchmark for llms on planning
    and reasoning about change). ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2206.10498*, 2022. ]{.ltx_bibblock}]{#bib.bib51}
-   [[Van Eyck & Müller (2012)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Gabriel Van Eyck and Martin Müller.
    ]{.ltx_bibblock} [Revisiting move groups in monte-carlo tree search.
    ]{.ltx_bibblock} [In *Advances in Computer Games: 13th International
    Conference, ACG 2011, Tilburg, The Netherlands, November 20-22,
    2011, Revised Selected Papers 13*, pp.  13--23. Springer, 2012.
    ]{.ltx_bibblock}]{#bib.bib52}
-   [[Wang et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Peiyi Wang, Lei Li, Zhihong Shao, RX Xu, Damai Dai, Yifei Li, Deli
    Chen, Y Wu, and Zhifang Sui. ]{.ltx_bibblock} [Math-shepherd: Verify
    and reinforce llms step-by-step without human annotations.
    ]{.ltx_bibblock} [*CoRR, abs/2312.08935*, 2023.
    ]{.ltx_bibblock}]{#bib.bib53}
-   [[Wang et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A
    Smith, Daniel Khashabi, and Hannaneh Hajishirzi. ]{.ltx_bibblock}
    [Self-instruct: Aligning language model with self generated
    instructions. ]{.ltx_bibblock} [*arXiv preprint arXiv:2212.10560*,
    2022. ]{.ltx_bibblock}]{#bib.bib54}
-   [[Wei et al. (2022)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia,
    Ed Chi, Quoc V Le, Denny Zhou, et al. ]{.ltx_bibblock}
    [Chain-of-thought prompting elicits reasoning in large language
    models. ]{.ltx_bibblock} [*Advances in neural information processing
    systems*, 35:24824--24837, 2022. ]{.ltx_bibblock}]{#bib.bib55}
-   [[Xie et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min-Yen Kan,
    Junxian He, and Michael Xie. ]{.ltx_bibblock} [Self-evaluation
    guided beam search for reasoning. ]{.ltx_bibblock} [*Advances in
    Neural Information Processing Systems*, 36, 2024.
    ]{.ltx_bibblock}]{#bib.bib56}
-   [[Xu et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng,
    Chongyang Tao, and Daxin Jiang. ]{.ltx_bibblock} [Wizardlm:
    Empowering large language models to follow complex instructions.
    ]{.ltx_bibblock} [*arXiv preprint arXiv:2304.12244*, 2023.
    ]{.ltx_bibblock}]{#bib.bib57}
-   [[Yao et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths,
    Yuan Cao, and Karthik Narasimhan. ]{.ltx_bibblock} [Tree of
    thoughts: Deliberate problem solving with large language models.
    ]{.ltx_bibblock} [*Advances in Neural Information Processing
    Systems*, 36, 2024. ]{.ltx_bibblock}]{#bib.bib58}
-   [[Yu et al. (2023)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu,
    Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu.
    ]{.ltx_bibblock} [Metamath: Bootstrap your own mathematical
    questions for large language models. ]{.ltx_bibblock} [*arXiv
    preprint arXiv:2309.12284*, 2023. ]{.ltx_bibblock}]{#bib.bib59}
-   [[Yuan et al. (2024a)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia
    Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, et al.
    ]{.ltx_bibblock} [Advancing llm reasoning generalists with
    preference trees. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2404.02078*, 2024a. ]{.ltx_bibblock}]{#bib.bib60}
-   [[Yuan et al. (2024b)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar
    Sukhbaatar, Jing Xu, and Jason Weston. ]{.ltx_bibblock}
    [Self-rewarding language models. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2401.10020*, 2024b. ]{.ltx_bibblock}]{#bib.bib61}
-   [[Zelikman et al. (2022)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah
    Goodman. ]{.ltx_bibblock} [Star: Bootstrapping reasoning with
    reasoning. ]{.ltx_bibblock} [*Advances in Neural Information
    Processing Systems*, 35:15476--15488, 2022.
    ]{.ltx_bibblock}]{#bib.bib62}
-   [[Zelikman et al. (2024)]{.ltx_tag .ltx_role_refnum
    .ltx_tag_bibitem} [ Eric Zelikman, Georges Harik, Yijia Shao, Varuna
    Jayasiri, Nick Haber, and Noah D Goodman. ]{.ltx_bibblock}
    [Quiet-star: Language models can teach themselves to think before
    speaking. ]{.ltx_bibblock} [*arXiv preprint arXiv:2403.09629*, 2024.
    ]{.ltx_bibblock}]{#bib.bib63}
-   [[Zhu et al. (2024)]{.ltx_tag .ltx_role_refnum .ltx_tag_bibitem} [
    Tinghui Zhu, Kai Zhang, Jian Xie, and Yu Su. ]{.ltx_bibblock}
    [Deductive beam search: Decoding deducible rationale for
    chain-of-thought reasoning. ]{.ltx_bibblock} [*arXiv preprint
    arXiv:2401.17686*, 2024. ]{.ltx_bibblock}]{#bib.bib64}
:::

::: {.ltx_pagination .ltx_role_newpage}
:::

::: {#A1 .section .ltx_appendix}
## [Appendix A ]{.ltx_tag .ltx_tag_appendix}Appendix {#appendix-a-appendix .ltx_title .ltx_title_appendix}

::: {#A1.SS1 .section .ltx_subsection}
### [A.1 ]{.ltx_tag .ltx_tag_subsection}Imagination, Searching, Criticizing and Learning Loop {#a.1-imagination-searching-criticizing-and-learning-loop .ltx_title .ltx_title_subsection}

::: {#algorithm1.11 .ltx_listing .ltx_lst_numbers_left .ltx_listing}
::: {#algorithm1.4.4 .ltx_listingline}
[Input]{#algorithm1.4.4.1 .ltx_text .ltx_font_bold} Initial dataset
$\mathcal{D}^{0} = {\{{({\mathbf{x}}_{i}^{0},{\mathbf{y}}_{i}^{0})}\mid{i \in {\lbrack N\rbrack}}\}}$,
policy model $\pi_{\theta}^{0}$, reward model $R$, number of
self-improving training loop $K$
:::

::: {#algorithm1.5.5 .ltx_listingline}
[Output]{#algorithm1.5.5.1 .ltx_text .ltx_font_bold} $\theta^{k}$
:::

::: {#algorithm1.6.6 .ltx_listingline}
[for]{#algorithm1.6.6.2 .ltx_text
.ltx_font_bold} *$k\leftarrow{1,\ldots,K}$* [do]{#algorithm1.6.6.3
.ltx_text .ltx_font_bold}
:::

::: {#algorithm1.7.7 .ltx_listingline}
  [ ]{.ltx_rule
style="width:1px;height:100%;background:black;display:inline-block;"}   
Generate synthetic prompts
${\lbrack{\mathbf{x}}^{k}\rbrack} = {\text{SYN}{(\pi_{\theta}^{k - 1},\mathcal{D}^{k - 1})}}$
:::

::: {#algorithm1.9.9 .ltx_listingline}
  [ ]{.ltx_rule
style="width:1px;height:100%;background:black;display:inline-block;"}   Collect
trajectories with search algorithm, *e.g.,* MCTS guided by $R$.
${\lbrack{\hat{\mathbf{y}}}^{k}\rbrack} = {\text{MCTS}{(\pi_{\theta}^{k - 1},{\lbrack{\mathbf{x}}^{k}\rbrack})}}$
:::

::: {#algorithm1.10.10 .ltx_listingline}
  [ ]{.ltx_rule
style="width:1px;height:100%;background:black;display:inline-block;"}   Construct
dataset
$\mathcal{D}^{k} = {\{{({\mathbf{x}}^{k},{\hat{\mathbf{y}}}^{k})}\}}$
:::

::: {#algorithm1.11.11 .ltx_listingline}
  [ ]{.ltx_rule
style="width:1px;height:100%;background:black;display:inline-block;"}   Update
policy
$\theta^{k} = {{\arg{\min_{\theta}L}}{(\pi_{\theta}^{k - 1},\mathcal{D}^{k})}}$
:::

::: {#algorithm1.11.12 .ltx_listingline}
end for
:::

::: {#algorithm1.11.13 .ltx_listingline}
:::
:::

[[Algorithm 1]{#algorithm1.13.1.1 .ltx_text .ltx_font_bold} ]{.ltx_tag
.ltx_tag_float}LLM self-improving loop

::: {#A1.SS1.p1 .ltx_para}
The algorithm is shown in Algorithm [[1]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#algorithm1 "In A.1 Imagination, Searching, Criticizing and Learning Loop ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::
:::

::: {#A1.SS2 .section .ltx_subsection}
### [A.2 ]{.ltx_tag .ltx_tag_subsection}Option-level MCTS {#a.2-option-level-mcts .ltx_title .ltx_title_subsection}

![[Figure 3: ]{.ltx_tag .ltx_tag_figure}An overview of the four
operations of $\eta$[Mcts]{#A1.F3.6.1 .ltx_text .ltx_font_smallcaps}. A
node is selected, expanded, simulated with fast rollout policy until a
terminal node is reached, then the signals from value function,
[PRM]{#A1.F3.7.2 .ltx_text .ltx_font_typewriter} and [ORM]{#A1.F3.8.3
.ltx_text .ltx_font_typewriter} are backpropagated.](x2.png){#A1.F3.g1
.ltx_graphics .ltx_centering .ltx_img_landscape width="830"
height="286"}

::: {#A1.SS2.p1 .ltx_para}
As illustrated in Figure [[3]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.F3 "Figure 3 ‣ A.2 Option-level MCTS ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref},
option-level MCTS consists of the following operations:

-   [[•]{.ltx_tag .ltx_tag_item}]{#A1.I1.i1}
    ::: {#A1.I1.i1.p1 .ltx_para}
    [Selection]{#A1.I1.i1.p1.1.1 .ltx_text .ltx_font_bold} Starting from
    the root node, we iteratively select the child node based on
    Equation [LABEL:eqs:ucb]{.ltx_ref .ltx_missing_label .ltx_ref_self}.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#A1.I1.i2}
    ::: {#A1.I1.i2.p1 .ltx_para}
    [Expansion]{#A1.I1.i2.p1.2.1 .ltx_text .ltx_font_bold} Once an
    expandable leaf node is selected, a new node is generated by
    starting with the previous state of the parent node as the initial
    option state. The option is then sampled using the policy $\pi$, and
    its completion is determined by the termination function $\beta$.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#A1.I1.i3}
    ::: {#A1.I1.i3.p1 .ltx_para}
    [Simulation]{#A1.I1.i3.p1.1.1 .ltx_text .ltx_font_bold} The scaled
    reward of the newly expanded node, as well as some simulated future
    trajectories are evaluated using the feedback functions, which is
    discussed in §[[4.4]{.ltx_text
    .ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.SS4 "4.4 Critic ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
    :::
-   [[•]{.ltx_tag .ltx_tag_item}]{#A1.I1.i4}
    ::: {#A1.I1.i4.p1 .ltx_para}
    [Backpropagation]{#A1.I1.i4.p1.1.1 .ltx_text .ltx_font_bold} The
    average value of the newly generated node and all its ancestors is
    updated using the scaled reward from the evaluation step. Meanwhile,
    the visit counts for these nodes are also increased by one.
    :::
:::
:::

::: {#A1.SS3 .section .ltx_subsection}
### [A.3 ]{.ltx_tag .ltx_tag_subsection}Importance-Based Adaptive Branching Under Uniform Distribution {#a.3-importance-based-adaptive-branching-under-uniform-distribution .ltx_title .ltx_title_subsection}

::: {#A1.SS3.p1 .ltx_para}
Let
$V = {\{{v_{\phi}^{\pi}{({\mathbf{s}}_{t},{\mathbf{o}}_{t}^{1})}},{v_{\phi}^{\pi}{({\mathbf{s}}_{t},{\mathbf{o}}_{t}^{2})}},\ldots,{v_{\phi}^{\pi}{({\mathbf{s}}_{t},{\mathbf{o}}_{t}^{m_{t}})}}\}}$
be a set of $m_{t}$ values that are uniformly distributed. If the
maximum and minimum values from $V$ are $v_{\max}$ and $v_{\min}$, the
average gap between two consecutive values is given by
$\frac{v_{\max} - v_{\min}}{m_{t} - 1}$. The upper bound of expected
minimum distances from a new value $v_{\text{new}}$ to any value from
$V$ is achieved when $v_{\text{new}}$ is consistently positioned at the
midpoint between two consecutive values, and it is given by
$\frac{v_{\max} - v_{\min}}{2{({m_{t} - 1})}}$.
:::

::: {#A1.SS3.p2 .ltx_para}
Since ${v_{\max} - v_{\min}} = {2I{({\mathbf{s}}_{t})}}$ for a uniform
distribution, we can conclude that
${E_{\phi}{(t)}} \leq \frac{I{({\mathbf{s}}_{t})}}{m_{t} - 1}$.
:::

::: {#A1.Thmtheorem1 .ltx_theorem .ltx_theorem_theorem}
###### [[Theorem ]{#A1.Thmtheorem1.1.1.1 .ltx_text .ltx_font_bold}[[4.1]{.ltx_text .ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#S4.Thmtheorem1 "Theorem 4.1. ‣ 4.3.2 Importance-Based Adaptive Branching ‣ 4.3 𝜂Mcts ‣ 4 AlphaLLM ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref .ltx_font_bold}]{.ltx_tag .ltx_tag_theorem}[.]{#A1.Thmtheorem1.2.2 .ltx_text .ltx_font_bold} {#theorem-4.1.-1 .ltx_title .ltx_runin .ltx_title_theorem}

::: {#A1.Thmtheorem1.p1 .ltx_para}
[The optimal branching factor $m_{t}$ in a tree search is set such that
$m_{t} - 1$ is proportional to the node importance
$I{(\mathbf{s}_{t})}$, under the condition
$\frac{I{(\mathbf{s}_{t})}}{m_{t} - 1} \leq \epsilon$.]{#A1.Thmtheorem1.p1.4.4
.ltx_text .ltx_font_italic}
:::
:::

::: {#A1.SS3.12 .ltx_proof}
###### Proof. {#proof. .ltx_title .ltx_runin .ltx_font_italic .ltx_title_proof}

::: {#A1.SS3.1.p1 .ltx_para}
We can have the optimization problem as:

  -- ------------------------------------------------------------ --------------------------------------------------------- --
     [minimize:]{#A1.Ex1.2.1.1.1 .ltx_text .ltx_markedasmath}     $\sum m_{t}$                                              
     [subject to:]{#A1.Ex2.2.1.1.1 .ltx_text .ltx_markedasmath}   $\frac{I{({\mathbf{s}}_{t})}}{m_{t} - 1} \leq \epsilon$   
  -- ------------------------------------------------------------ --------------------------------------------------------- --
:::

::: {#A1.SS3.2.p2 .ltx_para}
Introduce the Lagrange multiplier $\lambda_{t}$ for each constraint:
:::

::: {#A1.SS3.3.p3 .ltx_para}
  -- ----------------------------------------------------------------------------------------------------------------------------------------- --
     $${L{(m_{t},\lambda_{t})}} = {{\sum m_{t}} + {\sum{\lambda_{t}\left( {{\epsilon{({m_{t} - 1})}} - {I{({\mathbf{s}}_{t})}}} \right)}}}$$   
  -- ----------------------------------------------------------------------------------------------------------------------------------------- --
:::

::: {#A1.SS3.4.p4 .ltx_para}
Now, let's find the gradient of the Lagrangian with respect to $m_{t}$
and $\lambda_{t}$ and set them to zero:
:::

::: {#A1.SS3.5.p5 .ltx_para}
  -- ------------------------- --------------------------------------------------------------- --
     $\nabla_{m_{t}}L$         $= {1 + {\epsilon\lambda_{t}}} = 0$                             
     $\nabla_{\lambda_{t}}L$   $= {{\epsilon{({m_{t} - 1})}} - {I{({\mathbf{s}}_{t})}}} = 0$   
  -- ------------------------- --------------------------------------------------------------- --
:::

::: {#A1.SS3.6.p6 .ltx_para}
From the first equation, we get:
:::

::: {#A1.SS3.7.p7 .ltx_para}
  -- ------------------------------------------ --
     $$\lambda_{t} = {- \frac{1}{\epsilon}}$$   
  -- ------------------------------------------ --
:::

::: {#A1.SS3.8.p8 .ltx_para}
Substitute this value of $\lambda_{t}$ into the second equation:
:::

::: {#A1.SS3.9.p9 .ltx_para}
  -- --------------------------------------------------------------- --
     $${{\epsilon{({m_{t} - 1})}} - {I{({\mathbf{s}}_{t})}}} = 0$$   
  -- --------------------------------------------------------------- --
:::

::: {#A1.SS3.10.p10 .ltx_para}
Solving for $m_{t}$, we get:
:::

::: {#A1.SS3.11.p11 .ltx_para}
  -- ---------------------------------------------------------- --
     $$m_{t} = {\frac{I{({\mathbf{s}}_{t})}}{\epsilon} + 1}$$   
  -- ---------------------------------------------------------- --
:::

::: {#A1.SS3.12.p12 .ltx_para}
Thus, $m_{t} - 1$ is proportional to the node importance
$I{({\mathbf{s}}_{t})}$. ∎
:::
:::
:::

::: {#A1.SS4 .section .ltx_subsection}
### [A.4 ]{.ltx_tag .ltx_tag_subsection}Importance-Based Adaptive Branching Under Gaussian Distribution {#a.4-importance-based-adaptive-branching-under-gaussian-distribution .ltx_title .ltx_title_subsection}

::: {#A1.SS4.p1 .ltx_para}
If we assume that
$v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{j}\rbrack})}$
and
$v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}$
are independent and identically distributed Gaussian random variables:

  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --
     $${{v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{j}\rbrack})}},{v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}}} \sim {\mathcal{N}{(\mu,\sigma^{2})}}$$   
  -- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ --

The difference
$D_{ij} = {{v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{j}\rbrack})}} - {v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}}}$
will follow a normal distribution with:

  -- -------------------------------------------------- --
     $$D_{ij} \sim {\mathcal{N}{(0,{2\sigma^{2}})}}$$   
  -- -------------------------------------------------- --

To find the expected minimum absolute difference between
$v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{j}\rbrack})}$
and the closest
$v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}$,
we need to consider the distribution of the minimum of $m_{t}$ Gaussian
differences.
:::

::: {#A1.SS4.p2 .ltx_para}
The expected minimum value of $m_{t}$ absolute differences can be
approximated using properties of order statistics for Gaussian
distributions.
:::

::: {#A1.SS4.p3 .ltx_para}
For a set of $m_{t}$ independent normal random variables with variance
$2\sigma^{2}$, the expected minimum absolute difference,
${\mathbb{E}}{\lbrack{\min_{i}{|D_{ij}|}}\rbrack}$, can be approximated
by:

  -- ----------------------------------------------------------------- --
     $${E_{\phi}{(t)}} \approx \frac{\sigma\sqrt{2}}{\sqrt{m_{t}}}$$   
  -- ----------------------------------------------------------------- --

This approximation arises from the fact that the expected minimum value
of the absolute deviations of normally distributed random variables
scales with the inverse of the square root of the number of samples.
:::

::: {#A1.SS4.p4 .ltx_para}
Then, assume the range of the $m_{t}$ samples are
$R_{m} = max{(v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})} - min{(v_{\phi}^{\pi}{({\lbrack{\mathbf{s}}_{t},{\mathbf{o}}_{t}^{i}\rbrack})}}}$,
the the expected range ${\mathbb{E}}{\lbrack R_{m}\rbrack}$ of $m_{t}$
samples from a normal distribution can be approximated using properties
of extreme values of Gaussian distributions. The range $R_{m}$ can be
approximated as:

  -- --------------------------------------------------------- --
     $$R_{m} \approx {\sigma{({z_{0.9995} - z_{0.0005}})}}$$   
  -- --------------------------------------------------------- --

where $z_{p}$ is the p-th percentile of the standard normal
distribution. It can converge to

  -- -------------------------------------------------------------------------------------------------------------------- --
     $$R_{m} \approx {\sigma\sqrt{2{\ln{(m_{t})}}}\left( {2 - \frac{\ln{({\ln{(m_{t})}})}}{4{\ln{(m_{t})}}}} \right)}$$   
  -- -------------------------------------------------------------------------------------------------------------------- --

For simplicity, we can approximate the range using the primary term,
which captures the dominant behavior:

  -- -------------------------------------------------- --
     $$R_{m} \approx {\sigma\sqrt{2{\ln{(m_{t})}}}}$$   
  -- -------------------------------------------------- --

Then we have

  -- ------------------------------------------------------------------------------------------------- --
     $${E_{\phi}{(t)}} \approx {\frac{\sqrt{2}}{\sqrt{m_{t}}}\frac{R_{m}}{\sqrt{2{\ln{(m_{t})}}}}}$$   
  -- ------------------------------------------------------------------------------------------------- --

Knowing that for all distributions,

  -- -------------------------------------------------- --
     $${I{({\mathbf{s}}_{t})}} \geq \frac{R_{m}}{2}$$   
  -- -------------------------------------------------- --

We have

  -- ------------------------------------------------------------------------ --
     $${E_{\phi}{(t)}} \leq \frac{I{(s_{t})}}{\sqrt{m_{t}{\ln{(m_{t})}}}}$$   
  -- ------------------------------------------------------------------------ --

Then to find the optimal $m_{t}$, the optimization problem is

  -- ------------------------------------------------------------- --------------------------------------------------------------- --
     [minimize:]{#A1.Ex18.2.1.1.1 .ltx_text .ltx_markedasmath}     $\sum m_{t}$                                                    
     [subject to:]{#A1.Ex19.2.1.1.1 .ltx_text .ltx_markedasmath}   $\frac{I{(s_{t})}}{\sqrt{m_{t}{\ln{(m_{t})}}}} \leq \epsilon$   
  -- ------------------------------------------------------------- --------------------------------------------------------------- --
:::

::: {#A1.SS4.p5 .ltx_para}
To solve this optimization problem, we can first rewrite the constraint
in terms of $m_{t}$.

  -- -------------------------------------------------------------------- --
     $${m_{t}{\ln{(m_{t})}}} \geq \frac{I^{2}{(s_{t})}}{\epsilon^{2}}$$   
  -- -------------------------------------------------------------------- --
:::

::: {#A1.SS4.p6 .ltx_para}
Now, let's define a new function ${g{(m_{t})}} = {m_{t}{\ln{(m_{t})}}}$.
We want to find the minimum $m_{t}$ such that
${g{(m_{t})}} \geq \frac{I^{2}{(s_{t})}}{\epsilon^{2}}$. To do this, we
can find the derivative of $g{(m_{t})}$ and set it to zero to find the
critical points.
:::

::: {#A1.SS4.p7 .ltx_para}
  -- ------------------------------------------------------------------------------------------------ --
     $${g^{\prime}{(m_{t})}} = {\frac{d}{dm_{t}}{({m_{t}{\ln{(m_{t})}}})}} = {{\ln{(m_{t})}} + 1}$$   
  -- ------------------------------------------------------------------------------------------------ --
:::

::: {#A1.SS4.p8 .ltx_para}
Setting the derivative to zero:
:::

::: {#A1.SS4.p9 .ltx_para}
  -- ---------------------------- --
     $${\ln{(m_{t})}} = {- 1}$$   
  -- ---------------------------- --
:::

::: {#A1.SS4.p10 .ltx_para}
  -- --------------------- --
     $$m_{t} = e^{- 1}$$   
  -- --------------------- --
:::

::: {#A1.SS4.p11 .ltx_para}
However, this critical point corresponds to a minimum of the function
$g{(m_{t})}$, and we are interested in the minimum $m_{t}$ that
satisfies the constraint
${g{(m_{t})}} \geq \frac{I^{2}{(s_{t})}}{\epsilon^{2}}$. Since the
function $g{(m_{t})}$ is increasing for $m_{t} > e^{- 1}$, we can find
the minimum $m_{t}$ by setting
${g{(m_{t})}} = \frac{I^{2}{(s_{t})}}{\epsilon^{2}}$ and solving for
$m_{t}$:
:::

::: {#A1.SS4.p12 .ltx_para}
  -- ----------------------------------------------------------------- --
     $${m_{t}{\ln{(m_{t})}}} = \frac{I^{2}{(s_{t})}}{\epsilon^{2}}$$   
  -- ----------------------------------------------------------------- --

This can not be solved directly, but we can still observe that there is
a positive correlation between $m_{t}$ and $I{({\mathbf{s}}_{t})}$.
:::
:::

::: {#A1.SS5 .section .ltx_subsection}
### [A.5 ]{.ltx_tag .ltx_tag_subsection}Prompt Templates {#a.5-prompt-templates .ltx_title .ltx_title_subsection}

::: {#A1.SS5.SSS1 .section .ltx_subsubsection}
#### [A.5.1 ]{.ltx_tag .ltx_tag_subsubsection}PRM {#a.5.1-prm .ltx_title .ltx_title_subsubsection}

::: {#A1.SS5.SSS1.p1 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iOTEuMjEiIGlkPSJBMS5TUzUuU1NTMS5wMS5waWMxIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB3aWR0aD0iNjAwIj48ZyBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsOTEuMjEpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBmaWxsPSIjNDA0MDQwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggZD0iTSAwIDUuOTEgTCAwIDg1LjMgQyAwIDg4LjU3IDIuNjQgOTEuMjEgNS45MSA5MS4yMSBMIDU5NC4wOSA5MS4yMSBDIDU5Ny4zNiA5MS4yMSA2MDAgODguNTcgNjAwIDg1LjMgTCA2MDAgNS45MSBDIDYwMCAyLjY0IDU5Ny4zNiAwIDU5NC4wOSAwIEwgNS45MSAwIEMgMi42NCAwIDAgMi42NCAwIDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGw9IiNGMkYyRjIiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBkPSJNIDEuOTcgNS45MSBMIDEuOTcgODUuMyBDIDEuOTcgODcuNDggMy43MyA4OS4yNCA1LjkxIDg5LjI0IEwgNTk0LjA5IDg5LjI0IEMgNTk2LjI3IDg5LjI0IDU5OC4wMyA4Ny40OCA1OTguMDMgODUuMyBMIDU5OC4wMyA1LjkxIEMgNTk4LjAzIDMuNzMgNTk2LjI3IDEuOTcgNTk0LjA5IDEuOTcgTCA1LjkxIDEuOTcgQyAzLjczIDEuOTcgMS45NyAzLjczIDEuOTcgNS45MSBaIiBzdHlsZT0ic3Ryb2tlOm5vbmUiPjwvcGF0aD48L2c+PGcgZmlsbC1vcGFjaXR5PSIxLjAiIHRyYW5zZm9ybT0ibWF0cml4KDEuMCAwLjAgMC4wIDEuMCAyMS42NSAxMy43OCkiPjxmb3JlaWdub2JqZWN0IGNvbG9yPSIjMDAwMDAwIiBoZWlnaHQ9IjYzLjY1IiBvdmVyZmxvdz0idmlzaWJsZSIgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgMCAxNi42KSIgd2lkdGg9IjU1Ni42OSI+CjxzcGFuIGNsYXNzPSJsdHhfaW5saW5lLWJsb2NrIGx0eF9taW5pcGFnZSBsdHhfYWxpZ25fYm90dG9tIiBpZD0iQTEuU1M1LlNTUzEucDEucGljMS4xLjEuMS4xLjEiIHN0eWxlPSJ3aWR0aDo0MDIuM3B0OyI+CjxzcGFuIGNsYXNzPSJsdHhfcCIgaWQ9IkExLlNTNS5TU1MxLnAxLnBpYzEuMS4xLjEuMS4xLjEiPiMjI1lvdSBhcmUgZ2l2ZW4gYSBtYXRoIHByb2JsZW0sIGZvbGxvd2VkIGJ5IGEgc3RlcC1ieS1zdGVwIHJlYXNvbmluZyBwcm9jZXNzLiBZb3VyIHRhc2sgaXMgdG8gcmVhZCB0aGUgcHJvYmxlbSBjYXJlZnVsbHksIHVuZGVyc3RhbmQgdGhlIHNvbHZpbmcgc3RlcHMsIGFuZCBjaGVjayB0aGUgY29ycmVjdG5lc3Mgb2YgdGhlIGxhc3QgcmVhc29uaW5nIHN0ZXAuIE91dHB1dCDigJlUcnVl4oCZIGlmIHRoZSBsYXN0IHN0ZXAgaXMgY29ycmVjdCwgYW5kIOKAmUZhbHNl4oCZIG90aGVyd2lzZS5cblxuIyMjIFN0YXRlXG57PHNwYW4gY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIGlkPSJBMS5TUzUuU1NTMS5wMS5waWMxLjEuMS4xLjEuMS4xLjEiPnN0YXRlPC9zcGFuPn1cblxuIyMjQWN0aW9uXG57PHNwYW4gY2xhc3M9Imx0eF90ZXh0IGx0eF9mb250X3R5cGV3cml0ZXIiIGlkPSJBMS5TUzUuU1NTMS5wMS5waWMxLjEuMS4xLjEuMS4xLjIiPm9wdGlvbjwvc3Bhbj59XG5cbiMjI0Fzc2Vzc21lbnRcbns8c3BhbiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgaWQ9IkExLlNTNS5TU1MxLnAxLnBpYzEuMS4xLjEuMS4xLjEuMyI+dGV4dHVhbCByZXdhcmQ8L3NwYW4+fTwvc3Bhbj4KPC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==){#A1.SS5.SSS1.p1.pic1
.ltx_picture}
:::
:::

::: {#A1.SS5.SSS2 .section .ltx_subsubsection}
#### [A.5.2 ]{.ltx_tag .ltx_tag_subsubsection}ORM {#a.5.2-orm .ltx_title .ltx_title_subsubsection}

::: {#A1.SS5.SSS2.p1 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iMTQxLjAyIiBpZD0iQTEuU1M1LlNTUzIucDEucGljMSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgd2lkdGg9IjYwMCI+PGcgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDE0MS4wMikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIGZpbGw9IiM0MDQwNDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBkPSJNIDAgNS45MSBMIDAgMTM1LjEyIEMgMCAxMzguMzggMi42NCAxNDEuMDIgNS45MSAxNDEuMDIgTCA1OTQuMDkgMTQxLjAyIEMgNTk3LjM2IDE0MS4wMiA2MDAgMTM4LjM4IDYwMCAxMzUuMTIgTCA2MDAgNS45MSBDIDYwMCAyLjY0IDU5Ny4zNiAwIDU5NC4wOSAwIEwgNS45MSAwIEMgMi42NCAwIDAgMi42NCAwIDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGw9IiNGMkYyRjIiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBkPSJNIDEuOTcgNS45MSBMIDEuOTcgMTM1LjEyIEMgMS45NyAxMzcuMjkgMy43MyAxMzkuMDUgNS45MSAxMzkuMDUgTCA1OTQuMDkgMTM5LjA1IEMgNTk2LjI3IDEzOS4wNSA1OTguMDMgMTM3LjI5IDU5OC4wMyAxMzUuMTIgTCA1OTguMDMgNS45MSBDIDU5OC4wMyAzLjczIDU5Ni4yNyAxLjk3IDU5NC4wOSAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuNjUgMTMuNzgpIj48Zm9yZWlnbm9iamVjdCBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIxMTMuNDYiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDE2LjYpIiB3aWR0aD0iNTU2LjY5Ij4KPHNwYW4gY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIGlkPSJBMS5TUzUuU1NTMi5wMS5waWMxLjEuMS4xLjEuMSIgc3R5bGU9IndpZHRoOjQwMi4zcHQ7Ij4KPHNwYW4gY2xhc3M9Imx0eF9wIiBpZD0iQTEuU1M1LlNTUzIucDEucGljMS4xLjEuMS4xLjEuMSI+IyMjQXNzZXNzIGEgc29sdXRpb24gaW5jbHVkaW5nIGZpbmFsIGFuc3dlciB0byBhIGdpdmVuIG1hdGggcHJvYmxlbSBieSBmb2xsb3dpbmcgYmVsb3cgc3RlcHMuXG4tIEV2YWx1YXRlIHRoZSBtZXRob2QgdXNlZCBmb3Igc29sdmluZyB0aGUgcHJvYmxlbS5cbi0gUmV2aWV3IGVhY2ggY2FsY3VsYXRpb24gc3RlcCBmb3IgYWNjdXJhY3kuIENoZWNrIGZvciBjb21wdXRhdGlvbmFsIGVycm9ycywgaW5jb3JyZWN0IGZvcm11bGEgYXBwbGljYXRpb25zLCBvciBhcml0aG1ldGljIG1pc3Rha2VzLlxuLSBUaGUgc29sdXRpb24gc2hvdWxkIHVzZSBhbGwgdGhlIGluZm9ybWF0aW9uIHByb3ZpZGVkIGluIHRoZSBxdWVzdGlvbi5cbi0gRXhhbWluZSB0aGUgZmluYWwgYW5zd2VyIGZvciBjb3JyZWN0bmVzcywgY29uc2lkZXJpbmcgdGhlIGNhbGN1bGF0aW9ucyBhbmQgbWV0aG9kIHVzZWQuXG4uXG5cbiMjIyBQcm9tcHRcbns8c3BhbiBjbGFzcz0ibHR4X3RleHQgbHR4X2ZvbnRfdHlwZXdyaXRlciIgaWQ9IkExLlNTNS5TU1MyLnAxLnBpYzEuMS4xLjEuMS4xLjEuMSI+cHJvbXB0PC9zcGFuPn1cblxuIyMjVHJhamVjdG9yeVxuezxzcGFuIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBpZD0iQTEuU1M1LlNTUzIucDEucGljMS4xLjEuMS4xLjEuMS4yIj50cmFqZWN0b3J5PC9zcGFuPn1cblxuIyMjQXNzZXNzbWVudFxuezxzcGFuIGNsYXNzPSJsdHhfdGV4dCBsdHhfZm9udF90eXBld3JpdGVyIiBpZD0iQTEuU1M1LlNTUzIucDEucGljMS4xLjEuMS4xLjEuMS4zIj50ZXh0dWFsIHJld2FyZDwvc3Bhbj59PC9zcGFuPgo8L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+){#A1.SS5.SSS2.p1.pic1
.ltx_picture}
:::
:::

::: {#A1.SS5.SSS3 .section .ltx_subsubsection}
#### [A.5.3 ]{.ltx_tag .ltx_tag_subsubsection}Policy Finetuning {#a.5.3-policy-finetuning .ltx_title .ltx_title_subsubsection}

::: {#A1.SS5.SSS3.p1 .ltx_para}
For MATH experiments that take a WizardMath V1.0 70B as the policy, we
adopt their proposed system prompt for self-improving. For GSM8K
experiments taking Llama2 70B pretrain as the policy, we use the
following system prompt.
:::

::: {#A1.SS5.SSS3.p2 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iNTgiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB3aWR0aD0iNjAwIj48ZyBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNTgpIG1hdHJpeCgxIDAgMCAtMSAwIDApIj48ZyBmaWxsPSIjNDA0MDQwIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggZD0iTSAwIDUuOTEgTCAwIDUyLjA5IEMgMCA1NS4zNiAyLjY0IDU4IDUuOTEgNTggTCA1OTQuMDkgNTggQyA1OTcuMzYgNTggNjAwIDU1LjM2IDYwMCA1Mi4wOSBMIDYwMCA1LjkxIEMgNjAwIDIuNjQgNTk3LjM2IDAgNTk0LjA5IDAgTCA1LjkxIDAgQyAyLjY0IDAgMCAyLjY0IDAgNS45MSBaIiBzdHlsZT0ic3Ryb2tlOm5vbmUiPjwvcGF0aD48L2c+PGcgZmlsbD0iI0YyRjJGMiIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIGQ9Ik0gMS45NyA1LjkxIEwgMS45NyA1Mi4wOSBDIDEuOTcgNTQuMjcgMy43MyA1Ni4wMyA1LjkxIDU2LjAzIEwgNTk0LjA5IDU2LjAzIEMgNTk2LjI3IDU2LjAzIDU5OC4wMyA1NC4yNyA1OTguMDMgNTIuMDkgTCA1OTguMDMgNS45MSBDIDU5OC4wMyAzLjczIDU5Ni4yNyAxLjk3IDU5NC4wOSAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuNjUgMTMuNzgpIj48Zm9yZWlnbm9iamVjdCBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSIzMC40NCIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTYuNikiIHdpZHRoPSI1NTYuNjkiPgo8c3BhbiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgaWQ9IkExLlNTNS5TU1MzLnAyLnBpYzEuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yIiBzdHlsZT0id2lkdGg6NDAyLjNwdDsiPgo8c3BhbiBjbGFzcz0ibHR4X3AiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yIj5BIGNoYXQgYmV0d2VlbiBhIGN1cmlvdXMgdXNlciBhbmQgYW4gYXJ0aWZpY2lhbCBpbnRlbGxpZ2VuY2UgYXNzaXN0YW50LlxuClRoZSBhc3Npc3RhbnQgZ2l2ZXMgaGVscGZ1bCwgZGV0YWlsZWQsIGFuZCBwb2xpdGUgYW5zd2VycyB0byB0aGUgdXNlcuKAmXMgcXVlc3Rpb25zLlxuClVzZXI6IDxtYXRoIGFsdHRleHQ9IntcYm17eH19X3tpfSIgY2xhc3M9Imx0eF9NYXRoIiBkaXNwbGF5PSJpbmxpbmUiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEiPjxzZW1hbnRpY3MgaWQ9IkExLlNTNS5TU1MzLnAyLnBpYzEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEuMWEiPjxtc3ViIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMSIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEuY21tbCI+PG1pIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMS4yIiB4cmVmPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMS4yLmNtbWwiPvCdkpk8L21pPjxtaSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEuMyIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEuMy5jbW1sIj5pPC9taT48L21zdWI+PGFubm90YXRpb24teG1sIGVuY29kaW5nPSJNYXRoTUwtQ29udGVudCIgaWQ9IkExLlNTNS5TU1MzLnAyLnBpYzEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEuMWIiPjxhcHBseSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEuY21tbCIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEiPjxjc3ltYm9sIGNkPSJhbWJpZ3VvdXMiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMS4xLmNtbWwiIHhyZWY9IkExLlNTNS5TU1MzLnAyLnBpYzEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEuMS4xIj5zdWJzY3JpcHQ8L2NzeW1ib2w+PGNpIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMS4yLmNtbWwiIHhyZWY9IkExLlNTNS5TU1MzLnAyLnBpYzEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEubTEuMS4xLjIiPvCdkpk8L2NpPjxjaSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xLjEuMy5jbW1sIiB4cmVmPSJBMS5TUzUuU1NTMy5wMi5waWMxLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLm0xLjEuMS4zIj7wnZGWPC9jaT48L2FwcGx5PjwvYW5ub3RhdGlvbi14bWw+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4IiBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xYyI+e1xibXt4fX1fe2l9PC9hbm5vdGF0aW9uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LWxsYW1hcHVuIiBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS4xLjEuMS5tMS4xZCI+Ym9sZF9pdGFsaWNfeCBzdGFydF9QT1NUU1VCU0NSSVBUIGl0YWxpY19pIGVuZF9QT1NUU1VCU0NSSVBUPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD5cbgpBc3Npc3RhbnQ6IDxtYXRoIGFsdHRleHQ9IntcYm17eX19X3tpfSIgY2xhc3M9Imx0eF9NYXRoIiBkaXNwbGF5PSJpbmxpbmUiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEiPjxzZW1hbnRpY3MgaWQ9IkExLlNTNS5TU1MzLnAyLnBpYzEuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIubTIuMWEiPjxtc3ViIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMSIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEuY21tbCI+PG1pIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMS4yIiB4cmVmPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMS4yLmNtbWwiPvCdkpo8L21pPjxtaSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEuMyIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEuMy5jbW1sIj5pPC9taT48L21zdWI+PGFubm90YXRpb24teG1sIGVuY29kaW5nPSJNYXRoTUwtQ29udGVudCIgaWQ9IkExLlNTNS5TU1MzLnAyLnBpYzEuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIubTIuMWIiPjxhcHBseSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEuY21tbCIgeHJlZj0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEiPjxjc3ltYm9sIGNkPSJhbWJpZ3VvdXMiIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMS4xLmNtbWwiIHhyZWY9IkExLlNTNS5TU1MzLnAyLnBpYzEuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIubTIuMS4xIj5zdWJzY3JpcHQ8L2NzeW1ib2w+PGNpIGlkPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMS4yLmNtbWwiIHhyZWY9IkExLlNTNS5TU1MzLnAyLnBpYzEuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIubTIuMS4xLjIiPvCdkpo8L2NpPjxjaSBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xLjEuMy5jbW1sIiB4cmVmPSJBMS5TUzUuU1NTMy5wMi5waWMxLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLm0yLjEuMS4zIj7wnZGWPC9jaT48L2FwcGx5PjwvYW5ub3RhdGlvbi14bWw+PGFubm90YXRpb24gZW5jb2Rpbmc9ImFwcGxpY2F0aW9uL3gtdGV4IiBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xYyI+e1xibXt5fX1fe2l9PC9hbm5vdGF0aW9uPjxhbm5vdGF0aW9uIGVuY29kaW5nPSJhcHBsaWNhdGlvbi94LWxsYW1hcHVuIiBpZD0iQTEuU1M1LlNTUzMucDIucGljMS4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi4yLjIuMi5tMi4xZCI+Ym9sZF9pdGFsaWNfeSBzdGFydF9QT1NUU1VCU0NSSVBUIGl0YWxpY19pIGVuZF9QT1NUU1VCU0NSSVBUPC9hbm5vdGF0aW9uPjwvc2VtYW50aWNzPjwvbWF0aD48L3NwYW4+Cjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=){#A1.SS5.SSS3.p2.pic1
.ltx_picture}
:::
:::
:::

::: {#A1.SS6 .section .ltx_subsection}
### [A.6 ]{.ltx_tag .ltx_tag_subsection}MCTS Details {#a.6-mcts-details .ltx_title .ltx_title_subsection}

::: {#A1.SS6.p1 .ltx_para}
We set the MCTS parameters in Table [[5]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.T5 "Table 5 ‣ A.6 MCTS Details ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::

  -------------------------------------- -- ----------------------------------------------------------- ----------------------------------------------------------- ----------------------------------------------------------- -----------------------------------------------------------
  [Method]{#A1.T5.8.9.1.1.1 .ltx_text}      GSM8K                                                                                                                   MATH                                                        
                                            [Small]{#A1.T5.8.10.2.2.1 .ltx_text .ltx_font_typewriter}   [Large]{#A1.T5.8.10.2.3.1 .ltx_text .ltx_font_typewriter}   [Small]{#A1.T5.8.10.2.4.1 .ltx_text .ltx_font_typewriter}   [Large]{#A1.T5.8.10.2.5.1 .ltx_text .ltx_font_typewriter}
  $c$                                       1.0                                                         1.5                                                         1.0                                                         1.0
  $\alpha$                                  1.0                                                         1.0                                                         1.0                                                         1.0
  $c_{\text{max}}{(0)}$                     60                                                          60                                                          60                                                          60
  $c_{\text{max}}{(t)}$ where $t > 0$       10                                                          10                                                          10                                                          10
  $c_{\text{min}}{(0)}$                     10                                                          40                                                          10                                                          20
  $c_{\text{min}}{(t)}$ where $t > 0$       2                                                           2                                                           3                                                           3
  -------------------------------------- -- ----------------------------------------------------------- ----------------------------------------------------------- ----------------------------------------------------------- -----------------------------------------------------------

[Table 5: ]{.ltx_tag .ltx_tag_table}Parameters for MCTS. The Small/Large
means small \#rollout and small \#rollout
:::

::: {#A1.SS7 .section .ltx_subsection}
### [A.7 ]{.ltx_tag .ltx_tag_subsection}Additional Ablations {#a.7-additional-ablations .ltx_title .ltx_title_subsection}

::: {#A1.SS7.SSS0.Px1 .section .ltx_paragraph}
##### Fast-rollout model {#fast-rollout-model .ltx_title .ltx_title_paragraph}

::: {#A1.SS7.SSS0.Px1.p1 .ltx_para}
Using Llama-2-70b instead of Abel-7B-002 improves performance by
reducing bias from a smaller model, but Abel-002-7B is faster with
similar computational resources due to higher concurrency and quicker
processing. The details can be found in Table [[6]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.T6 "Table 6 ‣ Fast-rollout model ‣ A.7 Additional Ablations ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}.
:::

  Model         Acc (%)   Speed (s)
  ------------- --------- -----------
  Abel-002-7B   87.0      16.8
  Llama-2-70B   87.3      38.1

[Table 6: ]{.ltx_tag .ltx_tag_table}Ablation study over different
fast-rollout models on GSM8K.
:::
:::

::: {#A1.SS8 .section .ltx_subsection}
### [A.8 ]{.ltx_tag .ltx_tag_subsection}Search Comparison {#a.8-search-comparison .ltx_title .ltx_title_subsection}

  ------------------------------------------------------------------------------------------ ------------------------------------------- --------------------------------------------------------------- ------------------------------------------------------------- --------------------------------------------------------------- -------------------------------------------------------------
  [Method]{#A1.T7.1.2.1.1.1 .ltx_text}                                                       [\#Responses]{#A1.T7.1.2.1.2.1 .ltx_text}   GSM8K                                                                                                                         MATH                                                            
                                                                                                                                         [\#Rollouts]{#A1.T7.1.3.2.1.1 .ltx_text .ltx_font_typewriter}   [Accuracy]{#A1.T7.1.3.2.2.1 .ltx_text .ltx_font_typewriter}   [\#Rollouts]{#A1.T7.1.3.2.3.1 .ltx_text .ltx_font_typewriter}   [Accuracy]{#A1.T7.1.3.2.4.1 .ltx_text .ltx_font_typewriter}
  Greedy                                                                                     1                                           4.6                                                             57.8                                                          9.9                                                             20.7
  [Self-consistency]{#A1.T7.1.5.4.1.1 .ltx_text}                                             10                                          46                                                              67.4                                                          99                                                              22.5
                                                                                             30                                          137                                                             74.2                                                          299                                                             27.3
                                                                                             50                                          229                                                             75.4                                                          499                                                             28.8
  [Re-ranking]{#A1.T7.1.8.7.1.1 .ltx_text}                                                   10                                          46                                                              80.8                                                          99                                                              34.1
                                                                                             30                                          137                                                             86.3                                                          299                                                             39.0
                                                                                             50                                          229                                                             87.7                                                          499                                                             42.0
  [$\eta$[Mcts]{#A1.T7.1.1.1.1.1 .ltx_text .ltx_font_smallcaps}]{#A1.T7.1.1.1.1 .ltx_text}   \-                                          55                                                              87.0                                                          223                                                             45.4
                                                                                             \-                                          230                                                             88.9                                                          341                                                             48.7
  ------------------------------------------------------------------------------------------ ------------------------------------------- --------------------------------------------------------------- ------------------------------------------------------------- --------------------------------------------------------------- -------------------------------------------------------------

[Table 7: ]{.ltx_tag .ltx_tag_table}Comparative results of various
searching method on GSM8K and MATH.

::: {#A1.SS8.p1 .ltx_para}
Table [[7]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.T7 "Table 7 ‣ A.8 Search Comparison ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
presents the performance of various methods applied to different number
of responses, from 10 to 50. Our analysis confirms several key findings:
1) Reranking utilizing [ORM]{#A1.SS8.p1.2.1 .ltx_text
.ltx_font_typewriter} consistently outperforms self-consistency
techniques, indicating that [ORM]{#A1.SS8.p1.2.2 .ltx_text
.ltx_font_typewriter} is capable of generating meaningful signals for
searching. 2) $\eta$[Mcts]{#A1.SS8.p1.2.3 .ltx_text .ltx_font_smallcaps}
demonstrates superior performance while requiring significantly fewer
rollouts. For instance, on the MATH dataset, $\eta$[Mcts]{#A1.SS8.p1.2.4
.ltx_text .ltx_font_smallcaps} achieves better results with only half
the number of rollouts compared to reranking. Additionally, we evaluated
the performance of BFS on the GSM8K only, where it requires 87.9
rollouts to achieve a score of 80.6. These results suggest that our
design of an efficient MCTS in [AlphaLLM]{#A1.SS8.p1.2.5 .ltx_text
.ltx_font_smallcaps} can serve as an effective policy improvement
operation, enabling the search for high-quality trajectories with
reduced computational cost.
:::
:::

::: {#A1.SS9 .section .ltx_subsection}
### [A.9 ]{.ltx_tag .ltx_tag_subsection}Rollout Example {#a.9-rollout-example .ltx_title .ltx_title_subsection}

::: {#A1.SS9.p1 .ltx_para}
Consider the following GSM-like question:
:::

::: {#A1.SS9.p2 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iOTEuMjEiIGlkPSJBMS5TUzkucDIucGljMSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgd2lkdGg9IjYwMCI+PGcgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDkxLjIxKSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgZmlsbD0iIzQwNDA0MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIGQ9Ik0gMCA1LjkxIEwgMCA4NS4zIEMgMCA4OC41NyAyLjY0IDkxLjIxIDUuOTEgOTEuMjEgTCA1OTQuMDkgOTEuMjEgQyA1OTcuMzYgOTEuMjEgNjAwIDg4LjU3IDYwMCA4NS4zIEwgNjAwIDUuOTEgQyA2MDAgMi42NCA1OTcuMzYgMCA1OTQuMDkgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIHN0eWxlPSJzdHJva2U6bm9uZSI+PC9wYXRoPjwvZz48ZyBmaWxsPSIjRjJGMkYyIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDg1LjMgQyAxLjk3IDg3LjQ4IDMuNzMgODkuMjQgNS45MSA4OS4yNCBMIDU5NC4wOSA4OS4yNCBDIDU5Ni4yNyA4OS4yNCA1OTguMDMgODcuNDggNTk4LjAzIDg1LjMgTCA1OTguMDMgNS45MSBDIDU5OC4wMyAzLjczIDU5Ni4yNyAxLjk3IDU5NC4wOSAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuNjUgMTMuNzgpIj48Zm9yZWlnbm9iamVjdCBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI2My42NSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTYuNikiIHdpZHRoPSI1NTYuNjkiPgo8c3BhbiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgaWQ9IkExLlNTOS5wMi5waWMxLjEuMS4xLjEuMSIgc3R5bGU9IndpZHRoOjQwMi4zcHQ7Ij4KPHNwYW4gY2xhc3M9Imx0eF9wIiBpZD0iQTEuU1M5LnAyLnBpYzEuMS4xLjEuMS4xLjEiPlF1ZXN0aW9uOiBTYW5keeKAmXMgbW9udGhseSBwaG9uZSBiaWxsIGV4cGVuc2UgaXMgZXF1YWwgdG8gdGVuIHRpbWVzIGhlciBhZ2Ugbm93LiBJbiB0d28geWVhcnMsIFNhbmR5IHdpbGwgYmUgdGhyZWUgdGltZXMgYXMgb2xkIGFzIEtpbS4gSWYgS2ltIGlzIGN1cnJlbnRseSB4IHllYXJzIG9sZCwgY2FsY3VsYXRlIFNhbmR54oCZcyBtb250aGx5IHBob25lIGJpbGwgZXhwZW5zZS5cbklmIHdlIGtub3cgdGhlIGFuc3dlciB0byB0aGUgYWJvdmUgcXVlc3Rpb24gaXMgMzQwLCB3aGF0IGlzIHRoZSB2YWx1ZSBvZiB0aGUgdW5rbm93biB2YXJpYWJsZSB4P1xuPC9zcGFuPgo8L3NwYW4+PC9mb3JlaWdub2JqZWN0PjwvZz48L2c+PC9zdmc+){#A1.SS9.p2.pic1
.ltx_picture}
:::

::: {#A1.SS9.p3 .ltx_para}
A node in the second layer could have the following content:
:::

::: {#A1.SS9.p4 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iOTEuMjEiIGlkPSJBMS5TUzkucDQucGljMSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgd2lkdGg9IjYwMCI+PGcgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDkxLjIxKSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgZmlsbD0iIzQwNDA0MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIGQ9Ik0gMCA1LjkxIEwgMCA4NS4zIEMgMCA4OC41NyAyLjY0IDkxLjIxIDUuOTEgOTEuMjEgTCA1OTQuMDkgOTEuMjEgQyA1OTcuMzYgOTEuMjEgNjAwIDg4LjU3IDYwMCA4NS4zIEwgNjAwIDUuOTEgQyA2MDAgMi42NCA1OTcuMzYgMCA1OTQuMDkgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIHN0eWxlPSJzdHJva2U6bm9uZSI+PC9wYXRoPjwvZz48ZyBmaWxsPSIjRjJGMkYyIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDg1LjMgQyAxLjk3IDg3LjQ4IDMuNzMgODkuMjQgNS45MSA4OS4yNCBMIDU5NC4wOSA4OS4yNCBDIDU5Ni4yNyA4OS4yNCA1OTguMDMgODcuNDggNTk4LjAzIDg1LjMgTCA1OTguMDMgNS45MSBDIDU5OC4wMyAzLjczIDU5Ni4yNyAxLjk3IDU5NC4wOSAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuNjUgMTMuNzgpIj48Zm9yZWlnbm9iamVjdCBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI2My42NSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTYuNikiIHdpZHRoPSI1NTYuNjkiPgo8c3BhbiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgaWQ9IkExLlNTOS5wNC5waWMxLjEuMS4xLjEuMSIgc3R5bGU9IndpZHRoOjQwMi4zcHQ7Ij4KPHNwYW4gY2xhc3M9Imx0eF9wIiBpZD0iQTEuU1M5LnA0LnBpYzEuMS4xLjEuMS4xLjEiPkFuc3dlcjogV2Uga25vdyB0aGF0IFNhbmR54oCZcyBtb250aGx5IHBob25lIGJpbGwgaXMgMTAgdGltZXMgaGVyIGFnZS4gSW4gdHdvIHllYXJzLCBTYW5keSB3aWxsIGJlIDMgdGltZXMgYXMgb2xkIGFzIEtpbS4gVGhlIHN1bSBvZiBTYW5keeKAmXMgYWdlIG5vdyBhbmQgMiB5ZWFycyBpcyAzIHRpbWVzIHRoZSBzdW0gb2YgS2lt4oCZcyBhZ2Ugbm93IGFuZCB0d28geWVhcnMuXG5TYW5keeKAmXMgYWdlIG5vdyBpcyAzNDAvMTAgPSAmbHQ7Jmx0OzM0MC8xMD0zNCZndDsmZ3Q7MzQuIEluIHR3byB5ZWFycywgU2FuZHnigJlzIGFnZSB3aWxsIGJlIDM0ICsgMiA9ICZsdDsmbHQ7MzQrMj0zNiZndDsmZ3Q7MzYuXG48L3NwYW4+Cjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=){#A1.SS9.p4.pic1
.ltx_picture}
:::

::: {#A1.SS9.p5 .ltx_para}
The parent of this node has the content:
:::

::: {#A1.SS9.p6 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iNzQuNiIgaWQ9IkExLlNTOS5wNi5waWMxIiBvdmVyZmxvdz0idmlzaWJsZSIgdmVyc2lvbj0iMS4xIiB3aWR0aD0iNjAwIj48ZyBmaWxsPSIjMDAwMDAwIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMC40cHQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsNzQuNikgbWF0cml4KDEgMCAwIC0xIDAgMCkiPjxnIGZpbGw9IiM0MDQwNDAiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBkPSJNIDAgNS45MSBMIDAgNjguNyBDIDAgNzEuOTYgMi42NCA3NC42IDUuOTEgNzQuNiBMIDU5NC4wOSA3NC42IEMgNTk3LjM2IDc0LjYgNjAwIDcxLjk2IDYwMCA2OC43IEwgNjAwIDUuOTEgQyA2MDAgMi42NCA1OTcuMzYgMCA1OTQuMDkgMCBMIDUuOTEgMCBDIDIuNjQgMCAwIDIuNjQgMCA1LjkxIFoiIHN0eWxlPSJzdHJva2U6bm9uZSI+PC9wYXRoPjwvZz48ZyBmaWxsPSIjRjJGMkYyIiBmaWxsLW9wYWNpdHk9IjEuMCI+PHBhdGggZD0iTSAxLjk3IDUuOTEgTCAxLjk3IDY4LjcgQyAxLjk3IDcwLjg3IDMuNzMgNzIuNjQgNS45MSA3Mi42NCBMIDU5NC4wOSA3Mi42NCBDIDU5Ni4yNyA3Mi42NCA1OTguMDMgNzAuODcgNTk4LjAzIDY4LjcgTCA1OTguMDMgNS45MSBDIDU5OC4wMyAzLjczIDU5Ni4yNyAxLjk3IDU5NC4wOSAxLjk3IEwgNS45MSAxLjk3IEMgMy43MyAxLjk3IDEuOTcgMy43MyAxLjk3IDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGwtb3BhY2l0eT0iMS4wIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAgMC4wIDAuMCAxLjAgMjEuNjUgMTMuNzgpIj48Zm9yZWlnbm9iamVjdCBjb2xvcj0iIzAwMDAwMCIgaGVpZ2h0PSI0Ny4wNSIgb3ZlcmZsb3c9InZpc2libGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMCAwIC0xIDAgMTYuNikiIHdpZHRoPSI1NTYuNjkiPgo8c3BhbiBjbGFzcz0ibHR4X2lubGluZS1ibG9jayBsdHhfbWluaXBhZ2UgbHR4X2FsaWduX2JvdHRvbSIgaWQ9IkExLlNTOS5wNi5waWMxLjEuMS4xLjEuMSIgc3R5bGU9IndpZHRoOjQwMi4zcHQ7Ij4KPHNwYW4gY2xhc3M9Imx0eF9wIiBpZD0iQTEuU1M5LnA2LnBpYzEuMS4xLjEuMS4xLjEiPkFuc3dlcjogV2Uga25vdyB0aGF0IFNhbmR54oCZcyBtb250aGx5IHBob25lIGJpbGwgaXMgMTAgdGltZXMgaGVyIGFnZS4gSW4gdHdvIHllYXJzLCBTYW5keSB3aWxsIGJlIDMgdGltZXMgYXMgb2xkIGFzIEtpbS4gVGhlIHN1bSBvZiBTYW5keeKAmXMgYWdlIG5vdyBhbmQgMiB5ZWFycyBpcyAzIHRpbWVzIHRoZSBzdW0gb2YgS2lt4oCZcyBhZ2Ugbm93IGFuZCB0d28geWVhcnMuXG48L3NwYW4+Cjwvc3Bhbj48L2ZvcmVpZ25vYmplY3Q+PC9nPjwvZz48L3N2Zz4=){#A1.SS9.p6.pic1
.ltx_picture}
:::

::: {#A1.SS9.p7 .ltx_para}
And one of its fast-rollout paths could be:
:::

::: {#A1.SS9.p8 .ltx_para .ltx_noindent}
![](data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ibHR4X3BpY3R1cmUiIGhlaWdodD0iNTgiIGlkPSJBMS5TUzkucDgucGljMSIgb3ZlcmZsb3c9InZpc2libGUiIHZlcnNpb249IjEuMSIgd2lkdGg9IjYwMCI+PGcgZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjAuNHB0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLDU4KSBtYXRyaXgoMSAwIDAgLTEgMCAwKSI+PGcgZmlsbD0iIzQwNDA0MCIgZmlsbC1vcGFjaXR5PSIxLjAiPjxwYXRoIGQ9Ik0gMCA1LjkxIEwgMCA1Mi4wOSBDIDAgNTUuMzYgMi42NCA1OCA1LjkxIDU4IEwgNTk0LjA5IDU4IEMgNTk3LjM2IDU4IDYwMCA1NS4zNiA2MDAgNTIuMDkgTCA2MDAgNS45MSBDIDYwMCAyLjY0IDU5Ny4zNiAwIDU5NC4wOSAwIEwgNS45MSAwIEMgMi42NCAwIDAgMi42NCAwIDUuOTEgWiIgc3R5bGU9InN0cm9rZTpub25lIj48L3BhdGg+PC9nPjxnIGZpbGw9IiNGMkYyRjIiIGZpbGwtb3BhY2l0eT0iMS4wIj48cGF0aCBkPSJNIDEuOTcgNS45MSBMIDEuOTcgNTIuMDkgQyAxLjk3IDU0LjI3IDMuNzMgNTYuMDMgNS45MSA1Ni4wMyBMIDU5NC4wOSA1Ni4wMyBDIDU5Ni4yNyA1Ni4wMyA1OTguMDMgNTQuMjcgNTk4LjAzIDUyLjA5IEwgNTk4LjAzIDUuOTEgQyA1OTguMDMgMy43MyA1OTYuMjcgMS45NyA1OTQuMDkgMS45NyBMIDUuOTEgMS45NyBDIDMuNzMgMS45NyAxLjk3IDMuNzMgMS45NyA1LjkxIFoiIHN0eWxlPSJzdHJva2U6bm9uZSI+PC9wYXRoPjwvZz48ZyBmaWxsLW9wYWNpdHk9IjEuMCIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wIDAuMCAwLjAgMS4wIDIxLjY1IDEzLjc4KSI+PGZvcmVpZ25vYmplY3QgY29sb3I9IiMwMDAwMDAiIGhlaWdodD0iMzAuNDQiIG92ZXJmbG93PSJ2aXNpYmxlIiB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAwIDE2LjYpIiB3aWR0aD0iNTU2LjY5Ij4KPHNwYW4gY2xhc3M9Imx0eF9pbmxpbmUtYmxvY2sgbHR4X21pbmlwYWdlIGx0eF9hbGlnbl9ib3R0b20iIGlkPSJBMS5TUzkucDgucGljMS4xLjEuMS4xLjEiIHN0eWxlPSJ3aWR0aDo0MDIuM3B0OyI+CjxzcGFuIGNsYXNzPSJsdHhfcCIgaWQ9IkExLlNTOS5wOC5waWMxLjEuMS4xLjEuMS4xIj5UaGUgc3VtIG9mIFNhbmR54oCZcyBhZ2Ugbm93IGFuZCAyIHllYXJzIGlzIDM2LiBUaGUgc3VtIG9mIEtpbeKAmXMgYWdlIG5vdyBhbmQgdHdvIHllYXJzIGlzIHggKyAyLlxuMzYgPSAzKHggKyAyKVxuNiA9IDN4ICsgNlxuM3ggPSAzMFxueCA9IDEwXG4gIyMjIyAxMDwvc3Bhbj4KPC9zcGFuPjwvZm9yZWlnbm9iamVjdD48L2c+PC9nPjwvc3ZnPg==){#A1.SS9.p8.pic1
.ltx_picture}
:::
:::

::: {#A1.SS10 .section .ltx_subsection}
### [A.10 ]{.ltx_tag .ltx_tag_subsection}Critic Performance {#a.10-critic-performance .ltx_title .ltx_title_subsection}

::: {#A1.SS10.p1 .ltx_para}
We evaluated the performance of the value function and
[PRM]{#A1.SS10.p1.1.1 .ltx_text .ltx_font_typewriter} on the GSM8K test
set. Table [[8]{.ltx_text
.ltx_ref_tag}](https://arxiv.org/html/2404.12253v2#A1.T8 "Table 8 ‣ A.10 Critic Performance ‣ Appendix A Appendix ‣ Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing"){.ltx_ref}
presents a comparison of these models in terms of precision, recall, and
Expected Calibration Error (ECE). Results indicate that the value
function achieves higher precision and better calibration, while
[PRM]{#A1.SS10.p1.1.2 .ltx_text .ltx_font_typewriter} demonstrates a
superior recall.
:::

  Model                                                    Precision   Recall   ECE
  -------------------------------------------------------- ----------- -------- -------
  Value Function                                           0.82        0.79     0.032
  [PRM]{#A1.T8.1.3.2.1.1 .ltx_text .ltx_font_typewriter}   0.62        0.90     0.375

[Table 8: ]{.ltx_tag .ltx_tag_table}Performance comparison of the Value
Function model and [PRM]{#A1.T8.3.1 .ltx_text .ltx_font_typewriter} on
the GSM8K test set.
:::

::: {#A1.SS11 .section .ltx_subsection}
### [A.11 ]{.ltx_tag .ltx_tag_subsection}Compute Resources {#a.11-compute-resources .ltx_title .ltx_title_subsection}

::: {#A1.SS11.p1 .ltx_para}
Our experiments were conducted using NVIDIA A100 40GB GPUs. Serving
models based on Llama-2-70B or WizardMath-70B required 4 GPUs, while
serving Llama-2-7B and Abel-002-7B was possible on a single GPU.
Training the 70B models required 64 GPUs.
:::
:::

::: {#A1.SS12 .section .ltx_subsection}
### [A.12 ]{.ltx_tag .ltx_tag_subsection}Limitations and Future Work {#a.12-limitations-and-future-work .ltx_title .ltx_title_subsection}

::: {#A1.SS12.p1 .ltx_para}
Despite the promising results demonstrated by [AlphaLLM]{#A1.SS12.p1.1.1
.ltx_text .ltx_font_smallcaps} in this study, there are several
limitations that requires further exploration. ([i]{#A1.SS12.p1.1.2
.ltx_text .ltx_font_italic}) Our current implementation employs
relatively simple methods for generating synthetic prompts. Future
iterations of [AlphaLLM]{#A1.SS12.p1.1.3 .ltx_text .ltx_font_smallcaps}
should explore advanced techniques, such as Self-Instruct, to create
both diverse and model capability-awared prompts. ([ii]{#A1.SS12.p1.1.4
.ltx_text .ltx_font_italic}) Although [AlphaLLM]{#A1.SS12.p1.1.5
.ltx_text .ltx_font_smallcaps} demonstrates improvements over base
models, its performance in greedy sampling is substantially inferior to
that observed when decoded with $\eta$[Mcts]{#A1.SS12.p1.1.6 .ltx_text
.ltx_font_smallcaps}. This indicates that the full potential of MCTS for
self-improvement in LLMs has not yet been fully realized. Two potential
factors contributing to this issue have been identified: a) the
self-improvement loop may not be leveraging sufficient data; and b) the
base model may be limited in its capacity for rapid learning. Addressing
these concerns could lead to more significant improvemens.
([iii]{#A1.SS12.p1.1.7 .ltx_text .ltx_font_italic}) In our existing
framework, the critic models remain static. We will explore mechanisms
to continually update critic models to adapt to new policy models. This
will help ensure the discriminator-generator gap and improve the overall
training dynamics. ([iv]{#A1.SS12.p1.1.8 .ltx_text .ltx_font_italic})
The evaluation of [AlphaLLM]{#A1.SS12.p1.1.9 .ltx_text
.ltx_font_smallcaps} has been limited to mathematical reasoning tasks.
To verify the generalizability and broader applicability of the
framework, future research will need to extend its application to other
domains.
:::

::: {.ltx_pagination .ltx_role_newpage}
:::
:::
:::

::: {#Ax1 .section .ltx_appendix}
## NeurIPS Paper Checklist {#neurips-paper-checklist .ltx_title .ltx_title_appendix}

::: {#Ax1.p1 .ltx_para}
1.  [[1.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i1}
    ::: {#Ax1.I2.i1.p1 .ltx_para}
    [Claims]{#Ax1.I2.i1.p1.1.1 .ltx_text .ltx_font_bold}
    :::
2.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix1}
    ::: {#Ax1.I2.ix1.p1 .ltx_para}
    Question: Do the main claims made in the abstract and introduction
    accurately reflect the paper's contributions and scope?
    :::
3.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix2}
    ::: {#Ax1.I2.ix2.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix2.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
4.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix3}
    ::: {#Ax1.I2.ix3.p1 .ltx_para}
    Justification: Yes the claims are accurately made.
    :::
5.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix4}
    ::: {#Ax1.I2.ix4.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix4.I1.i1}
        ::: {#Ax1.I2.ix4.I1.i1.p1 .ltx_para}
        The answer NA means that the abstract and introduction do not
        include the claims made in the paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix4.I1.i2}
        ::: {#Ax1.I2.ix4.I1.i2.p1 .ltx_para}
        The abstract and/or introduction should clearly state the claims
        made, including the contributions made in the paper and
        important assumptions and limitations. A No or NA answer to this
        question will not be perceived well by the reviewers.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix4.I1.i3}
        ::: {#Ax1.I2.ix4.I1.i3.p1 .ltx_para}
        The claims made should match theoretical and experimental
        results, and reflect how much the results can be expected to
        generalize to other settings.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix4.I1.i4}
        ::: {#Ax1.I2.ix4.I1.i4.p1 .ltx_para}
        It is fine to include aspirational goals as motivation as long
        as it is clear that these goals are not attained by the paper.
        :::
    :::
6.  [[2.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i2}
    ::: {#Ax1.I2.i2.p1 .ltx_para}
    [Limitations]{#Ax1.I2.i2.p1.1.1 .ltx_text .ltx_font_bold}
    :::
7.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix5}
    ::: {#Ax1.I2.ix5.p1 .ltx_para}
    Question: Does the paper discuss the limitations of the work
    performed by the authors?
    :::
8.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix6}
    ::: {#Ax1.I2.ix6.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix6.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
9.  [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix7}
    ::: {#Ax1.I2.ix7.p1 .ltx_para}
    Justification: Yes we discussed the limitations in Appendix.
    :::
10. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8}
    ::: {#Ax1.I2.ix8.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i1}
        ::: {#Ax1.I2.ix8.I1.i1.p1 .ltx_para}
        The answer NA means that the paper has no limitation while the
        answer No means that the paper has limitations, but those are
        not discussed in the paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i2}
        ::: {#Ax1.I2.ix8.I1.i2.p1 .ltx_para}
        The authors are encouraged to create a separate \"Limitations\"
        section in their paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i3}
        ::: {#Ax1.I2.ix8.I1.i3.p1 .ltx_para}
        The paper should point out any strong assumptions and how robust
        the results are to violations of these assumptions (e.g.,
        independence assumptions, noiseless settings, model
        well-specification, asymptotic approximations only holding
        locally). The authors should reflect on how these assumptions
        might be violated in practice and what the implications would
        be.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i4}
        ::: {#Ax1.I2.ix8.I1.i4.p1 .ltx_para}
        The authors should reflect on the scope of the claims made,
        e.g., if the approach was only tested on a few datasets or with
        a few runs. In general, empirical results often depend on
        implicit assumptions, which should be articulated.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i5}
        ::: {#Ax1.I2.ix8.I1.i5.p1 .ltx_para}
        The authors should reflect on the factors that influence the
        performance of the approach. For example, a facial recognition
        algorithm may perform poorly when image resolution is low or
        images are taken in low lighting. Or a speech-to-text system
        might not be used reliably to provide closed captions for online
        lectures because it fails to handle technical jargon.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i6}
        ::: {#Ax1.I2.ix8.I1.i6.p1 .ltx_para}
        The authors should discuss the computational efficiency of the
        proposed algorithms and how they scale with dataset size.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i7}
        ::: {#Ax1.I2.ix8.I1.i7.p1 .ltx_para}
        If applicable, the authors should discuss possible limitations
        of their approach to address problems of privacy and fairness.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix8.I1.i8}
        ::: {#Ax1.I2.ix8.I1.i8.p1 .ltx_para}
        While the authors might fear that complete honesty about
        limitations might be used by reviewers as grounds for rejection,
        a worse outcome might be that reviewers discover limitations
        that aren't acknowledged in the paper. The authors should use
        their best judgment and recognize that individual actions in
        favor of transparency play an important role in developing norms
        that preserve the integrity of the community. Reviewers will be
        specifically instructed to not penalize honesty concerning
        limitations.
        :::
    :::
11. [[3.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i3}
    ::: {#Ax1.I2.i3.p1 .ltx_para}
    [Theory Assumptions and Proofs]{#Ax1.I2.i3.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
12. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix9}
    ::: {#Ax1.I2.ix9.p1 .ltx_para}
    Question: For each theoretical result, does the paper provide the
    full set of assumptions and a complete (and correct) proof?
    :::
13. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix10}
    ::: {#Ax1.I2.ix10.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix10.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
14. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix11}
    ::: {#Ax1.I2.ix11.p1 .ltx_para}
    Justification: We provide the assumptions and proofs for the Theorem
    4.1. and other theoretical results.
    :::
15. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12}
    ::: {#Ax1.I2.ix12.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i1}
        ::: {#Ax1.I2.ix12.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not include theoretical
        results.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i2}
        ::: {#Ax1.I2.ix12.I1.i2.p1 .ltx_para}
        All the theorems, formulas, and proofs in the paper should be
        numbered and cross-referenced.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i3}
        ::: {#Ax1.I2.ix12.I1.i3.p1 .ltx_para}
        All assumptions should be clearly stated or referenced in the
        statement of any theorems.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i4}
        ::: {#Ax1.I2.ix12.I1.i4.p1 .ltx_para}
        The proofs can either appear in the main paper or the
        supplemental material, but if they appear in the supplemental
        material, the authors are encouraged to provide a short proof
        sketch to provide intuition.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i5}
        ::: {#Ax1.I2.ix12.I1.i5.p1 .ltx_para}
        Inversely, any informal proof provided in the core of the paper
        should be complemented by formal proofs provided in appendix or
        supplemental material.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix12.I1.i6}
        ::: {#Ax1.I2.ix12.I1.i6.p1 .ltx_para}
        Theorems and Lemmas that the proof relies upon should be
        properly referenced.
        :::
    :::
16. [[4.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i4}
    ::: {#Ax1.I2.i4.p1 .ltx_para}
    [Experimental Result Reproducibility]{#Ax1.I2.i4.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
17. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix13}
    ::: {#Ax1.I2.ix13.p1 .ltx_para}
    Question: Does the paper fully disclose all the information needed
    to reproduce the main experimental results of the paper to the
    extent that it affects the main claims and/or conclusions of the
    paper (regardless of whether the code and data are provided or not)?
    :::
18. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix14}
    ::: {#Ax1.I2.ix14.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix14.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
19. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix15}
    ::: {#Ax1.I2.ix15.p1 .ltx_para}
    Justification: We provided the hyoerparameters to reproduce the
    results.
    :::
20. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16}
    ::: {#Ax1.I2.ix16.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i1}
        ::: {#Ax1.I2.ix16.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not include experiments.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i2}
        ::: {#Ax1.I2.ix16.I1.i2.p1 .ltx_para}
        If the paper includes experiments, a No answer to this question
        will not be perceived well by the reviewers: Making the paper
        reproducible is important, regardless of whether the code and
        data are provided or not.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i3}
        ::: {#Ax1.I2.ix16.I1.i3.p1 .ltx_para}
        If the contribution is a dataset and/or model, the authors
        should describe the steps taken to make their results
        reproducible or verifiable.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i4}
        ::: {#Ax1.I2.ix16.I1.i4.p1 .ltx_para}
        Depending on the contribution, reproducibility can be
        accomplished in various ways. For example, if the contribution
        is a novel architecture, describing the architecture fully might
        suffice, or if the contribution is a specific model and
        empirical evaluation, it may be necessary to either make it
        possible for others to replicate the model with the same
        dataset, or provide access to the model. In general. releasing
        code and data is often one good way to accomplish this, but
        reproducibility can also be provided via detailed instructions
        for how to replicate the results, access to a hosted model
        (e.g., in the case of a large language model), releasing of a
        model checkpoint, or other means that are appropriate to the
        research performed.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i5}
        ::: {#Ax1.I2.ix16.I1.i5.p1 .ltx_para}
        While NeurIPS does not require releasing code, the conference
        does require all submissions to provide some reasonable avenue
        for reproducibility, which may depend on the nature of the
        contribution. For example

        1.  [[(a)]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i5.I1.i1}
            ::: {#Ax1.I2.ix16.I1.i5.I1.i1.p1 .ltx_para}
            If the contribution is primarily a new algorithm, the paper
            should make it clear how to reproduce that algorithm.
            :::
        2.  [[(b)]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i5.I1.i2}
            ::: {#Ax1.I2.ix16.I1.i5.I1.i2.p1 .ltx_para}
            If the contribution is primarily a new model architecture,
            the paper should describe the architecture clearly and
            fully.
            :::
        3.  [[(c)]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i5.I1.i3}
            ::: {#Ax1.I2.ix16.I1.i5.I1.i3.p1 .ltx_para}
            If the contribution is a new model (e.g., a large language
            model), then there should either be a way to access this
            model for reproducing the results or a way to reproduce the
            model (e.g., with an open-source dataset or instructions for
            how to construct the dataset).
            :::
        4.  [[(d)]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix16.I1.i5.I1.i4}
            ::: {#Ax1.I2.ix16.I1.i5.I1.i4.p1 .ltx_para}
            We recognize that reproducibility may be tricky in some
            cases, in which case authors are welcome to describe the
            particular way they provide for reproducibility. In the case
            of closed-source models, it may be that access to the model
            is limited in some way (e.g., to registered users), but it
            should be possible for other researchers to have some path
            to reproducing or verifying the results.
            :::
        :::
    :::
21. [[5.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i5}
    ::: {#Ax1.I2.i5.p1 .ltx_para}
    [Open access to data and code]{#Ax1.I2.i5.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
22. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix17}
    ::: {#Ax1.I2.ix17.p1 .ltx_para}
    Question: Does the paper provide open access to the data and code,
    with sufficient instructions to faithfully reproduce the main
    experimental results, as described in supplemental material?
    :::
23. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix18}
    ::: {#Ax1.I2.ix18.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix18.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
24. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix19}
    ::: {#Ax1.I2.ix19.p1 .ltx_para}
    Justification: The code is available at
    https://github.com/YeTianJHU/AlphaLLM.
    :::
25. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20}
    ::: {#Ax1.I2.ix20.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i1}
        ::: {#Ax1.I2.ix20.I1.i1.p1 .ltx_para}
        The answer NA means that paper does not include experiments
        requiring code.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i2}
        ::: {#Ax1.I2.ix20.I1.i2.p1 .ltx_para}
        Please see the NeurIPS code and data submission guidelines
        (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more
        details.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i3}
        ::: {#Ax1.I2.ix20.I1.i3.p1 .ltx_para}
        While we encourage the release of code and data, we understand
        that this might not be possible, so "No" is an acceptable
        answer. Papers cannot be rejected simply for not including code,
        unless this is central to the contribution (e.g., for a new
        open-source benchmark).
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i4}
        ::: {#Ax1.I2.ix20.I1.i4.p1 .ltx_para}
        The instructions should contain the exact command and
        environment needed to run to reproduce the results. See the
        NeurIPS code and data submission guidelines
        (<https://nips.cc/public/guides/CodeSubmissionPolicy>) for more
        details.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i5}
        ::: {#Ax1.I2.ix20.I1.i5.p1 .ltx_para}
        The authors should provide instructions on data access and
        preparation, including how to access the raw data, preprocessed
        data, intermediate data, and generated data, etc.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i6}
        ::: {#Ax1.I2.ix20.I1.i6.p1 .ltx_para}
        The authors should provide scripts to reproduce all experimental
        results for the new proposed method and baselines. If only a
        subset of experiments are reproducible, they should state which
        ones are omitted from the script and why.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i7}
        ::: {#Ax1.I2.ix20.I1.i7.p1 .ltx_para}
        At submission time, to preserve anonymity, the authors should
        release anonymized versions (if applicable).
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix20.I1.i8}
        ::: {#Ax1.I2.ix20.I1.i8.p1 .ltx_para}
        Providing as much information as possible in supplemental
        material (appended to the paper) is recommended, but including
        URLs to data and code is permitted.
        :::
    :::
26. [[6.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i6}
    ::: {#Ax1.I2.i6.p1 .ltx_para}
    [Experimental Setting/Details]{#Ax1.I2.i6.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
27. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix21}
    ::: {#Ax1.I2.ix21.p1 .ltx_para}
    Question: Does the paper specify all the training and test details
    (e.g., data splits, hyperparameters, how they were chosen, type of
    optimizer, etc.) necessary to understand the results?
    :::
28. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix22}
    ::: {#Ax1.I2.ix22.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix22.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
29. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix23}
    ::: {#Ax1.I2.ix23.p1 .ltx_para}
    Justification: Yes training and test details are mentioned.
    :::
30. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix24}
    ::: {#Ax1.I2.ix24.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix24.I1.i1}
        ::: {#Ax1.I2.ix24.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not include experiments.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix24.I1.i2}
        ::: {#Ax1.I2.ix24.I1.i2.p1 .ltx_para}
        The experimental setting should be presented in the core of the
        paper to a level of detail that is necessary to appreciate the
        results and make sense of them.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix24.I1.i3}
        ::: {#Ax1.I2.ix24.I1.i3.p1 .ltx_para}
        The full details can be provided either with the code, in
        appendix, or as supplemental material.
        :::
    :::
31. [[7.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i7}
    ::: {#Ax1.I2.i7.p1 .ltx_para}
    [Experiment Statistical Significance]{#Ax1.I2.i7.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
32. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix25}
    ::: {#Ax1.I2.ix25.p1 .ltx_para}
    Question: Does the paper report error bars suitably and correctly
    defined or other appropriate information about the statistical
    significance of the experiments?
    :::
33. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix26}
    ::: {#Ax1.I2.ix26.p1 .ltx_para}
    Answer: [\[No\] ]{#Ax1.I2.ix26.p1.1.1 .ltx_text
    style="color:#FF8000;"}
    :::
34. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix27}
    ::: {#Ax1.I2.ix27.p1 .ltx_para}
    Justification: Error bars are not included in our experiment results
    due to the high computational cost.
    :::
35. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28}
    ::: {#Ax1.I2.ix28.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i1}
        ::: {#Ax1.I2.ix28.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not include experiments.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i2}
        ::: {#Ax1.I2.ix28.I1.i2.p1 .ltx_para}
        The authors should answer \"Yes\" if the results are accompanied
        by error bars, confidence intervals, or statistical significance
        tests, at least for the experiments that support the main claims
        of the paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i3}
        ::: {#Ax1.I2.ix28.I1.i3.p1 .ltx_para}
        The factors of variability that the error bars are capturing
        should be clearly stated (for example, train/test split,
        initialization, random drawing of some parameter, or overall run
        with given experimental conditions).
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i4}
        ::: {#Ax1.I2.ix28.I1.i4.p1 .ltx_para}
        The method for calculating the error bars should be explained
        (closed form formula, call to a library function, bootstrap,
        etc.)
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i5}
        ::: {#Ax1.I2.ix28.I1.i5.p1 .ltx_para}
        The assumptions made should be given (e.g., Normally distributed
        errors).
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i6}
        ::: {#Ax1.I2.ix28.I1.i6.p1 .ltx_para}
        It should be clear whether the error bar is the standard
        deviation or the standard error of the mean.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i7}
        ::: {#Ax1.I2.ix28.I1.i7.p1 .ltx_para}
        It is OK to report 1-sigma error bars, but one should state it.
        The authors should preferably report a 2-sigma error bar than
        state that they have a 96% CI, if the hypothesis of Normality of
        errors is not verified.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i8}
        ::: {#Ax1.I2.ix28.I1.i8.p1 .ltx_para}
        For asymmetric distributions, the authors should be careful not
        to show in tables or figures symmetric error bars that would
        yield results that are out of range (e.g. negative error rates).
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix28.I1.i9}
        ::: {#Ax1.I2.ix28.I1.i9.p1 .ltx_para}
        If error bars are reported in tables or plots, The authors
        should explain in the text how they were calculated and
        reference the corresponding figures or tables in the text.
        :::
    :::
36. [[8.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i8}
    ::: {#Ax1.I2.i8.p1 .ltx_para}
    [Experiments Compute Resources]{#Ax1.I2.i8.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
37. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix29}
    ::: {#Ax1.I2.ix29.p1 .ltx_para}
    Question: For each experiment, does the paper provide sufficient
    information on the computer resources (type of compute workers,
    memory, time of execution) needed to reproduce the experiments?
    :::
38. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix30}
    ::: {#Ax1.I2.ix30.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix30.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
39. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix31}
    ::: {#Ax1.I2.ix31.p1 .ltx_para}
    Justification: We provide the information of the compute resources
    we used in the Appendix.
    :::
40. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix32}
    ::: {#Ax1.I2.ix32.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix32.I1.i1}
        ::: {#Ax1.I2.ix32.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not include experiments.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix32.I1.i2}
        ::: {#Ax1.I2.ix32.I1.i2.p1 .ltx_para}
        The paper should indicate the type of compute workers CPU or
        GPU, internal cluster, or cloud provider, including relevant
        memory and storage.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix32.I1.i3}
        ::: {#Ax1.I2.ix32.I1.i3.p1 .ltx_para}
        The paper should provide the amount of compute required for each
        of the individual experimental runs as well as estimate the
        total compute.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix32.I1.i4}
        ::: {#Ax1.I2.ix32.I1.i4.p1 .ltx_para}
        The paper should disclose whether the full research project
        required more compute than the experiments reported in the paper
        (e.g., preliminary or failed experiments that didn't make it
        into the paper).
        :::
    :::
41. [[9.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i9}
    ::: {#Ax1.I2.i9.p1 .ltx_para}
    [Code Of Ethics]{#Ax1.I2.i9.p1.1.1 .ltx_text .ltx_font_bold}
    :::
42. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix33}
    ::: {#Ax1.I2.ix33.p1 .ltx_para}
    Question: Does the research conducted in the paper conform, in every
    respect, with the NeurIPS Code of Ethics
    <https://neurips.cc/public/EthicsGuidelines>?
    :::
43. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix34}
    ::: {#Ax1.I2.ix34.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix34.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
44. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix35}
    ::: {#Ax1.I2.ix35.p1 .ltx_para}
    Justification: Yes the research conform NeurIPS Code of Ethics.
    :::
45. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix36}
    ::: {#Ax1.I2.ix36.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix36.I1.i1}
        ::: {#Ax1.I2.ix36.I1.i1.p1 .ltx_para}
        The answer NA means that the authors have not reviewed the
        NeurIPS Code of Ethics.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix36.I1.i2}
        ::: {#Ax1.I2.ix36.I1.i2.p1 .ltx_para}
        If the authors answer No, they should explain the special
        circumstances that require a deviation from the Code of Ethics.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix36.I1.i3}
        ::: {#Ax1.I2.ix36.I1.i3.p1 .ltx_para}
        The authors should make sure to preserve anonymity (e.g., if
        there is a special consideration due to laws or regulations in
        their jurisdiction).
        :::
    :::
46. [[10.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i10}
    ::: {#Ax1.I2.i10.p1 .ltx_para}
    [Broader Impacts]{#Ax1.I2.i10.p1.1.1 .ltx_text .ltx_font_bold}
    :::
47. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix37}
    ::: {#Ax1.I2.ix37.p1 .ltx_para}
    Question: Does the paper discuss both potential positive societal
    impacts and negative societal impacts of the work performed?
    :::
48. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix38}
    ::: {#Ax1.I2.ix38.p1 .ltx_para}
    Answer: [\[N/A\] ]{#Ax1.I2.ix38.p1.1.1 .ltx_text
    style="color:#808080;"}
    :::
49. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix39}
    ::: {#Ax1.I2.ix39.p1 .ltx_para}
    Justification: This work primarily focuses on foundational research
    in algorithm improvement and, as such, does not have a direct
    societal impact.
    :::
50. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40}
    ::: {#Ax1.I2.ix40.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i1}
        ::: {#Ax1.I2.ix40.I1.i1.p1 .ltx_para}
        The answer NA means that there is no societal impact of the work
        performed.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i2}
        ::: {#Ax1.I2.ix40.I1.i2.p1 .ltx_para}
        If the authors answer NA or No, they should explain why their
        work has no societal impact or why the paper does not address
        societal impact.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i3}
        ::: {#Ax1.I2.ix40.I1.i3.p1 .ltx_para}
        Examples of negative societal impacts include potential
        malicious or unintended uses (e.g., disinformation, generating
        fake profiles, surveillance), fairness considerations (e.g.,
        deployment of technologies that could make decisions that
        unfairly impact specific groups), privacy considerations, and
        security considerations.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i4}
        ::: {#Ax1.I2.ix40.I1.i4.p1 .ltx_para}
        The conference expects that many papers will be foundational
        research and not tied to particular applications, let alone
        deployments. However, if there is a direct path to any negative
        applications, the authors should point it out. For example, it
        is legitimate to point out that an improvement in the quality of
        generative models could be used to generate deepfakes for
        disinformation. On the other hand, it is not needed to point out
        that a generic algorithm for optimizing neural networks could
        enable people to train models that generate Deepfakes faster.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i5}
        ::: {#Ax1.I2.ix40.I1.i5.p1 .ltx_para}
        The authors should consider possible harms that could arise when
        the technology is being used as intended and functioning
        correctly, harms that could arise when the technology is being
        used as intended but gives incorrect results, and harms
        following from (intentional or unintentional) misuse of the
        technology.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix40.I1.i6}
        ::: {#Ax1.I2.ix40.I1.i6.p1 .ltx_para}
        If there are negative societal impacts, the authors could also
        discuss possible mitigation strategies (e.g., gated release of
        models, providing defenses in addition to attacks, mechanisms
        for monitoring misuse, mechanisms to monitor how a system learns
        from feedback over time, improving the efficiency and
        accessibility of ML).
        :::
    :::
51. [[11.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i11}
    ::: {#Ax1.I2.i11.p1 .ltx_para}
    [Safeguards]{#Ax1.I2.i11.p1.1.1 .ltx_text .ltx_font_bold}
    :::
52. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix41}
    ::: {#Ax1.I2.ix41.p1 .ltx_para}
    Question: Does the paper describe safeguards that have been put in
    place for responsible release of data or models that have a high
    risk for misuse (e.g., pretrained language models, image generators,
    or scraped datasets)?
    :::
53. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix42}
    ::: {#Ax1.I2.ix42.p1 .ltx_para}
    Answer: [\[N/A\] ]{#Ax1.I2.ix42.p1.1.1 .ltx_text
    style="color:#808080;"}
    :::
54. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix43}
    ::: {#Ax1.I2.ix43.p1 .ltx_para}
    Justification: The paper has no such risks.
    :::
55. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix44}
    ::: {#Ax1.I2.ix44.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix44.I1.i1}
        ::: {#Ax1.I2.ix44.I1.i1.p1 .ltx_para}
        The answer NA means that the paper poses no such risks.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix44.I1.i2}
        ::: {#Ax1.I2.ix44.I1.i2.p1 .ltx_para}
        Released models that have a high risk for misuse or dual-use
        should be released with necessary safeguards to allow for
        controlled use of the model, for example by requiring that users
        adhere to usage guidelines or restrictions to access the model
        or implementing safety filters.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix44.I1.i3}
        ::: {#Ax1.I2.ix44.I1.i3.p1 .ltx_para}
        Datasets that have been scraped from the Internet could pose
        safety risks. The authors should describe how they avoided
        releasing unsafe images.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix44.I1.i4}
        ::: {#Ax1.I2.ix44.I1.i4.p1 .ltx_para}
        We recognize that providing effective safeguards is challenging,
        and many papers do not require this, but we encourage authors to
        take this into account and make a best faith effort.
        :::
    :::
56. [[12.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i12}
    ::: {#Ax1.I2.i12.p1 .ltx_para}
    [Licenses for existing assets]{#Ax1.I2.i12.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
57. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix45}
    ::: {#Ax1.I2.ix45.p1 .ltx_para}
    Question: Are the creators or original owners of assets (e.g., code,
    data, models), used in the paper, properly credited and are the
    license and terms of use explicitly mentioned and properly
    respected?
    :::
58. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix46}
    ::: {#Ax1.I2.ix46.p1 .ltx_para}
    Answer: [\[Yes\] ]{#Ax1.I2.ix46.p1.1.1 .ltx_text
    style="color:#0000FF;"}
    :::
59. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix47}
    ::: {#Ax1.I2.ix47.p1 .ltx_para}
    Justification: The datasets and models used in this paper are
    properly cited.
    :::
60. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48}
    ::: {#Ax1.I2.ix48.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i1}
        ::: {#Ax1.I2.ix48.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not use existing assets.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i2}
        ::: {#Ax1.I2.ix48.I1.i2.p1 .ltx_para}
        The authors should cite the original paper that produced the
        code package or dataset.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i3}
        ::: {#Ax1.I2.ix48.I1.i3.p1 .ltx_para}
        The authors should state which version of the asset is used and,
        if possible, include a URL.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i4}
        ::: {#Ax1.I2.ix48.I1.i4.p1 .ltx_para}
        The name of the license (e.g., CC-BY 4.0) should be included for
        each asset.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i5}
        ::: {#Ax1.I2.ix48.I1.i5.p1 .ltx_para}
        For scraped data from a particular source (e.g., website), the
        copyright and terms of service of that source should be
        provided.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i6}
        ::: {#Ax1.I2.ix48.I1.i6.p1 .ltx_para}
        If assets are released, the license, copyright information, and
        terms of use in the package should be provided. For popular
        datasets,
        [paperswithcode.com/datasets](paperswithcode.com/datasets){.ltx_ref
        .ltx_url .ltx_font_typewriter} has curated licenses for some
        datasets. Their licensing guide can help determine the license
        of a dataset.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i7}
        ::: {#Ax1.I2.ix48.I1.i7.p1 .ltx_para}
        For existing datasets that are re-packaged, both the original
        license and the license of the derived asset (if it has changed)
        should be provided.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix48.I1.i8}
        ::: {#Ax1.I2.ix48.I1.i8.p1 .ltx_para}
        If this information is not available online, the authors are
        encouraged to reach out to the asset's creators.
        :::
    :::
61. [[13.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i13}
    ::: {#Ax1.I2.i13.p1 .ltx_para}
    [New Assets]{#Ax1.I2.i13.p1.1.1 .ltx_text .ltx_font_bold}
    :::
62. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix49}
    ::: {#Ax1.I2.ix49.p1 .ltx_para}
    Question: Are new assets introduced in the paper well documented and
    is the documentation provided alongside the assets?
    :::
63. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix50}
    ::: {#Ax1.I2.ix50.p1 .ltx_para}
    Answer: [\[N/A\] ]{#Ax1.I2.ix50.p1.1.1 .ltx_text
    style="color:#808080;"}
    :::
64. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix51}
    ::: {#Ax1.I2.ix51.p1 .ltx_para}
    Justification: We didn't release new assets.
    :::
65. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix52}
    ::: {#Ax1.I2.ix52.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix52.I1.i1}
        ::: {#Ax1.I2.ix52.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not release new assets.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix52.I1.i2}
        ::: {#Ax1.I2.ix52.I1.i2.p1 .ltx_para}
        Researchers should communicate the details of the
        dataset/code/model as part of their submissions via structured
        templates. This includes details about training, license,
        limitations, etc.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix52.I1.i3}
        ::: {#Ax1.I2.ix52.I1.i3.p1 .ltx_para}
        The paper should discuss whether and how consent was obtained
        from people whose asset is used.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix52.I1.i4}
        ::: {#Ax1.I2.ix52.I1.i4.p1 .ltx_para}
        At submission time, remember to anonymize your assets (if
        applicable). You can either create an anonymized URL or include
        an anonymized zip file.
        :::
    :::
66. [[14.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i14}
    ::: {#Ax1.I2.i14.p1 .ltx_para}
    [Crowdsourcing and Research with Human Subjects]{#Ax1.I2.i14.p1.1.1
    .ltx_text .ltx_font_bold}
    :::
67. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix53}
    ::: {#Ax1.I2.ix53.p1 .ltx_para}
    Question: For crowdsourcing experiments and research with human
    subjects, does the paper include the full text of instructions given
    to participants and screenshots, if applicable, as well as details
    about compensation (if any)?
    :::
68. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix54}
    ::: {#Ax1.I2.ix54.p1 .ltx_para}
    Answer: [\[N/A\] ]{#Ax1.I2.ix54.p1.1.1 .ltx_text
    style="color:#808080;"}
    :::
69. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix55}
    ::: {#Ax1.I2.ix55.p1 .ltx_para}
    Justification: This paper does not involve crowdsourcing nor
    research with human subjects.
    :::
70. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix56}
    ::: {#Ax1.I2.ix56.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix56.I1.i1}
        ::: {#Ax1.I2.ix56.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not involve
        crowdsourcing nor research with human subjects.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix56.I1.i2}
        ::: {#Ax1.I2.ix56.I1.i2.p1 .ltx_para}
        Including this information in the supplemental material is fine,
        but if the main contribution of the paper involves human
        subjects, then as much detail as possible should be included in
        the main paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix56.I1.i3}
        ::: {#Ax1.I2.ix56.I1.i3.p1 .ltx_para}
        According to the NeurIPS Code of Ethics, workers involved in
        data collection, curation, or other labor should be paid at
        least the minimum wage in the country of the data collector.
        :::
    :::
71. [[15.]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.i15}
    ::: {#Ax1.I2.i15.p1 .ltx_para}
    [Institutional Review Board (IRB) Approvals or Equivalent for
    Research with Human Subjects]{#Ax1.I2.i15.p1.1.1 .ltx_text
    .ltx_font_bold}
    :::
72. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix57}
    ::: {#Ax1.I2.ix57.p1 .ltx_para}
    Question: Does the paper describe potential risks incurred by study
    participants, whether such risks were disclosed to the subjects, and
    whether Institutional Review Board (IRB) approvals (or an equivalent
    approval/review based on the requirements of your country or
    institution) were obtained?
    :::
73. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix58}
    ::: {#Ax1.I2.ix58.p1 .ltx_para}
    Answer: [\[N/A\] ]{#Ax1.I2.ix58.p1.1.1 .ltx_text
    style="color:#808080;"}
    :::
74. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix59}
    ::: {#Ax1.I2.ix59.p1 .ltx_para}
    Justification: This paper does not involve crowdsourcing nor
    research with human subjects.
    :::
75. [[]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix60}
    ::: {#Ax1.I2.ix60.p1 .ltx_para}
    Guidelines:

    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix60.I1.i1}
        ::: {#Ax1.I2.ix60.I1.i1.p1 .ltx_para}
        The answer NA means that the paper does not involve
        crowdsourcing nor research with human subjects.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix60.I1.i2}
        ::: {#Ax1.I2.ix60.I1.i2.p1 .ltx_para}
        Depending on the country in which research is conducted, IRB
        approval (or equivalent) may be required for any human subjects
        research. If you obtained IRB approval, you should clearly state
        this in the paper.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix60.I1.i3}
        ::: {#Ax1.I2.ix60.I1.i3.p1 .ltx_para}
        We recognize that the procedures for this may vary significantly
        between institutions and locations, and we expect authors to
        adhere to the NeurIPS Code of Ethics and the guidelines for
        their institution.
        :::
    -   [[•]{.ltx_tag .ltx_tag_item}]{#Ax1.I2.ix60.I1.i4}
        ::: {#Ax1.I2.ix60.I1.i4.p1 .ltx_para}
        For initial submissions, do not include any information that
        would break anonymity (if applicable), such as the institution
        conducting the review.
        :::
    :::
:::

::: {.ltx_pagination .ltx_role_newpage}
:::
:::
:::

::: {.ltx_page_logo}
Generated on Tue Dec 10 18:17:52 2024 by [[L[a]{.ltx_font_smallcaps
style="position:relative; bottom:2.2pt;"}T[e]{.ltx_font_smallcaps
style="font-size:120%;position:relative; bottom:-0.2ex;"}]{style="letter-spacing:-0.2em; margin-right:0.1em;"}[XML]{style="font-size:90%; position:relative; bottom:-0.2ex;"}![Mascot
Sammy](lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/){.ltx_LaTeXML_logo}
:::
:::
