# Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

## Abstract

Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs. The code is available at <https://github.com/YeTianJHU/AlphaLLM>.

# Introduction

Hello

## Some Section

Some text before the wrapped table. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam convallis libero in consequat.

<span id="tab:option_critic_example" label="tab:option_critic_example">\[tab:option\_critic\_example\]</span>

<div id="tab:option_critic_example">

| Method       | `Acc` | `#Rollout` |
|:-------------|:-----:|:----------:|
| emcts        | 45.4  |    148     |
| w/o option   | 44.1  |    198     |
| orm w/o tool | 38.8  |    201     |

Comparison results of options formulation on MATH (example).

</div>

<div id="tab:demos">

| Task     | *π*<sub>base</sub><sup>strong</sup> | *π*<sub>base</sub><sup>weak</sup> | *π*<sup>strong</sup>                                    | High-quality demonstrations |
|:---------|:------------------------------------|:----------------------------------|:--------------------------------------------------------|:----------------------------|
| Code     | Deepseek-7B-Coder                   | Pythia-1B                         | *π*<sub>base</sub><sup>strong</sup>, SFT on GPT-4 T=1   | GPT-4 T=0                   |
| MATH     | Deepseek-7B-Math                    | Pythia-1B                         | *π*<sub>base</sub><sup>strong</sup>                     | GPT-4 T=0                   |
| Critique | Deepseek-7B-Coder                   | Pythia-1B                         | *π*<sub>base</sub><sup>strong</sup>, SFT + Iterated DPO | Reference critiques         |
| MMLU     | Mistral-7B                          | Pythia-7B                         | Ground-truth labels                                     | Ground-truth labels         |

Summary of the models and policies used for each task. We study the sensitivity of the results to these choices in Appendix <a href="#sec:invariance" data-reference-type="ref" data-reference="sec:invariance">[sec:invariance]</a>. We rely on pre-trained models from the Deepseek  and Pythia  families, as well as Mistral-7B and GPT-4 .

</div>

<div id="tab:option">

| `Search Node`  |                                                                              `Example`                                                                               |    `Termination`     |
|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|
|  Token-level   |            *y*<sub>0</sub> → *y*<sub>1</sub> → *y*<sub>2</sub> → *y*<sub>3</sub> → *y*<sub>5</sub> → *y*<sub>6</sub> → *y*<sub>7</sub> → *y*<sub>8</sub>             |        token         |
| Sentence-level |   *y*<sub>0</sub>*y*<sub>1</sub>*y*<sub>2</sub>  → *y*<sub>4</sub>*y*<sub>5</sub>*y*<sub>6</sub>  → *y*<sub>7</sub>*y*<sub>8</sub>*y*<sub>9</sub>*y*<sub>10</sub>    |       new line       |
|  Option-level  | *y*<sub>0</sub>  → *y*<sub>1</sub>*y*<sub>2</sub>  → *y*<sub>4</sub>*y*<sub>5</sub>*y*<sub>6</sub> *y*<sub>7</sub>*y*<sub>8</sub>*y*<sub>9</sub>  → *y*<sub>10</sub> | termination function |

Comparative illustration of token-level, sentence-level, and option-level MCTS search nodes. *y* denotes a token sampled from the policy model. The arrow → represents the transition from one search node to the subsequent node within the search process.

</div>

Some text after the wrapped table. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

## Another Section with a standard table

<div id="table:ablation_critic_example">

| Model          | Precision | Recall |  ECE  |
|:---------------|:---------:|:------:|:-----:|
| Value Function |   0.82    |  0.79  | 0.032 |
| prm            |   0.62    |  0.90  | 0.375 |

Performance comparison of the Value Function model and on the GSM8K test set.

</div>

# References

<div class="thebibliography">

64 urlstyle

David Abel, Dilip Arumugam, Lucas Lehnert, and Michael Littman State abstractions for lifelong reinforcement learning In *International Conference on Machine Learning*, pp. 10–19. PMLR, 2018. **Abstract:** In lifelong reinforcement learning, agents must effectively transfer knowledge across tasks while simultaneously addressing exploration, credit as- signment, and generalization. State abstraction can help overcome these hurdles by compressing the representation used by an agent, thereby re- ducing the computational and statistical burdens of learning. To this end, we here develop theory to compute and use state abstractions in lifelong reinforcement learning. We introduce two new classes of abstractions: (1) transitive state abstrac- tions, whose optimal form can be computed efﬁ- ciently, and (2) PAC state abstractions, which are guaranteed to hold with respect to a distribution of tasks. We show that the joint family of transi- tive PAC abstractions can be acquired efﬁciently, preserve near optimal-behavior, and experimen- tally reduce sample complexity in simple domains, thereby yielding a family of desirable abstractions for use in lifelong reinforcement learning. Along with these positive results, we show that there are pathological cases where state abstractions can negatively impact performance. (@abel2018state)

</div>

[1] Equal Contribution; Corresponding Author
