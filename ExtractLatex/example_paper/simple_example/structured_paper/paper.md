# Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

## Abstract

Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs. The code is available at <https://github.com/YeTianJHU/AlphaLLM>.

# Introduction

Hello

<div id="tab:length_repdiv_LLaMA2_NEFT">

|                             |               | Alpaca (*α* = 5) | Evol-Instruct (*α* = 5) | ShareGPT (*α* = 10) | OpenPlatypus (*α* = 15) |
|:----------------------------|:--------------|:----------------:|:-----------------------:|:-------------------:|:-----------------------:|
| **Character** **Lengths**   | Training data |      270.31      |         1356.43         |       1276.76       |         649.39          |
|                             | -2 7B         |      375.22      |         864.06          |       1011.28       |         1100.98         |
|                             | \+            |     1061.89      |         1403.59         |       1496.86       |         1694.26         |
| **Whitespace** **Lengths**  | -2 7B         |       60.5       |         138.99          |       161.04        |         170.41          |
|                             | \+            |      169.36      |         225.56          |       234.99        |         264.12          |
| **2-Gram** **Repetition %** | -2 7B         |       1.49       |          3.87           |        4.82         |          2.73           |
|                             | \+            |       1.72       |          3.79           |        4.58         |          3.21           |
| **Log-Diversity**           | -2 7B         |      15.97       |          10.65          |        8.40         |          9.96           |
|                             | \+            |      16.41       |          10.77          |        8.60         |          9.64           |

(**Row 1**) Avg. Character lengths of `AlpacaEval` responses from -2 models finetuned on different datasets. We also report average output length for each dataset (though we trained with max sequence length of 512). increases average length. (**Row 2**) Whitespace-tokenized lengths of generations. (**Row 3**) 2-Gram repetition rates. (**Row 4**) Log-Diversity measures.

</div>

# References

<div class="thebibliography">

David Abel, Dilip Arumugam, Lucas Lehnert, and Michael Littman State abstractions for lifelong reinforcement learning In *International Conference on Machine Learning*, pp. 10–19. PMLR, 2018. **Abstract:** In lifelong reinforcement learning, agents must effectively transfer knowledge across tasks while simultaneously addressing exploration, credit as- signment, and generalization. State abstraction can help overcome these hurdles by compressing the representation used by an agent, thereby re- ducing the computational and statistical burdens of learning. To this end, we here develop theory to compute and use state abstractions in lifelong reinforcement learning. We introduce two new classes of abstractions: (1) transitive state abstrac- tions, whose optimal form can be computed efﬁ- ciently, and (2) PAC state abstractions, which are guaranteed to hold with respect to a distribution of tasks. We show that the joint family of transi- tive PAC abstractions can be acquired efﬁciently, preserve near optimal-behavior, and experimen- tally reduce sample complexity in simple domains, thereby yielding a family of desirable abstractions for use in lifelong reinforcement learning. Along with these positive results, we show that there are pathological cases where state abstractions can negatively impact performance. (@abel2018state)

</div>

[1] Equal Contribution; Corresponding Author
