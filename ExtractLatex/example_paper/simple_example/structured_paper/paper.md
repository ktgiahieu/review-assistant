# Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing

## Abstract

Despite the impressive capabilities of Large Language Models (LLMs) on various tasks, they still struggle with scenarios that involves complex reasoning and planning. Self-correction and self-learning emerge as viable solutions, employing strategies that allow LLMs to refine their outputs and learn from self-assessed rewards. Yet, the efficacy of LLMs in self-refining its response, particularly in complex reasoning and planning task, remains dubious. In this paper, we introduce for the self-improvements of LLMs, which integrates Monte Carlo Tree Search (MCTS) with LLMs to establish a self-improving loop, thereby enhancing the capabilities of LLMs without additional annotations. Drawing inspiration from the success of AlphaGo, addresses the unique challenges of combining MCTS with LLM for self-improvement, including data scarcity, the vastness search spaces of language tasks, and the subjective nature of feedback in language tasks. is comprised of prompt synthesis component, an efficient MCTS approach tailored for language tasks, and a trio of critic models for precise feedback. Our experimental results in mathematical reasoning tasks demonstrate that significantly enhances the performance of LLMs without additional annotations, showing the potential for self-improvement in LLMs. The code is available at <https://github.com/YeTianJHU/AlphaLLM>.

# Introduction

Hello

<div class="minipage">

<div class="center">

<table>
<tbody>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td colspan="4" style="text-align: center;">Accuracies (%)</td>
</tr>
<tr>
<td style="text-align: left;">Method</td>
<td style="text-align: center;">FLOPs (G)</td>
<td style="text-align: center;">Joint</td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;">Avg</td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><span>0.68</span></td>
<td style="text-align: center;"><span>48.2</span></td>
<td style="text-align: center;"><span>97.0</span></td>
<td style="text-align: center;"><span>45.1</span></td>
<td style="text-align: center;"><span>71.0</span></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"><span>0.68</span></td>
<td style="text-align: center;"><span>48.4</span></td>
<td style="text-align: center;"><span>49.1</span></td>
<td style="text-align: center;"><span>96.1</span></td>
<td style="text-align: center;"><span>72.6</span></td>
</tr>
<tr>
<td style="text-align: left;">W. Avg <span>(Eq. <a href="#eq:wavg" data-reference-type="ref" data-reference="eq:wavg">[eq:wavg]</a>)</span></td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;"><span>43.0</span></td>
<td style="text-align: center;"><span>54.1</span></td>
<td style="text-align: center;"><span>67.5</span></td>
<td style="text-align: center;"><span>60.8</span></td>
</tr>
<tr>
<td style="text-align: left;">Git Re-Basin<span class="math inline"><sup>‡</sup></span></td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;"><span>46.2</span></td>
<td style="text-align: center;"><span>76.8</span></td>
<td style="text-align: center;"><span>82.7</span></td>
<td style="text-align: center;"><span>79.8</span></td>
</tr>
<tr>
<td style="text-align: left;">Permute <span>(Eq. <a href="#eq:rebasin" data-reference-type="ref" data-reference="eq:rebasin">[eq:rebasin]</a>)</span></td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;"><span>58.4</span></td>
<td style="text-align: center;"><span>86.6</span></td>
<td style="text-align: center;"><span>87.4</span></td>
<td style="text-align: center;"><span>87.4</span></td>
</tr>
<tr>
<td style="text-align: left;"><span><strong>ZipIt!</strong></span><span class="math inline"><sub>20/20</sub></span></td>
<td style="text-align: center;">0.68</td>
<td style="text-align: center;"><strong>79.1</strong></td>
<td style="text-align: center;"><strong>92.9</strong></td>
<td style="text-align: center;"><strong>91.2</strong></td>
<td style="text-align: center;"><strong>92.1</strong></td>
</tr>
<tr>
<td style="text-align: left;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: left;"><span><strong>ZipIt!</strong></span><span class="math inline"><sub>13/20</sub></span></td>
<td style="text-align: center;">0.91</td>
<td style="text-align: center;"><strong>83.8</strong></td>
<td style="text-align: center;"><strong>95.1</strong></td>
<td style="text-align: center;"><strong>94.1</strong></td>
<td style="text-align: center;"><strong>94.6</strong></td>
</tr>
</tbody>
</table>

</div>

</div>

# References

<div class="thebibliography">

David Abel, Dilip Arumugam, Lucas Lehnert, and Michael Littman State abstractions for lifelong reinforcement learning In *International Conference on Machine Learning*, pp. 10–19. PMLR, 2018. **Abstract:** In lifelong reinforcement learning, agents must effectively transfer knowledge across tasks while simultaneously addressing exploration, credit as- signment, and generalization. State abstraction can help overcome these hurdles by compressing the representation used by an agent, thereby re- ducing the computational and statistical burdens of learning. To this end, we here develop theory to compute and use state abstractions in lifelong reinforcement learning. We introduce two new classes of abstractions: (1) transitive state abstrac- tions, whose optimal form can be computed efﬁ- ciently, and (2) PAC state abstractions, which are guaranteed to hold with respect to a distribution of tasks. We show that the joint family of transi- tive PAC abstractions can be acquired efﬁciently, preserve near optimal-behavior, and experimen- tally reduce sample complexity in simple domains, thereby yielding a family of desirable abstractions for use in lifelong reinforcement learning. Along with these positive results, we show that there are pathological cases where state abstractions can negatively impact performance. (@abel2018state)

</div>

[^1]: Equal Contribution; †Corresponding Author
