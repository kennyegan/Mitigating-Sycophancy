# Provenance: patching-ablation-dissociation-brief

## Search Strategy
- 4 alphaXiv semantic/keyword searches across sufficiency/necessity, circuit redundancy, self-repair, and backup heads
- 3 web searches covering activation patching methodology, circuit redundancy limitations, and current discourse
- 10 full paper reports retrieved via alpha_get_paper
- 1 additional semantic search for backup/compensation literature

## Papers Read in Full (AI Report)
1. 2307.15771 — The Hydra Effect (McGrath et al., DeepMind)
2. 2402.15390 — Explorations of Self-Repair (Rushing & Nanda, UT Austin)
3. 2407.08734 — Circuit Faithfulness Metrics Not Robust (Miller et al., FAR AI)
4. 2404.15255 — How to Use and Interpret Activation Patching (Heimersheim & Nanda)
5. 2309.16042 — Best Practices of Activation Patching (Zhang & Nanda, UC Berkeley)
6. 2407.15166 — Adversarial Circuit Evaluation (uit de Bos & Garriga-Alonso, FAR AI)
7. 2403.17806 — Have Faith in Faithfulness (Hanna et al., Amsterdam/Technion)
8. 2411.16105 — Adaptive Circuit Behavior (Nainani et al., UMass)
9. 2602.16698 — Causality is Key (Joshi et al., Mila/BU)
10. Local project paper.md (Egan, 2026)

## Papers Identified but Not Read in Full
- 2311.17030 — Interpretability Illusion for Subspace Activation Patching
- 2602.16740 — Quantifying LLM Attention-Head Stability
- 2602.06801 — Non-Identifiability of Steering Vectors
- 1905.10650 — Are Sixteen Heads Really Better than One
- 2304.14997 — Towards Automated Circuit Discovery (Conmy et al.)

## Key Claims Verification
| Claim in Brief | Source | Verified How |
|----------------|--------|--------------|
| Hydra effect: compensation ~70% of direct effect | McGrath et al. 2023 | Paper report: "slope between direct effect and compensatory response is less than one (e.g., around 0.7)" |
| LayerNorm accounts for ~30% of self-repair | Rushing & Nanda 2024 | Paper report: "approximately 30% of a head's direct effect on average" |
| 53.5% of claims have rung mismatch | Joshi et al. 2026 | Paper report: "Approximately half (53.5%) of the claims exhibited a gap score greater than 0" |
| Top-10 ablation null for sycophancy: +0.5pp Llama-3, +1.0pp Mistral | Egan 2026 | Direct from paper.md Sections 5.7 and 5.10 |
| IOI circuit: 100% node overlap on prompt variants | Nainani et al. 2024 | Paper report Table 2 |

## Limitations
- Focused on transformer LLMs; no coverage of vision models or other architectures
- Self-repair quantification (30% LayerNorm, 70% total) from specific models/tasks; may not generalize to all settings
- Did not read Optimal Ablation (NeurIPS 2024 spotlight) which may have additional formal results
- The "widespread" claim for dissociation is supported by multiple papers but most use narrow task distributions (IOI, factual recall); only Egan (2026) and Rushing & Nanda (2024) test on broader distributions
