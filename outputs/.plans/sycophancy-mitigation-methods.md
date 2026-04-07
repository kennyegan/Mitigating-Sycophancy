# Comparison Plan: sycophancy-mitigation-methods

## Sources

| ID | Source | Authors | Year |
|----|--------|---------|------|
| A | **Mitigating Sycophancy in LLMs: A Mechanistic Investigation** (local project, `paper.md`) | Kenny Egan | 2026 |
| B | **Ask Don't Tell: Reducing Sycophancy in Large Language Models** (arXiv:2602.23971) | Dubois, Ududec, Summerfield, Luettgau (UK AISI) | 2026 |

## Focus

**Methodological novelty** — how each paper's approach to understanding/mitigating sycophancy differs in technique, experimental design, and what new things it contributes to the field.

## Comparison Dimensions

1. **Core methodology** — mechanistic interpretability (probes, patching, ablation, steering) vs. controlled behavioral experiments (factorial input framing)
2. **Sycophancy definition & measurement** — logit-based compliance gap vs. rubric-based LLM-as-judge scoring
3. **Causal identification strategy** — activation patching + ablation (internal circuits) vs. content-matched prompt manipulation (input framing)
4. **Mitigation approach** — inference-time (steering/ablation) → training-time (DPO) vs. input-level reframing (question conversion, perspective shift)
5. **Models studied** — open-weight (Llama-3-8B, Mistral-7B) vs. closed frontier (GPT-4o, GPT-5, Sonnet-4.5)
6. **Domain coverage** — opinion + factual + reasoning vs. subjective opinion only
7. **Key novel claims** — what each paper uniquely contributes
8. **Limitations & caveats** — generalizability, scope, threats to validity

## Expected Output Structure

1. Summary of each source
2. Comparison matrix (source × dimension)
3. Mermaid diagram: methodology architecture comparison
4. Agreement / Disagreement / Uncertainty analysis
5. Synthesis: complementarity of approaches
6. Sources section with URLs

## Deliverable

`outputs/sycophancy-mitigation-methods-comparison.md`
