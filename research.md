The review is complete and saved at `outputs/sycophancy-mech-interp-review.md`. Here's the full deliverable summary:

---

## Peer Review Complete

**Slug:** `sycophancy-mech-interp`
**Verdict:** **5 — Weak Accept (revisions required)**
**Output:** `outputs/sycophancy-mech-interp-review.md`

---

### What the review covers

**Strengths identified:**
- Neutral-transfer probe design is methodologically rigorous and the format-confound caution is a genuine contribution
- Patching-to-ablation dissociation is well-confirmed across 3 ablation tiers and 2 architectures
- Cross-architecture Mistral replication substantially strengthens generalizability claims
- DPO probe re-analysis (§5.11) is the most novel finding — mechanistically grounding what DPO does
- Statistical practices (Wilson CIs, BH correction, permutation tests) are unusually careful for a mech-interp paper

**Major issues found:**
1. **Three missing mechanistic sycophancy papers** never cited: Chen et al. 2025 (arXiv:2409.01658, path patching for sycophancy), Li et al. 2025 (arXiv:2508.02087, logit-lens + patching on Llama-3), O'Brien et al. 2026 (arXiv:2601.18939, SAEs for sycophancy)
2. **Unaddressed layer conflict** with Li et al. 2025 — they find sycophancy emerges in late layers (16–23), this paper finds it in early layers (1–5), same model
3. **"First mechanistic evidence" overstated** — Lee et al. (ICML 2024) and Yang et al. (EMNLP 2025) already did mechanistic DPO analysis for toxicity
4. **Unexplained 34/100 → 100/100 patching success jump** — potential selection bias in head identification

**Moderate issues:** Mistral 99.8% factual sycophancy needs format verification; Heimersheim & Nanda 2024 missing from sufficiency/necessity discussion; Panickssery et al. cited in text but absent from `.bib`; "Other" probe category undefined

---

### Sources inspected

- https://arxiv.org/abs/2508.02087 — Li et al. 2025 (mechanistic sycophancy, contradictory layers)
- https://arxiv.org/abs/2409.01658 — Chen et al. 2025 (pinpoint tuning, path patching)
- https://arxiv.org/abs/2601.18939 — O'Brien et al. 2026 (SAE + probe sycophancy)
- https://arxiv.org/abs/2404.15255 — Heimersheim & Nanda 2024 (activation patching interpretation)
- https://proceedings.mlr.press/v235/rushing24a.html — Rushing & Nanda 2024 (self-repair, ICML)
- https://arxiv.org/abs/2401.01967 — Lee et al. 2024 (mechanistic DPO + toxicity, ICML oral)
- https://arxiv.org/abs/2411.06424 — Yang et al. 2025 (DPO + toxicity, EMNLP)
- https://arxiv.org/abs/2312.06681 — Panickssery et al. 2023 (steering vectors for sycophancy)
- https://arxiv.org/abs/2307.15771 — McGrath et al. 2023 (Hydra Effect, self-repair)