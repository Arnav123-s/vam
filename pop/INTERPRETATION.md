# Interpretation track: testing the owner's original ideas

This directory is a **separate track** from the main VAM build. Its premise, from the owner
(2026-11-04): the published implementation may have **mis-extracted** several of the original
ideas. Where something doesn't work, the failure may sit in the *interpretation* or the
*implementation* — not necessarily the idea.

Methodology: for each hypothesis below, build the **most faithful minimal version of the
owner's original formulation** (quoted from F/T/N and the sera-research-raw archive), test it
against the extracted version, and let the numbers decide which interpretation is right.
A failed interpretation is logged with *why*, not tuned until it passes.

## H1. Population-scale density matrices (Page limit per node)

**Owner's formulation** (F spec, verbatim): *"we have millions or billions of vectors"*;
*"a density matrix can become an overlying layer — like a topological map or a legend on a map."*
N-L2: the Page limit dictates *"how many vectors we need to span one equation"* — the basis
**per node**, pruned past the Page turnaround. N-L5: *"Multiple density matrices assemble into
a multi-scale MPO"* with blankets *"between vectors, between matrices, and across the tensors."*

**What was extracted instead:** Page-rank of **one** byte-Hankel matrix; one coherent mixture;
nested context tables. The population was never built.

**Testable claim:** a population of small density matrices — one per context node, each with its
basis sized by the Marchenko–Pastur edge on its *local* statistics, insulated by blankets,
summarized by an overlying "legend" layer — outperforms a single flat model on multi-scale
structure, **and** its organization stabilizes as data grows.

**Phase 1** (`runs/phase1_regime_flip.py`): 8-node population vs flat order-2 baseline on a
regime-flip source (slow regime × fast order-2, with the fast pattern *reversed* per regime —
provably ambiguous to any flat context model). Metrics: prequential bits/bit, per-node Page
rank, prediction-drift stability.

## H2. Governor as criticality, not cache management

**Owner's formulation** (N-L6, verbatim): *"keep the engine operating permanently at the edge of
criticality"* — a global entropy monitor against Bekenstein–Hawking capacity; *"halts volume
expansion, purges micro-noise through boundary dissipative sinks, and **radiates completed
invariants outward**."*

**What was extracted instead:** per-level table occupancy, doorkeepers, verify thresholds,
eviction protection. The outward radiation was never built; the halt had a bug (inert).

**Testable claim (Phase 2):** a global entropy monitor that (a) tracks total system entropy vs
capacity, (b) purges noise past the Page point, (c) *emits completed invariants outward* as new
stable structure — keeps a growing population at criticality measurably better than a
cache-manager governor. Metrics: entropy vs capacity over time, invariant-emission rate,
prediction quality at scale.

## H3. Stability curriculum (teaching order)

**Owner's formulation** (T12.3 part ii): ordered information must *"change and stabilize the
density–weight organization."* F spec: *"The lowest bit layer serves as an immutable,
hardware-native bedrock... the layers above it actively perceive, adapt, and mix infinitely."*
As layers and numbers increase, the system gets **more stable and better organized, not random.**

**What was extracted instead:** transfer (A-then-B vs B-only bits/byte). The wrong criterion.

**Testable claim (Phase 1, measured alongside H1):** with the base layer frozen and upper layers
plastic, the population's organization **stabilizes** as data grows — per-node prediction drift
falls, Page ranks settle, node assignments stop churning. Metrics: drift(t), rank-settle time,
churn rate. Teaching order = curriculum (same thing): staged input is the mechanism.

## H4. Tensors as the knowledge substrate

**Owner's formulation** (F spec): *"Coarse-grained macroscopic concepts entangle with
fine-grained microscopic features using unitary disentanglers and isometric coarse-graining
operators"* — MERA-like; *"A density matrix at Layer X can represent higher-level density
layers at Layer Y in superposition."*

**What was extracted instead:** MPO as a compression/approximation trick for the chain; no
VAM-1 result ever used a truncated run; text runs used nested tables, never the tensor network.

**Testable claim (Phase 3):** the population's nodes assemble into a tensor network where
cross-scale entanglement *is* the knowledge organization — coarse nodes disentangle shared
structure from fine nodes. Metrics: entanglement across cuts vs prediction quality; ablation
of disentanglers.

## H5. Collapse as navigation

**Owner's formulation** (F spec, verbatim): *"we scope out everything that token could fire
using Feynman, and with Newton's law, we pull out the needle from the haystack"* —
macro-scope (Feynman over the field) → Newton threads the attractor → micro-scope (Feynman on
the closest matrix) → Newton picks the minimal step → emit → ***every token that emerges
changes the entire algorithm*** via Markov transitions.

**What was extracted instead:** `hbar` sampler settings on a fixed predictor. V7 bounds
*sampling*, which is not this idea.

**Testable claim (Phase 4):** a navigation loop — scope, thread, minimal-step, emit,
transform — where each emission re-conditions the whole population (not re-enters a bit into a
fixed predictor). Metrics: multi-step task success vs the sampling loop; system-transform
measure (how much the population re-organizes per emission).

## Track rules

1. Every run is saved with its numbers; protocols are short but written before runs.
2. A failed interpretation is logged with the *reason* (extraction? implementation? idea?) —
   never tuned until it passes.
3. Nothing here touches `src/`, `experiments/`, or the main build. This track is separate.
4. The owner rules on interpretations; the numbers rule on performance.
