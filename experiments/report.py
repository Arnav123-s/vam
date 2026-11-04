"""Generate RESULTS.md from the saved result files (experiments/PROTOCOL.md, reporting rules).

    python experiments/report.py

Every number in RESULTS.md is read from experiments/results/*.json; the pass
or fail of each pre-registered criterion is computed here, never typed.
"""

from __future__ import annotations

import json

from _common import RESULTS, ROOT

TOL = 0.01


def e1_section(d: dict) -> list[str]:
    out = ["## E1. Known sources", "",
           f"Run at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}, "
           f"2^{d['provenance']['log2_bits']} bits per source, seed {d['provenance']['seed']}.", "",
           "Loss is bits per bit over the last quarter of the stream (prequential). "
           "*Floor* is the best any fixed rule on the last k bits can do; *KT* is Bayesian counting "
           "with the same k bits; *oracle* knows the source. *Counting* is VAM-1 with the count-dependent "
           "friction of decision D10, which T does not state; *literal* is the T12.1 law as written "
           "(constant friction, dt = 0.05).", "",
           "Every window row is massless VAM-1. At zero mass the chain is an exact sliding window, so these "
           "rows measure a classical k-bit context model with one moving weight per context; they do not "
           "exercise the coherent field. Only the mass sweep does.", ""]
    crit1, crit2, literal = [], [], []
    for row in d["context_sweep"]:
        out += [f"### {row['source']}", "",
                f"Oracle {row['oracle']['last_quarter']:.4f}"
                + (f", entropy rate {row['entropy_rate']:.4f}" if row["entropy_rate"] is not None else ""), "",
                "| k | floor | KT | VAM-1 counting (D10) | VAM-1 literal |", "|---|---|---|---|---|"]
        for e in row["by_k"]:
            cons, kt, floor = e["consolidating"]["last_quarter"], e["kt"]["last_quarter"], e["window_floor"]
            out.append(f"| {e['k']} | {floor:.4f} | {kt:.4f} | {cons:.4f} | {e['constant']['last_quarter']:.4f} |")
            if e["k"] <= 8:
                crit1.append((row["source"], e["k"], cons - floor))
                literal.append((row["source"], e["k"], e["constant"]["last_quarter"] - floor))
            crit2.append((row["source"], e["k"], cons - kt))
        out.append("")
    ok1 = all(abs(x) <= TOL for *_, x in crit1)
    ok2 = all(x <= TOL for *_, x in crit2)
    worst1 = max(crit1, key=lambda r: abs(r[2]))
    worst2 = max(crit2, key=lambda r: r[2])
    above = max(literal, key=lambda r: r[2])
    below = min(literal, key=lambda r: r[2])
    ok_lit = all(abs(x) <= TOL for *_, x in literal)
    out += ["### Pre-registered criteria", "",
            "Both were registered for the counting law, then called \"consolidating\" (see the post-audit "
            "revision in `PROTOCOL.md`). Because that law is a counting rule, criterion 2 is close to true by "
            "construction.", "",
            f"1. Counting law within {TOL} of the window floor for k ≤ 8, every source: "
            f"**{'pass' if ok1 else 'fail'}** (largest gap {worst1[2]:+.4f}, {worst1[0]}, k = {worst1[1]}).",
            f"2. Counting law within {TOL} of KT or better, every source and k: "
            f"**{'pass' if ok2 else 'fail'}** (largest excess {worst2[2]:+.4f}, {worst2[0]}, k = {worst2[1]}).",
            "", "Not pre-registered, added after the audit: the literal T12.1 law measured against criterion 1 "
            f"would {'pass' if ok_lit else 'fail'}. Its largest excess over the floor is {above[2]:+.4f} "
            f"({above[0]}, k = {above[1]}), and it falls furthest below the floor by {below[2]:+.4f} "
            f"({below[0]}, k = {below[1]}). With a constant rate its weights keep moving with recent "
            "history: it settles above the floor where learning has converged, and beats the fixed-rule "
            "floor where recent history carries information the window does not.",
            "", "### Mass sweep (pointer engine, c = 6 cells, k = 4, counting law, full loop)", "",
            "Each bit is followed by an emission at the active port and its re-entry (T14 steps 11-12); the "
            "emitted bits travel back along the chain, and the mass mixes them into the context.", "",
            "| source | θ | VAM-1 loss | fixed-rule floor | oracle |", "|---|---|---|---|---|"]
    for r in d["mass_sweep"]:
        out.append(f"| {r['source']} | {r['theta']} | {r['vam1']['last_quarter']:.4f} | "
                   f"{r['window_floor']:.4f} | {r['oracle']['last_quarter']:.4f} |")
    return [*out, ""]


def law_name(law: str) -> str:
    if law == "counting":
        return "counting law (D10)"
    return f"literal law, rate {law.split('_')[1]}"


def describe(spec: dict) -> str:
    if spec["machine"] == "window":
        return f"window k = {spec['k']}, {law_name(spec['law'])}"
    prior = f", prior {spec['prior']:g}" if spec["law"] == "counting" else ""
    nfa = []
    if "depths" in spec:
        nfa.append("levels " + "/".join(str(d) for d in spec["depths"]) + " bits")
    if spec.get("coherence", 0):
        nfa.append(f"coherence {spec['coherence']:g}")
    if spec.get("verify", 1) > 1:
        nfa.append(f"verify {spec['verify']}")
    if spec.get("governor"):
        nfa.append(f"governor (protect {spec.get('protect', 1.0):g} bits, sink rate {spec.get('sink_rate', 1.0):g})")
    return f"nested levels, {law_name(spec['law'])}{prior}" + ("".join(f", {x}" for x in nfa))


def e1b_section(d: dict, e1: dict | None) -> list[str]:
    out = ["## E1b. Nested levels on the known sources", "",
           f"Run at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}, on the E1 streams. "
           "Last-quarter bits per bit; *best window* is the best fixed-window VAM-1 of E1 (counting law); N16 "
           "and N64 use the counting law too.", "",
           "| source | oracle | best window | N16 | N64 |", "|---|---|---|---|---|"]
    best = {}
    if e1 is not None:
        for row in e1["context_sweep"]:
            best[row["source"]] = min(e["consolidating"]["last_quarter"] for e in row["by_k"])
    gaps = []
    for r in d["rows"]:
        bw = best.get(r["source"])
        out.append(f"| {r['source']} | {r['oracle']['last_quarter']:.4f} | "
                   f"{'' if bw is None else f'{bw:.4f}'} | {r['N16']['last_quarter']:.4f} | "
                   f"{r['N64']['last_quarter']:.4f} |")
        if bw is not None:
            gaps.append((r["source"], r["N16"]["last_quarter"] - bw))
    if gaps:
        worst = max(gaps, key=lambda g: g[1])
        ok = all(g <= TOL for _, g in gaps)
        out += ["", f"Criterion: N16 within {TOL} of the best window on every source: **{'pass' if ok else 'fail'}** "
                f"(largest excess {worst[1]:+.4f}, {worst[0]})."]
    return [*out, ""]


def e2_run(name: str, run: dict) -> list[str]:
    arms, t, tr = run["arms"], run["t12_3"], run["transfer"]
    title = {"literal": "Best configuration under the literal T12.1 law (the specification's machine)",
             "best_overall": "Best configuration overall"}[name]
    out = [f"### {title}: {describe(run['spec'])}", "", "Frozen bits per byte (lower is better):", "",
           "| arm | A test | A unseen books | B test | B unseen books |", "|---|---|---|---|---|"]
    for arm, r in arms.items():
        r = r if arm != "shuffled_A_then_B" else {k: v for k, v in r.items() if k != "after_A"}
        out.append(f"| {arm} | {r['A_test']:.3f} | {r['A_heldout']:.3f} | {r['B_test']:.3f} | "
                   f"{r['B_heldout']:.3f} |")
    p3, drift = t["part3"], t["part2_weight_drift_per_MiB"]
    out += ["", f"Transfer (B test after A then B, minus B only): {tr['A_then_B_minus_B_only']:+.4f} bpb; "
            f"with sentence-shuffled A: {tr['shuffled_A_then_B_minus_B_only']:+.4f} bpb.", "",
            f"T12.3 part 2, weight drift per MiB of Stage A: first {drift[0]:.2f}, last {drift[-1]:.2f}.", "",
            f"T12.3 part 3, true against word-shuffled text: A test {p3['A_test']:.3f} against "
            f"{p3['A_test_word_shuffled']:.3f}; unseen books {p3['A_heldout']:.3f} against "
            f"{p3['A_heldout_word_shuffled']:.3f}.", "",
            "Generation from the end-of-A state (greedy and sampled, 48 bytes each), verbatim:", ""]
    for g in run["generation"]:
        out += [f"- prompt `{g['prompt']!r}`", f"  - greedy `{g['max']!r}`", f"  - sampled `{g['sample']!r}`"]
    return [*out, ""]


def e2_section(d: dict) -> list[str]:
    a1, s1 = d["t12_3_part1"]["A"], d["t12_3_part1"]["sentence_shuffled_A"]
    out = ["## E2. Textbooks first, then unstructured text", "",
           f"Run at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}. Stream sizes in bytes: "
           + ", ".join(f"{k} {v:,}" for k, v in d["stream_bytes"].items()) + ".", "",
           "Configurations tried on A-validation (frozen bpb):", "",
           *[f"- {describe(r)}: {r['A_validation_bpb']:.3f}" for r in d["selection"]], "",
           f"T12.3 part 1, over the {a1['k']}-bit contexts of Stage A as read: share of visits to familiar "
           f"contexts {a1['share_of_visits_to_familiar_contexts']:.4f} in textbook order against "
           f"{s1['share_of_visits_to_familiar_contexts']:.4f} sentence-shuffled; visitation entropy "
           f"{a1['entropy_bits']:.3f} against {s1['entropy_bits']:.3f} bits (a weak test; see the protocol).", ""]
    if d["reference_lzma"]:
        out += ["Reference (lzma, conditional on the training split): "
                + ", ".join(f"{k} {v:.3f}" for k, v in d["reference_lzma"].items()) + " bpb.", ""]
    for name, run in d["runs"].items():
        out += e2_run(name, run)
    return out


AREA_READ_MAX = 1e-6        # PROTOCOL E3 Amendment 1 (A2): discarded weight at or below this is read for the area law


def instrument_label(st: dict) -> str:
    """PROTOCOL E3 Amendment 1 (A1): which Part 4 settings are the machine's own instrument, and what the rest are."""
    own = st["hbar"] == 1 and st["form"] == "euclidean" and st["mode"] == "sample" and not st["verified"]
    if own:
        return "the machine's own output instrument (T10.1)"
    if st["hbar"] == 0:
        why = "shortest path over -ln p (T15.3); form and mode have no effect"
    elif st["form"] == "feynman":
        why = "phase built from the probability"
    elif st["hbar"] < 1:
        why = "tempering of path probabilities (T14 step 7)"
    elif st["mode"] == "collapse":
        why = "greedy decision rule"
    else:
        why = "the machine's prediction"
    if st["verified"]:
        why += "; evidence threshold on the readout"
    return f"chosen realization: {why}"


def reported_once(settings: list[dict]) -> list[dict]:
    """At hbar = 0 the form and mode change nothing: keep the first setting of each ``verified`` value."""
    seen, out = set(), []
    for st in settings:
        if st["hbar"] == 0:
            if st["verified"] in seen:
                continue
            seen.add(st["verified"])
        out.append(st)
    return out


def verdict(ok) -> str:
    return "**pass**" if ok else "**fail**"


def e3_part1(p: dict) -> list[str]:
    l1, l2, l3, l7 = p["N_L1"], p["N_L2"], p["N_L3"], p["N_L7"]
    flow = max([*l1["flow_max_difference_reading"].values(), l1["flow_max_difference_generation"]])
    out = ["### Part 1. Designed behaviours", "",
           "| layer | criterion (protocol) | measured | result |", "|---|---|---|---|",
           f"| N-L1 register | cap min(Bekenstein, word) = 64 bits holds the deepest level | cap "
           f"{l1['register_cap_bits']} bits, deepest level {l1['deepest_level_bits']} bits | "
           f"{verdict(l1['criteria']['a_register'])} |",
           f"| N-L1 flow | integrated ODE equals the machine to 1e-8 (reading and generation) | largest difference "
           f"{flow:.1e} | {verdict(l1['criteria']['b_flow'])} |",
           f"| N-L1 loss | -ln q = integral of Tr(K R) to 1e-10 | {l1['loss_integral_max_difference']:.1e} | "
           f"{verdict(l1['criteria']['c_integral'])} |",
           f"| N-L2 sources | Page rank = hidden states when resolvable, never more | see below | "
           f"{verdict(l2['criteria']['a_known_sources'])} |",
           f"| N-L3 readout | coherent readout in [eps, 1 - eps] | [{l3['min']:.6g}, {l3['max']:.6g}], eps "
           f"{l3['eps']:.6g} | {verdict(l3['criteria']['bounded'])} |"]
    cb = p["N_L4_L5_L6_chain_blankets"]
    sv = p["N_L5_L6_text_survival"]
    out += [f"| N-L4 blankets | untruncated engines contain a disturbance to 1e-12 | see below | "
            f"{verdict(cb['criteria']['untruncated_cone_leak'])} |",
            f"| N-L5/N-L6 text | plain keeps < 2%, verify 2 and governor keep >= 97% | plain "
            f"{sv['rows']['plain']['after']:.3f}, verify 2 {sv['rows']['verify_2']['after']:.3f} of "
            f"{sv['rows']['verify_2']['before']:.3f}, governor {sv['rows']['governor']['after']:.3f} of "
            f"{sv['rows']['governor']['before']:.3f} | {verdict(sv['criteria']['survival'])} |",
            f"| N-L7 hbar = 0 | equals the select=max planner bit for bit | "
            f"{'equal' if l7['hbar0_equals_max_planner'] else 'differs'} | {verdict(l7['criteria']['hbar0'])} |",
            f"| N-L7 hbar = 1 | Z_0 equals the prediction | largest difference {l7['hbar1_Z_minus_prediction']:.1e} | "
            f"{verdict(l7['criteria']['hbar1'])} |", ""]
    out += ["N-L2, the Page rank of known sources (400,000 bits, past and future windows of k bits):", "",
            "| source | k | hidden states | Page rank | weakest direction | cutoff | resolvable | result |",
            "|---|---|---|---|---|---|---|---|"]
    for r in l2["known_sources"]:
        weak = f"{r['weakest_direction']:.4f}" if "weakest_direction" in r else ""
        cut = f"{r['cutoff']:.4f}" if "cutoff" in r else ""
        res = "" if "resolvable" not in r else ("yes" if r["resolvable"] else "no")
        out.append(f"| {r['source']} | {r['window']} | {r['states']} | {r['rank']} | {weak} | {cut} | {res} | "
                   f"{verdict(r['pass'])} |")
    out += ["", f"N-L2 on text: the Page rank of the byte Hankel matrix (previous d bytes, next byte) of the first "
            f"{l2['text_bytes']:,} bytes of A-train. Turnaround (largest rank): "
            f"**{l2['turnaround_bytes']} bytes**.", "",
            "| d (bytes) | contexts | rank | cutoff |", "|---|---|---|---|",
            *[f"| {r['depth_bytes']} | {r['rows']:,} | {r['rank']} | {r['cutoff']:.4f} |"
              for r in l2["text_depth_profile"]], ""]
    c = cb["config"]
    out += [f"N-L4, N-L5, N-L6 on the chain ({c['cells']} cells, context {c['context']}, theta {c['theta']}, "
            f"{c['bits']} bits of English, then {c['events']} forced events after disturbing cell {c['cell']}):", "",
            "| engine | discarded weight | cone leak | CMI per interior cell (bits) | S2 / qubits | largest bond | "
            "seconds |", "|---|---|---|---|---|---|---|"]
    for name, e in cb["engines"].items():
        out.append(f"| {name} | {e['discarded']:.2e} | {e['cone_leak']:.1e} | "
                   f"{' '.join(f'{x:.3f}' for x in e['cmi_bits'])} | {e['renyi2_per_qubit_end']:.3f} | "
                   f"{max(e['bonds'])} | {e['seconds']} |")
    fd = p["N_L5_L6_chain_fidelity"]
    c = fd["config"]
    out += ["", f"Fidelity of the governed engine ({c['cells']} cells, context {c['context']}, theta {c['theta']}, "
            f"{c['bits']} bits), against the {fd['reference']}; {fd['ungoverned_reference']}:", "",
            "| engine | largest prediction difference | largest bond | mean bond | discarded per event | seconds |",
            "|---|---|---|---|---|---|"]
    for n in fd["seconds_by_engine"]:
        diff = fd["max_abs_q_difference_vs_reference"].get(n)
        out.append(f"| {n} | {'reference' if diff is None else f'{diff:.1e}'} | {fd['largest_bond'][n]} | "
                   f"{fd['mean_bond'][n]:.1f} | {fd['discarded_per_event'][n]:.1e} | {fd['seconds_by_engine'][n]} |")
    out += ["", "Area law (largest operator entanglement of any cut after 40 bits):", "",
            "Read for the area law only when the discarded weight is at most 1e-6 (Amendment 1); other rows are "
            "truncation-bound. The quantity is operator entanglement of vec(rho) across the cut. Governed rows are "
            "bounded by their sinks.", "",
            "| cells | engine | largest cut entropy (bits) | middle cut (bits) | largest bond | discarded | read |",
            "|---|---|---|---|---|---|---|",
            *[f"| {r['cells']} | {r['engine']} | {r['max_cut_entropy_bits']:.3f} | "
              f"{r['middle_cut_entropy_bits']:.3f} | "
              f"{r['largest_bond']} | {r['discarded']:.1e} | "
              f"{'yes' if r['discarded'] <= AREA_READ_MAX else 'truncation-bound'} |"
              for r in p["N_L5_area_law"]["rows"]], "",
            "What the sinks change (the exact machine with and without them, same stream and emissions):", "",
            "| cells | context | theta | g | bits per bit, plain | with sinks | largest prediction change |",
            "|---|---|---|---|---|---|---|",
            *[f"| {r['cells']} | {r['context']} | {r['theta']} | {r['dephasing']} | {r['plain_bits_per_bit']:.5f} | "
              f"{r['sinks_bits_per_bit']:.5f} | {r['max_abs_q_change']:.1e} |"
              for r in p["N_L6_chain_sink_cost"]["rows"]],
            ""]
    return out


def e3_section(d: dict, e2: dict | None) -> list[str]:
    p2, p3, p4 = d["part2"], d["part3"], d["part4"]
    out = ["## E3. The nested fractal architecture, layer by layer", "",
           f"Run at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}, {d['seconds'] / 3600:.1f} hours. "
           "Protocol: `experiments/PROTOCOL.md`, E3.", ""]
    out += e3_part1(d["part1"])
    out += ["### Part 2. Configuration, chosen on A-validation only", "",
            f"Reproducibility: E2's best overall configuration gives {p2['start_A_validation_bpb']:.6f} bpb on "
            "A-validation "
            f"(E2: {p2['e2_start_bpb']:.6f}): {verdict(p2['reproduced'])}.", ""]
    rows = {json.dumps(r["spec"], sort_keys=True): r for r in p2["rows"]}
    for st in p2["stages"]:
        out += [f"Stage {st['stage']}:", ""]
        for c in st["candidates"]:
            r = rows[json.dumps(c, sort_keys=True)]
            mark = " (chosen)" if c == st["chosen"] else ""
            out.append(f"- {describe(c)}: {r['A_validation_bpb']:.4f}{mark}")
        out.append("")
    out += [f"Chosen: **{describe(p2['chosen'])}**.", "", "### Part 3. The whole machine on E2's arms", "",
            "Frozen bits per byte (lower is better); E2's best overall configuration in brackets:", "",
            "| arm | A test | A unseen books | B test | B unseen books |", "|---|---|---|---|---|"]
    ref = (e2 or {}).get("runs", {}).get("best_overall", {}).get("arms", {})
    for arm, r in p3["arms"].items():
        r2 = ref.get(arm, {})
        cell = lambda k, r=r, r2=r2: f"{r[k]:.3f}" + (f" [{r2[k]:.3f}]" if k in r2 else "")
        out.append(f"| {arm} | {cell('A_test')} | {cell('A_heldout')} | {cell('B_test')} | {cell('B_heldout')} |")
    tr, t = p3["transfer"], p3["t12_3"]
    p3t = t["part3"]
    out += ["", f"Transfer (B test after A then B, minus B only): {tr['A_then_B_minus_B_only']:+.4f} bpb; with "
            f"sentence-shuffled A: {tr['shuffled_A_then_B_minus_B_only']:+.4f} bpb.", "",
            f"T12.3 part 3, true against word-shuffled text: A test {p3t['A_test']:.3f} against "
            f"{p3t['A_test_word_shuffled']:.3f}; unseen books {p3t['A_heldout']:.3f} against "
            f"{p3t['A_heldout_word_shuffled']:.3f}.", ""]
    if e2 is not None and e2.get("reference_lzma"):
        out += ["Reference (lzma, conditional on the training split): "
                + ", ".join(f"{k} {v:.3f}" for k, v in e2["reference_lzma"].items()) + " bpb.", ""]
    out += ["Counters at the end of each arm (evictions, doorkeeper refusals, protected knocks, occupancy per level):",
            ""]
    for arm, r in p3["arms"].items():
        c = r.get("counters")
        if c:
            occ = ", ".join(f"level {k}: {v:.3f}" for k, v in c["occupancy"].items())
            out.append(f"- {arm}: {c['evictions']:,} evictions, {c['refused']:,} refused, {c['protected']:,} "
                       f"protected; {occ}")
    out += ["", "Generation from the end-of-A state with E2's planner (greedy and sampled, 48 bytes), verbatim:", ""]
    for g in p3["generation"]:
        out += [f"- prompt `{g['prompt']!r}`", f"  - greedy `{g['max']!r}`", f"  - sampled `{g['sample']!r}`"]
    if p4:
        out += ["", f"### Part 4. Writing with the path integral (N-L7), {p4['prompts']} prompts, "
                f"{p4['bytes']} bytes each",
                "", "| hbar | form | mode | verified | repeated 8-byte windows | words in A-train vocabulary | "
                "printable | valid UTF-8 | own bits per byte | what the setting is |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        shown = reported_once(p4["settings"])
        for st in shown:
            words = "" if st["words_in_A_train_vocabulary"] is None else f"{st['words_in_A_train_vocabulary']:.3f}"
            out.append(f"| {st['hbar']:g} | {st['form']} | {st['mode']} | {'on' if st['verified'] else 'off'} | "
                       f"{st['repeated_8byte_windows']:.3f} | {words} | {st['printable_bytes']:.3f} | "
                       f"{st['valid_utf8']:.3f} | {st['own_bits_per_byte']:.3f} | {instrument_label(st)} |")
        out += ["", "At hbar = 0 the form and mode have no effect, so those settings are reported once.",
                "", "Outputs, verbatim:", ""]
        for st in shown:
            out += [f"<details><summary>hbar {st['hbar']:g}, {st['form']}, {st['mode']}, verified "
                    f"{'on' if st['verified'] else 'off'}</summary>", ""]
            out += [f"- `{o['prompt']!r}` → `{o['text']!r}`" for o in st["outputs"]]
            out += ["", "</details>", ""]
    return [*out, ""]


def e3_t4_section(t4: dict | None, e3: dict) -> list[str]:
    """PROTOCOL E3 Amendment 4: measurements added after the result, labelled so."""
    if t4 is None:
        return []
    out = ["### Added after the result: T4.2 and T4.3 (PROTOCOL E3 Amendment 4)", "",
           f"Run at commit `{t4['provenance']['commit']}`, {t4['provenance']['utc']}. E3's numbers above stand as run; "
           "these were added after them and are measurements only.", ""]
    tx = t4["T4_2_text"]
    part1 = {r["depth_bytes"]: r for r in e3["part1"]["N_L2"]["text_depth_profile"]}
    out += [f"**T4.2. N-L2 on text with a count floor of {tx['floor']}** (fixed before the script read PG-19: the number "
            f"of values a next byte takes). The first {tx['text_bytes']:,} bytes of A-train; the null is the same bytes "
            "shuffled. A measurement only: ranks on a filtered row set fall with depth because rows are removed, so "
            "they are never used to cut depths.", "",
            "| d (bytes) | contexts kept | continuations kept | samples kept | rank | cutoff | null rank | Part 1 rank (no floor) |",
            "|---|---|---|---|---|---|---|---|"]
    for a, n in zip(tx["A_train"], tx["byte_shuffled_null"], strict=True):
        share = a["samples_kept"] / a["samples"] if a["samples"] else 0.0
        rank = "n/a" if a["rank"] is None else a["rank"]
        cut = "" if a["cutoff"] is None else f"{a['cutoff']:.3f}"
        nr = "n/a" if n["rank"] is None else n["rank"]
        p1 = part1.get(a["depth_bytes"], {}).get("rank", "")
        out.append(f"| {a['depth_bytes']} | {a['rows_kept']:,} of {a['rows']:,} | {a['cols_kept']} of {a['cols']} | "
                   f"{share:.3f} | {rank} | {cut} | {nr} | {p1} |")
    ch = t4["T4_3_chain"]
    c = ch["config"]
    p1c = e3["part1"]["N_L4_L5_L6_chain_blankets"]["engines"]
    out += ["", f"**T4.3. The governed chain with the cache trigger** ({c['cells']} cells, context {c['context']}, theta "
            f"{c['theta']}, {c['bits']} bits of English, {c['events']} forced events after disturbing cell {c['cell']}, as "
            f"Part 1's blankets; cache cap {c['cache_cap']}). `cache_trigger_*` starts the sinks when the largest bond "
            "reaches half of the cap (or the cap itself), instead of at the Page fraction of the register.", "",
            "| engine | discarded weight | cone leak | largest CMI (bits) | S2 / qubits | largest bond | sinks from event | "
            "seconds |", "|---|---|---|---|---|---|---|---|"]
    rows = [(f"Part 1: {n}", p1c[n]) for n in ("governed_cache_bond", "governed_page_threshold") if n in p1c]
    rows += list(ch["engines"].items())
    for name, e in rows:
        out.append(f"| {name} | {e['discarded']:.2e} | {e['cone_leak']:.1e} | {max(e['cmi_bits']):.3f} | "
                   f"{e['renyi2_per_qubit_end']:.3f} | {max(e['bonds'])} | {e['governed_since']} | {e['seconds']} |")
    return [*out, ""]


def e4_section(d: dict, e3: dict | None, e2: dict | None) -> list[str]:
    """E4 (PROTOCOL E4): every number from e4.json; E3's from e3.json; the reading rules computed here, never typed."""
    p0, p1, p2 = d["part0"], d["part1"], d["part2"]
    out = ["## E4. N above one equation, on text", "",
           f"Run at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}, {d['seconds'] / 3600:.1f} hours. "
           "Protocol: `experiments/PROTOCOL.md`, E4; design: `DESIGN.md` section 11.", "",
           "### Part 0. Gate: identity and reproduction", "",
           f"With the upper levels off the machine equals E3's bit for bit: {verdict(p0['criteria']['identity'])}. E3's chosen "
           f"A-validation {p0['e3_saved_A_validation_bpb']:.10f}, recomputed {p0['recomputed_A_validation_bpb']:.10f}: "
           f"{verdict(p0['criteria']['reproduction'])}. The upper-off path of the E4 loop runs at "
           f"{p0['median_rate_ratio_upper_off_to_nested']:.3f} of the nested machine's rate (median of paired, interleaved "
           "runs; E3's own loop is untouched).", "",
           "### Part 1. Designed behaviours", "",
           "| layer | criterion | measured | result |", "|---|---|---|---|"]
    l1, l2, l3, l4, l5, l6, l7 = (p1[k] for k in ("N_L1", "N_L2", "N_L3", "N_L4", "N_L5", "N_L6", "N_L7"))
    out += [
        f"| N-L1 | registers at most 64 bits, arity at most 8, state constant and under 128 bytes | arity {l1['arity']}, state "
        f"{l1['state_bytes']} bytes; units {l1['units']:,}, mean {l1['mean_unit_bytes']:.2f} bytes, code collisions "
        f"{l1['code_collisions']} ({l1['collision_share']:.2e}) | {verdict(l1['criteria']['bounds_and_state'])} |",
        f"| N-L2 | the null never exceeds rank 1 (a measurement; gates nothing) | see below | "
        f"{verdict(l2['criteria']['null_never_above_rank_1'])} |",
        f"| N-L3 | readout in the hull and in [eps, 1 - eps]; nesting exact | [{l3['min']:.6g}, {l3['max']:.6g}]; nesting "
        f"{l3['nesting_max_deviation']:.1e} | {verdict(l3['criteria']['hull_and_bounded'] and l3['criteria']['nesting'])} |",
        f"| N-L4 | a flipped byte's effect ends and stays ended | median span {l4['median_span_bytes']:.0f} bytes, largest "
        f"{l4['max_span_bytes']} | {verdict(l4['criteria']['effect_ends_and_stays_ended'])} |",
        f"| N-L5 | noise stays out; storage constant | verify 3: " + ", ".join(f"{k} {v:.4f}" for k, v in
                                                                              l5["occupancy_verify3"].items())
        + f" (verify 1: " + ", ".join(f"{k} {v:.2f}" for k, v in l5["occupancy_verify1"].items()) + ") | "
        f"{verdict(all(l5['criteria'].values()))} |",
        f"| N-L6 | survival; global occupancy at most 1; the raise changes refusals at verify 2, 3, 4 | "
        f"{l6['survival']['plain']['after']}/{l6['survival']['plain']['before']} plain, "
        f"{l6['survival']['verify_2']['after']}/{l6['survival']['verify_2']['before']} verify 2, "
        f"{l6['survival']['governor']['after']}/{l6['survival']['governor']['before']} governor | "
        f"{verdict(all(l6['criteria'].values()))} |",
        f"| N-L7 | instrument equals the prediction; hbar 0 equals the planner | {l7['hbar1_Z_minus_prediction']:.1e} | "
        f"{verdict(all(l7['criteria'].values()))} |", ""]
    if not l6["criteria"]["raise_changes_every_verify"]:
        out += ["The raised bar changed no refusal count at some verify setting, so by the protocol it is removed.", ""]
    out += [f"N-L2, Page rank of (upper-level context, next byte) with a count floor of {l2['floor']} on the first "
            f"{l2['text_bytes']:,} bytes of A-train, and the byte-shuffled null (a measurement; no level is cut by it):", "",
            "| level | A-train rank | cutoff | contexts kept | null rank |", "|---|---|---|---|---|"]
    nulls = {r["level"]: r for r in l2["rows"] if r["data"] == "byte_shuffled_null"}
    for r in (r for r in l2["rows"] if r["data"] == "A_train"):
        cut = "n/a" if r["cutoff"] is None else f"{r['cutoff']:.3f}"
        na = lambda v: "n/a" if v is None else v
        out.append(f"| {r['level']} | {na(r['rank'])} | {cut} | {r['rows_kept']:,} of {r['rows']:,} | "
                   f"{na(nulls[r['level']]['rank'])} |")
    cuts = p1["N_L5_cuts"]
    out += ["", f"Label entropy per cut: {cuts['note']}; mean {cuts['mean_label_entropy_bits']}, largest "
            f"{cuts['max_label_entropy_bits']} bits over {cuts['samples']:,} samples. The value is zero because the "
            "coherence is 1.0, which makes the label state pure; it is not the bond-2 bound at work.", "",
            "`n/a` in the table: no context of that level reaches the count floor, so no rank is defined.", ""]
    out += ["### Part 2. Configuration, chosen on A-validation only", ""]
    rows = {json.dumps(r["spec"], sort_keys=True): r for r in p2["rows"]}
    for st in p2["stages"]:
        out += [f"Stage {st['stage']}:", ""]
        for c in st["candidates"]:
            r = rows[json.dumps(c, sort_keys=True)]
            mark = " (chosen)" if c == st["chosen"] else ""
            out.append(f"- `{ {k: v for k, v in c.items() if k.startswith('hier_') or k in ('sink_rate',)} }`: "
                       f"{r['A_validation_bpb']:.4f}, {r['seconds']} s{mark}")
        out.append("")
    out += [f"Chosen: `{p2['chosen']}`.", ""]
    a3, b3 = d["part3a"], d["part3b"]
    e3a = (e3 or {}).get("part3", {}).get("arms", {})
    arms_ = {"A_only": a3["A_only"], "A_then_B": a3["A_then_B"], "B_only": b3["B_only"],
             "shuffled_A_then_B": b3["shuffled_A_then_B"]}
    out += ["### Part 3. The whole machine on E2's arms", "", "Frozen bits per byte; E3's chosen machine in brackets:", "",
            "| arm | A test | A unseen books | B test | B unseen books |", "|---|---|---|---|---|"]
    for arm, r in arms_.items():
        r3 = e3a.get(arm, {})
        cell = lambda k, r=r, r3=r3: f"{r[k]:.3f}" + (f" [{r3[k]:.3f}]" if k in r3 else "")
        out.append(f"| {arm} | {cell('A_test')} | {cell('A_heldout')} | {cell('B_test')} | {cell('B_heldout')} |")
    ctrl = d["part4"]
    out += ["", "### Part 4. Controls (A only arm)", "", "| machine | A test | A unseen books | B test | B unseen books |",
            "|---|---|---|---|---|"]
    for name, r in (("E4 chosen", a3["A_only"]), ("C1 equal storage and compute, no extra span",
                                                  ctrl["C1_equal_storage_no_extra_span"]),
                    ("C2 order destroyed", ctrl["C2_order_destroyed"])):
        out.append(f"| {name} | {r['A_test']:.3f} | {r['A_heldout']:.3f} | {r['B_test']:.3f} | {r['B_heldout']:.3f} |")
    if e3a:
        r3 = e3a["A_only"]
        out.append(f"| C0 upper levels off (E3, saved) | {r3['A_test']:.3f} | {r3['A_heldout']:.3f} | {r3['B_test']:.3f} | "
                   f"{r3['B_heldout']:.3f} |")
    p5 = d["part5"]
    out += ["", "### Part 5. A-test by window class (the longest of 32, 16, 8 bytes before the byte that occurs in A-train)", "",
            "| class | share of bytes | E3 bpb | E4 bpb | E3 bits | E4 bits | gain (bits) |", "|---|---|---|---|---|---|---|"]
    gains = {}
    for c in ("0", "8", "16", "32", "all"):
        a, b = p5["E3"][c], p5["E4"][c]
        gains[c] = a["bits"] - b["bits"]
        fmt = lambda v: "" if v is None else f"{v:.3f}"
        out.append(f"| {c} | {a['share']:.4f} | {fmt(a['bpb'])} | {fmt(b['bpb'])} | {a['bits']:,.0f} | {b['bits']:,.0f} | "
                   f"{gains[c]:,.0f} |")
    out.append("")
    # reading rules
    ref = e3a.get("A_only", {})
    e4a = a3["A_only"]
    lz = (e2 or {}).get("reference_lzma", {}).get("lzma_A_test_given_A_train")
    d2 = lambda x, y: None if x is None or y is None else y - x
    both = lambda a, b, t: (a is not None and b is not None and a >= t and b >= t)
    g_test, g_held = d2(e4a["A_test"], ref.get("A_test")), d2(e4a["A_heldout"], ref.get("A_heldout"))
    c1, c2 = ctrl["C1_equal_storage_no_extra_span"], ctrl["C2_order_destroyed"]
    long_gain, total_gain = gains["16"] + gains["32"], gains["all"]
    share = long_gain / gains["all"] if gains["all"] > 0 else None
    tr = a3["A_then_B"]["B_test"] - b3["B_only"]["B_test"]
    tr_sh = b3["shuffled_A_then_B"]["B_test"] - b3["B_only"]["B_test"]
    forget = a3["A_then_B"]["A_test"] - e4a["A_test"]
    out += ["### Reading rules (PROTOCOL E4, fixed before the run)", "",
            f"- R1 (headline): E4 A-test {e4a['A_test']:.4f} bpb; E3 {ref.get('A_test', float('nan')):.4f}; lzma "
            f"{lz if lz is None else f'{lz:.4f}'}.",
            f"- R2 (the hierarchy helps, at least 0.01 on A-test and unseen books): gains "
            f"{'n/a' if g_test is None else f'{g_test:+.4f}'} and {'n/a' if g_held is None else f'{g_held:+.4f}'}: "
            f"{verdict(both(g_test, g_held, 0.01))}.",
            f"- R3 (span, not capacity; at least 0.005 over C1): {c1['A_test'] - e4a['A_test']:+.4f} and "
            f"{c1['A_heldout'] - e4a['A_heldout']:+.4f}: "
            f"{verdict(both(c1['A_test'] - e4a['A_test'], c1['A_heldout'] - e4a['A_heldout'], 0.005))}.",
            f"- R4 (order; at least 0.005 over C2): {c2['A_test'] - e4a['A_test']:+.4f} and "
            f"{c2['A_heldout'] - e4a['A_heldout']:+.4f}: "
            f"{verdict(both(c2['A_test'] - e4a['A_test'], c2['A_heldout'] - e4a['A_heldout'], 0.005))}.",
            f"- R5 (long matches; more than half of the gain in classes 16 and 32): "
            f"{'no gain to share' if share is None else f'{share:.2f} of {total_gain:,.0f} bits'}: "
            f"{verdict(share is not None and share > 0.5)}.",
            f"- R6: transfer (A then B minus B only) {tr:+.4f}; sentence-shuffled control {tr_sh:+.4f}; forgetting of A "
            f"after B {forget:+.4f}.", ""]
    if d.get("part3c"):
        sh2 = d["part3c"]["shuffled_A_then_B"]["B_test"] - b3["B_only"]["B_test"]
        out += [f"Second shuffle seed: {sh2:+.4f}; the spread of the two controls is {abs(sh2 - tr_sh):.4f}.", ""]
    else:
        out += ["Unit 3c (the second shuffle seed) was not run, so R6's shuffled control has one seed and no spread.", ""]
    out += ["### Part 6. Writing", "",
            "The instrument row (hbar 1, euclidean, sample, unverified), E4 above and E3 below, verbatim (JSON-escaped):", ""]
    inst = next(s_ for s_ in d["part6"]["settings"] if s_["hbar"] == 1.0 and s_["mode"] == "sample")
    e3p4 = [s_ for s_ in ((e3 or {}).get("part4") or {}).get("settings", []) if instrument_label(s_).startswith("the machine")]
    for i, o in enumerate(inst["outputs"]):
        out.append(f"{i + 1}. prompt `{json.dumps(o['prompt'])}`")
        out.append(f"   - E4 `{json.dumps(o['text'])}`")
        if e3p4 and i < len(e3p4[0]["outputs"]):
            out.append(f"   - E3 `{json.dumps(e3p4[0]['outputs'][i]['text'])}`")
    out += ["", "| setting | repeated 8-byte windows | words in A-train vocabulary | printable | valid UTF-8 | own bits per byte |",
            "|---|---|---|---|---|---|"]
    for s_ in d["part6"]["settings"]:
        w = "" if s_["words_in_A_train_vocabulary"] is None else f"{s_['words_in_A_train_vocabulary']:.3f}"
        out.append(f"| hbar {s_['hbar']:g}, {s_['form']}, {s_['mode']} ({instrument_label(s_)}) | "
                   f"{s_['repeated_8byte_windows']:.3f} | {w} | {s_['printable_bytes']:.3f} | {s_['valid_utf8']:.3f} | "
                   f"{s_['own_bits_per_byte']:.3f} |")
    return [*out, ""]


def e4_t_section(t: dict | None, d: dict, e3: dict | None) -> list[str]:
    """PROTOCOL E4 Amendment 2: E3's base with sink rate 0, added after the result (R2 and R5 read cleanly)."""
    if t is None:
        return []
    c = t["E3_base_sink_rate_0"]
    a4, c0 = d["part3a"]["A_only"], (e3 or {})["part3"]["arms"]["A_only"]
    out = ["### Added after the result: E3's base with sink rate 0 (PROTOCOL E4 Amendment 2)", "",
           f"Run at commit `{t['provenance']['commit']}`, {t['provenance']['utc']}. E4's chosen machine also dropped the sinks, so "
           "E4 against E3 (R2, R5) mixes the hierarchy with that change; this control is the E3 machine (upper levels off) with "
           "sink rate 0 on the same splits. It changes no E4 number and selects nothing.", "",
           "| machine (A only arm) | A test | A unseen books | B test | B unseen books |", "|---|---|---|---|---|"]
    for name, r in (("E3 as chosen (sink rate 0.1, saved)", c0), ("E3 base, sink rate 0 (this control)", c),
                    ("E4 chosen", a4)):
        out.append(f"| {name} | {r['A_test']:.3f} | {r['A_heldout']:.3f} | {r['B_test']:.3f} | {r['B_heldout']:.3f} |")
    g = (c["A_test"] - a4["A_test"], c["A_heldout"] - a4["A_heldout"])
    s_ = (c0["A_test"] - c["A_test"], c0["A_heldout"] - c["A_heldout"])
    ok = min(g) >= 0.01
    out += ["", f"- Removing the sinks from E3 (rate 0.1 to 0) changes A-test by {s_[0]:+.4f} and unseen books by {s_[1]:+.4f} "
            "(positive: lower bpb).",
            f"- R2 read cleanly (E4 against E3 at the same sink rate; at least 0.01 on A-test and unseen books): "
            f"{g[0]:+.4f} and {g[1]:+.4f}: {verdict(ok)}.", ""]
    p5 = d["part5"]
    out += ["A-test by window class: the control against E4:", "",
            "| class | share of bytes | control bpb | E4 bpb | control bits | E4 bits | gain (bits) |", "|---|---|---|---|---|---|---|"]
    gains = {}
    for k in ("0", "8", "16", "32", "all"):
        a, b = c["decomposition"][k], p5["E4"][k]
        gains[k] = a["bits"] - b["bits"]
        out.append(f"| {k} | {a['share']:.4f} | {a['bpb']:.3f} | {b['bpb']:.3f} | {a['bits']:,.0f} | {b['bits']:,.0f} | "
                   f"{gains[k]:,.0f} |")
    gains_all = gains["all"]
    share = (gains["16"] + gains["32"]) / gains_all if gains_all > 0 else None
    out += ["", f"- R5 read cleanly (more than half of E4's gain over this control in classes 16 and 32): "
            f"{'no gain to share' if share is None else f'{share:.2f} of {gains_all:,.0f} bits'}: "
            f"{verdict(share is not None and share > 0.5)}.", ""]
    return out


# ---------------------------------------------------------------------------------------------------
# E5: criteria P1 to P3 (PROTOCOL E5.7), computed from e5.json and the question file, never typed
# ---------------------------------------------------------------------------------------------------

E5_ALPHA, E5_CHANCE, E5_D = 0.025, 0.25, 0.10


def binom_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exactly."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(max(k, 0), n + 1))


def e5_threshold(n: int, alpha: float = E5_ALPHA, chance: float = E5_CHANCE) -> int:
    """X: the smallest count whose binomial tail under ``chance`` is at most ``alpha``."""
    return next(i for i in range(n + 1) if binom_tail(n, i, chance) <= alpha)


def e5_correct(costs: dict, questions: list[dict], key: str = "with_stem") -> dict:
    """Question id to correct (True/False): the cheapest option is chosen, a tie for the cheapest counts as wrong."""
    out = {}
    for q in questions:
        c = costs[q["id"]][key]
        best = min(c)
        out[q["id"]] = c.count(best) == 1 and c.index(best) == q["answer_index"]
    return out


def e5_paired(t: dict, c: dict, ids: list[str]) -> dict:
    """The sign test of T against C on the questions in ``ids``: b = only T right, c = only C right."""
    b = sum(t[i] and not c[i] for i in ids)
    w = sum(c[i] and not t[i] for i in ids)
    n = len(ids)
    gap = (sum(t[i] for i in ids) - sum(c[i] for i in ids)) / n if n else 0.0
    p = binom_tail(b + w, b, 0.5) if b + w else 1.0
    return {"n": n, "t_correct": sum(t[i] for i in ids), "c_correct": sum(c[i] for i in ids), "only_t": b,
            "only_c": w, "gap": gap, "p_sign": p, "sign_pass": bool(b + w and p <= E5_ALPHA and b > w),
            "pass": bool(b + w and p <= E5_ALPHA and b > w and gap >= E5_D)}


def e5_analysis(d: dict, questions: list[dict]) -> dict:
    """Everything E5 reports about the questions, from the saved costs: counts, P1, P2, P3, the pre-registered secondary
    analyses and the stem-free baseline."""
    arms = d["arms"]
    ok = {a: e5_correct(arms[a]["questions"], questions) for a in arms}
    base = {a: e5_correct(arms[a]["questions"], questions, "no_stem") for a in arms}
    kinds = {k: [q["id"] for q in questions if q["kind"] == k] for k in ("recall", "inference")}
    out = {"N": {k: len(v) for k, v in kinds.items()}, "X": {k: e5_threshold(len(v)) for k, v in kinds.items()},
           "correct": {a: {k: sum(ok[a][i] for i in v) for k, v in kinds.items()} for a in ok},
           "baseline_correct": {a: {k: sum(base[a][i] for i in v) for k, v in kinds.items()} for a in base}}
    if "T" in ok:
        out["P1"] = {k: {"N": len(v), "X": out["X"][k], "correct": out["correct"]["T"][k],
                         "p": binom_tail(len(v), out["correct"]["T"][k], E5_CHANCE),
                         "pass": out["correct"]["T"][k] >= out["X"][k]} for k, v in kinds.items()}
    if "T" in ok and "C" in ok:
        out["P2"] = e5_paired(ok["T"], ok["C"], kinds["recall"])
        out["P3"] = e5_paired(ok["T"], ok["C"], kinds["inference"])
        qd = {q["id"]: q for q in questions}
        subsets = {
            "answer not the most frequent option in T": lambda q: not q["answer_most_frequent_in_T"],
            "no distractor absent from all of T": lambda q: all(c > 0 for i, c in enumerate(q["option_counts_T"])
                                                               if i != q["answer_index"]),
            "answer of at least 4 bytes (inference co-occurrence checked)": lambda q: q["option_bytes"] >= 4}
        out["secondary"] = {name: {k: e5_paired(ok["T"], ok["C"], [i for i in v if f(qd[i])])
                                   for k, v in kinds.items() if k == "inference" or "inference" not in name}
                            for name, f in subsets.items()}
        works = sorted({q["work"] for q in questions})
        out["per_work"] = {w: e5_paired(ok["T"], ok["C"], [q["id"] for q in questions if q["work"] == w])
                           for w in works}
    return out


def e5_section(d: dict, questions: list[dict]) -> list[str]:
    """E5 (PROTOCOL E5): every number from e5.json and the question file; the criteria computed here."""
    pv = d["provenance"]
    a = e5_analysis(d, questions)
    out = ["## E5. Does the taught machine use what it was taught?", "",
           f"Run at commit `{pv['commit']}`, {pv['utc']}, {d['seconds'] / 3600:.1f} hours"
           f"{' (smoke run: not a result)' if pv['smoke'] else ''}. "
           f"Protocol: `experiments/PROTOCOL.md`, E5. Questions `{pv['questions_sha256'][:16]}`, preamble "
           f"`{pv['preamble']!r}`. E5 bounds \"uses taught knowledge\"; it cannot show understanding.", "",
           "### Questions answered (cheapest option, a tie counts wrong; chance 0.25)", "",
           "| arm | recall correct | inference correct | stem-free baseline, recall | stem-free baseline, inference |",
           "|---|---|---|---|---|"]
    n = a["N"]
    for arm in a["correct"]:
        c, b = a["correct"][arm], a["baseline_correct"][arm]
        out.append(f"| {arm} | {c['recall']} / {n['recall']} | {c['inference']} / {n['inference']} | "
                   f"{b['recall']} / {n['recall']} | {b['inference']} / {n['inference']} |")
    out += ["", "### Criteria", "", "| criterion | measured | result |", "|---|---|---|"]
    if "P1" in a:
        p1 = a["P1"]["recall"]
        out.append(f"| P1: T correct on at least X = {p1['X']} of N = {p1['N']} recall questions | {p1['correct']} "
                   f"(exact tail {p1['p']:.4g}) | {verdict(p1['pass'])} |")
        q1 = a["P1"]["inference"]
        out.append(f"| P1 on inference (descriptive): at least X = {q1['X']} of N = {q1['N']} | {q1['correct']} "
                   f"(exact tail {q1['p']:.4g}) | {verdict(q1['pass'])} |")
    for name, label in (("P2", "recall"), ("P3", "inference")):
        if name in a:
            r = a[name]
            out.append(f"| {name}: T beats C on {label}: sign test p <= {E5_ALPHA} and gap >= {E5_D} | only T right "
                       f"{r['only_t']}, only C right {r['only_c']}, p {r['p_sign']:.4g}, gap {r['gap']:+.3f} "
                       f"(T {r['t_correct']}, C {r['c_correct']} of {r['n']}) | {verdict(r['pass'])} |")
    if "secondary" in a:
        out += ["", "### Pre-registered secondary analyses (reported beside the criteria, never replacing them)", "",
                "| subset | kind | n | T correct | C correct | only T | only C | p (sign) | gap |",
                "|---|---|---|---|---|---|---|---|---|"]
        for name, per in a["secondary"].items():
            for k, r in per.items():
                out.append(f"| {name} | {k} | {r['n']} | {r['t_correct']} | {r['c_correct']} | {r['only_t']} | "
                           f"{r['only_c']} | {r['p_sign']:.4g} | {r['gap']:+.3f} |")
        out += ["", "Per source work (descriptive, both kinds together):", "",
                "| work | n | T correct | C correct | only T | only C |", "|---|---|---|---|---|---|"]
        for w, r in a["per_work"].items():
            out.append(f"| {w} | {r['n']} | {r['t_correct']} | {r['c_correct']} | {r['only_t']} | {r['only_c']} |")
    out += ["", "### Capacity and held-out retention (measurements, not criteria)", ""]
    for arm, rec in d["arms"].items():
        out += [f"Arm {arm}: after each stage, the capacity counters, and frozen bits per byte on the held-out part of "
                "every stage read so far.", "",
                "| stage | trained bytes | train bpb | evictions | refused | protected | global occupancy | seconds "
                "| held-out bpb by stage |", "|---|---|---|---|---|---|---|---|---|"]
        for s in rec["order"]:
            r = rec["stages"][s]
            c = r["counters"]
            held = ", ".join(f"{k} {v:.4f}" for k, v in r["heldout_bpb"].items())
            out.append(f"| {s} | {r['trained_bytes']} | {r['train_bpb']:.4f} | {c['evictions']} | {c['refused']} | "
                       f"{c['protected']} | {c['global_occupancy']:.4f} | {r['seconds']} | {held} |")
        g = rec["gates"]
        sc = g["snapshot_equals_copy"]
        out += ["", f"Pre-run gates: snapshot equals a full copy bit for bit on {sc['items']} questions: "
                f"{verdict(sc['bit_identical'])}; the four table hashes unchanged by frozen reads: "
                f"{verdict(g['tables_unchanged'])}.", ""]
    return out


# ---------------------------------------------------------------------------------------------------
# E6: the criterion C1 and the reading rules (PROTOCOL E6.7), computed from e6.json, never typed
# ---------------------------------------------------------------------------------------------------

E6_MARGIN = 0.02            # C1: class-0 bits per byte at least this much below E4's and the rank-1 arm's
E6_TIE = 0.0005             # E4's tie band (e4_hier.TIE): E4+O worse than E4 on A-test by more than this "loses"


def e6_bootstrap(d: dict, draws: int = 1000, seed: int = 0) -> dict:
    """By-book bootstrap of the class-0 difference (E4+O minus each comparator): books of A-test resampled."""
    import numpy as np
    arms = d["arms"]
    bits = {a: np.array(arms[a]["class0_per_book"]["bits"]) for a in arms}
    n = np.array(arms["E4"]["class0_per_book"]["bytes"], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(draws, len(n)))
    out = {}
    for ref in ("E4", "E4+O rank 1"):
        diff = (bits["E4+O"][idx].sum(1) - bits[ref][idx].sum(1)) / np.maximum(n[idx].sum(1), 1.0)
        out[ref] = {"mean": float(diff.mean()), "low": float(np.percentile(diff, 2.5)),
                    "high": float(np.percentile(diff, 97.5)),
                    "books": len(n), "draws": draws}
    return out


def e6_analysis(d: dict) -> dict:
    arms = d["arms"]
    c0 = {a: arms[a]["A_test_decomposition"]["0"]["bpb"] for a in arms}
    allb = {a: arms[a]["A_test_decomposition"]["all"]["bpb"] for a in arms}
    c1_e4 = c0["E4"] - c0["E4+O"]
    c1_r1 = c0["E4+O rank 1"] - c0["E4+O"]
    out = {"class0_bpb": c0, "all_bpb": allb, "class0_gain_over_E4": c1_e4, "class0_gain_over_rank1": c1_r1,
           "C1": bool(c1_e4 >= E6_MARGIN and c1_r1 >= E6_MARGIN),
           "A_test_gain_over_E4": allb["E4"] - allb["E4+O"],
           "A_test_gain_over_rank1": allb["E4+O rank 1"] - allb["E4+O"],
           "heldout_gain_over_E4": arms["E4"]["A_heldout_bpb"] - arms["E4+O"]["A_heldout_bpb"],
           "loses": bool(allb["E4+O"] > allb["E4"] + E6_TIE)}
    out["reading"] = ("C1 passes" if out["C1"] else
                      ("E6 loses: E4+O is worse than E4 on A-test by more than the tie band" if out["loses"]
                       else "C1 fails; E4+O is not worse than E4 on A-test"))
    return out


def e6_section(d: dict) -> list[str]:
    """E6 (PROTOCOL E6): every number from e6.json; the criterion computed here."""
    pv, a = d["provenance"], e6_analysis(d)
    arms = d["arms"]
    out = ["## E6. The shared operator level O", "",
           f"Run at commit `{pv['commit']}`, {pv['utc']}, {d['seconds'] / 3600:.1f} hours"
           f"{' (smoke run: not a result)' if pv['smoke'] else ''}. Protocol: `experiments/PROTOCOL.md`, E6; "
           "design: `DESIGN.md` section 12, V2. E6 bounds \"O helps on unseen contexts\"; it does not test relations, "
           "sums or the loop.", "",
           "### Bits per byte (frozen; lower is better)", "",
           "| arm | train | A-test | A-heldout (unseen books) | class 0 | class 8 | class 16 | class 32 |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in arms.items():
        dec = r["A_test_decomposition"]
        cells = " | ".join("n/a" if dec[c]["bpb"] is None else f"{dec[c]['bpb']:.4f}" for c in ("0", "8", "16", "32"))
        out.append(f"| {name} | {r['train_bpb']:.4f} | {dec['all']['bpb']:.4f} | {r['A_heldout_bpb']:.4f} | {cells} |")
    shares = arms["E4"]["A_test_decomposition"]
    out += ["", "Share of A-test bytes by window class: " + ", ".join(
        f"class {c} {shares[c]['share']:.3f}" for c in ("0", "8", "16", "32")) + ".", "",
        "### Criterion C1 (class-0 bits per byte, as DESIGN 12.4)", "",
        "| comparison | gain (bits per byte, positive = E4+O lower) | required | result |", "|---|---|---|---|",
        f"| E4+O against E4 | {a['class0_gain_over_E4']:+.4f} | {E6_MARGIN} "
        f"| {verdict(a['class0_gain_over_E4'] >= E6_MARGIN)} |",
        f"| E4+O against E4+O at rank 1 | {a['class0_gain_over_rank1']:+.4f} | {E6_MARGIN} | "
        f"{verdict(a['class0_gain_over_rank1'] >= E6_MARGIN)} |",
        "", f"C1: {verdict(a['C1'])}. Reading: {a['reading']}.", "",
        f"All bytes of A-test: E4+O against E4 {a['A_test_gain_over_E4']:+.4f}, against rank 1 "
        f"{a['A_test_gain_over_rank1']:+.4f}; unseen books against E4 {a['heldout_gain_over_E4']:+.4f}.", ""]
    b = e6_bootstrap(d)
    out += ["By-book bootstrap of the class-0 difference, E4+O minus comparator (bits per byte, 1000 draws, seed 0; "
            "an uncertainty for the margin, not a criterion): " + "; ".join(
                f"against {k}: {v['mean']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}] over {v['books']} books"
                for k, v in b.items()) + ".", ""]
    out += ["### O's own measures", "", "| arm | compiles | rank k (min, median, max) | Page rank (min, median, max) | "
            "negative-mass share on A-test | bytes with a negative part | state resets (learning, A-test) | "
            "worst node excess (bits, bound 2) |", "|---|---|---|---|---|---|---|---|"]
    for name in ("E4+O", "E4+O rank 1"):
        r = arms[name]
        ks = sorted(c.get("k", 1) for c in r["compiles"]) or [0]
        ps = sorted(c.get("page_rank", 1) for c in r["compiles"]) or [0]
        def med(v):
            return v[len(v) // 2]

        nm, nl = r["negative_mass_A_test"], r["negative_mass_learning"]
        out.append(f"| {name} | {len(r['compiles'])} | {ks[0]}, {med(ks)}, {ks[-1]} | {ps[0]}, {med(ps)}, {ps[-1]} | "
                   f"{nm['mean_negative_share']:.4f} | {nm['share_of_bytes_with_negative']:.4f} | "
                   f"{int(nl['resets'])}, {int(nm['resets'])} | {r['stage_regret']['max_excess_bits']:.3f} |")
    st = arms["E4+O"]["store"]
    out += ["", "The resets of the rank-1 arm are no-ops (its state is constant). Count store: "
            f"{st['positions_c3']} positions, largest C3 cell {st['max_cell']}, {st['occupied_buckets']} of "
            f"{st['buckets']} past buckets occupied, {st['distinct_contexts']} distinct 4-byte contexts "
            f"({st['contexts_per_occupied_bucket']:.1f} per occupied bucket).", "",
            "### Gates (PROTOCOL E6.5)", "",
            f"(a) With O off the path equals E4 bit for bit on {d['part0']['bytes']} bytes of A-train (learning and "
            f"frozen): {verdict(d['part0']['bit_identical'])}.",
            "(e) Frozen reads leave the tables, the count store and the operators unchanged: "
            + verdict(all(r["tables_unchanged_by_frozen_reads"] for r in arms.values())) + ".",
            "(f) The stage costs at most 2 bits per node against its input: "
            + verdict(all(r["stage_regret"]["max_excess_bits"] <= 2.0 + 1e-6
                          for r in arms.values() if "stage_regret" in r))
            + ".", ""]
    rep = arms["E4"].get("e4_reproduction")
    if rep:
        out += [f"E4 reproduction: A-test {rep['A_test']:.10f} against the saved {rep['saved_A_test']:.10f} "
                f"(difference {rep['abs_difference']:.1e}): {verdict(rep['abs_difference'] <= 1e-9)}.", ""]
    out += ["Gates (b), (c), (d) and (g) are tests (`tests/test_operator.py`)."]
    return out


def e6_diag_section(e6: dict, ra: dict | None, rb: dict | None) -> list[str]:
    """E6 Amendment 1 (added after the result): R-a and R-b beside E6, each read with C1's arithmetic."""
    base = e6_analysis(e6)
    out = ["### E6 Amendment 1: diagnostic reruns (added after the result; C1 as registered stands)", "",
           "Each rerun is the E4+O arm only, against the E4 and rank-1 arms of `e6.json` (same splits and seeds). "
           "C1's arithmetic is applied to each; the verdict of E6 is unchanged by them.", "",
           "| run | change | class 0 | class 0 gain over E4 | class 0 gain over rank 1 | C1 | all bytes of A-test | "
           "A-heldout | k (min, median, max) | negative share (A-test) | bytes with a negative part | "
           "resets (A-test) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    runs = [("E6", "none (past_bits 12, Page rank)", e6["arms"]["E4+O"], base)]
    for label, d in (("R-a", ra), ("R-b", rb)):
        if d is None:
            continue
        both = {"arms": {**{k: e6["arms"][k] for k in ("E4", "E4+O rank 1")}, "E4+O": d["arms"]["E4+O"]}}
        runs.append((label, d["rerun"] + (" (smoke: not a result)" if d["provenance"]["smoke"] else ""),
                     d["arms"]["E4+O"], e6_analysis(both)))
    for label, change, r, a in runs:
        ks = sorted(c.get("k", 1) for c in r["compiles"])
        nm = r["negative_mass_A_test"]
        out.append(
            f"| {label} | {change} | {a['class0_bpb']['E4+O']:.4f} | {a['class0_gain_over_E4']:+.4f} | "
            f"{a['class0_gain_over_rank1']:+.4f} | {verdict(a['C1'])} | {a['all_bpb']['E4+O']:.4f} | "
            f"{r['A_heldout_bpb']:.4f} | {ks[0]}, {ks[len(ks) // 2]}, {ks[-1]} | "
            f"{nm['mean_negative_share']:.4f} | {nm['share_of_bytes_with_negative']:.4f} | {nm['resets']:.0f} |")
    c0 = e6["arms"]["E4+O rank 1"]["A_test_decomposition"]["0"]["bpb"]
    out += ["", f"Comparators from E6: class 0 of E4 {base['class0_bpb']['E4']:.4f}, of rank 1 {c0:.4f}.", ""]
    ran = [d for d in (ra, rb) if d is not None]
    if len(ran) == 2:
        above = all(e6_analysis({"arms": {**{k: e6["arms"][k] for k in ("E4", "E4+O rank 1")},
                                          "E4+O": d["arms"]["E4+O"]}})["class0_gain_over_rank1"] <= 0 for d in ran)
        out += ["Stopping rule (Amendment 1): " + (
            "both reruns leave O at or above rank 1 on class 0: the reading is \"spectral O at L <= 4 learns nothing "
            "beyond the byte unigram on this corpus\", and tuning stops." if above else
            "at least one rerun puts O below rank 1 on class 0; the manager rules the next step."), ""]
    for d in ran:
        st = d["arms"]["E4+O"].get("store")
        if st:
            out.append(f"{d['rerun']}: store of {st['bytes']} bytes, {st['occupied_buckets']} of {st['buckets']} "
                       f"buckets occupied, {st['contexts_per_occupied_bucket']:.1f} contexts per occupied bucket, "
                       f"largest C3 cell {st['max_cell']}"
                       f"{'' if not d.get('c3_rows') else ', C3 over ' + str(d['c3_rows']) + ' rows'}; "
                       f"run time {d['seconds'] / 60:.0f} min.")
    return out


# ---------------------------------------------------------------------------------------------------
# E7: the criterion C-E7 and the reading rules (PROTOCOL E7.7), computed from e7.json, never typed
# ---------------------------------------------------------------------------------------------------

E7_BAR = 0.90               # C-E7: mean top-1 of E4 + F5 on determined held-out items at h = 0.2, each stage
E7_MARGIN = 0.5             # ... and above E4's mean on the same items by at least this much


def e7_collect(d: dict) -> dict:
    """``out[(stage, h, arm, relation)][seed]``: top-1 on determined, undetermined, all held-out and seen items, the
    mean bits of the correct candidate, the share determined, the share on which the stage was active."""
    import numpy as np
    out: dict = {}
    for u in d["units"].values():
        held = [r for r in u["items"] if not r["seen"]]
        det = [r for r in held if r["determined"]]
        und = [r for r in held if r["determined"] is False]
        act = [r["active"] for r in held if r["active"] is not None]

        def m(rows, key="top1"):
            return float(np.mean([r[key] for r in rows])) if rows else None

        out.setdefault((u["stage"], u["h"], u["arm"], u["relation"]), {})[u["seed"]] = {
            "det": m(det), "und": m(und), "all": m(held), "seen": u["seen_top1"], "bits": m(held, "bits_correct"),
            "share_det": (len(det) / len(held)) if (det or und) else None,
            "active": float(np.mean(act)) if act else None, "n_det": len(det), "n_und": len(und),
            "classified": u["classified"]}
    return out


def _agg(vals: list) -> str:
    import numpy as np
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    return f"{np.mean(v):.3f} ({min(v):.3f}, {np.median(v):.3f}, {max(v):.3f})"


def e7_analysis(d: dict) -> dict:
    """C-E7 per stage at h = 0.2, from the units: the means over the evaluation draws."""
    import numpy as np
    col = e7_collect(d)
    out = {"stages": {}}
    ok = True
    for stage in (1, 2):
        f5, e4 = col.get((stage, 0.2, "E4+F5", "sum")), col.get((stage, 0.2, "E4", "sum"))
        if not f5 or not e4:
            out["stages"][stage] = None
            ok = False
            continue
        det5 = [v["det"] for v in f5.values() if v["det"] is not None]
        det4 = [e4[s]["det"] for s, v in f5.items() if v["det"] is not None and e4.get(s, {}).get("det") is not None]
        mean5, mean4 = float(np.mean(det5)), float(np.mean(det4))
        passes = bool(mean5 >= E7_BAR and mean5 - mean4 >= E7_MARGIN)
        out["stages"][stage] = {"F5_mean": mean5, "F5_min": min(det5), "E4_mean": mean4, "draws": len(det5),
                                "pass": passes}
        ok = ok and passes
    out["C_E7"] = bool(ok)
    return out


def e7_section(d: dict) -> list[str]:
    """E7 (PROTOCOL E7): every number from e7.json; the criterion computed here. Synthetic data."""
    pv, a, col = d["provenance"], e7_analysis(d), e7_collect(d)
    out = ["## E7. Rotation families on a synthetic relation (synthetic data)", "",
           f"Run at commit `{pv['commit']}`, {pv['utc']}, {d['seconds'] / 3600:.1f} hours"
           f"{' (smoke run: not a result)' if pv['smoke'] else ''}; the family stage enters the chain at slot "
           f"`{d.get('slot', 'top')}`. Protocol: `experiments/PROTOCOL.md`, E7; design: `DESIGN.md` section 12, "
           "V3 to V5. F5 is the V5 route (masked CP decomposition, a chosen realization), F4 the exact V4 lattice "
           "route (an "
           "upper bound and the classifier), E4+O the linear-operator control of V3. Top-1 is the lowest-cost "
           "candidate among all results of the stage, from the machine's own probabilities. Each cell is mean "
           "(min, median, max) over the evaluation draws.", "",
           "### Held-out top-1 (higher is better)", "",
           "| stage | h | arm | determined | undetermined | all held-out | seen equations | bits of the correct "
           "result | stage active | share determined |", "|---|---|---|---|---|---|---|---|---|---|"]
    for stage in (1, 2):
        for h in (0.2, 0.4):
            for arm in ("E4", "E4+F5", "E4+F4", "E4+O"):
                runs = col.get((stage, h, arm, "sum"))
                if not runs:
                    continue
                v = list(runs.values())
                out.append(f"| {stage} | {h} | {arm} | {_agg([x['det'] for x in v])} | {_agg([x['und'] for x in v])} | "
                           f"{_agg([x['all'] for x in v])} | {_agg([x['seen'] for x in v])} | "
                           f"{_agg([x['bits'] for x in v])} | {_agg([x['active'] for x in v])} | "
                           f"{_agg([x['share_det'] for x in v])} |")
    out += ["", "### Criterion C-E7 (E4 + F5, h = 0.2, determined held-out items)", "",
            "| stage | draws | E4 + F5 mean | minimum over draws | E4 mean (same items) | result |",
            "|---|---|---|---|---|---|"]
    for stage in (1, 2):
        r = a["stages"].get(stage)
        if r is None:
            out.append(f"| {stage} | not run | | | | n/a |")
        else:
            out.append(f"| {stage} | {r['draws']} | {r['F5_mean']:.3f} | {r['F5_min']:.3f} | {r['E4_mean']:.3f} | "
                       f"{verdict(r['pass'])} |")
    out += ["", f"Bar: mean at least {E7_BAR} and at least {E7_MARGIN} above E4's mean, at both stages. "
            f"C-E7: {verdict(a['C_E7'])}.", ""]
    ctl = [(arm, col.get((1, 0.2, arm, "random"))) for arm in ("E4", "E4+F5", "E4+F4")]
    if any(r for _, r in ctl):
        out += ["### Relation control (stage 1, h = 0.2, a random commutative table in place of the sum)", "",
                "| arm | held-out top-1 (chance 1/19 = 0.053) | stage active |", "|---|---|---|"]
        for arm, runs in ctl:
            if runs:
                v = list(runs.values())
                out.append(f"| {arm} | {_agg([x['all'] for x in v])} | {_agg([x['active'] for x in v])} |")
        out.append("")
    fits = [f for u in d["units"].values() if u["arm"] == "E4+F5" for f in u.get("fits", [])]
    if fits:
        out += [f"F5 fits: {len(fits)}; mean residual {sum(f['mean_residual'] for f in fits) / len(fits):.4f}, tokens "
                f"dropped by the 0.05 rule {sum(f['dropped'] for f in fits)} in total; "
                f"fit time {sum(f['seconds'] for f in fits) / len(fits):.1f} s on average.", ""]
    ab = d.get("ablation_after0")
    if ab:
        out += ["### Slot ablation (E7 Amendment 1): the stage after level 0, as first written (stage 1, h = 0.2)", "",
                "| arm | determined held-out top-1 | all held-out top-1 | bits of the correct result |",
                "|---|---|---|---|"]
        for arm in ("E4+F5", "E4+F4"):
            runs = ab.get((1, 0.2, arm, "sum"))
            if runs:
                v = list(runs.values())
                out.append(f"| {arm} | {_agg([x['det'] for x in v])} | {_agg([x['all'] for x in v])} | "
                           f"{_agg([x['bits'] for x in v])} |")
        out.append("")
    out += ["### Gates (PROTOCOL E7.5)", ""]
    p0 = d["part0"]
    out += [f"(a) With the stages off the path equals E4 bit for bit on {p0['gate_a']['bytes']} bytes: "
            f"{verdict(p0['gate_a']['bit_identical'])}.",
            f"(b) The lattice code equals the Hermite normal form on {p0['gate_b']['systems']} random systems and is "
            f"exact when complete: {verdict(p0['gate_b']['agree'] == p0['gate_b']['systems'])}.",
            f"(c) V5's closed case: F5 top-1 {p0['gate_c']['mod10_top1']:.2f} on addition modulo 10 and "
            f"{p0['gate_c']['stage1_complete_top1']:.2f} on stage 1 with nothing held out: "
            f"{verdict(p0['gate_c']['mod10_top1'] >= 1.0 and p0['gate_c']['stage1_complete_top1'] >= 0.99)}.",
            f"(h) The parser agrees with the text rule: {verdict(p0['gate_h']['checked'])}.",
            "(d) soundness of F4 on the classified items, the frozen-read half of (e) (tables, equations and evidences "
            "unchanged), (f) stage regret at most 2 bits per node and (g) E4's seen-equation top-1 at least 0.99 are "
            "checked in every unit and stop the run when they fail; the causality half of (e) is a test only "
            "(`tests/test_families.py`).", ""]
    return out


# ---------------------------------------------------------------------------------------------------
# E6b: the criterion C-E6b and the reading rules (PROTOCOL E6b.7), computed from e6b.json, never typed
# ---------------------------------------------------------------------------------------------------

E6B_MARGIN = 0.02           # C-E6b: each of four differences at least this many bits per byte
E6B_TIE = 0.0005            # E4's tie band


def e6b_bootstrap(d: dict, key: str, draws: int = 1000, seed: int = 0) -> dict:
    """By-book bootstrap of the set-``key`` difference of E4+K against each comparator (bits per byte)."""
    import numpy as np
    arms = d["arms"]
    bits = {a: np.array(arms[a]["per_book"][key]["bits"]) for a in arms}
    n = np.array(arms["E4"]["per_book"][key]["bytes"], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(draws, len(n)))
    out = {}
    for ref in ("E4", "E4+K random"):
        diff = (bits["E4+K"][idx].sum(1) - bits[ref][idx].sum(1)) / np.maximum(n[idx].sum(1), 1.0)
        out[ref] = {"mean": float(diff.mean()), "low": float(np.percentile(diff, 2.5)),
                    "high": float(np.percentile(diff, 97.5)), "books": len(n)}
    return out


def e6b_analysis(d: dict) -> dict:
    arms = d["arms"]
    sets = {k: {a: arms[a]["set_bpb"][k] for a in arms} for k in ("class0", "U2", "U1", "class0_U2")}
    allb = {a: arms[a]["A_test_decomposition"]["all"]["bpb"] for a in arms}
    gains = {}
    for k in ("class0", "U2"):
        gains[k] = {"over_E4": sets[k]["E4"] - sets[k]["E4+K"], "over_random": sets[k]["E4+K random"] - sets[k]["E4+K"]}
    c = all(g[x] >= E6B_MARGIN for g in gains.values() for x in ("over_E4", "over_random"))
    out = {"sets": sets, "all": allb, "gains": gains, "C": bool(c),
           "A_test_gain_over_E4": allb["E4"] - allb["E4+K"],
           "A_test_gain_over_random": allb["E4+K random"] - allb["E4+K"],
           "heldout_gain_over_E4": arms["E4"]["A_heldout_bpb"] - arms["E4+K"]["A_heldout_bpb"],
           "heldout_gain_over_random": arms["E4+K random"]["A_heldout_bpb"] - arms["E4+K"]["A_heldout_bpb"],
           "loses": bool(allb["E4+K"] > allb["E4"] + E6B_TIE),
           "capacity_only": bool(abs(allb["E4+K"] - allb["E4+K random"]) <= E6B_TIE)}
    out["reading"] = ("C-E6b passes" if c else
                      "E6b loses: E4+K is worse than E4 on A-test by more than the tie band" if out["loses"] else
                      "C-E6b fails; E4+K is within 0.0005 of the random control on A-test: the effect is capacity, "
                      "not classes" if out["capacity_only"] else
                      "C-E6b fails; E4+K is not worse than E4 on A-test")
    return out


def e6b_section(d: dict) -> list[str]:
    """E6b (PROTOCOL E6b): every number from e6b.json; the criterion computed here."""
    pv, a, arms = d["provenance"], e6b_analysis(d), d["arms"]
    out = ["## E6b. Word-class levels on text", "",
           f"Run at commit `{pv['commit']}`, {pv['utc']}, {d['seconds'] / 3600:.1f} hours"
           f"{' (smoke run: not a result)' if pv['smoke'] else ''}; V = {pv.get('V', 'n/a')}. Protocol: "
           "`experiments/PROTOCOL.md`, E6b; design: `DESIGN.md` section 12, V2'. E4+K has the class levels K1, K2, K4; "
           "E4+K random has the same levels with a random class map of the same rows per class. U2 are the A-test "
           f"bytes whose last-2-unit context never occurs in A-train ({d['set_shares']['U2']:.3f} of A-test).", "",
           "### Bits per byte (frozen; lower is better)", "",
           "| arm | train | A-test | A-heldout (unseen books) | class 0 | U2 | U1 | class 0 and U2 |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in arms.items():
        sb = r["set_bpb"]
        out.append(f"| {name} | {r['train_bpb']:.4f} | {r['A_test_decomposition']['all']['bpb']:.4f} | "
                   f"{r['A_heldout_bpb']:.4f} | {sb['class0']:.4f} | {sb['U2']:.4f} | {sb['U1']:.4f} | "
                   f"{sb['class0_U2']:.4f} |")
    out += ["", "### Criterion C-E6b (E4+K against E4 and the random control; bits per byte, positive = E4+K lower)",
            "", "| set | against E4 | against the control | required | result |", "|---|---|---|---|---|"]
    for k, label in (("class0", "class 0"), ("U2", "U2")):
        g = a["gains"][k]
        ok = g["over_E4"] >= E6B_MARGIN and g["over_random"] >= E6B_MARGIN
        out.append(f"| {label} | {g['over_E4']:+.4f} | {g['over_random']:+.4f} | {E6B_MARGIN} | {verdict(ok)} |")
    out += ["", f"C-E6b: {verdict(a['C'])}. Reading: {a['reading']}.", "",
            f"All bytes of A-test: E4+K against E4 {a['A_test_gain_over_E4']:+.4f}, against the control "
            f"{a['A_test_gain_over_random']:+.4f}; unseen books: {a['heldout_gain_over_E4']:+.4f} and "
            f"{a['heldout_gain_over_random']:+.4f}.", ""]
    rows = []
    for k, label in (("class0", "class 0"), ("U2", "U2")):
        b = e6b_bootstrap(d, k)
        for ref, v in b.items():
            rows.append(f"{label}, E4+K minus {ref}: {v['mean']:+.4f} [{v['low']:+.4f}, {v['high']:+.4f}] over "
                        f"{v['books']} books")
    out += ["By-book bootstrap of the differences (bits per byte, 1000 draws, seed 0; an uncertainty, not a "
            "criterion): " + "; ".join(rows) + ".", "", "### The class maps", "",
            "| arm | compiles | seconds per compile (mean) | rows clustered (last) | churn between compiles (mean) | "
            "A-test unit tokens in the map |", "|---|---|---|---|---|---|"]
    for name in ("E4+K", "E4+K random"):
        r = arms[name]
        cs = r["compiles"]
        recorded = name == "E4+K" or any("real_churn" in c for c in cs)       # the first run copied the real churn
        churn = [c["churn"] for c in cs if "churn" in c]
        churn_cell = (f"{sum(churn) / len(churn):.3f}" if churn else "n/a") if recorded else "not recorded"
        out.append(f"| {name} | {len(cs)} | {sum(c['seconds'] for c in cs) / max(1, len(cs)):.1f} | "
                   f"{cs[-1]['rows'] if cs else 'n/a'} | {churn_cell} | "
                   f"{r.get('A_test_units', {}).get('in_map', 0.0):.3f} |")
    if arms["E4+K"].get("final_map_rows_per_class") and arms["E4+K random"].get("final_map_rows_per_class_active"):
        import numpy as np
        qs = [0, 25, 50, 75, 100]
        out += ["", "Rows per class at the end (min, q1, median, q3, max): real "
                f"{np.percentile(arms['E4+K']['final_map_rows_per_class'], qs).round(0).tolist()}, control "
                f"{np.percentile(arms['E4+K random']['final_map_rows_per_class_active'], qs).round(0).tolist()}."]
    rep = arms["E4"].get("e4_reproduction")
    out += ["", "### Gates (PROTOCOL E6b.5)", ""]
    p0 = d["part0"]
    out += [f"(a), (b) With the K levels off the path equals E4 bit for bit on {p0['gate_a_b']['bytes']} bytes and its "
            f"E4 keys equal hier's: {verdict(p0['gate_a_b']['bit_identical'] and p0['gate_a_b']['e4_keys_equal'])}.",
            f"(c), (e) The compile recovers 8 known classes with purity {p0['gate_c_e']['purity']:.3f} (at least 0.9) "
            f"and the control has the real row counts: {verdict(p0['gate_c_e']['purity'] >= 0.9)}.",
            "(d) Frozen reads leave the tables, the count store and the map unchanged: "
            f"{verdict(all(r['tables_unchanged_by_frozen_reads'] for r in arms.values()))}."]
    if rep:
        out.append(f"E4 reproduction: A-test {rep['A_test']:.10f} against the saved {rep['saved_A_test']:.10f} "
                   f"(difference {rep['abs_difference']:.1e}): {verdict(rep['abs_difference'] <= 1e-9)}.")
    out += ["Gates (b) (keys against the Python writing), (c) (determinism) and (d) (causality) are also tests "
            "(`tests/test_wordclass.py`).", ""]
    return out


def e6b_mix_analysis(mix: dict, e6b: dict) -> dict:
    """C-E6b's arithmetic on the mixtures (E6b Amendment 1): M against E4 and against M random."""
    m, r, e4 = mix["arms"]["M"], mix["arms"]["M random"], e6b["arms"]["E4"]
    gains = {}
    for k in ("class0", "U2"):
        gains[k] = {"over_E4": e4["set_bpb"][k] - m["set_bpb"][k], "over_random": r["set_bpb"][k] - m["set_bpb"][k]}
    c = all(g[x] >= E6B_MARGIN for g in gains.values() for x in ("over_E4", "over_random"))
    a_e4 = e4["A_test_decomposition"]["all"]["bpb"]
    a_m, a_r = m["A_test_decomposition"]["all"]["bpb"], r["A_test_decomposition"]["all"]["bpb"]
    out = {"gains": gains, "C": bool(c), "A_test": {"E4": a_e4, "M": a_m, "M random": a_r},
           "A_test_gain_over_E4": a_e4 - a_m, "A_test_gain_over_random": a_r - a_m,
           "heldout_gain_over_E4": e4["A_heldout_bpb"] - m["A_heldout_bpb"],
           "heldout_gain_over_random": r["A_heldout_bpb"] - m["A_heldout_bpb"],
           "loses": bool(a_m > a_e4 + E6B_TIE), "capacity_only": bool(abs(a_m - a_r) <= E6B_TIE)}
    out["reading"] = ("C-E6b passes for the mixture" if c else
                      "the mixture loses: M is worse than E4 on A-test by more than the tie band" if out["loses"] else
                      "C-E6b fails for the mixture; M is within 0.0005 of M random on A-test"
                      if out["capacity_only"] else
                      "C-E6b fails for the mixture; M is not worse than E4 on A-test")
    return out


def e6b_mix_section(mix: dict, e6b: dict) -> list[str]:
    """E6b Amendment 1 (added after the result): the top mixture, beside the original C-E6b."""
    import numpy as np
    pv, a = mix["provenance"], e6b_mix_analysis(mix, e6b)
    arms = {"E4": e6b["arms"]["E4"], "E4+K": e6b["arms"]["E4+K"], "E4+K random": e6b["arms"]["E4+K random"],
            **mix["arms"]}
    out = ["### E6b Amendment 1: the top mixture (added after the result; the original C-E6b stands)", "",
           f"Run at commit `{pv['commit']}`, {pv['utc']}, {mix['seconds'] / 3600:.1f} hours"
           f"{' (smoke run: not a result)' if pv['smoke'] else ''}. M mixes E4 and E4+K, M random mixes E4 and E4+K "
           "random, with one node per class of the deepest level contributing to the K machine's readout (at most 28 "
           "bits against the better input). Same streams, seeds and constants as the E6b run.", "",
           "| arm | train | A-test | A-heldout | class 0 | U2 | U1 | class 0 and U2 |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in arms.items():
        sb = r["set_bpb"]
        tr = r["train_bpb"]["mix"] if isinstance(r["train_bpb"], dict) else r["train_bpb"]
        out.append(f"| {name} | {tr:.4f} | {r['A_test_decomposition']['all']['bpb']:.4f} | {r['A_heldout_bpb']:.4f} | "
                   f"{sb['class0']:.4f} | {sb['U2']:.4f} | {sb['U1']:.4f} | {sb['class0_U2']:.4f} |")
    out += ["", "C-E6b's arithmetic on the mixtures (bits per byte, positive = M lower):", "",
            "| set | M against E4 | M against M random | required | result |", "|---|---|---|---|---|"]
    for k, label in (("class0", "class 0"), ("U2", "U2")):
        g = a["gains"][k]
        ok = g["over_E4"] >= E6B_MARGIN and g["over_random"] >= E6B_MARGIN
        out.append(f"| {label} | {g['over_E4']:+.4f} | {g['over_random']:+.4f} | {E6B_MARGIN} | {verdict(ok)} |")
    out += ["", f"C-E6b for the mixture: {verdict(a['C'])}. Reading: {a['reading']}.", "",
            f"All bytes of A-test: M against E4 {a['A_test_gain_over_E4']:+.4f}, against M random "
            f"{a['A_test_gain_over_random']:+.4f}; unseen books: {a['heldout_gain_over_E4']:+.4f} and "
            f"{a['heldout_gain_over_random']:+.4f}.", ""]
    rows = []
    for k, label in (("class0", "class 0"), ("U2", "U2")):
        bits_m = np.array(mix["arms"]["M"]["per_book"][k]["bits"])
        bits_r = np.array(mix["arms"]["M random"]["per_book"][k]["bits"])
        bits_e = np.array(e6b["arms"]["E4"]["per_book"][k]["bits"])
        n = np.array(e6b["arms"]["E4"]["per_book"][k]["bytes"], dtype=float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(n), size=(1000, len(n)))
        for ref, bref in (("E4", bits_e), ("M random", bits_r)):
            diff = (bits_m[idx].sum(1) - bref[idx].sum(1)) / np.maximum(n[idx].sum(1), 1.0)
            rows.append(f"{label}, M minus {ref}: {diff.mean():+.4f} [{np.percentile(diff, 2.5):+.4f}, "
                        f"{np.percentile(diff, 97.5):+.4f}]")
    out += ["By-book bootstrap of the differences (bits per byte, 1000 draws, seed 0; an uncertainty, not a "
            "criterion): " + "; ".join(rows) + ".", ""]
    for name in ("M", "M random"):
        r = mix["arms"][name]
        nd = r["nodes"]
        tot = sum(nd["bits"]) or 1.0
        used = [(i, nd["bits"][i] / tot, nd["evidence"][i]) for i in range(len(nd["bits"])) if nd["bits"][i] > 0]
        out.append(f"{name}: mixture regret at most {r['regret']['max_excess_bits']:.3f} bits at a node (bound 2) over "
                   f"{r['regret']['nodes_used']} nodes; nodes (deepest level: share of bits, final evidence in bits, "
                   "positive = prefers K): " + ", ".join(f"{i}: {sh:.3f}, {ev:+.0f}" for i, sh, ev in used) + ".")
        rep = r.get("reproduction")
        if rep:
            out.append(f"Components reproduce results/e6b.json: largest difference {max(rep.values()):.1e}.")
    out.append("")
    return out


def main() -> None:
    lines = ["# VAM results", "",
             "Generated by `experiments/report.py` from `experiments/results/*.json`; do not edit by hand. "
             "The protocol, fixed before the runs, is `experiments/PROTOCOL.md`.", ""]
    def load(name: str) -> dict | None:
        path = RESULTS / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    e1 = load("e1")
    sections = (("E1", e1, e1_section), ("E1b", load("e1b"), lambda d: e1b_section(d, e1)),
                ("E2", load("e2"), e2_section), ("E3", load("e3"),
                                       lambda d: e3_section(d, load("e2")) + e3_t4_section(load("e3_t4"), d)),
                ("E4", load("e4"), lambda d: e4_section(d, load("e3"), load("e2"))
                                       + e4_t_section(load("e4_t"), d, load("e3"))))
    e5 = load("e5")
    e5_questions = ROOT / "experiments" / "data" / "e5" / "questions.json"
    if e5 is not None and e5_questions.exists():
        from hashlib import sha256
        if sha256(e5_questions.read_bytes()).hexdigest() != e5["provenance"]["questions_sha256"]:
            raise SystemExit("e5.json was run on a different question file than data/e5/questions.json")
        sections += (("E5", e5, lambda d: e5_section(d, json.loads(e5_questions.read_text())["questions"])),)
    else:
        sections += (("E5", None, None),)
    sections += (("E6", load("e6"),
                  lambda d: e6_section(d) + e6_diag_section(d, load("e6_ra"), load("e6_rb"))
                  if load("e6_ra") or load("e6_rb") else e6_section(d)),)
    def e7_with_ablation(d: dict) -> list[str]:
        abl = load("e7_after0")
        return e7_section({**d, "ablation_after0": e7_collect(abl) if abl else None})

    sections += (("E7", load("e7"), e7_with_ablation),)
    sections += (("E6b", load("e6b"),
                  lambda d: e6b_section(d) + (e6b_mix_section(load("e6b_mix"), d) if load("e6b_mix") else [])),)
    for name, data, section in sections:
        lines += section(data) if data is not None else [f"## {name}: not run yet", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
