"""Generate RESULTS.md from the saved result files (experiments/PROTOCOL.md, reporting rules).

    python experiments/report.py

Every number in RESULTS.md is read from experiments/results/*.json; the pass
or fail of each pre-registered criterion is computed here, never typed.
"""

from __future__ import annotations

import itertools
import json
import subprocess

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
            f"Verified support: contexts and continuations seen at least {l2.get('text_min_count', 1)} times in "
            "each half of every split.", "",
            "| d (bytes) | verified contexts | share of positions outside the support | rank | split-half edge | "
            "cutoff |", "|---|---|---|---|---|---|",
            *[f"| {r['depth_bytes']} | {r['rows']:,} | {r.get('unverified_share', 0.0):.3f} | {r['rank']} | "
              f"{r['split_edge']:.4f} | {r['cutoff']:.4f} |" for r in l2["text_depth_profile"]], ""]
    prof = l2["text_depth_profile"]
    out += [f"The deeper the context, the more positions fall outside the verified support "
            f"({prof[0].get('unverified_share', 0.0):.1%} at d = {prof[0]['depth_bytes']}, "
            f"{prof[-1].get('unverified_share', 0.0):.1%} at d = {prof[-1]['depth_bytes']}), so the deep ranks are "
            "measured on fewer contexts and the turnaround is a proposal, not a measurement of where structure "
            "ends. It feeds one Part 2 candidate only, which A-validation accepts or rejects.", ""]
    for mc, o in sorted(l2.get("text_other_min_counts", {}).items(), key=lambda t: int(t[0])):
        out.append(f"- With contexts seen at least {mc} times: ranks {', '.join(map(str, o['ranks']))}; "
                   f"turnaround {o['turnaround_bytes']} bytes.")
    if l2.get("text_other_min_counts"):
        out.append("")
    first = p.get("N_L2_first_measurement")
    if first is not None:
        out += [f"First measurement (protocol amendment): {first['note']}. Its ranks: "
                + ", ".join(f"d = {r['depth_bytes']}: {r['rank']}" for r in first["text_depth_profile"]) + ".", ""]
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
    secs = fd.get("seconds_by_engine", {})
    for n in fd["largest_bond"]:
        diff = fd["max_abs_q_difference_vs_reference"].get(n)
        out.append(f"| {n} | {'reference' if diff is None else f'{diff:.1e}'} | {fd['largest_bond'][n]} | "
                   f"{fd['mean_bond'][n]:.1f} | {fd['discarded_per_event'][n]:.1e} | {secs.get(n, '')} |")
    out += ["", "Area law (largest operator entanglement of any cut after 40 bits):", "",
            "| cells | engine | largest cut entropy (bits) | middle cut (bits) | largest bond | discarded |",
            "|---|---|---|---|---|---|",
            *[f"| {r['cells']} | {r['engine']} | {r['max_cut_entropy_bits']:.3f} | "
              f"{r['middle_cut_entropy_bits']:.3f} | "
              f"{r['largest_bond']} | {r['discarded']:.1e} |" for r in p["N_L5_area_law"]["rows"]], "",
            "What the sinks change (the exact machine with and without them, same stream and emissions):", "",
            "| cells | context | theta | g | bits per bit, plain | with sinks | largest prediction change |",
            "|---|---|---|---|---|---|---|",
            *[f"| {r['cells']} | {r['context']} | {r['theta']} | {r['dephasing']} | {r['plain_bits_per_bit']:.5f} | "
              f"{r['sinks_bits_per_bit']:.5f} | {r['max_abs_q_change']:.1e} |"
              for r in p["N_L6_chain_sink_cost"]["rows"]],
            ""]
    return out


def code_changes(a: str, b: str) -> list[str] | None:
    """Files under src/ and experiments/ (results aside) that differ between two commits; None if unknown."""
    try:
        res = subprocess.run(["git", "diff", "--name-only", a.split("+")[0], b.split("+")[0], "--", "src",
                              "experiments", ":!experiments/results"], cwd=ROOT, capture_output=True, text=True,
                             check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [f.removeprefix("vam/") for f in res.stdout.split()]


def e3_provenance(d: dict) -> list[str]:
    """The commit each part ran at, and what changed in the code between those commits."""
    parts = [(f"Part 1, {k}", v) for k, v in d["part1"].items()]
    parts += [("Part 2", d["part2"]), ("Parts 3 and 4", d["part3"])]
    out = ["| part | commit | finished (UTC) | seconds |", "|---|---|---|---|"]
    order = []
    for name, v in parts:
        c = v.get("commit", "not recorded")
        note = v.get("commit_note") or ("stamped after the run" if "T" not in v.get("utc", "T") else "")
        shown = "not recorded" if c == "not recorded" else f"`{c[:7]}`" + ("+uncommitted" if "+" in c else "")
        out.append(f"| {name} | {shown}{f' ({note})' if note else ''} | {v.get('utc', '')} | "
                   f"{v.get('seconds', '')} |")
        if c != "not recorded" and c not in order:
            order.append(c)
    order.sort(key=lambda c: min(v.get("utc", "") for _, v in parts if v.get("commit") == c))
    out += [""]
    for a, b in itertools.pairwise(order):
        files = code_changes(a, b)
        what = ("unknown" if files is None else "no code (documentation only)" if not files
                else ", ".join(f"`{f}`" for f in files))
        out.append(f"- `{a[:7]}` to `{b[:7]}` changed: {what}.")
    return [*out, ""]


def e3_section(d: dict, e2: dict | None) -> list[str]:
    p2, p3, p4 = d["part2"], d["part3"], d["part4"]
    out = ["## E3. The nested fractal architecture, layer by layer", "",
           f"Saved at commit `{d['provenance']['commit']}`, {d['provenance']['utc']}; {d['seconds'] / 3600:.1f} "
           "hours of computation. Protocol: `experiments/PROTOCOL.md`, E3. The run was resumed from its "
           "checkpoint, so the parts ran at different commits:", ""]
    out += e3_provenance(d)
    out += e3_part1(d["part1"])
    rep = ("not checked (smoke streams)" if p2["reproduced"] is None else verdict(p2["reproduced"]))
    out += ["### Part 2. Configuration, chosen on A-validation only", "",
            f"Reproducibility: E2's best overall configuration gives {p2['start_A_validation_bpb']:.6f} bpb on "
            "A-validation "
            f"(E2: {p2['e2_start_bpb']:.6f}): {rep}.", ""]
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
                "printable | valid UTF-8 | own bits per byte |", "|---|---|---|---|---|---|---|---|---|"]
        for st in p4["settings"]:
            words = "" if st["words_in_A_train_vocabulary"] is None else f"{st['words_in_A_train_vocabulary']:.3f}"
            out.append(f"| {st['hbar']:g} | {st['form']} | {st['mode']} | {'on' if st['verified'] else 'off'} | "
                       f"{st['repeated_8byte_windows']:.3f} | {words} | {st['printable_bytes']:.3f} | "
                       f"{st['valid_utf8']:.3f} | {st['own_bits_per_byte']:.3f} |")
        name = lambda st: (f"hbar {st['hbar']:g}, {st['form']}, {st['mode']}, verified "
                           f"{'on' if st['verified'] else 'off'}")
        groups: dict = {}
        for st in p4["settings"]:
            groups.setdefault(tuple(o["text"] for o in st["outputs"]), []).append(st)
        same = [g for g in groups.values() if len(g) > 1]
        out += ["", f"{len(groups)} of the {len(p4['settings'])} settings give distinct outputs. At hbar = 0 only "
                "the least-action path has weight, so the form (Euclidean or Feynman) and the mode (collapse or "
                "sample) cannot change the step: those rows repeat one setting, not independent evidence.", ""]
        if same:
            out += ["Settings with identical outputs:", ""]
            out += [f"- {'; '.join(name(st) for st in g)}" for g in same]
        out += ["", "Outputs, verbatim (once per distinct setting):", ""]
        for g in groups.values():
            out += [f"<details><summary>{'; '.join(name(st) for st in g)}</summary>", ""]
            out += [f"- `{o['prompt']!r}` → `{o['text']!r}`" for o in g[0]["outputs"]]
            out += ["", "</details>", ""]
    return [*out, ""]


def e3_explore_section(d: dict) -> list[str]:
    ch = d["chosen"]
    out = ["### E3, exploration past the grid's edge (post hoc)", "",
           "Chosen after Part 2 had ended, because its best sink rate and protection were the smallest in the "
           "grid; A-validation only; it changes nothing E3 chose (protocol amendment). Frozen A-validation bpb "
           f"of the chosen configuration with one setting changed (reproduces Part 2 exactly: "
           f"{'yes' if d['reproduced'] else 'no'}):", "",
           "| protect (bits) | sink rate | A-validation bpb | protected knocks | evictions |", "|---|---|---|---|---|"]
    for r in d["rows"]:
        sp = r["spec"]
        mark = " (Part 2's choice)" if sp == ch else ""
        out.append(f"| {sp['protect']:g} | {sp['sink_rate']:g}{mark} | {r['A_validation_bpb']:.4f} | "
                   f"{r['protected']:,} | {r['evictions']:,} |")
    return [*out, ""]


def main() -> None:
    lines = ["# VAM results", "",
             "Generated by `experiments/report.py` from `experiments/results/*.json`; do not edit by hand. "
             "The protocol, fixed before the runs, is `experiments/PROTOCOL.md`.", ""]
    def load(name: str) -> dict | None:
        path = RESULTS / f"{name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    e1 = load("e1")
    sections = (("E1", e1, e1_section), ("E1b", load("e1b"), lambda d: e1b_section(d, e1)),
                ("E2", load("e2"), e2_section), ("E3", load("e3"), lambda d: e3_section(d, load("e2"))))
    for name, data, section in sections:
        lines += section(data) if data is not None else [f"## {name}: not run yet", ""]
    explore = load("e3_explore")
    if explore is not None:
        lines += e3_explore_section(explore)
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(ROOT / "RESULTS.md")


if __name__ == "__main__":
    main()
