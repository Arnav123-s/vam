"""E6b diagnosis: read-only decomposition of results/e6b.json (saved numbers only). Run from vam/."""

import json

d = json.load(open("experiments/results/e6b.json"))
A = d["arms"]
e4, k, r = A["E4"], A["E4+K"], A["E4+K random"]
WC = ("0", "8", "16", "32")
share = {c: e4["A_test_decomposition"][c]["share"] for c in WC}
print("window class share:", {c: round(v, 4) for c, v in share.items()})
print("difference to E4 in bits per byte (positive = worse), and its contribution to the A-test total (share x diff)")
for name, a in (("E4+K", k), ("E4+K random", r)):
    row, tot = {}, 0.0
    for c in WC:
        diff = a["A_test_decomposition"][c]["bpb"] - e4["A_test_decomposition"][c]["bpb"]
        row[c] = (round(diff, 4), round(share[c] * diff, 4))
        tot += share[c] * diff
    all_diff = a["A_test_decomposition"]["all"]["bpb"] - e4["A_test_decomposition"]["all"]["bpb"]
    print(name, row, "sum", round(tot, 4), "A-test all", round(all_diff, 4))
s = e4["set_share"]
print("set shares:", {x: round(v, 4) for x, v in s.items()})


def c0_outside_u2(a: dict) -> float:
    b = a["set_bpb"]
    return (s["class0"] * b["class0"] - s["class0_U2"] * b["class0_U2"]) / (s["class0"] - s["class0_U2"])


o4, ok, orr = c0_outside_u2(e4), c0_outside_u2(k), c0_outside_u2(r)
print(f"class 0 outside U2 (exact 2-unit context seen): E4 {o4:.4f}, E4+K {ok:.4f} ({ok - o4:+.4f}), "
      f"control {orr:.4f} ({orr - o4:+.4f})")
b4, bk, br = (x["set_bpb"]["class0_U2"] for x in (e4, k, r))
print(f"class 0 inside U2: E4 {b4:.4f}, E4+K {bk:.4f} ({bk - b4:+.4f}), control {br:.4f} ({br - b4:+.4f})")
for key in ("U1", "U2"):
    dk, dr = k["set_bpb"][key] - e4["set_bpb"][key], r["set_bpb"][key] - e4["set_bpb"][key]
    print(f"{key}: E4+K {dk:+.4f}, control {dr:+.4f} against E4")
du2 = k["set_bpb"]["U2"] - e4["set_bpb"]["U2"]
u2_out = (s["U2"] * du2 - s["class0_U2"] * (bk - b4)) / (s["U2"] - s["class0_U2"])
print(f"U2 outside class 0: E4+K minus E4 {u2_out:+.4f}")
tk, tr = k["train_bpb"] - e4["train_bpb"], r["train_bpb"] - e4["train_bpb"]
print(f"train (prequential): E4+K {tk:+.4f}, control {tr:+.4f}; "
      f"unseen books: E4+K {k['A_heldout_bpb'] - e4['A_heldout_bpb']:+.4f}, "
      f"control {r['A_heldout_bpb'] - e4['A_heldout_bpb']:+.4f}")
for name, a in (("E4", e4), ("E4+K", k), ("E4+K random", r)):
    c = a["counters"]
    print(name, "evictions", c["evictions"], "refused", c["refused"], "protected", c["protected"],
          "global occupancy", round(c["global_occupancy"], 3), "train seconds", round(a["train_seconds"]))
print("E4 occupancy by hashed level", {x: round(v, 2) for x, v in e4["counters"]["occupancy"].items()})
print("E4+K occupancy by hashed level", {x: round(v, 2) for x, v in k["counters"]["occupancy"].items()})
cs = k["compiles"]
churn = sorted(c["churn"] for c in cs[1:])
mean_s = sum(c["seconds"] for c in cs) / len(cs)
print("compiles", len(cs), "mean seconds", round(mean_s, 1), "churn first, median, last", round(cs[1]["churn"], 3),
      round(churn[len(churn) // 2], 3), round(cs[-1]["churn"], 3), "A-test unit tokens in the map",
      round(k["A_test_units"]["in_map"], 3))
ce, ck, cr = (a["learning_curve_bpb_per_MiB"] for a in (e4, k, r))
dk = [round(x - y, 3) for x, y in zip(ck, ce, strict=True)]
dr = [round(x - y, 3) for x, y in zip(cr, ce, strict=True)]
print("learning curve, E4+K minus E4 per MiB:", dk)
print("learning curve, control minus E4 per MiB:", dr)
behind_k = [i for i, x in enumerate(dk) if x > 0]
behind_r = [i for i, x in enumerate(dr) if x > 0]
print("MiB at which E4+K is behind E4:", behind_k, "; control behind:", behind_r)
same = all(a.get("churn") == b.get("churn") for a, b in zip(k["compiles"], r["compiles"], strict=True))
print("control compile churn equals the real map's (it is copied from the real map's info):", same)
