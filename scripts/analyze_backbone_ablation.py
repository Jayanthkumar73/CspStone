"""
Post-hoc analysis of backbone ablation JSON results.
Generates:
  1. Paper-ready LaTeX tables (QMUL + SCface)
  2. Delta-over-bicubic summary (honest: shows where SR helps AND hurts)
  3. Score-gap comparison across backbones
  4. Console-friendly ASCII summary

Usage:
  python scripts/analyze_backbone_ablation.py \
    --results_dir results/backbone_ablation
"""

import json, argparse
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX: QMUL table  (mirrors Table 9 in paper, extended)
# ─────────────────────────────────────────────────────────────────────────────

def latex_qmul(data: dict) -> str:
    rows = []
    for backbone, methods in data.items():
        bic_tar1 = methods.get("Bicubic", {}).get("tar_at_far", {}).get("0.01", 0) * 100
        ada_tar1 = methods.get("AdaFace-SR", {}).get("tar_at_far", {}).get("0.01", 0) * 100
        delta    = ada_tar1 - bic_tar1

        for method, r in methods.items():
            t    = r["tar_at_far"]
            gap  = r.get("score_gap", 0)
            auc  = r["auc"]
            is_best = method == "AdaFace-SR" and delta > 0

            row = (
                f"  {backbone.replace('_', r'-')} & {method} & "
                f"{auc:.1f} & "
                f"{t.get('0.3', t.get('0.30', 0))*100:.1f} & "
                f"{t.get('0.1', t.get('0.10', 0))*100:.1f} & "
                f"{t.get('0.01', 0)*100:.1f} & "
                f"{t.get('0.001', 0)*100:.1f} & "
                f"{gap:.3f} \\\\"
            )
            if is_best:
                row = r"  \rowcolor{best}" + "\n" + row
            rows.append(row)
        rows.append(r"  \midrule")

    body = "\n".join(rows[:-1])  # drop trailing \midrule
    return rf"""
\begin{{table*}}[!t]
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{%
  Backbone Ablation on QMUL-SurvFace.
  AdaFace-SR (SR branch trained with frozen ArcFace iResNet-50 identity loss)
  is evaluated against a bicubic baseline across three independent FR backbones.
  \textbf{{Gap}} = genuine$-$impostor score mean.
  $\checkmark$/$\times$ in TAR@1\% column indicates improvement/degradation
  over bicubic for that backbone.
}}
\label{{tab:backbone_ablation_qmul}}
\centering
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}llccccccc@{{}}}}
\toprule
\textbf{{FR Backbone}} & \textbf{{Method}}
  & \textbf{{AUC}} & \textbf{{@30\%}} & \textbf{{@10\%}}
  & \textbf{{@1\%}} & \textbf{{@0.1\%}} & \textbf{{Score Gap}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX: SCface table
# ─────────────────────────────────────────────────────────────────────────────

def latex_scface(data: dict) -> str:
    rows = []
    for backbone, methods in data.items():
        for method, r in methods.items():
            row = (
                f"  {backbone.replace('_', r'-')} & {method} & "
                f"{r.get(16, r.get('16', 0)):.1f} & "
                f"{r.get(24, r.get('24', 0)):.1f} & "
                f"{r.get(32, r.get('32', 0)):.1f} & "
                f"{r.get('native', 0):.1f} \\\\"
            )
            rows.append(row)
        rows.append(r"  \midrule")

    body = "\n".join(rows[:-1])
    return rf"""
\begin{{table}}[!t]
\renewcommand{{\arraystretch}}{{1.3}}
\caption{{%
  Backbone Ablation on SCface d3 Rank-1 (\%).
  Native-resolution column validates gate closure across all backbones.
}}
\label{{tab:backbone_ablation_scface}}
\centering
\begin{{tabular}}{{@{{}}llcccc@{{}}}}
\toprule
\textbf{{FR Backbone}} & \textbf{{Method}}
  & \textbf{{16\,px}} & \textbf{{24\,px}} & \textbf{{32\,px}} & \textbf{{Native}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def summary(qmul: dict, scface: dict | None):
    W = 72
    print("\n" + "═" * W)
    print("  BACKBONE ABLATION ANALYSIS SUMMARY")
    print("═" * W)

    # QMUL
    print("\n  QMUL-SurvFace  (TAR@FAR thresholds)\n")
    hdr = f"  {'Backbone':<22} {'Method':<14} {'AUC':>6} {'@10%':>7} {'@1%':>7} {'@0.1%':>7} {'Gap':>6}"
    print(hdr)
    print("  " + "─" * (W - 2))
    for backbone, methods in qmul.items():
        for method, r in methods.items():
            t = r["tar_at_far"]
            print(f"  {backbone:<22} {method:<14} "
                  f"{r['auc']:>6.1f} "
                  f"{t.get('0.1', t.get('0.10', 0))*100:>7.1f} "
                  f"{t.get('0.01', 0)*100:>7.1f} "
                  f"{t.get('0.001', 0)*100:>7.1f} "
                  f"{r.get('score_gap', 0):>6.3f}")
        print()

    # Delta over bicubic
    print("  ── Δ AdaFace-SR vs Bicubic (TAR@FAR=1%) ──\n")
    all_positive = True
    for backbone, methods in qmul.items():
        bic = methods.get("Bicubic", {}).get("tar_at_far", {}).get("0.01", 0) * 100
        ada = methods.get("AdaFace-SR", {}).get("tar_at_far", {}).get("0.01", 0) * 100
        d   = ada - bic
        tag = "✓ improves" if d > 0 else "✗ regresses"
        if d < 0:
            all_positive = False
        print(f"  {backbone:<22}  bic={bic:.1f}%  ada={ada:.1f}%  "
              f"Δ={d:+.1f}pp  {tag}")

    # Honest interpretation
    print()
    if all_positive:
        print("  FINDING: AdaFace-SR consistently improves over bicubic")
        print("  across ALL tested backbones → backbone-independent.")
    else:
        print("  FINDING: AdaFace-SR improves for some backbones but not all.")
        print("  This is consistent with Level-3 mismatch (Section VI-B):")
        print("  SR corrections trained with ArcFace identity loss do not")
        print("  uniformly satisfy other backbone decision boundaries at")
        print("  strict FAR thresholds. This motivates backbone-specific SR")
        print("  training as future work.")

    # SCface
    if scface:
        print("\n\n  SCface d3 Rank-1 (%) — native resolution (gate closure test)\n")
        print(f"  {'Backbone':<22} {'Method':<14} {'16px':>7} {'24px':>7} {'32px':>7} {'Native':>7}")
        print("  " + "─" * (W - 2))
        for backbone, methods in scface.items():
            for method, r in methods.items():
                print(f"  {backbone:<22} {method:<14} "
                      f"{r.get(16, r.get('16', 0)):>7.1f} "
                      f"{r.get(24, r.get('24', 0)):>7.1f} "
                      f"{r.get(32, r.get('32', 0)):>7.1f} "
                      f"{r.get('native', 0):>7.1f}")
            print()

        print("  Native-resolution test: if the gate closes correctly, all")
        print("  backbones should show near-zero degradation vs bicubic.")
        for backbone, methods in scface.items():
            bic = methods.get("Bicubic", {}).get("native", 0)
            ada = methods.get("AdaFace-SR", {}).get("native", 0)
            gap = ada - bic
            tag = "PASS (≤0.5pp)" if abs(gap) <= 0.5 else "MARGINAL" if abs(gap) <= 1.5 else "FAIL"
            print(f"  {backbone:<22}  bic={bic:.1f}%  ada={ada:.1f}%  Δ={gap:+.1f}pp  [{tag}]")

    print("\n" + "═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/backbone_ablation")
    args = p.parse_args()

    rd = Path(args.results_dir)

    qmul   = load(rd / "qmul_results.json")   if (rd / "qmul_results.json").exists()   else {}
    scface = load(rd / "scface_results.json") if (rd / "scface_results.json").exists() else None

    summary(qmul, scface)

    if qmul:
        tex = latex_qmul(qmul)
        out = rd / "table_qmul.tex"
        out.write_text(tex)
        print(f"LaTeX (QMUL)   → {out}")

    if scface:
        tex = latex_scface(scface)
        out = rd / "table_scface.tex"
        out.write_text(tex)
        print(f"LaTeX (SCface) → {out}")


if __name__ == "__main__":
    main()