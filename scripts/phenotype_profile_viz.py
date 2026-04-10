"""
Mock visualization of CharmPheno phenotype profiles.

Generates a synthetic example showing:
  (left)  a small set of discovered phenotypes, each a weighted list of
          top-N conditions (the beta_k rows of the HDP output);
  (right) one patient's phenotype profile (the theta_d vector) as a
          horizontal bar plot of phenotype loadings.

No real data is used. Outputs phenotype_profile_viz.png in this directory.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = "phenotype_profile_viz.png"

# Synthetic phenotypes: each is a name + top conditions with weights (sum to ~1)
phenotypes = [
    ("Metabolic Syndrome", [
        ("Type 2 diabetes", 0.32),
        ("Essential hypertension", 0.24),
        ("Hyperlipidemia", 0.18),
        ("Obesity", 0.14),
        ("Long-term insulin use", 0.08),
        ("Fatty liver", 0.04),
    ]),
    ("Cardiovascular", [
        ("Coronary artery disease", 0.28),
        ("Atrial fibrillation", 0.22),
        ("Heart failure", 0.18),
        ("Anticoagulation monitoring", 0.13),
        ("Chest pain, unspecified", 0.11),
        ("Syncope", 0.08),
    ]),
    ("Chronic Kidney Disease", [
        ("CKD stage 3", 0.30),
        ("Anemia of CKD", 0.20),
        ("Electrolyte disorder", 0.16),
        ("Proteinuria", 0.14),
        ("Secondary hyperparathyroidism", 0.12),
        ("Fluid overload", 0.08),
    ]),
    ("Respiratory / COPD", [
        ("COPD", 0.34),
        ("Chronic cough", 0.20),
        ("Tobacco use disorder", 0.16),
        ("Bronchodilator therapy", 0.14),
        ("Acute exacerbation of COPD", 0.10),
        ("Sleep apnea", 0.06),
    ]),
    ("Depression / Anxiety", [
        ("Major depressive disorder", 0.30),
        ("Generalized anxiety", 0.22),
        ("SSRI therapy", 0.18),
        ("Insomnia", 0.14),
        ("Adjustment disorder", 0.10),
        ("Counseling encounter", 0.06),
    ]),
    ("Osteoarthritis / MSK", [
        ("Osteoarthritis, knee", 0.28),
        ("Low back pain", 0.22),
        ("NSAID therapy", 0.16),
        ("Physical therapy encounter", 0.14),
        ("Osteoarthritis, hip", 0.12),
        ("Joint injection", 0.08),
    ]),
    ("Acute Infection", [
        ("Acute URI", 0.26),
        ("Antibiotic therapy", 0.22),
        ("Fever, unspecified", 0.18),
        ("Cough", 0.16),
        ("Pharyngitis", 0.10),
        ("Sinusitis", 0.08),
    ]),
    ("Preventive Care", [
        ("Annual wellness visit", 0.30),
        ("Immunization", 0.22),
        ("Screening mammogram", 0.16),
        ("Cholesterol screening", 0.14),
        ("Colonoscopy screening", 0.10),
        ("Hemoglobin A1c screening", 0.08),
    ]),
]

# One patient's phenotype profile (theta_d) — loads on metabolic syndrome,
# cardiovascular, and CKD; near-zero on everything else. Includes a small
# uncertainty bar for each to signal the Bayesian posterior.
rng = np.random.default_rng(3)
theta = np.array([0.34, 0.26, 0.22, 0.03, 0.05, 0.06, 0.01, 0.03])
theta = theta / theta.sum()
theta_err = np.array([0.04, 0.05, 0.04, 0.015, 0.02, 0.025, 0.01, 0.015])

# --- Plot ---
fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
ax_left = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[0, 1])

# Left panel: phenotype "cards" with top conditions
ax_left.set_xlim(0, 10)
ax_left.set_ylim(0, len(phenotypes))
ax_left.invert_yaxis()
ax_left.axis("off")
ax_left.set_title(
    "Discovered phenotypes (top conditions by weight)",
    fontsize=13, fontweight="bold", pad=12,
)

card_colors = plt.cm.tab10(np.linspace(0, 1, len(phenotypes)))

for i, (name, conds) in enumerate(phenotypes):
    y_top = i + 0.05
    y_bot = i + 0.95
    # Card background
    box = FancyBboxPatch(
        (0.05, y_top), 9.9, y_bot - y_top,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=card_colors[i], edgecolor="#444", alpha=0.12, linewidth=0.8,
    )
    ax_left.add_patch(box)
    # Phenotype name
    ax_left.text(
        0.25, i + 0.27, f"Phenotype {i+1}: {name}",
        fontsize=10.5, fontweight="bold", va="center",
    )
    # Conditions as inline weighted list
    cond_text = "   ".join(f"{c} ({w:.2f})" for c, w in conds[:5])
    ax_left.text(
        0.25, i + 0.68, cond_text,
        fontsize=8.2, va="center", color="#333", style="italic",
    )

# Right panel: per-patient phenotype profile bar chart with error bars
names = [p[0] for p in phenotypes]
y_pos = np.arange(len(names))
bars = ax_right.barh(
    y_pos, theta, xerr=theta_err,
    color=card_colors, edgecolor="#333", linewidth=0.6,
    error_kw=dict(ecolor="#333", capsize=3, lw=1),
)
ax_right.set_yticks(y_pos)
ax_right.set_yticklabels(names, fontsize=9.5)
ax_right.invert_yaxis()
ax_right.set_xlim(0, max(theta + theta_err) * 1.15)
ax_right.set_xlabel("Phenotype loading  (θₖ, with 95% credible interval)", fontsize=10)
ax_right.set_title(
    "Patient phenotype profile",
    fontsize=13, fontweight="bold", pad=12,
)
ax_right.grid(axis="x", linestyle=":", alpha=0.5)
ax_right.set_axisbelow(True)
for spine in ("top", "right"):
    ax_right.spines[spine].set_visible(False)

# Numeric labels at bar tips
for bar, val in zip(bars, theta):
    ax_right.text(
        bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}", va="center", fontsize=8.5, color="#222",
    )

fig.suptitle(
    "CharmPheno: Discovered phenotypes and a per-patient profile (illustrative; synthetic data)",
    fontsize=14, fontweight="bold",
)

fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
