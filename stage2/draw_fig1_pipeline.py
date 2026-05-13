"""
Fig 1 — Full System Pipeline  (Stage 1 → 2 → 3)
실행: python draw_fig1_pipeline.py
출력: fig1_pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(18, 13))
ax.set_xlim(0, 18)
ax.set_ylim(0, 13)
ax.axis("off")
fig.patch.set_facecolor("white")

C_BORDER = "#90A4AE"


def box(x, y, w, h, fc, ec=C_BORDER, lw=1.5, r=0.15):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc))


def txt(x, y, s, fs=9, fw="normal", c="black",
        ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
            ha=ha, va=va, fontstyle=style, linespacing=1.4)


def arr(x1, y1, x2, y2, c="#546E7A", lw=2.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=18))


# ══════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════
ax.text(9.0, 12.72,
        "Fig 1.  System Pipeline for Fire Evacuation Guidance",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#212121",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#ECEFF1",
                  edgecolor=C_BORDER, lw=1.5))

# ══════════════════════════════════════════════════════
# ROW 1 — Three-stage pipeline
# ══════════════════════════════════════════════════════
# Stage 1
box(0.3, 10.75, 4.4, 1.75, "#2E7D32", ec="#1B5E20", lw=2.5, r=0.22)
txt(2.5, 12.17, "Stage 1", fs=14, fw="bold", c="white")
txt(2.5, 11.72, "Floor Plan Parsing  ->  Grid Map", fs=10, c="white")
txt(2.5, 11.3,  "CAD / image input", fs=8.5, c="#A5D6A7")
txt(2.5, 10.97, "Completed  (Sangsang Hall B2F)", fs=8, c="#81C784", style="italic")

arr(4.7, 11.62, 5.3, 11.62, c="#546E7A")

# Stage 2
box(5.3, 10.75, 7.2, 1.75, "#283593", ec="#1A237E", lw=2.5, r=0.22)
txt(8.9, 12.17, "Stage 2 — RL Training", fs=13, fw="bold", c="white")
txt(8.9, 11.72, "Autoregressive MDP  +  RecurrentPPO  +  4-Stage Curriculum", fs=9.5, c="#90CAF9")
txt(8.9, 11.3,  "Grid Map Env  |  Scenario Random Sampling  |  K_MAX = 64 steps/tick", fs=8.5, c="#BBDEFB")

arr(12.5, 11.62, 13.1, 11.62, c="#546E7A")

# Stage 3
box(13.1, 10.75, 4.6, 1.75, "#4527A0", ec="#311B92", lw=2.5, r=0.22)
txt(15.4, 12.17, "Stage 3", fs=14, fw="bold", c="white")
txt(15.4, 11.72, "Unity 3D Visualization", fs=10, c="white")
txt(15.4, 11.3,  ".onnx policy transfer", fs=8.5, c="#CE93D8")
txt(15.4, 10.97, "Crowd simulation", fs=8, c="#E1BEE7", style="italic")

txt(9.0, 10.5, "Stage 2 Detail:  4-Stage Curriculum Scenarios  +  Training Variables",
    fs=8.5, c="#546E7A", style="italic")

# ══════════════════════════════════════════════════════
# ROW 2 — 4-Stage Curriculum
# ══════════════════════════════════════════════════════
scenarios = [
    ("#5D4037", "#3E2723",
     "Stage 1",  "20 Agents",
     "Fixed fire origin\nBoth exits open\nBasic escape learning"),
    ("#6D4C41", "#4E342E",
     "Stage 2",  "40 Agents",
     "2 random fires\nIncreased spread speed\nSmoke vision blocking"),
    ("#7B3F00", "#4E1F00",
     "Stage 3",  "40 Agents",
     "Exit A blocked by fire\nDetour path search\nFastest spread (v=0.18)"),
    ("#7B1B1B", "#4A0F0F",
     "Stage 4",  "40 Agents",
     "Full-area fire spread\nCorridor collapse\nIrregular navigation"),
]
for i, (fc, ec, stage, n_agents, desc) in enumerate(scenarios):
    sx = 0.3 + i * 4.4
    box(sx, 8.15, 4.0, 2.0, fc, ec=ec, lw=2, r=0.22)
    txt(sx + 2.0, 9.83, stage,    fs=11, fw="bold", c="white")
    txt(sx + 2.0, 9.48, n_agents, fs=9,  c="#FFCC80")
    for j, line in enumerate(desc.split("\n")):
        txt(sx + 2.0, 9.05 - j * 0.30, line, fs=8, c="#EEEEEE")
    if i < 3:
        arr(sx + 4.0, 9.15, sx + 4.5, 9.15, c="#795548", lw=2.5)

txt(9.0, 7.92,
    "Curriculum advance:  survival rate >= 80%  |  rolling window = 50 episodes  "
    "|  resets to Stage 1 on restart",
    fs=7.8, c="#546E7A", style="italic")

# ══════════════════════════════════════════════════════
# ROW 3 — Per-episode Random Variables (3 columns)
# ══════════════════════════════════════════════════════
var_data = [
    ("Fire Parameters", [
        "Initial position  (random cell)",
        "Count:  1 ~ N fires",
        "Spread probability  v",
        "Smoke vision radius",
    ]),
    ("Agent Parameters", [
        "Initial position  (random)",
        "Population count  (20 or 40)",
        "Sequential decision order",
        "LSTM hidden state  (h, c)",
    ]),
    ("Environment Parameters", [
        "Exit open / blocked status",
        "Corridor collapse position",
        "Obstacle placement",
        "Scenario stage  (1 ~ 4)",
    ]),
]
for i, (title, items) in enumerate(var_data):
    sx = 0.3 + i * 5.9
    box(sx, 5.2, 5.5, 2.45, "#37474F", ec="#263238", lw=1.8, r=0.2)
    txt(sx + 2.75, 7.32, title, fs=10, fw="bold", c="white")
    for j, item in enumerate(items):
        txt(sx + 0.35, 6.9 - j * 0.38, f"• {item}", fs=8, c="#ECEFF1", ha="left")

txt(9.0, 5.05,
    "All parameters sampled randomly each episode to improve generalization",
    fs=8, c="#546E7A", style="italic")

# ══════════════════════════════════════════════════════
# ROW 4 — Observation Space (25-dim)
# ══════════════════════════════════════════════════════
box(0.3, 3.1, 17.4, 1.65, "#E3F2FD", ec="#1565C0", lw=1.8, r=0.15)
txt(9.0, 4.5, "Observation Space  (25-dim per agent per autoregressive step)",
    fs=10, fw="bold", c="#0D47A1")

obs_groups = [
    ("Cell Features  (8-dim)", "#C5CAE9", "#3949AB",
     "fire_near  smoke_near  density  active_flag  |  dist_A  dist_B  row  col"),
    ("Global Features  (15-dim)  F1~F15", "#FFF9C4", "#F57F17",
     "fire threats  congestion  panic  distances  |  fire centroid  approach speed"),
    ("Position  (2-dim)", "#FCE4EC", "#AD1457",
     "slot_idx / K_MAX   |   n_active / K_MAX"),
]
widths = [5.7, 6.5, 4.5]
xs     = [0.45, 6.35, 13.05]
for (title, fc, ec, desc), w, ox in zip(obs_groups, widths, xs):
    box(ox, 3.18, w, 1.35, fc, ec=ec, lw=1.5, r=0.12)
    txt(ox + w/2, 4.25, title, fs=9, fw="bold", c=ec)
    txt(ox + w/2, 3.74, desc,  fs=7.8, c="#37474F")

# ══════════════════════════════════════════════════════
# ROW 5 — Reward Design
# ══════════════════════════════════════════════════════
# Positive
box(0.3, 0.45, 7.9, 2.35, "#1B5E20", ec="#0A3D0A", lw=2, r=0.22)
txt(4.25, 2.5,  "+ Positive Reward",      fs=12, fw="bold",   c="white")
txt(4.25, 2.07, "Survival rate of escaped agents",  fs=9,  c="#A5D6A7")
txt(4.25, 1.7,  "+1.0 per escaped agent  +  fast-escape time bonus",  fs=8.5, c="#C8E6C9")
txt(4.25, 1.32, "Guided to correct exit  /  Cleared fire zone",       fs=8,   c="#81C784")

# Negative
box(9.8, 0.45, 7.9, 2.35, "#7F1515", ec="#4A0F0F", lw=2, r=0.22)
txt(13.75, 2.5,  "- Negative Reward",     fs=12, fw="bold",   c="white")
txt(13.75, 2.07, "Agent death penalty",   fs=9,  c="#FFAB91")
txt(13.75, 1.7,  "-1.0 per dead agent  -  panic-weighted step penalty",  fs=8.5, c="#FFCCBC")
txt(13.75, 1.32, "Guided into fire zone  /  Max-step timeout penalty",   fs=8,   c="#FF8A65")

txt(9.0, 1.7, "vs", fs=14, fw="bold", c="#546E7A")

# Action bar
box(0.3, 0.04, 17.4, 0.38, "#ECEFF1", ec=C_BORDER, lw=1, r=0.08)
txt(9.0, 0.23,
    "Action:  Discrete(2)  =  Guide to Exit A  /  Guide to Exit B"
    "          (one action per agent per autoregressive step)",
    fs=8.2, c="#37474F")

plt.tight_layout(pad=0.2)
out = "fig1_pipeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out}")
