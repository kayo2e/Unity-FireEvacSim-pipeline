"""
Fig 3 — BFS-guided Autoregressive MDP: Hybrid Decision Architecture
실행: python draw_fig3_bfs_ar.py
출력: fig3_bfs_ar.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis("off")
fig.patch.set_facecolor("white")

C_BORDER = "#90A4AE"


def box(x, y, w, h, fc, ec=C_BORDER, lw=1.5, r=0.12):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc))


def txt(x, y, s, fs=9, fw="normal", c="black",
        ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
            ha=ha, va=va, fontstyle=style, linespacing=1.4)


def arr(x1, y1, x2, y2, c="#546E7A", lw=1.6, label="", ldy=0.18):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=lw, mutation_scale=14))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + ldy, label, fontsize=7.5, color="#37474F",
                ha="center", style="italic")


# ══════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════
ax.text(9.0, 9.72,
        "Fig 3.  BFS-guided Autoregressive MDP — Hybrid Decision Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#212121",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECEFF1",
                  edgecolor=C_BORDER, lw=1.5))


# ══════════════════════════════════════════════════════
# SECTION A — BFS Preprocessing  (x: 0.2 ~ 5.0)
# ══════════════════════════════════════════════════════
box(0.2, 1.25, 4.8, 8.1, "#E3F2FD", ec="#1565C0", lw=2)
txt(2.6, 9.12, "BFS Preprocessing", fs=11, fw="bold", c="#0D47A1")
txt(2.6, 8.78, "_bfs_dist  (fire-aware BFS from exits)", fs=8.5, c="#1565C0", style="italic")

# ── Mini grid 6×6 ──
GX, GY, CS = 0.38, 5.45, 0.50
N = 6
fire_cells   = {(0, 0), (0, 1), (1, 0)}
exit_a_cells = {(0, 5)}
exit_b_cells = {(5, 4)}
agent_cells  = {(2, 2), (3, 3), (4, 1)}
light_cells  = {(1, 4), (2, 3), (2, 4), (3, 2), (3, 4)}

bfs_d = {
    (0, 0): "X", (0, 1): "X", (1, 0): "X",
    (0, 2): 3,   (0, 3): 2,   (0, 4): 1,   (0, 5): "A",
    (1, 1): 5,   (1, 2): 4,   (1, 3): 3,   (1, 4): 2,   (1, 5): 1,
    (2, 0): 6,   (2, 1): 5,   (2, 2): 4,   (2, 3): 3,   (2, 4): 2,   (2, 5): 2,
    (3, 0): 5,   (3, 1): 4,   (3, 2): 3,   (3, 3): 3,   (3, 4): 2,   (3, 5): 3,
    (4, 0): 4,   (4, 1): 3,   (4, 2): 2,   (4, 3): 2,   (4, 4): 1,   (4, 5): 2,
    (5, 0): 4,   (5, 1): 3,   (5, 2): 2,   (5, 3): 1,   (5, 4): "B", (5, 5): 1,
}

for r in range(N):
    for c in range(N):
        cx = GX + c * CS
        cy = GY + (N - 1 - r) * CS
        if (r, c) in fire_cells:
            fc_c, ec_g = "#EF5350", "#C62828"
        elif (r, c) in exit_a_cells:
            fc_c, ec_g = "#66BB6A", "#2E7D32"
        elif (r, c) in exit_b_cells:
            fc_c, ec_g = "#42A5F5", "#1565C0"
        elif (r, c) in agent_cells:
            fc_c, ec_g = "#FFA726", "#E65100"
        elif (r, c) in light_cells:
            fc_c, ec_g = "#FFF9C4", "#F9A825"
        else:
            fc_c, ec_g = "#FAFAFA", "#BDBDBD"

        ax.add_patch(mpatches.Rectangle(
            (cx, cy), CS - 0.04, CS - 0.04,
            lw=0.9, ec=ec_g, fc=fc_c))

        if (r, c) in bfs_d:
            d = bfs_d[(r, c)]
            is_special = (r, c) in fire_cells or (r, c) in exit_a_cells or (r, c) in exit_b_cells
            txt_c = "white" if is_special else (
                "#F9A825" if (r, c) in light_cells else "#37474F")
            ax.text(cx + (CS - 0.04) / 2, cy + (CS - 0.04) / 2,
                    str(d), ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txt_c)

# Grid legend
legend_items = [("#EF5350", "Fire"), ("#FFA726", "Agent"),
                ("#66BB6A", "Exit A"), ("#42A5F5", "Exit B"), ("#FFF9C4", "Light cell")]
for i, (lc, lt) in enumerate(legend_items):
    lx = 0.38 + i * 0.88
    ax.add_patch(mpatches.Rectangle((lx, 5.28), 0.17, 0.13, fc=lc, lw=0, zorder=3))
    ax.text(lx + 0.21, 5.345, lt, fontsize=6.5, va="center")

# ── _compute_path_slots output ──
box(0.35, 1.38, 4.45, 3.65, "#E8EAF6", ec="#3949AB", lw=1.3, r=0.1)
txt(2.6, 4.82, "_compute_path_slots()", fs=9.5, fw="bold", c="#283593")
txt(2.6, 4.52, "Trace BFS path per agent -> collect light cells", fs=8, c="#37474F")
txt(2.6, 4.25, "Sort by dist to exit  (closest = Slot 0):", fs=8, c="#37474F")

slots_info = [
    ("Slot 0", "cell (1,4)", "dist = 2", "#E8F5E9", "#2E7D32"),
    ("Slot 1", "cell (2,3)", "dist = 3", "#F1F8E9", "#388E3C"),
    ("Slot 2", "cell (2,4)", "dist = 4", "#F9FBE7", "#558B2F"),
    ("Slot 3", "cell (3,2)", "dist = 5", "#FFFDE7", "#F9A825"),
    ("  ...  ", "  ...  ",   "...",       "#FAFAFA", "#9E9E9E"),
]
for i, (sl, cell, dist, sfc, sec) in enumerate(slots_info):
    sy = 3.88 - i * 0.47
    box(0.5, sy - 0.16, 4.15, 0.35, sfc, ec=sec, lw=0.8, r=0.06)
    txt(1.35, sy, sl,   fs=8,   fw="bold", c=sec)
    txt(2.6,  sy, cell, fs=7.8, c="#37474F")
    txt(4.3,  sy, dist, fs=7.8, fw="bold", c=sec)


# ══════════════════════════════════════════════════════
# SECTION B — Autoregressive LSTM  (x: 6.2 ~ 12.1)
# ══════════════════════════════════════════════════════
box(6.2, 1.25, 5.9, 8.1, "#E8F5E9", ec="#2E7D32", lw=2)
txt(9.15, 9.12, "Autoregressive LSTM Decisions  (K_MAX steps / tick)", fs=10.5, fw="bold", c="#1B5E20")

slot_ys    = [7.3, 5.45, 3.6]
slot_shade = ["#A5D6A7", "#C8E6C9", "#DCEDC8"]

for i, (sy, sfc) in enumerate(zip(slot_ys, slot_shade)):
    # slot label
    box(6.3, sy + 0.42, 0.62, 0.58, sfc, ec="#43A047", lw=1.1, r=0.07)
    txt(6.61, sy + 0.71, f"Slot\n{i}", fs=7.5, fw="bold", c="#1B5E20")

    # obs
    box(7.02, sy + 0.08, 1.5, 0.92, "#E3F2FD", ec="#1565C0", lw=1.2, r=0.08)
    txt(7.77, sy + 0.68, f"obs_{i}  (25-dim)", fs=8, fw="bold", c="#0D47A1")
    txt(7.77, sy + 0.3,  "cell(8)+global(15)+pos(2)", fs=7, c="#1565C0")

    arr(8.52, sy + 0.54, 8.84, sy + 0.54, c="#1565C0", lw=1.2)

    # FC
    box(8.84, sy + 0.17, 0.82, 0.74, "#F1F8E9", ec="#388E3C", lw=1.2, r=0.07)
    txt(9.25, sy + 0.54, "FC(64)\n x2", fs=7.8, c="#2E7D32")

    arr(9.66, sy + 0.54, 9.96, sy + 0.54, c="#388E3C", lw=1.2)

    # LSTM
    box(9.96, sy + 0.04, 1.05, 1.0, "#EDE7F6", ec="#6A1B9A", lw=1.5, r=0.08)
    txt(10.49, sy + 0.54, "LSTM\n(256)", fs=8.5, fw="bold", c="#4A148C")

    arr(11.01, sy + 0.54, 11.32, sy + 0.54, c="#6A1B9A", lw=1.2)

    # action
    box(11.32, sy + 0.17, 0.72, 0.74, "#FFCDD2", ec="#C62828", lw=1.2, r=0.07)
    txt(11.68, sy + 0.68, f"a_{i}", fs=8.5, fw="bold", c="#B71C1C")
    txt(11.68, sy + 0.32, "{0,1}", fs=7.5, c="#B71C1C")

# LSTM (h, c) arrows between slots
for i in range(len(slot_ys) - 1):
    y_top = slot_ys[i] + 0.04
    y_bot = slot_ys[i + 1] + 1.04
    ax.annotate("", xy=(10.49, y_bot), xytext=(10.49, y_top),
                arrowprops=dict(arrowstyle="->", color="#7B1FA2", lw=1.8,
                                connectionstyle="arc3,rad=-0.5"))
    txt(9.98, (y_top + y_bot) / 2, "(h,c)", fs=8, c="#7B1FA2", style="italic")

# dots
txt(9.15, 2.95, "•  •  •", fs=14, c="#43A047")

# note
box(6.3, 1.38, 5.7, 1.22, "#F9FBE7", ec="#33691E", lw=1, r=0.08)
txt(9.15, 2.3, "pos feature = [slot_idx / K_MAX,  n_active / K_MAX]", fs=8, c="#33691E")
txt(9.15, 1.85,
    "LSTM maintains consistency across all K decisions within one tick",
    fs=7.8, c="#33691E", style="italic")


# ══════════════════════════════════════════════════════
# SECTION C — BFS Execution  (x: 13.1 ~ 17.8)
# ══════════════════════════════════════════════════════
box(13.1, 1.25, 4.7, 8.1, "#FFF8E1", ec="#F57F17", lw=2)
txt(15.45, 9.12, "BFS Execution  (_dir_toward)", fs=11, fw="bold", c="#E65100")

# Action = 0  →  Exit A
box(13.25, 6.25, 4.4, 2.65, "#E8F5E9", ec="#2E7D32", lw=1.5, r=0.12)
txt(15.45, 8.67, "action = 0   ->   Exit A", fs=10, fw="bold", c="#1B5E20")
box(13.4, 7.2, 4.1, 0.65, "#C8E6C9", ec="#43A047", lw=1, r=0.07)
txt(15.45, 7.525, "dist_map  =  _dist_to_exit_A", fs=8.5, c="#2E7D32", style="italic")
box(13.4, 6.35, 4.1, 0.77, "white", ec="#43A047", lw=0.9, r=0.06)
txt(15.45, 6.82, "argmin  dist_map[N / S / E / W neighbor]", fs=8, c="#212121")
txt(15.45, 6.5,  "=>  guidance direction toward Exit A", fs=8, c="#2E7D32", fw="bold")

# Action = 1  →  Exit B
box(13.25, 3.3, 4.4, 2.65, "#E3F2FD", ec="#1565C0", lw=1.5, r=0.12)
txt(15.45, 5.72, "action = 1   ->   Exit B", fs=10, fw="bold", c="#0D47A1")
box(13.4, 4.25, 4.1, 0.65, "#BBDEFB", ec="#1976D2", lw=1, r=0.07)
txt(15.45, 4.575, "dist_map  =  _dist_to_exit_B", fs=8.5, c="#1565C0", style="italic")
box(13.4, 3.4, 4.1, 0.77, "white", ec="#1976D2", lw=0.9, r=0.06)
txt(15.45, 3.87, "argmin  dist_map[N / S / E / W neighbor]", fs=8, c="#212121")
txt(15.45, 3.55, "=>  guidance direction toward Exit B", fs=8, c="#1565C0", fw="bold")

# Key insight
box(13.25, 1.38, 4.4, 1.7, "#FFF3E0", ec="#E65100", lw=1.5, r=0.12)
txt(15.45, 2.82, "Key Insight", fs=10, fw="bold", c="#E65100")
txt(15.45, 2.45, "RL decides WHICH exit    (learned)", fs=8.5, c="#BF360C")
txt(15.45, 2.1,  "BFS computes direction  (rule-based)", fs=8.5, c="#BF360C")
txt(15.45, 1.7,  "Hybrid: policy + fire-aware navigation", fs=8, c="#E65100", style="italic")


# ══════════════════════════════════════════════════════
# ARROWS between sections
# ══════════════════════════════════════════════════════
arr(5.0, 6.5, 6.2, 6.5, c="#37474F", lw=2.2, label="path_slots + obs", ldy=0.2)
arr(12.1, 6.5, 13.1, 6.5, c="#37474F", lw=2.2, label="actions [a_0..a_K]", ldy=0.2)


# ══════════════════════════════════════════════════════
# BOTTOM BAR — Role Separation
# ══════════════════════════════════════════════════════
box(0.2, 0.05, 17.6, 1.0, "#ECEFF1", ec=C_BORDER, lw=1.2, r=0.1)

box(0.38, 0.13, 5.7, 0.8, "#FCE4EC", ec="#C62828", lw=1.3, r=0.08)
txt(0.74, 0.53, "RL (RecurrentPPO):", fs=9, fw="bold", c="#B71C1C", ha="left")
txt(3.2, 0.75, "HIGH-LEVEL — Which exit?", fs=8.5, fw="bold", c="#C62828")
txt(3.2, 0.47, "Discrete(2)  |  LSTM context  |  Trained by PPO", fs=7.5, c="#37474F")
txt(3.2, 0.2,  "2^K joint space  ->  K x Discrete(2)  sequential", fs=7.5, c="#C62828", style="italic")

box(6.3, 0.13, 5.8, 0.8, "#E3F2FD", ec="#1565C0", lw=1.3, r=0.08)
txt(6.68, 0.53, "BFS (Rule-based):", fs=9, fw="bold", c="#0D47A1", ha="left")
txt(9.3, 0.75, "LOW-LEVEL — Which direction?", fs=8.5, fw="bold", c="#1565C0")
txt(9.3, 0.47, "Fire-aware BFS  |  No training  |  Always optimal", fs=7.5, c="#37474F")
txt(9.3, 0.2,  "argmin dist_map[neighbor]  per slot", fs=7.5, c="#1565C0", style="italic")

box(12.3, 0.13, 5.3, 0.8, "#E8F5E9", ec="#2E7D32", lw=1.3, r=0.08)
txt(14.95, 0.75, "Combined Result", fs=8.5, fw="bold", c="#1B5E20")
txt(14.95, 0.47, "Tractable  +  Interpretable  +  Scalable", fs=7.5, c="#37474F")
txt(14.95, 0.2,  "AutoregressivePPO  +  BFS  =  Novel Framework", fs=7.5, c="#2E7D32", style="italic")


plt.tight_layout(pad=0.2)
plt.savefig("fig3_bfs_ar.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved -> fig3_bfs_ar.png")
