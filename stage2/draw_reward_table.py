"""
Reward structure table
실행: python draw_reward_table.py
출력: reward_table.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.2)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── 헤더 ──
col_x   = [0.15, 3.8, 6.7, 9.9, 12.3]   # left edges
col_w   = [3.55, 2.8, 3.1, 2.3, 1.6]
headers = ["Event", "Condition", "Formula", "Value", "Type"]
H_FC, H_EC = "#37474F", "#263238"

for x, w, h in zip(col_x, col_w, headers):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, 5.55), w - 0.08, 0.52,
        boxstyle="round,pad=0.06", linewidth=1.4,
        edgecolor=H_EC, facecolor=H_FC))
    ax.text(x + (w - 0.08) / 2, 5.81, h,
            ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")

# ── 행 데이터 ──────────────────────────────────────────────────────────
#  (event, condition, formula, value, type, row_fc)
rows = [
    ("Agent escapes",
     "Reached exit cell",
     "+20.0 per agent",
     "+20.0",
     "Positive",
     "#E8F5E9", "#2E7D32"),

    ("BFS progress bonus",
     "Each step while alive",
     "delta * 0.2 * urgency\nurgency = 1 + (t/T)*2",
     "+variable",
     "Positive",
     "#F1F8E9", "#388E3C"),

    ("Fire retreats from exit",
     "Fire dist to exit increases",
     "(cur_dist - prev_dist) * 1.5\n(Exit A + Exit B)",
     "+variable",
     "Positive",
     "#E3F2FD", "#1565C0"),

    ("Agent dies in fire",
     "Agent on fire cell",
     "-20.0 per agent",
     "-20.0",
     "Negative",
     "#FFEBEE", "#C62828"),

    ("Fire approaches exit",
     "Fire dist to exit decreases",
     "(cur_dist - prev_dist) * 1.5\n(Exit A + Exit B)",
     "-variable",
     "Negative",
     "#FFF3E0", "#E65100"),

    ("Exit imbalance penalty",
     "Both exits safe (fire_dist > 5)\n& n_alive > 1",
     "-|near_A - near_B| / n_alive",
     "up to -1.0",
     "Negative",
     "#FFF8E1", "#F9A825"),

    ("Episode end: not escaped",
     "Truncated or all done",
     "-15.0 per not-escaped agent",
     "-15.0 x k",
     "Negative",
     "#FFEBEE", "#B71C1C"),

    ("Episode end: both exits used",
     "escaped_A > 0 and escaped_B > 0",
     "+15.0 (one-time bonus)",
     "+15.0",
     "Positive",
     "#E8F5E9", "#1B5E20"),
]

ROW_H = 0.59
for i, (ev, cond, form, val, rtype, fc, ec) in enumerate(rows):
    y = 5.55 - (i + 1) * (ROW_H + 0.02) + 0.06
    # row background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.15, y), 13.68, ROW_H,
        boxstyle="round,pad=0.04", linewidth=1.0,
        edgecolor=ec, facecolor=fc))

    # type badge
    badge_fc = "#C8E6C9" if rtype == "Positive" else "#FFCDD2"
    badge_ec = "#2E7D32" if rtype == "Positive" else "#C62828"
    badge_c  = "#1B5E20" if rtype == "Positive" else "#B71C1C"
    ax.add_patch(mpatches.FancyBboxPatch(
        (12.33, y + 0.12), 1.42, 0.34,
        boxstyle="round,pad=0.05", linewidth=1.1,
        edgecolor=badge_ec, facecolor=badge_fc))
    ax.text(12.33 + 0.71, y + 0.29, rtype,
            ha="center", va="center",
            fontsize=8, fontweight="bold", color=badge_c)

    cy = y + ROW_H / 2
    # event
    ax.text(0.25, cy, ev, ha="left", va="center", fontsize=9, fontweight="bold", color="#212121")
    # condition (may be 2 lines)
    ax.text(3.85, cy, cond, ha="left", va="center", fontsize=8, color="#37474F", linespacing=1.35)
    # formula
    ax.text(6.75, cy, form, ha="left", va="center", fontsize=8,
            color="#212121", fontstyle="italic", linespacing=1.35)
    # value
    val_c = "#1B5E20" if val.startswith("+") else "#B71C1C"
    ax.text(10.95, cy, val, ha="center", va="center",
            fontsize=9, fontweight="bold", color=val_c)

# ── 제목 ──
ax.text(7.0, 6.08,
        "Reward Function Design  (env_core.py)",
        ha="center", va="center", fontsize=13, fontweight="bold", color="#212121",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#ECEFF1",
                  edgecolor="#90A4AE", lw=1.4))

plt.tight_layout(pad=0.15)
plt.savefig("reward_table.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved -> reward_table.png")
