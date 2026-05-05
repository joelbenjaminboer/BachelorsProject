#!/usr/bin/env python3
"""
Comprehensive visualization of the IMU_Intent_Encoder transformer architecture.

Usage:
    python visualize_model.py
    python visualize_model.py --d_model 256 --num_heads 8 --num_layers 6
    python visualize_model.py --output my_model.png
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize IMU_Intent_Encoder architecture")
    p.add_argument("--input_features",   type=int,   default=6,       help="IMU sensor axes")
    p.add_argument("--context_length",   type=int,   default=25,     help="Input sequence length")
    p.add_argument("--forecast_horizon", type=int,   default=10,      help="Prediction output steps")
    p.add_argument("--d_model",          type=int,   default=256,     help="Embedding dimension")
    p.add_argument("--num_heads",        type=int,   default=8,       help="Attention heads")
    p.add_argument("--num_layers",       type=int,   default=6,       help="Encoder depth")
    p.add_argument("--dim_feedforward",  type=int,   default=1024,    help="FFN hidden size")
    p.add_argument("--dropout",          type=float, default=0.1,     help="Dropout probability")
    p.add_argument("--pe_base",          type=float, default=10000.0, help="Positional encoding base")
    p.add_argument("--output",           type=str,   default="model_visualization.png")
    return p.parse_args()


# ── Parameter counting ─────────────────────────────────────────────────────────

def count_params(cfg):
    F, D, L, Dff, H = (
        cfg.input_features, cfg.d_model, cfg.num_layers,
        cfg.dim_feedforward, cfg.forecast_horizon,
    )
    input_proj  = F * D + D
    cls_mask    = 2 * D
    # MHA: 4D²+4D  ·  LN1+LN2: 4D  ·  FFN: 2D·Dff+Dff+D
    per_layer   = 4*D*D + 9*D + 2*D*Dff + Dff
    transformer = L * per_layer
    final_ln    = 2 * D
    recon_head  = D * F + F
    reg_head    = D*Dff + Dff + Dff*H + H
    total       = input_proj + cls_mask + transformer + final_ln + recon_head + reg_head
    return {
        "Input Projection":    input_proj,
        "CLS + Mask Tokens":   cls_mask,
        f"Transformer (×{L})": transformer,
        "Final LayerNorm":     final_ln,
        "Reconstruction Head": recon_head,
        "Regression Head":     reg_head,
        "_total":              total,
    }


def fmt(n):
    if n >= 1_000_000: return f"{n/1e6:.2f}M"
    if n >= 1_000:     return f"{n/1e3:.1f}K"
    return str(n)


# ── Shared palette ─────────────────────────────────────────────────────────────

PAL = dict(
    input="#4A90D9", proj="#5BA55B", token="#E8A838",
    pe="#9B59B6",    xfm="#C0392B", norm="#1ABC9C",
    recon="#E67E22", reg="#2980B9", io="#607D8B",
)


# ── Box / arrow helpers ────────────────────────────────────────────────────────

def _box(ax, cx, cy, w, h, title, sub, color, fs_title=8.5, fs_sub=6.5):
    r = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="white", alpha=0.92, linewidth=1.8, zorder=2,
    )
    ax.add_patch(r)
    # title offset: push up when sub-label present so both have breathing room
    dy = 0.20 if sub else 0
    ax.text(cx, cy + dy, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", color="white", zorder=3)
    if sub:
        # sub placed well inside box bottom
        ax.text(cx, cy - 0.22, sub, ha="center", va="center",
                fontsize=fs_sub, color="white", alpha=0.92, zorder=3)


def _arr(ax, cx, y0, y1, lbl="", lbl_dx=0.20):
    """Vertical downward arrow from y0 to y1 with an optional right-side label."""
    ax.annotate("", xy=(cx, y1 + 0.07), xytext=(cx, y0 - 0.07),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.7), zorder=1)
    if lbl:
        ax.text(cx + lbl_dx, (y0 + y1) / 2, lbl,
                ha="left", va="center", fontsize=6.5, color="#555")


# ── Panel 1: Full architecture flow ───────────────────────────────────────────

def draw_architecture(ax, cfg):
    F, D, Nh, L, Dff, H = (
        cfg.input_features, cfg.d_model, cfg.num_heads,
        cfg.num_layers, cfg.dim_feedforward, cfg.forecast_horizon,
    )
    seq = cfg.context_length
    BH  = 0.82   # box height for all fixed blocks

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 21)
    ax.axis("off")
    ax.set_facecolor("#f7f9fc")

    cx = 5.0

    # ── Fixed upper blocks (1.6 unit center-to-center spacing) ──
    y_input, y_proj, y_cls, y_pe = 19.6, 18.0, 16.4, 14.8

    _box(ax, cx, y_input, 5.6, BH, "Input IMU Sequence",
         f"[B, {seq}, {F}]   (batch × time × features)", PAL["input"])
    _arr(ax, cx, y_input - BH/2, y_proj + BH/2, f"Linear({F} → {D})")

    _box(ax, cx, y_proj, 5.6, BH, "Input Projection",
         f"[B, {seq}, {D}]", PAL["proj"])
    _arr(ax, cx, y_proj - BH/2, y_cls + BH/2, "prepend CLS token")

    _box(ax, cx, y_cls, 5.6, BH, "CLS Token Prepend",
         f"[B, {seq+1}, {D}]", PAL["token"])
    _arr(ax, cx, y_cls - BH/2, y_pe + BH/2, f"sinusoidal PE + dropout")

    _box(ax, cx, y_pe, 5.6, BH, "Positional Encoding",
         f"[B, {seq+1}, {D}]  ·  base = {int(cfg.pe_base)}", PAL["pe"])

    # ── Transformer stack (dynamic height) ──
    avail  = y_pe - BH/2 - 4.6        # space reserved for LN + dual heads below
    lh     = min(0.92, avail / (L * 1.20))
    gap    = lh * 0.20
    y_top  = y_pe - BH/2 - 0.45 - lh / 2

    _arr(ax, cx, y_pe - BH/2, y_top + lh / 2)

    for i in range(L):
        yy = y_top - i * (lh + gap)
        _box(ax, cx, yy, 5.6, lh,
             f"Transformer Layer {i + 1}",
             f"Pre-LN · MHA ({Nh} heads, d_k={D // Nh}) · FFN ({D} → {Dff} → {D})",
             PAL["xfm"], fs_title=8.2, fs_sub=6.2)
        if i < L - 1:
            ax.annotate("", xy=(cx, yy - lh/2 - gap + 0.06),
                        xytext=(cx, yy - lh/2 - 0.06),
                        arrowprops=dict(arrowstyle="->", color="#444", lw=1.4), zorder=1)

    y_last = y_top - (L - 1) * (lh + gap)

    # ── Final LayerNorm ──
    y_ln = y_last - lh / 2 - 1.1
    _arr(ax, cx, y_last - lh / 2, y_ln + BH/2, "LayerNorm")
    _box(ax, cx, y_ln, 5.6, BH, "Final LayerNorm",
         f"[B, {seq+1}, {D}]", PAL["norm"])

    # ── Dual output heads ──
    y_fork = y_ln - BH / 2 - 0.3
    y_h    = y_fork - 1.6
    lx, rx = 2.6, 7.4

    ax.annotate("", xy=(lx, y_h + BH/2), xytext=(cx, y_fork),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.6,
                                connectionstyle="arc3,rad=0.28"), zorder=1)
    ax.text(lx - 0.65, (y_fork + y_h + BH/2) / 2, "[:,1:,:]",
            ha="right", va="center", fontsize=6.5, color="#555")

    ax.annotate("", xy=(rx, y_h + BH/2), xytext=(cx, y_fork),
                arrowprops=dict(arrowstyle="->", color="#444", lw=1.6,
                                connectionstyle="arc3,rad=-0.28"), zorder=1)
    ax.text(rx + 0.65, (y_fork + y_h + BH/2) / 2, "[:,0,:] CLS",
            ha="left", va="center", fontsize=6.5, color="#555")

    _box(ax, lx, y_h, 4.6, BH, "Reconstruction Head",
         f"Linear({D} → {F})  →  [B, {seq}, {F}]", PAL["recon"])
    _box(ax, rx, y_h, 4.6, BH, "Regression Head",
         f"Linear({D}→{Dff}) · ReLU · Linear({Dff}→{H})  →  [B,{H}]", PAL["reg"])

    ax.text(lx, y_h - BH/2 - 0.35, "MAE Pretraining",
            ha="center", va="top", fontsize=8, color=PAL["recon"],
            style="italic", fontweight="bold")
    ax.text(rx, y_h - BH/2 - 0.35, "Supervised Fine-tuning",
            ha="center", va="top", fontsize=8, color=PAL["reg"],
            style="italic", fontweight="bold")


# ── Panel 2: Single transformer layer internals ────────────────────────────────

def draw_layer_detail(ax, cfg):
    D, Nh, Dff = cfg.d_model, cfg.num_heads, cfg.dim_feedforward
    dk = D // Nh

    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 12.5)
    ax.axis("off")
    ax.set_facecolor("#f7f9fc")
    ax.set_title("Single Transformer Layer  (Pre-LN / norm_first=True)",
                 fontsize=9, fontweight="bold", color="#2c3e50", pad=8)

    cx    = 3.3    # centre of main column — shifted right to leave room for skip lines
    bw    = 5.0    # box width (left edge = cx-2.5 = 0.8)
    bh    = 0.72   # standard box height
    ah    = 0.52   # add-node box height
    sx    = 0.35   # x of the skip (residual) line — clear of box left edge (0.8)

    # ── y-positions (generous 1.5+ unit gaps) ──
    y_in   = 11.7
    y_ln1  =  9.9
    y_mha  =  7.9
    y_add1 =  6.5
    y_ln2  =  5.0
    y_ffn  =  3.0
    y_add2 =  1.6
    y_out  =  0.4

    # ── Skip (residual) lines drawn before boxes so boxes sit on top ──
    # Skip 1: from Input bottom → Add1 centre (horizontal arrow delivered at add1 level)
    ax.plot([sx, sx], [y_in - bh/2, y_add1],
            color="#bbb", lw=1.8, ls="--", zorder=0)
    ax.annotate("", xy=(cx - bw/2 + 0.05, y_add1),
                xytext=(sx, y_add1),
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.6), zorder=1)

    # Skip 2: from Add1 centre → Add2 centre
    ax.plot([sx, sx], [y_add1, y_add2],
            color="#bbb", lw=1.8, ls="--", zorder=0)
    ax.annotate("", xy=(cx - bw/2 + 0.05, y_add2),
                xytext=(sx, y_add2),
                arrowprops=dict(arrowstyle="->", color="#bbb", lw=1.6), zorder=1)

    # Skip labels (centered vertically in each skip span, to the left of skip line)
    ax.text(sx - 0.08, (y_in - bh/2 + y_add1) / 2, "skip\n①",
            ha="right", va="center", fontsize=6.5, color="#bbb", style="italic",
            linespacing=1.3)
    ax.text(sx - 0.08, (y_add1 + y_add2) / 2, "skip\n②",
            ha="right", va="center", fontsize=6.5, color="#bbb", style="italic",
            linespacing=1.3)

    # ── Main column boxes + arrows ──
    _box(ax, cx, y_in,  bw, bh,  "Input x",
         f"[B, T={cfg.context_length+1}, {D}]",                             PAL["io"])

    _arr(ax, cx, y_in  - bh/2, y_ln1 + bh/2)
    _box(ax, cx, y_ln1, bw, bh,  "LayerNorm₁",
         f"d_model = {D}",                                                   PAL["norm"])

    _arr(ax, cx, y_ln1 - bh/2, y_mha + bh/2)
    _box(ax, cx, y_mha, bw, bh,  "Multi-Head Self-Attention",
         f"{Nh} heads  ·  d_k = d_v = {dk}  ·  dropout = {cfg.dropout}",   PAL["xfm"])

    _arr(ax, cx, y_mha - bh/2, y_add1 + ah/2)
    _box(ax, cx, y_add1, 1.4, ah, "⊕  Add", "",                            PAL["io"], fs_title=9)

    _arr(ax, cx, y_add1 - ah/2, y_ln2 + bh/2)
    _box(ax, cx, y_ln2, bw, bh,  "LayerNorm₂",
         f"d_model = {D}",                                                   PAL["norm"])

    _arr(ax, cx, y_ln2 - bh/2, y_ffn + bh/2)
    _box(ax, cx, y_ffn, bw, bh,  "Feed-Forward Network",
         f"Linear({D}→{Dff})  ·  GELU  ·  Dropout  ·  Linear({Dff}→{D})", PAL["reg"])

    _arr(ax, cx, y_ffn - bh/2, y_add2 + ah/2)
    _box(ax, cx, y_add2, 1.4, ah, "⊕  Add", "",                            PAL["io"], fs_title=9)

    _arr(ax, cx, y_add2 - ah/2, y_out + bh/2)
    _box(ax, cx, y_out, bw, bh,  "Output",
         f"[B, T, {D}]  (same shape as input)",                              PAL["io"])


# ── Panel 3: Attention head decomposition ─────────────────────────────────────

def draw_attention_heads(ax, cfg):
    D, Nh = cfg.d_model, cfg.num_heads
    dk = D // Nh

    ax.set_xlim(-0.3, Nh + 0.3)
    ax.set_ylim(-1.5, 10.0)
    ax.axis("off")
    ax.set_facecolor("#f7f9fc")
    ax.set_title(f"Attention Head Decomposition\n"
                 f"d_model = {D}  →  {Nh} heads  ×  d_k = {dk}",
                 fontsize=9, fontweight="bold", color="#2c3e50", pad=8)

    cmap  = plt.cm.Set2(np.linspace(0, 0.85, Nh))
    bar_h = 5.8
    bar_y = 2.0

    for i in range(Nh):
        r = FancyBboxPatch(
            (i + 0.1, bar_y), 0.80, bar_h,
            boxstyle="round,pad=0.04",
            facecolor=cmap[i], edgecolor="white", alpha=0.90, linewidth=1.5,
        )
        ax.add_patch(r)
        cx_h = i + 0.5
        ax.text(cx_h, bar_y + bar_h * 0.78, f"Head {i+1}",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")
        ax.text(cx_h, bar_y + bar_h * 0.55, f"Q: {dk}d",
                ha="center", va="center", fontsize=7.5, color="white")
        ax.text(cx_h, bar_y + bar_h * 0.35, f"K: {dk}d",
                ha="center", va="center", fontsize=7.5, color="white")
        ax.text(cx_h, bar_y + bar_h * 0.15, f"V: {dk}d",
                ha="center", va="center", fontsize=7.5, color="white")

    ax.text(Nh / 2, bar_y + bar_h + 0.65,
            f"d_model = {D}  split equally across {Nh} heads",
            ha="center", va="center", fontsize=8, color="#2c3e50", fontweight="bold")

    scale    = 1 / math.sqrt(dk)
    seq      = cfg.context_length + 1
    attn_mem = seq * seq * Nh * 4 / 1e6

    stats = [
        f"Scale factor:  1/√{dk}  =  {scale:.4f}",
        f"Attn map: [B, {Nh}, {seq}, {seq}]  ≈  {attn_mem:.2f} MB  (fp32, B=1)",
    ]
    for i, s in enumerate(stats):
        ax.text(Nh / 2, bar_y - 0.55 - i * 0.65, s,
                ha="center", va="center", fontsize=7.5, color="#555",
                fontfamily="monospace")


# ── Panel 4: Sinusoidal PE heatmap ────────────────────────────────────────────

def draw_pe_heatmap(ax, cfg):
    D      = cfg.d_model
    base   = cfg.pe_base
    T      = min(cfg.context_length + 1, 80)
    show_d = min(D, 64)

    pos  = np.arange(T).reshape(-1, 1)
    dims = np.arange(0, D, 2)
    div  = np.exp(dims * (-math.log(base) / D))

    pe = np.zeros((T, D))
    pe[:, 0::2] = np.sin(pos * div)
    n_cos = D // 2
    pe[:, 1::2] = np.cos(pos * div[:n_cos])

    im = ax.imshow(pe[:, :show_d], aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1, origin="upper", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    ax.set_xlabel("Dimension index", fontsize=8, labelpad=5)
    ax.set_ylabel("Token position",  fontsize=8, labelpad=5)
    ax.set_title(
        f"Sinusoidal Positional Encoding\n"
        f"(first {show_d} of {D} dims,  {T} positions,  base = {int(base)})",
        fontsize=9, fontweight="bold", color="#2c3e50", pad=8,
    )
    ax.tick_params(labelsize=7)


# ── Panel 5: Parameter breakdown ──────────────────────────────────────────────

def draw_param_breakdown(ax, cfg):
    params = count_params(cfg)
    total  = params.pop("_total")
    labels = list(params.keys())
    values = list(params.values())
    colors = ["#4A90D9", "#E8A838", "#C0392B", "#1ABC9C", "#E67E22", "#2980B9"]

    bars = ax.barh(labels, values, color=colors[:len(labels)],
                   edgecolor="white", linewidth=1.6, alpha=0.90, height=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + total * 0.015,
                bar.get_y() + bar.get_height() / 2,
                fmt(val), va="center", fontsize=7.5, color="#333")

    ax.set_title(
        f"Parameter Breakdown\nTotal: {fmt(total)}  ·  ≈ {total*4/1e6:.1f} MB  (fp32)",
        fontsize=9, fontweight="bold", color="#2c3e50", pad=8,
    )
    ax.set_xlabel("Parameters", fontsize=8, labelpad=5)
    ax.tick_params(labelsize=7.5)
    ax.set_facecolor("#f7f9fc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt(int(x))))
    # 1.32× leaves room for value labels beyond the longest bar
    ax.set_xlim(0, max(values) * 1.32)


# ── Panel 6: Configuration summary table ──────────────────────────────────────

def draw_config_table(ax, cfg):
    params = count_params(cfg)
    total  = params["_total"]
    D, Nh  = cfg.d_model, cfg.num_heads

    rows = [
        ["input_features",   str(cfg.input_features),    f"IMU sensor axes"],
        ["context_length",   str(cfg.context_length),    f"input tokens  (+1 CLS = {cfg.context_length+1})"],
        ["forecast_horizon", str(cfg.forecast_horizon),  "regression output steps"],
        ["d_model",          str(D),                     "embedding dimension"],
        ["num_heads",        str(Nh),                    f"head_dim = {D//Nh}"],
        ["num_layers",       str(cfg.num_layers),        "stacked encoder layers"],
        ["dim_feedforward",  str(cfg.dim_feedforward),   f"FFN ratio = {cfg.dim_feedforward/D:.1f}× d_model"],
        ["dropout",          str(cfg.dropout),           "PE · MHA · FFN · reg head"],
        ["pe_base",          str(int(cfg.pe_base)),      "sinusoidal encoding base"],
        ["Total params",     fmt(total),                 f"≈ {total*4/1e6:.1f} MB  (fp32)"],
    ]

    ax.axis("off")
    ax.set_title("Configuration Summary", fontsize=9, fontweight="bold",
                 color="#2c3e50", pad=8)

    tbl = ax.table(cellText=rows, colLabels=["Parameter", "Value", "Note"],
                   cellLoc="left", loc="center", bbox=[0, 0.02, 1, 0.96])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.8)

    for j in range(3):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#eaf0fb" if i % 2 == 0 else "#ffffff"
        for j in range(3):
            tbl[i, j].set_facecolor(bg)
    for j in range(3):
        tbl[len(rows), j].set_facecolor("#d4edda")


# ── Layout + save ──────────────────────────────────────────────────────────────

def main():
    cfg = parse_args()

    if cfg.d_model % cfg.num_heads != 0:
        raise ValueError(
            f"d_model ({cfg.d_model}) must be divisible by num_heads ({cfg.num_heads})"
        )

    fig = plt.figure(figsize=(26, 19), facecolor="white")
    fig.suptitle(
        "IMU Intent Encoder — Architecture Visualization",
        fontsize=15, fontweight="bold", color="#2c3e50", y=0.987,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1, 1],
        hspace=0.45, wspace=0.30,
        left=0.03, right=0.97, top=0.955, bottom=0.03,
    )

    ax_arch  = fig.add_subplot(gs[:, 0])        # full-height left column
    ax_layer = fig.add_subplot(gs[0, 1])        # top-middle
    ax_heads = fig.add_subplot(gs[0, 2])        # top-right
    ax_pe    = fig.add_subplot(gs[1, 1])        # bottom-middle

    draw_architecture(ax_arch,  cfg)
    draw_layer_detail(ax_layer, cfg)
    draw_attention_heads(ax_heads, cfg)
    draw_pe_heatmap(ax_pe, cfg)

    # Bottom-right: param breakdown (top half) + config table (bottom half)
    gs_br = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[1, 2],
        hspace=0.60, height_ratios=[1.1, 1.3],
    )
    draw_param_breakdown(fig.add_subplot(gs_br[0]), cfg)
    draw_config_table(fig.add_subplot(gs_br[1]), cfg)

    plt.savefig(cfg.output, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Saved → {cfg.output}")
    plt.show()


if __name__ == "__main__":
    main()
