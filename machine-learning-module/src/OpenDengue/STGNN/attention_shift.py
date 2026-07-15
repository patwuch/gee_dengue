# attention_shift.py
"""
Visualise the month-to-month "shift" in where GAT attention is concentrated,
as trajectories of the attention-weighted spatial centroid(s).

For each prediction month, every node's in-strength is how much attention
other nodes place on it (sum of alpha over edges where it is the SOURCE — the
node being attended-to, see the direction convention noted in
explain_attention.py). Treating in-strength as a mass at that node's
position, a single whole-network centroid would be:

    centroid(month) = sum_i instrength_i(month) * position_i
                      -----------------------------------------
                            sum_i instrength_i(month)

but that collapses multi-region attention into one misleading average if the
mass is genuinely split across more than one place at once. Instead this
script runs weighted k-means (weights = in-strength) separately per month to
find up to k "attentional hubs", then greedily matches each month's clusters
to the previous month's by nearest centroid (Hungarian assignment) so a
cluster's identity — and therefore its trajectory — is tracked consistently
across time rather than being relabelled arbitrarily every month.

This is a proxy for shifting transmission focus, not a claim about actual
disease movement (see the caveat in the module docstring of
explain_attention.py: attention is not directly transmission).

Reads results/STGNN/<name>/attention_weights.npz (written by
explain_attention.py — run that first). No model or GPU needed here.

Run via Snakemake (script mode) or directly:
    python attention_shift.py --config config.yaml [--lag t-0] [--k 2]
"""

import colorsys
import json
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from choropleth import _node_names, _load_geometry


def _node_positions(node_names: list, gdf) -> np.ndarray:
    centroids = gdf.set_index("name").loc[node_names].geometry.centroid
    return np.array([(pt.x, pt.y) for pt in centroids])


def _node_boundary_rings(node_names: list, gdf, area_frac: float = 0.99,
                          max_parts: int = 15, tolerance: float = 0.03) -> list:
    """Simplified exterior rings per node, for a lightweight decorative
    basemap in the interactive HTML (not for precise geographic analysis).

    Some nodes (e.g. Viet Nam here) are MultiPolygons with thousands of tiny
    slivers/islets; keeping every part would blow up the embedded payload for
    negligible visual gain. Instead keep only the largest parts that
    cumulatively cover area_frac of the total area (capped at max_parts),
    then simplify each kept part's boundary.
    """
    boundaries = gdf.set_index("name").loc[node_names]
    rings_by_node = []
    for geom in boundaries.geometry:
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        parts.sort(key=lambda p: -p.area)
        total_area = sum(p.area for p in parts) or 1.0
        kept, running = [], 0.0
        for p in parts:
            kept.append(p)
            running += p.area
            if running / total_area >= area_frac or len(kept) >= max_parts:
                break

        node_rings = []
        for part in kept:
            simplified = part.simplify(tolerance, preserve_topology=True)
            node_rings.append([[round(x, 4), round(y, 4)] for x, y in simplified.exterior.coords])
        rings_by_node.append(node_rings)
    return rings_by_node


def compute_instrength(edges: np.ndarray, combined_by_window: np.ndarray,
                        n_nodes: int, lag_idx: int) -> np.ndarray:
    """In-strength per node at one lag step, for every month.

    combined_by_window: (W, T, E). Returns (W, N) — instrength[m, i] = how
    much attention node i receives from others at month m, lag t=lag_idx.
    Self-loops excluded: attending to yourself doesn't represent inter-node
    influence and would trivially inflate a node's own mass.
    """
    keep = edges[0] != edges[1]
    src  = edges[0][keep]
    W    = combined_by_window.shape[0]
    vals = combined_by_window[:, lag_idx, keep]  # (W, E_kept)

    instrength = np.zeros((W, n_nodes))
    for e, s in enumerate(src):
        instrength[:, s] += vals[:, e]
    return instrength


def attention_cluster_trajectories(instrength: np.ndarray, positions: np.ndarray,
                                    k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-means per month, with cluster identity tracked across months.

    Returns:
        centroids (W, k, 2)     — cluster centre per month, identity-aligned
        mass_frac (W, k)        — each cluster's share of that month's total
                                  in-strength (rows sum to 1)
    """
    W, N = instrength.shape
    centroids = np.zeros((W, k, 2))
    mass_frac = np.zeros((W, k))

    prev_order = None
    for m in range(W):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        km.fit(positions, sample_weight=instrength[m])
        centers = km.cluster_centers_          # (k, 2), arbitrary label order
        labels  = km.labels_
        mass    = np.array([instrength[m, labels == c].sum() for c in range(k)])
        mass    = mass / mass.sum()

        if prev_order is None:
            order = np.arange(k)
        else:
            # Match this month's raw cluster centres to last month's
            # identity-ordered centroids by nearest position (Hungarian on
            # pairwise distance) so cluster 0 this month is whichever raw
            # cluster is closest to cluster 0's previous location, etc.
            prev_centers = centroids[m - 1]
            cost = np.linalg.norm(prev_centers[:, None, :] - centers[None, :, :], axis=-1)
            _, order = linear_sum_assignment(cost)

        centroids[m] = centers[order]
        mass_frac[m] = mass[order]
        prev_order = order

    return centroids, mass_frac


def _lighten(rgb: tuple, amount: float) -> tuple:
    """amount in [0,1]: 0 = unchanged, 1 = near-white. Keeps hue fixed so a
    cluster's colour stays identifiable while lightness encodes time."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    l = l + (1 - l) * amount
    return colorsys.hls_to_rgb(h, l, s)


def plot_cluster_trajectories(centroids: np.ndarray, mass_frac: np.ndarray,
                               month_labels: list, node_names: list,
                               positions: np.ndarray, gdf, out_path: Path,
                               lag_label: str) -> None:
    """centroids (W, k, 2), mass_frac (W, k). Hue = cluster identity,
    lightness = time (light -> dark as months progress), marker size = the
    cluster's share of that month's total in-strength."""
    W, k, _ = centroids.shape
    fig, ax = plt.subplots(figsize=(9, 9))
    boundaries = gdf.set_index("name").loc[node_names]

    boundaries.boundary.plot(ax=ax, linewidth=0.4, color="0.75", zorder=1)
    ax.scatter(positions[:, 0], positions[:, 1], s=8, color="0.6", zorder=2)

    base_colors = matplotlib.colormaps["tab10"].colors[:k]

    for c in range(k):
        path = centroids[:, c, :]
        ax.plot(path[:, 0], path[:, 1], color=base_colors[c], linewidth=1,
                zorder=3, alpha=0.5)
        for m in range(W - 1):
            ax.annotate(
                "", xy=path[m + 1], xytext=path[m],
                arrowprops=dict(arrowstyle="-|>", color=base_colors[c], lw=1.4,
                                 shrinkA=0, shrinkB=0, alpha=0.8),
                zorder=4,
            )
        for m in range(W):
            # amount=0 at last month (full saturation), ->0.8 for the earliest
            # month (capped short of 1 so it stays visibly tinted, not white)
            amount = 0.8 * (1 - m / max(W - 1, 1))
            color = _lighten(base_colors[c], amount)
            size = 40 + 260 * mass_frac[m, c]
            ax.scatter(*path[m], color=color, s=size, edgecolor="black",
                       linewidth=0.4, zorder=5)
        ax.annotate(f"cluster {c}", path[-1], fontsize=8, xytext=(5, 5),
                    textcoords="offset points", fontweight="bold",
                    color=base_colors[c])

    # Zoom the main view to the trajectories (+ padding) — at full-map scale
    # the movement is a few degrees, invisible against a whole-region extent.
    # A locator inset (below) keeps the geographic context.
    all_xy = centroids.reshape(-1, 2)
    pad = max(0.5, 0.4 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1])))
    x0, x1 = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    y0, y1 = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"Attention-weighted cluster trajectories, k={k} (lag {lag_label})\n"
        "Colour = cluster identity, light→dark = "
        f"{month_labels[0]}→{month_labels[-1]}, size = share of attention mass",
        fontsize=10,
    )

    # Locator inset: full study region with a rectangle marking the zoomed area.
    inset = ax.inset_axes([0.02, 0.02, 0.28, 0.28])
    boundaries.boundary.plot(ax=inset, linewidth=0.2, color="0.8", zorder=1)
    inset.add_patch(matplotlib.patches.Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", linewidth=1
    ))
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("0.6")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


_SHIFT_INTERACTIVE_BODY = """
<style>
  .shift-root {
    --surface-1:      #fcfcfb;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --muted:          #898781;
    --gridline:       #e1e0d9;
    --border:         rgba(11,11,11,0.10);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--surface-1);
  }
  @media (prefers-color-scheme: dark) {
    .shift-root {
      --surface-1:      #1a1a19;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --muted:          #898781;
      --gridline:       #2c2c2a;
      --border:         rgba(255,255,255,0.10);
    }
  }
  :root[data-theme="dark"] .shift-root {
    --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
  }
  :root[data-theme="light"] .shift-root {
    --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
    --muted: #898781; --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
  }
  .shift-root { padding: 16px; box-sizing: border-box; }
  .shift-header { margin-bottom: 8px; }
  .shift-header h1 { font-size: 16px; margin: 0 0 4px; }
  .shift-header p { font-size: 12.5px; color: var(--text-secondary); margin: 0; max-width: 78ch; }
  .shift-toolbar {
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    margin: 12px 0; padding-bottom: 12px; border-bottom: 1px solid var(--gridline);
    font-size: 12.5px; color: var(--text-secondary);
  }
  .shift-toolbar label { display: flex; align-items: center; gap: 6px; }
  .shift-toolbar input[type="number"] {
    font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface-1); color: var(--text-primary);
    width: 4em;
  }
  .shift-warn { color: var(--muted); font-size: 11.5px; }
  .shift-timeline {
    display: flex; align-items: center; gap: 10px; margin: 0 0 12px;
    font-size: 12.5px; color: var(--text-secondary);
  }
  .shift-timeline input[type="range"] { flex: 1 1 auto; accent-color: #1f77b4; }
  .shift-timeline .lag-label {
    font-variant-numeric: tabular-nums; color: var(--text-primary); width: 10em; flex: none;
  }
  .shift-canvas-wrap { border: 1px solid var(--gridline); border-radius: 8px; }
  .shift-canvas-wrap svg { display: block; width: 100%; height: auto; }
  .shift-node-label { font-size: 9px; fill: var(--text-primary); pointer-events: none; }
  .shift-legend {
    display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px;
    font-size: 12px; color: var(--text-secondary);
  }
  .shift-legend .swatch {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 5px; vertical-align: middle;
  }
</style>

<div class="shift-root">
  <div class="shift-header">
    <h1>__TITLE__</h1>
    <p>Each dot is a weighted k-means cluster centre for one prediction month, where "weight" is how much attention a node
    receives from the rest of the network (in-strength) at the chosen lag. A cluster's colour is its identity, tracked
    month-to-month by matching each month's raw clusters to the previous month's nearest centres — so a colour means the
    same "attentional hub" throughout, not a new random grouping each month. Lightness encodes time (light = __FIRST_MONTH__,
    dark = __LAST_MONTH__); dot size is that cluster's share of the month's total attention mass. This is a proxy for
    shifting model attention, not a direct claim about disease transmission.</p>
  </div>
  <div class="shift-toolbar">
    <label>Clusters (k): <input type="number" id="shift-k-input" min="1" max="8" value="2"></label>
    <span class="shift-warn">k capped at 8 — cluster identity matching is brute-force and grows factorially with k.</span>
    <label><input type="checkbox" id="shift-dispersion-toggle"> Show dispersion (radius of gyration + node contributions)</label>
    <span class="shift-warn" id="shift-dispersion-month"></span>
  </div>
  <div class="shift-timeline">
    <span>Lag:</span>
    <input type="range" id="shift-lag-slider" min="0" max="0" value="0" step="1">
    <span class="lag-label" id="shift-lag-label">—</span>
  </div>
  <div class="shift-canvas-wrap">
    <svg id="shift-svg" viewBox="0 0 __VB_W__ __VB_H__" xmlns="http://www.w3.org/2000/svg"></svg>
  </div>
  <p class="shift-warn">A dashed ring is the weighted radius of gyration of each cluster's members around its centroid —
  small means the centroid sits on a real attention hotspot, large means it's an average of spatially distant nodes and
  may not correspond to any actual place. Lines from a centroid to member nodes are weighted by that node's share of the
  cluster's attention mass (nodes below 3% share are omitted to limit clutter). Click any dot to inspect that month.</p>
  <div class="shift-legend" id="shift-legend"></div>
</div>

<script type="application/json" id="shift-data">__DATA__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("shift-data").textContent);
  const nodes = data.nodes;                     // [{name, x, y}]
  const months = data.months;                   // [label, ...] length W
  const instrengthMean = data.instrengthMean;    // (W, N)
  const instrengthByLag = data.instrengthByLag;  // (T, W, N), index 0 = earliest lag
  const boundaries = data.boundaries;            // [ [ [[x,y],...], ... ], ... ] per node

  const W = months.length;
  const T = instrengthByLag.length;

  const svg = document.getElementById("shift-svg");
  const NS = "http://www.w3.org/2000/svg";

  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const VB_W = __VB_W__, VB_H = __VB_H__, PAD = __VB_PAD__;
  const sx = v => PAD + (v - xMin) / (xMax - xMin || 1) * (VB_W - 2 * PAD);
  const sy = v => VB_H - (PAD + (v - yMin) / (yMax - yMin || 1) * (VB_H - 2 * PAD));
  const positions = nodes.map(n => [n.x, n.y]);
  // Data-space-to-pixel scale, for converting a weighted-radius-of-gyration
  // (data units) into an SVG ring radius. x and y scales are near-identical
  // since the viewBox aspect ratio is chosen to match the data's, so either
  // axis works as a uniform scale factor.
  const dataToPx = (VB_W - 2 * PAD) / (xMax - xMin || 1);

  // ── Boundary silhouettes (decorative backdrop, not analytical) ─────────
  const gBoundaries = document.createElementNS(NS, "g");
  boundaries.forEach(nodeRings => {
    nodeRings.forEach(ring => {
      const d = ring.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`).join(" ") + " Z";
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "var(--gridline)");
      path.setAttribute("stroke-width", "0.6");
      gBoundaries.appendChild(path);
    });
  });
  svg.appendChild(gBoundaries);

  // ── Base context dots ──────────────────────────────────────────────────
  const gBase = document.createElementNS(NS, "g");
  nodes.forEach(n => {
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", sx(n.x));
    dot.setAttribute("cy", sy(n.y));
    dot.setAttribute("r", "2");
    dot.setAttribute("fill", "var(--muted)");
    dot.setAttribute("opacity", "0.35");
    const title = document.createElementNS(NS, "title");
    title.textContent = n.name;
    dot.appendChild(title);
    gBase.appendChild(dot);
  });
  svg.appendChild(gBase);
  // Dispersion (radius-of-gyration rings + spider lines) drawn beneath the
  // cluster dots/trajectories so the dots stay on top and clickable.
  const gDispersion = document.createElementNS(NS, "g");
  svg.appendChild(gDispersion);
  const gClusters = document.createElementNS(NS, "g");
  svg.appendChild(gClusters);

  // ── Lag slider ───────────────────────────────────────────────────────────
  // Slider value 0 = "mean over window", 1..T = lag index 0..T-1 (earliest..
  // most recent). A slider (not a dropdown) matters here: dragging through
  // it recomputes every cluster trajectory live, so you can see how much the
  // whole picture depends on which lookback depth you're using — e.g.
  // whether two hubs at t-0 merge into one at t-11. That's a lot harder to
  // notice one discrete dropdown click at a time.
  function lagLabel(sliderVal) {
    return sliderVal === 0 ? "mean over window" : `t-${T - sliderVal}`;
  }
  function lagIdxFromSlider(sliderVal) { return sliderVal === 0 ? -1 : sliderVal - 1; }

  // ── Seeded PRNG (deterministic clustering across recomputes) ───────────
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function pickWeightedIndex(weights, rand) {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = rand() * total;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }

  // Weighted Lloyd's algorithm with k-means++ init (weighted by mass), so
  // dense-attention nodes are more likely seed points — mirrors sklearn's
  // KMeans(sample_weight=...) used in the Python/static version of this plot.
  function weightedKMeans(points, weights, k, rand) {
    const n = points.length;
    const centers = [];
    centers.push(points[pickWeightedIndex(weights, rand)]);
    for (let c = 1; c < k; c++) {
      const d2 = points.map((p, i) => {
        const dmin = Math.min(...centers.map(ct => (p[0] - ct[0]) ** 2 + (p[1] - ct[1]) ** 2));
        return dmin * weights[i];
      });
      centers.push(points[pickWeightedIndex(d2, rand)]);
    }
    let assign = new Array(n).fill(0);
    for (let iter = 0; iter < 50; iter++) {
      let changed = false;
      for (let i = 0; i < n; i++) {
        let best = 0, bestDist = Infinity;
        for (let c = 0; c < k; c++) {
          const dx = points[i][0] - centers[c][0], dy = points[i][1] - centers[c][1];
          const d = dx * dx + dy * dy;
          if (d < bestDist) { bestDist = d; best = c; }
        }
        if (assign[i] !== best) { assign[i] = best; changed = true; }
      }
      const sums = Array.from({ length: k }, () => [0, 0, 0]);
      for (let i = 0; i < n; i++) {
        const c = assign[i];
        sums[c][0] += points[i][0] * weights[i];
        sums[c][1] += points[i][1] * weights[i];
        sums[c][2] += weights[i];
      }
      for (let c = 0; c < k; c++) {
        if (sums[c][2] > 0) centers[c] = [sums[c][0] / sums[c][2], sums[c][1] / sums[c][2]];
      }
      if (!changed && iter > 0) break;
    }
    const mass = new Array(k).fill(0);
    for (let i = 0; i < n; i++) mass[assign[i]] += weights[i];
    const totalMass = mass.reduce((a, b) => a + b, 0) || 1;
    return { centers, massFrac: mass.map(m => m / totalMass), assign };
  }

  // Brute-force permutation search (k <= 8, so <= 40320 permutations) to find
  // the relabelling of 0..k-1 that minimises a given cost matrix — the same
  // role scipy's linear_sum_assignment plays in the static plot. costMatrix[c]
  // is the cost of assigning identity-slot c to each candidate raw index.
  function bestPermutation(costMatrix) {
    const k = costMatrix.length;
    const idx = Array.from({ length: k }, (_, i) => i);
    let best = idx.slice(), bestCost = Infinity;
    function permute(a, l) {
      if (l === a.length) {
        let cost = 0;
        for (let c = 0; c < k; c++) cost += costMatrix[c][a[c]];
        if (cost < bestCost) { bestCost = cost; best = a.slice(); }
        return;
      }
      for (let i = l; i < a.length; i++) {
        [a[l], a[i]] = [a[i], a[l]];
        permute(a, l + 1);
        [a[l], a[i]] = [a[i], a[l]];
      }
    }
    permute(idx, 0);
    return best;
  }

  function dist(p, q) {
    const dx = p[0] - q[0], dy = p[1] - q[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  // Match this month's raw cluster centres to last month's identity-ordered
  // centroids, for chaining identity across the time axis within one run.
  function matchOrder(prevCenters, centers) {
    const k = centers.length;
    const cost = prevCenters.map(pc => centers.map(c => dist(pc, c)));
    return bestPermutation(cost);
  }

  // Match an entire new (W, k, 2) trajectory against a previous one by total
  // path distance summed over all W months, rather than comparing only one
  // boundary month. A single-point match (e.g. "new month 0 vs previous
  // month W-1") is fragile: those two months naturally sit in different
  // places even with nothing wrong, since attention genuinely moves month to
  // month, so matching on that one point alone can pick the wrong permutation
  // by chance. Summing over the whole shared shape is far more robust.
  function matchTrajectories(prevFull, newFull) {
    const k = newFull[0].length;
    const cost = Array.from({ length: k }, (_, c) =>
      Array.from({ length: k }, (_, cp) => {
        let s = 0;
        for (let m = 0; m < prevFull.length; m++) s += dist(prevFull[m][c], newFull[m][cp]);
        return s;
      })
    );
    return bestPermutation(cost);
  }

  // computeLocalTrajectories is a pure function of (k, weightsPerMonth): same
  // inputs always produce the same output, chaining identity only month-to-
  // month within this one run (month 0 always starts from k-means' own raw
  // label order). This determinism matters — re-rendering with unchanged
  // inputs (e.g. clicking a dot to change which month's dispersion is shown)
  // must reproduce an identical trajectory, or clusters visibly flip colour
  // for no data-driven reason.
  function computeLocalTrajectories(k, weightsPerMonth) {
    const rand = mulberry32(42);
    const centroids = [], massFracs = [], assigns = [];
    let prevCenters = null;
    for (let m = 0; m < W; m++) {
      const { centers, massFrac, assign } = weightedKMeans(positions, weightsPerMonth[m], k, rand);
      const order = prevCenters === null
        ? Array.from({ length: k }, (_, i) => i)
        : matchOrder(prevCenters, centers);
      const orderedCenters = order.map(i => centers[i]);
      const orderedMass = order.map(i => massFrac[i]);
      // order[c] = raw cluster feeding ordered slot c; invert it so each
      // node's raw assignment can be relabelled to the identity-tracked slot.
      const invOrder = new Array(k);
      order.forEach((rawIdx, c) => { invOrder[rawIdx] = c; });
      const orderedAssign = assign.map(rawIdx => invOrder[rawIdx]);
      centroids.push(orderedCenters);
      massFracs.push(orderedMass);
      assigns.push(orderedAssign);
      prevCenters = orderedCenters;
    }
    return { centroids, massFracs, assigns };
  }

  // Reference trajectory (already relabelled for display) carried across
  // separate computeTrajectories() calls — e.g. across lag-slider moves —
  // so a cluster that's spatially stable keeps its colour/index even though
  // each call's *local* labelling (above) restarts from scratch every time.
  // The relabelling is a whole-path match (matchTrajectories), not a
  // boundary-point chain, so it's robust to normal month-to-month movement
  // and, since computeLocalTrajectories is deterministic, repeated calls
  // with unchanged weights reproduce the exact same relabelling (fixed
  // point) instead of drifting.
  let referenceCentroids = null;
  let referenceK = null;

  function computeTrajectories(k, weightsPerMonth) {
    const local = computeLocalTrajectories(k, weightsPerMonth);
    if (referenceK !== k || referenceCentroids === null) {
      referenceCentroids = local.centroids;
      referenceK = k;
      return local;
    }
    const relabel = matchTrajectories(referenceCentroids, local.centroids);
    const invRelabel = new Array(k);
    relabel.forEach((localIdx, globalC) => { invRelabel[localIdx] = globalC; });

    const centroids = local.centroids.map(row => relabel.map(i => row[i]));
    const massFracs = local.massFracs.map(row => relabel.map(i => row[i]));
    const assigns = local.assigns.map(row => row.map(localIdx => invRelabel[localIdx]));

    referenceCentroids = centroids;
    referenceK = k;
    return { centroids, massFracs, assigns };
  }

  // ── Colour helpers (hue = cluster identity, lightness = time) ──────────
  const TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f",
  ];
  function hexToRgb01(hex) {
    const n = parseInt(hex.replace("#", ""), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  function rgbToHex01(rgb) {
    return "#" + rgb.map(v => Math.round(Math.max(0, Math.min(1, v)) * 255)
      .toString(16).padStart(2, "0")).join("");
  }
  function rgbToHls(r, g, b) {
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s; const l = (max + min) / 2;
    if (max === min) { h = s = 0; }
    else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        default: h = (r - g) / d + 4;
      }
      h /= 6;
    }
    return [h, l, s];
  }
  function hlsToRgb(h, l, s) {
    if (s === 0) return [l, l, l];
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
  }
  function lighten(hex, amount) {
    const [r, g, b] = hexToRgb01(hex);
    let [h, l, s] = rgbToHls(r, g, b);
    l = l + (1 - l) * amount;
    return rgbToHex01(hlsToRgb(h, l, s));
  }

  // Which month's dispersion (ring + spider lines) is currently shown.
  // Clicking any dot re-targets this instead of always showing the last
  // month, since the interesting month to sanity-check is whichever one
  // an outlier trajectory jump draws your eye to.
  let selectedMonth = W - 1;
  // Minimum share of a cluster's attention mass a node must hold to get its
  // own spider line — otherwise near-every node (assigned by nearest-centroid
  // geometry regardless of weight) would draw a line, drowning the signal.
  const SPIDER_MIN_SHARE = 0.03;

  function drawDispersion(c, color, centroid, month, assigns, weights) {
    const memberIdx = [];
    for (let i = 0; i < assigns.length; i++) if (assigns[i] === c) memberIdx.push(i);
    const clusterTotal = memberIdx.reduce((s, i) => s + weights[i], 0);
    if (clusterTotal <= 0) return;

    // Weighted radius of gyration: how spread out the mass-bearing members
    // are around the centroid. Small ring = centroid sits on a real
    // hotspot; large ring = the dot is an average of distant nodes and may
    // not correspond to any actual place (see explanation in chat / repo).
    let variance = 0;
    for (const i of memberIdx) {
      const dx = positions[i][0] - centroid[0], dy = positions[i][1] - centroid[1];
      variance += weights[i] * (dx * dx + dy * dy);
    }
    variance /= clusterTotal;
    const radiusPx = Math.sqrt(variance) * dataToPx;

    const ring = document.createElementNS(NS, "circle");
    ring.setAttribute("cx", sx(centroid[0]));
    ring.setAttribute("cy", sy(centroid[1]));
    ring.setAttribute("r", radiusPx.toFixed(2));
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", color);
    ring.setAttribute("stroke-width", "1");
    ring.setAttribute("stroke-dasharray", "3,2");
    ring.setAttribute("opacity", "0.7");
    const ringTitle = document.createElementNS(NS, "title");
    ringTitle.textContent =
      `${month} · cluster ${c} · weighted radius of gyration ≈ ${radiusPx > 0 ? "" : "0"}` +
      `${Math.sqrt(variance).toFixed(2)} (data units) — larger means this centroid ` +
      "averages spatially distant nodes rather than sitting on one hotspot";
    ring.appendChild(ringTitle);
    gDispersion.appendChild(ring);

    for (const i of memberIdx) {
      const share = weights[i] / clusterTotal;
      if (share < SPIDER_MIN_SHARE) continue;
      const line = document.createElementNS(NS, "line");
      line.setAttribute("x1", sx(centroid[0]));
      line.setAttribute("y1", sy(centroid[1]));
      line.setAttribute("x2", sx(positions[i][0]));
      line.setAttribute("y2", sy(positions[i][1]));
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", (0.5 + 2.5 * share).toFixed(2));
      line.setAttribute("opacity", (0.25 + 0.55 * share).toFixed(2));
      const lineTitle = document.createElementNS(NS, "title");
      lineTitle.textContent = `${nodes[i].name} · ${(share * 100).toFixed(1)}% of cluster ${c}'s mass this month`;
      line.appendChild(lineTitle);
      gDispersion.appendChild(line);
    }
  }

  function render(k, lagIdx) {
    while (gClusters.firstChild) gClusters.removeChild(gClusters.firstChild);
    while (gDispersion.firstChild) gDispersion.removeChild(gDispersion.firstChild);
    const weightsPerMonth = lagIdx === -1 ? instrengthMean : instrengthByLag[lagIdx];
    const { centroids, massFracs, assigns } = computeTrajectories(k, weightsPerMonth);
    selectedMonth = Math.min(selectedMonth, W - 1);
    const showDispersion = document.getElementById("shift-dispersion-toggle").checked;

    for (let c = 0; c < k; c++) {
      const color = TAB10[c % TAB10.length];
      const path = centroids.map(row => row[c]);

      const poly = document.createElementNS(NS, "polyline");
      poly.setAttribute("points", path.map(p => `${sx(p[0])},${sy(p[1])}`).join(" "));
      poly.setAttribute("fill", "none");
      poly.setAttribute("stroke", color);
      poly.setAttribute("stroke-width", "1");
      poly.setAttribute("opacity", "0.5");
      gClusters.appendChild(poly);

      for (let m = 0; m < W; m++) {
        const amount = 0.8 * (1 - m / Math.max(W - 1, 1));
        const dotColor = lighten(color, amount);
        const r = 4 + 9 * Math.sqrt(massFracs[m][c]);
        const dot = document.createElementNS(NS, "circle");
        dot.setAttribute("cx", sx(path[m][0]));
        dot.setAttribute("cy", sy(path[m][1]));
        dot.setAttribute("r", r.toFixed(2));
        dot.setAttribute("fill", dotColor);
        dot.setAttribute("stroke", m === selectedMonth ? "#e41a1c" : "black");
        dot.setAttribute("stroke-width", m === selectedMonth ? "1.6" : "0.4");
        dot.style.cursor = "pointer";
        dot.addEventListener("click", () => { selectedMonth = m; render(k, lagIdx); });
        const title = document.createElementNS(NS, "title");
        title.textContent = `${months[m]} · cluster ${c} · ${(massFracs[m][c] * 100).toFixed(1)}% of attention mass` +
          " (click to inspect dispersion for this month)";
        dot.appendChild(title);
        gClusters.appendChild(dot);
      }

      const last = path[W - 1];
      const label = document.createElementNS(NS, "text");
      label.setAttribute("class", "shift-node-label");
      label.setAttribute("x", sx(last[0]) + 6);
      label.setAttribute("y", sy(last[1]) - 6);
      label.setAttribute("fill", color);
      label.setAttribute("font-weight", "bold");
      label.textContent = `cluster ${c}`;
      gClusters.appendChild(label);

      if (showDispersion) {
        drawDispersion(c, color, centroids[selectedMonth][c], months[selectedMonth],
                        assigns[selectedMonth], weightsPerMonth[selectedMonth]);
      }
    }

    const monthLabelEl = document.getElementById("shift-dispersion-month");
    monthLabelEl.textContent = showDispersion
      ? `showing dispersion for ${months[selectedMonth]} (click a dot to change)`
      : "";

    const legend = document.getElementById("shift-legend");
    legend.innerHTML = "";
    legend.append(`Light → dark = ${months[0]} → ${months[W - 1]}. `);
    for (let c = 0; c < k; c++) {
      const span = document.createElement("span");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = TAB10[c % TAB10.length];
      span.appendChild(sw);
      span.append(`cluster ${c}`);
      legend.appendChild(span);
    }
  }

  const kInput = document.getElementById("shift-k-input");
  const lagSlider = document.getElementById("shift-lag-slider");
  const lagLabelEl = document.getElementById("shift-lag-label");
  const dispersionToggle = document.getElementById("shift-dispersion-toggle");
  lagSlider.min = 0; lagSlider.max = T; lagSlider.value = T; // default: most recent lag, t-0
  lagLabelEl.textContent = lagLabel(Number(lagSlider.value));

  function rerender() {
    const k = Math.max(1, Math.min(8, Number(kInput.value) || 1));
    kInput.value = String(k);
    const sliderVal = Number(lagSlider.value);
    lagLabelEl.textContent = lagLabel(sliderVal);
    render(k, lagIdxFromSlider(sliderVal));
  }
  kInput.addEventListener("change", rerender);
  lagSlider.addEventListener("input", rerender);
  dispersionToggle.addEventListener("change", rerender);

  rerender();
})();
</script>
"""


def write_interactive_shift(node_names: list, positions: np.ndarray,
                             month_labels: list, instrength_mean: np.ndarray,
                             instrength_by_lag: np.ndarray, gdf, out_path: Path) -> None:
    """Self-contained HTML: k-means clustering runs live in the browser (k
    and lag are both adjustable), so nothing about the number of attentional
    hubs is baked in at generation time. No server or external assets
    required."""
    nodes_json = [
        {"name": name, "x": float(x), "y": float(y)}
        for name, (x, y) in zip(node_names, positions)
    ]
    boundary_rings = _node_boundary_rings(node_names, gdf)
    payload = json.dumps({
        "nodes": nodes_json,
        "months": month_labels,
        "instrengthMean": np.round(instrength_mean, 6).tolist(),
        "instrengthByLag": np.round(instrength_by_lag, 6).tolist(),
        "boundaries": boundary_rings,
    }).replace("</", "<\\/")

    xs, ys = positions[:, 0], positions[:, 1]
    x_range = (xs.max() - xs.min()) or 1.0
    y_range = (ys.max() - ys.min()) or 1.0
    aspect = x_range / y_range
    if aspect >= 1:
        vb_w, vb_h = 1000, round(1000 / aspect)
    else:
        vb_w, vb_h = round(1000 * aspect), 1000

    body = (
        _SHIFT_INTERACTIVE_BODY
        .replace("__DATA__", payload)
        .replace("__TITLE__", "STGNN attention shift — cluster trajectories")
        .replace("__FIRST_MONTH__", month_labels[0])
        .replace("__LAST_MONTH__", month_labels[-1])
        .replace("__VB_W__", str(vb_w))
        .replace("__VB_H__", str(vb_h))
        .replace("__VB_PAD__", "40")
    )
    html = (
        "<!doctype html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n"
        "<title>STGNN attention shift</title>\n</head>\n<body>\n"
        + body +
        "\n</body>\n</html>\n"
    )
    out_path.write_text(html)


def attention_shift(out_dir: Path, csv_path: str, geom_path: str,
                     lag: str = "t-0", k: int = 2) -> None:
    npz = np.load(out_dir / "attention_weights.npz", allow_pickle=True)
    edges              = npz["edges"]
    combined_by_window = npz["combined_by_window"]  # (W, T, E)
    window_dates        = [pd.Timestamp(str(d)) for d in npz["window_dates"]]
    saved_node_names    = list(npz["node_names"])

    W, T, _ = combined_by_window.shape
    if lag == "mean":
        lag_idx = None
    else:
        offset = int(lag.split("-")[1])  # "t-0" -> 0, "t-11" -> 11
        lag_idx = T - 1 - offset
        if not (0 <= lag_idx < T):
            raise ValueError(f"lag {lag} out of range for T={T} (valid: t-0 .. t-{T-1})")

    node_names = _node_names(csv_path)
    assert node_names == saved_node_names, "node order mismatch vs saved attention_weights.npz"
    gdf       = _load_geometry(geom_path)
    positions = _node_positions(node_names, gdf)

    # Computed once for every lag regardless of --lag, since the interactive
    # HTML lets the lag (and k) be changed live in the browser.
    instrength_by_lag = np.stack([
        compute_instrength(edges, combined_by_window, len(node_names), t)
        for t in range(T)
    ])  # (T, W, N)
    instrength_mean = instrength_by_lag.mean(axis=0)  # (W, N)

    instrength = instrength_mean if lag_idx is None else instrength_by_lag[lag_idx]
    lag_label  = "mean over window" if lag_idx is None else lag

    centroids, mass_frac = attention_cluster_trajectories(instrength, positions, k=k)
    month_labels = [d.strftime("%b %Y") for d in window_dates]

    out_path = out_dir / "attention_shift_centroid.png"
    plot_cluster_trajectories(centroids, mass_frac, month_labels, node_names,
                               positions, gdf, out_path, lag_label)
    print(f"  -> {out_path}")

    csv_out = out_dir / "attention_shift_centroid.csv"
    rows = []
    for m, month in enumerate(month_labels):
        for c in range(k):
            rows.append({
                "month": month, "cluster": c,
                "centroid_x": centroids[m, c, 0], "centroid_y": centroids[m, c, 1],
                "mass_fraction": mass_frac[m, c],
            })
    pd.DataFrame(rows).to_csv(csv_out, index=False)
    print(f"  -> {csv_out}")

    html_path = out_dir / "attention_shift_interactive.html"
    write_interactive_shift(node_names, positions, month_labels,
                             instrength_mean, instrength_by_lag, gdf, html_path)
    print(f"  -> {html_path}")


def _resolve_data_paths(csv_path: str | None, geom_path: str | None) -> tuple[str, str]:
    if csv_path and geom_path:
        return csv_path, geom_path
    _root = Path(__file__).resolve().parent
    for _ in range(6):
        if (_root / ".git").exists():
            break
        _root = _root.parent
    return (
        csv_path or str(_root / "data/interim/machine-learning/SEA_dengue_env_monthly_2011-2018.csv"),
        geom_path or str(_root / "data/processed/dengue-infection/geoparquet/gaul_2024_sea_filtered.parquet"),
    )


def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(
        description="Visualise the month-to-month shift of the attention-weighted centroid."
    )
    ap.add_argument("--config", required=True, help="Path to config YAML.")
    ap.add_argument("--lag", default="t-0",
                    help="Which lag step to use: 't-0'..'t-<T-1>', or 'mean' to average over the window (default: t-0).")
    ap.add_argument("--k", type=int, default=2,
                    help="Number of attentional hubs to track per month (default: 2).")
    ap.add_argument("--csv-path", default=None)
    ap.add_argument("--geom-path", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = Path("results/STGNN") / cfg["name"]
    csv_path, geom_path = _resolve_data_paths(args.csv_path, args.geom_path)
    attention_shift(out_dir, csv_path, geom_path, lag=args.lag, k=args.k)


if __name__ == "__main__":
    if "snakemake" in globals():
        out_dir = Path(snakemake.params.results_dir)          # noqa: F821
        attention_shift(
            out_dir,
            csv_path=snakemake.params.csv_path,                # noqa: F821
            geom_path=snakemake.params.geom_path,               # noqa: F821
            lag=snakemake.params.get("lag", "t-0"),             # noqa: F821
            k=snakemake.params.get("k", 2),                     # noqa: F821
        )
    else:
        _main_cli()
