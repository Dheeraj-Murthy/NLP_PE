"""Full citation network around a judgment, drawn on a canvas.

Replaces st.graphviz_chart, which ran the whole Graphviz layout in the
browser's main thread: a few hundred nodes froze the tab, and it ran again
every time the tab was shown or resized. Here the layout is radial by hop
(the case in the middle, each hop on the next ring, cases placed next to
the case that brought them in), which is linear in the size of the graph
and done once; panning, zooming and switching tabs only repaint.

The data is the API's /graph/judgment/{id} response: nodes (id, label,
court, date, is_center, hop) and edges (source cites target).
"""

import json
from typing import Any, Dict

import streamlit as st


def render_network(data: Dict[str, Any], height: int = 640) -> None:
    graph = {"nodes": data.get("nodes", []), "edges": data.get("edges", []), "center": data.get("center_id")}
    # Case names come from the database: escape "<" so none can close the
    # <script> they are embedded in. The page only ever sets them as text.
    page = _TEMPLATE.replace("__GRAPH__", json.dumps(graph).replace("<", "\\u003c"))
    if hasattr(st, "iframe"):
        st.iframe(page, height=height)
    else:  # Streamlit before st.iframe
        import streamlit.components.v1 as components

        components.html(page, height=height)


_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #FFFFFF; --canvas: #FAFAFA; --border: #E4E4E7; --text: #09090B; --muted: #71717A;
    --cites: #2563EB; --citedby: #7C3AED; --other: #A1A1AA;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: Inter, system-ui, sans-serif; color: var(--text); background: var(--bg); }
  #wrap { position: relative; height: 100%; border: 1px solid var(--border); border-radius: 8px; background: var(--canvas); overflow: hidden; }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  canvas.over-node { cursor: pointer; }
  .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; }
  .toolbar button { font: 500 12px Inter, system-ui, sans-serif; color: var(--text); background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; cursor: pointer; }
  .toolbar button:hover { background: #F4F4F5; }
  .legend { position: absolute; top: 8px; left: 8px; display: flex; gap: 14px; font-size: 12px; color: var(--muted);
    background: var(--canvas); padding: 4px 6px; border-radius: 6px; }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 5px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
  .legend span.outline::before { background: var(--bg); border: 1.5px solid var(--muted); width: 8px; height: 8px; }
  .tip { position: absolute; pointer-events: none; background: var(--text); color: var(--bg); font-size: 12px; padding: 4px 8px;
    border-radius: 4px; max-width: 320px; display: none; }
  .card { position: absolute; left: 12px; bottom: 12px; max-width: 380px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px; font-size: 12.5px; box-shadow: 0 4px 16px rgba(0,0,0,.06); display: none; }
  .card h4 { margin: 0 0 4px; font-size: 13.5px; font-weight: 600; line-height: 1.35; }
  .card .meta { color: var(--muted); margin-bottom: 8px; }
  .card a { color: var(--text); font-weight: 500; margin-right: 14px; }
  .hint { position: absolute; right: 8px; bottom: 6px; font-size: 11px; color: var(--muted); background: var(--canvas); padding: 2px 6px; border-radius: 4px; }
</style>
</head>
<body>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div class="legend">
    <span style="--c: var(--cites)">Cases it cites</span>
    <span style="--c: var(--citedby)">Cases citing it</span>
    <span style="--c: var(--other)" class="outline">Further out</span>
  </div>
  <div class="toolbar"><button id="fit">Fit</button></div>
  <div class="tip" id="tip"></div>
  <div class="card" id="card"></div>
  <div class="hint">Hover or click a case · drag to pan · scroll to zoom</div>
</div>
<script>
const G = __GRAPH__;
const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
const tip = document.getElementById("tip"), card = document.getElementById("card");
const css = getComputedStyle(document.documentElement);
const v = name => css.getPropertyValue(name).trim();
const COLORS = { center: v("--text"), cites: v("--cites"), cited_by: v("--citedby"), other: v("--other") };

// ---- Graph ---------------------------------------------------------------
const byId = new Map(G.nodes.map(n => [n.id, Object.assign(n, { nbrs: new Set(), out: new Set() })]));
const edges = G.edges.filter(e => byId.has(e.source) && byId.has(e.target) && e.source !== e.target);
for (const e of edges) {
  byId.get(e.source).nbrs.add(e.target); byId.get(e.target).nbrs.add(e.source);
  byId.get(e.source).out.add(e.target);
}
const center = byId.get(G.center) || G.nodes.find(n => n.is_center) || G.nodes[0];

// Hop from the API; recompute by BFS if it is missing (older backend).
if (center && G.nodes.some(n => n.hop === undefined)) {
  for (const n of G.nodes) n.hop = Infinity;
  center.hop = 0;
  const queue = [center];
  while (queue.length) {
    const n = queue.shift();
    for (const id of n.nbrs) { const m = byId.get(id); if (m.hop === Infinity) { m.hop = n.hop + 1; queue.push(m); } }
  }
  const far = Math.max(0, ...G.nodes.filter(n => n.hop !== Infinity).map(n => n.hop)) + 1;
  for (const n of G.nodes) if (n.hop === Infinity) n.hop = far;
}

// ---- Radial layout -------------------------------------------------------
// Ring h holds the cases h hops out. Hop 1 is split into what the case cites
// and what cites it; every further case sits next to the case (one ring in)
// that brought it in, so each first-hop case gets its own wedge. Each case
// is a capsule pointing outward from its ring, like a spoke, so a crowded
// ring packs capsules side by side instead of on top of each other.
const FONT = "500 12px Inter, system-ui, sans-serif";
const CAP_H = 20, CAP_PAD = 9, MAX_TEXT_W = 190, RING_GAP = 70, SLOT = CAP_H + 6;

function fitLabel(label) {
  if (ctx.measureText(label).width <= MAX_TEXT_W) return label;
  let s = label;
  while (s.length > 1 && ctx.measureText(s + "…").width > MAX_TEXT_W) s = s.slice(0, -1);
  return s.trimEnd() + "…";
}

function layout() {
  if (!center) return;
  ctx.font = FONT;
  for (const n of G.nodes) {
    n.text = fitLabel(n.label);
    n.len = ctx.measureText(n.text).width + CAP_PAD * 2;
    n.kids = 0;
  }
  const rings = [];
  for (const n of G.nodes) (rings[n.hop] = rings[n.hop] || []).push(n);
  center.x = center.y = 0; center.angle = 0; center.side = "center";
  center.ux = 1; center.uy = 0;
  center.mx = 0; center.my = 0;  // the case's own capsule lies flat, centred
  center.ox = center.oy = 0;
  let radius = center.len / 2, prevLen = 0;
  for (let h = 1; h < rings.length; h++) {
    const ring = rings[h] || [];
    for (const n of ring) {
      if (h === 1) {
        n.side = center.out.has(n.id) ? "cites" : "cited_by";
        n.parent = center;
      } else {
        // Of its neighbours one ring in, join the one with the fewest cases so
        // far: wedges stay even, so each case lands near the case that
        // brought it in.
        let best = null;
        for (const id of n.nbrs) {
          const m = byId.get(id);
          if (m.hop === h - 1 && (!best || m.kids < best.kids || (m.kids === best.kids && m.angle < best.angle))) best = m;
        }
        n.parent = best || center;
        n.parent.kids++;
        n.side = n.parent.side;
      }
    }
    if (h === 1) ring.sort((a, b) => (a.side > b.side) - (a.side < b.side) || b.nbrs.size - a.nbrs.size || a.id - b.id);
    else ring.sort((a, b) => a.parent.angle - b.parent.angle || b.nbrs.size - a.nbrs.size || a.id - b.id);
    // Clear the previous ring's capsules, and leave a capsule's width per case around the ring.
    radius = Math.max(radius + prevLen + RING_GAP, (ring.length * SLOT) / (2 * Math.PI));
    prevLen = Math.max(0, ...ring.map(n => n.len));
    ring.forEach((n, i) => {
      n.angle = (i + 0.5) / ring.length * 2 * Math.PI;
      n.ux = Math.cos(n.angle - Math.PI / 2); n.uy = Math.sin(n.angle - Math.PI / 2);
      n.x = n.ux * radius; n.y = n.uy * radius;                                  // inner end, on the ring
      n.ox = n.x + n.ux * n.len; n.oy = n.y + n.uy * n.len;                      // outer end
      n.mx = n.x + n.ux * n.len / 2; n.my = n.y + n.uy * n.len / 2;              // middle
    });
  }
  for (const n of G.nodes) {
    const strong = n === center ? COLORS.center : COLORS[n.side] || COLORS.other;
    // The case and its direct citations are filled; further cases are outlined in their wedge's colour.
    n.fill = n.hop <= 1 ? strong : v("--bg");
    n.stroke = strong;
    n.ink = n.hop <= 1 ? "#FFFFFF" : v("--text");
  }
}

// Where an edge meets a case: links outward leave from the outer end,
// links inward or around the ring arrive at the inner end.
function end(n, other) {
  if (n === center) return [0, 0];
  return other.hop > n.hop ? [n.ox, n.oy] : [n.x, n.y];
}

let treeEdges = [], crossEdges = [], crossAlpha = 0.2;
function relayout() {
  layout();
  const isTree = e => byId.get(e.source).parent === byId.get(e.target) || byId.get(e.target).parent === byId.get(e.source);
  treeEdges = edges.filter(isTree);
  crossEdges = edges.filter(e => !isTree(e));
  crossAlpha = Math.max(0.03, Math.min(0.2, 0.2 * Math.sqrt(150 / Math.max(1, crossEdges.length))));
}
relayout();

// ---- Drawing -------------------------------------------------------------
let W = 0, H = 0, dpr = 1;
let t = { x: 0, y: 0, k: 1 };
let hover = null, selected = null, queued = false;

function resize() {
  const r = cv.getBoundingClientRect();
  if (!r.width || !r.height) return false;  // hidden tab: draw when shown
  dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  return true;
}

function pill(x, y, w, h) {
  const r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arc(x + w - r, y + r, r, -Math.PI / 2, Math.PI / 2);
  ctx.lineTo(x + r, y + h);
  ctx.arc(x + r, y + r, r, Math.PI / 2, Math.PI * 1.5);
  ctx.closePath();
}

function draw() {
  queued = false;
  if (!W) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.setTransform(dpr * t.k, 0, 0, dpr * t.k, dpr * t.x, dpr * t.y);

  const focus = selected || hover;
  const lit = focus ? new Set([focus.id, ...focus.nbrs]) : null;

  // Each case's link to the case that brought it in draws the tree; every
  // other link is a faint web, fainter the more there are. Each set is one
  // path, so thousands of edges are two strokes. The focused case's edges
  // go on top with arrows.
  ctx.lineWidth = 1 / t.k;
  for (const [set, alpha] of [[crossEdges, crossAlpha], [treeEdges, 0.45]]) {
    ctx.strokeStyle = `rgba(113,113,122,${focus ? alpha * 0.35 : alpha})`;
    ctx.beginPath();
    for (const e of set) {
      const a = byId.get(e.source), b = byId.get(e.target);
      ctx.moveTo(...end(a, b)); ctx.lineTo(...end(b, a));
    }
    ctx.stroke();
  }
  if (focus) {
    ctx.strokeStyle = ctx.fillStyle = "rgba(9,9,11,0.7)";
    ctx.lineWidth = 1.4 / t.k;
    for (const e of edges) {
      if (e.source !== focus.id && e.target !== focus.id) continue;
      const a = byId.get(e.source), b = byId.get(e.target);
      arrow(end(a, b), end(b, a));
    }
  }

  // Capsules. Text is skipped when too small to read, and capsules off
  // screen are skipped altogether.
  ctx.font = FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const readable = 12 * t.k >= 6;
  for (const n of G.nodes) {
    const sx = n.mx * t.k + t.x, sy = n.my * t.k + t.y, reach = n.len * t.k;
    if (sx < -reach || sx > W + reach || sy < -reach || sy > H + reach) continue;
    // Cases outside the focus fade to outlines, so their names stay legible.
    const dim = lit && !lit.has(n.id);
    ctx.globalAlpha = dim ? 0.3 : 1;
    ctx.save();
    ctx.translate(n.mx, n.my);
    // Keep text upright: capsules on the left half are turned around.
    let a = Math.atan2(n.uy, n.ux);
    if (n.ux < -1e-6) a += Math.PI;
    ctx.rotate(a);
    pill(-n.len / 2, -CAP_H / 2, n.len, CAP_H);
    ctx.fillStyle = dim ? v("--bg") : n.fill;
    ctx.fill();
    ctx.lineWidth = (n === selected ? 2.5 : 1.2) / t.k;
    ctx.strokeStyle = n === selected ? v("--text") : n.stroke;
    ctx.stroke();
    if (readable) {
      ctx.fillStyle = dim ? v("--text") : n.ink;
      ctx.fillText(n.text, 0, 1);
    }
    ctx.restore();
  }
  ctx.globalAlpha = 1;
}

function arrow([ax, ay], [bx, by]) {
  const dx = bx - ax, dy = by - ay, len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, s = 7 / t.k;
  ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx - ux * s - uy * s * 0.5, by - uy * s + ux * s * 0.5);
  ctx.lineTo(bx - ux * s + uy * s * 0.5, by - uy * s - ux * s * 0.5);
  ctx.fill();
}

function redraw() { if (!queued) { queued = true; requestAnimationFrame(draw); } }

// Fit everything, or (first) only the case and its direct citations: a
// whole two- or three-hop network fitted to the frame is too small to read,
// while the first ring is what one opens the view for.
function fit(firstRing = false) {
  if (!G.nodes.length || !W) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of G.nodes) {
    if (firstRing && n.hop > 1) continue;
    const xs = n === center ? [-n.len / 2, n.len / 2] : [n.x, n.ox], ys = n === center ? [0, 0] : [n.y, n.oy];
    x0 = Math.min(x0, ...xs); x1 = Math.max(x1, ...xs); y0 = Math.min(y0, ...ys); y1 = Math.max(y1, ...ys);
  }
  const pad = 30;
  const k = Math.min(1.5, (W - pad * 2) / Math.max(1, x1 - x0), (H - pad * 2) / Math.max(1, y1 - y0));
  t = { k, x: W / 2 - (x0 + x1) / 2 * k, y: H / 2 - (y0 + y1) / 2 * k };
  redraw();
}

// ---- Interaction ---------------------------------------------------------
// A point is on a capsule if, in the capsule's own frame, it is within its
// length and height (never less than a few screen pixels, to stay clickable).
function nodeAt(px, py) {
  const x = (px - t.x) / t.k, y = (py - t.y) / t.k;
  const minHalf = 4 / t.k;
  for (let i = G.nodes.length - 1; i >= 0; i--) {  // topmost first
    const n = G.nodes[i], dx = x - n.mx, dy = y - n.my;
    const along = dx * n.ux + dy * n.uy, across = -dx * n.uy + dy * n.ux;
    if (Math.abs(along) <= Math.max(n.len / 2, minHalf) && Math.abs(across) <= Math.max(CAP_H / 2, minHalf)) return n;
  }
  return null;
}

const esc = s => String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function showCard(n) {
  const meta = [n.court, n.date, n.hop ? `${n.hop} hop${n.hop > 1 ? "s" : ""} out` : null].filter(Boolean).join(" · ");
  card.innerHTML = `<h4>${esc(n.label)}</h4><div class="meta">#${n.id}${meta ? " · " + esc(meta) : ""} · ${n.nbrs.size} connection${n.nbrs.size === 1 ? "" : "s"} shown</div>` +
    `<a href="Document?type=judgment&id=${n.id}" target="_blank" rel="noopener">Open judgment ↗</a>` +
    (n !== center ? `<a href="Citation_Graph?focus=${n.id}" target="_blank" rel="noopener">Map this case ↗</a>` : "");
  card.style.display = "block";
}

let drag = null, dragged = false;
cv.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, tx: t.x, ty: t.y }; dragged = false; });
window.addEventListener("pointermove", e => {
  const r = cv.getBoundingClientRect();
  if (drag) {
    if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) { dragged = true; cv.classList.add("dragging"); }
    t.x = drag.tx + e.clientX - drag.x;
    t.y = drag.ty + e.clientY - drag.y;
    tip.style.display = "none";
    redraw();
    return;
  }
  const n = nodeAt(e.clientX - r.left, e.clientY - r.top);
  if (n !== hover) { hover = n; redraw(); }
  cv.classList.toggle("over-node", !!n);
  if (n) {
    tip.textContent = n.label + (n.date ? ` · ${n.date}` : "");
    tip.style.left = Math.min(e.clientX - r.left + 12, W - 330) + "px";
    tip.style.top = (e.clientY - r.top + 14) + "px";
    tip.style.display = "block";
  } else tip.style.display = "none";
});
window.addEventListener("pointerup", () => { drag = null; cv.classList.remove("dragging"); });
cv.addEventListener("pointerleave", () => { if (hover) { hover = null; redraw(); } tip.style.display = "none"; });
cv.addEventListener("click", e => {
  if (dragged) return;
  const r = cv.getBoundingClientRect();
  const n = nodeAt(e.clientX - r.left, e.clientY - r.top);
  selected = n;
  if (n) showCard(n); else card.style.display = "none";
  redraw();
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = Math.min(6, Math.max(0.05, t.k * Math.exp(-e.deltaY * 0.0015)));
  t.x = mx - (mx - t.x) * (k / t.k);
  t.y = my - (my - t.y) * (k / t.k);
  t.k = k;
  redraw();
}, { passive: false });
document.getElementById("fit").onclick = () => fit();

// The iframe has no size while its tab is hidden: fit the first time it gets
// one, and afterwards only repaint (no layout work) on resize.
// Resizing the canvas clears it, so repaint straight away, not next frame.
let fitted = false;
new ResizeObserver(() => {
  if (!resize()) return;
  if (!fitted) { fitted = true; fit(true); }
  draw();
}).observe(cv);
// Labels were measured before Inter loaded: measure again, and refit unless
// the view has been moved already.
let moved = false;
cv.addEventListener("pointerdown", () => { moved = true; });
cv.addEventListener("wheel", () => { moved = true; });
(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => {
  relayout();
  if (fitted && !moved) fit(true);
  if (W) draw();
});
</script>
</body>
</html>
"""
