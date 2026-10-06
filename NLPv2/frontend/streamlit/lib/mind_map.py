"""Mind-map view of a judgment's citations, in the style of NotebookLM: the
case at the root, branches that open and close on click, everything else
folded away. Rendered client-side as plain SVG (no extra packages), so only
what is expanded is ever drawn.

The tree comes from the API's /graph/judgment/{id}/mind-map endpoint:
nodes with kind "case", "branch" (Cites / Cited by), "group" (a relationship
type) or "more" (cases left out), each with optional "children".
"""

import json
from typing import Any, Dict

import streamlit as st


def render_mind_map(tree: Dict[str, Any], height: int = 640) -> None:
    # Case names come from the database: escape "<" so none can close the
    # <script> they are embedded in. The page itself only ever sets them as
    # text, never as HTML.
    page = _TEMPLATE.replace("__TREE__", json.dumps(tree).replace("<", "\\u003c"))
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
    --root-bg: #18181B; --root-text: #FFFFFF;
    --cites: #2563EB; --cites-soft: #EFF6FF; --cites-line: #BFDBFE;
    --citedby: #7C3AED; --citedby-soft: #F5F3FF; --citedby-line: #DDD6FE;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: Inter, system-ui, sans-serif; color: var(--text); background: var(--bg); }
  #wrap { position: relative; height: 100%; border: 1px solid var(--border); border-radius: 8px; background: var(--canvas); overflow: hidden; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; user-select: none; }
  svg.dragging { cursor: grabbing; }
  .link { fill: none; stroke-width: 1.6; }
  .node { cursor: pointer; }
  .node rect.box { stroke-width: 1.2; transition: filter .15s; }
  .node:hover rect.box { filter: drop-shadow(0 2px 4px rgba(0,0,0,.12)); }
  .node.selected rect.box { stroke-width: 2.2; }
  .node text { font-size: 13px; dominant-baseline: middle; pointer-events: none; }
  .node .toggle circle { stroke-width: 1.2; }
  .node .toggle text { font-size: 10.5px; font-weight: 600; text-anchor: middle; }
  .toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 6px; }
  .toolbar button { font: 500 12px Inter, system-ui, sans-serif; color: var(--text); background: var(--bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; cursor: pointer; }
  .toolbar button:hover { background: #F4F4F5; }
  .legend { position: absolute; top: 12px; left: 12px; display: flex; gap: 14px; font-size: 12px; color: var(--muted); }
  .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; background: var(--c); }
  .card { position: absolute; left: 12px; bottom: 12px; max-width: 380px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px; font-size: 12.5px; box-shadow: 0 4px 16px rgba(0,0,0,.06); display: none; }
  .card h4 { margin: 0 0 4px; font-size: 13.5px; font-weight: 600; line-height: 1.35; }
  .card .meta { color: var(--muted); margin-bottom: 8px; }
  .card a { color: var(--text); font-weight: 500; margin-right: 14px; }
  .hint { position: absolute; right: 12px; bottom: 10px; font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<div id="wrap">
  <svg id="svg"><g id="viewport"><g id="links"></g><g id="nodes"></g></g></svg>
  <div class="legend">
    <span style="--c: var(--cites)">Cites</span>
    <span style="--c: var(--citedby)">Cited by</span>
  </div>
  <div class="toolbar">
    <button id="expand">Expand all</button>
    <button id="collapse">Collapse</button>
    <button id="fit">Fit</button>
  </div>
  <div class="card" id="card"></div>
  <div class="hint">Click a node to open or close it · drag to pan · scroll to zoom</div>
</div>
<script>
const TREE = __TREE__;

const H_GAP = 56, V_GAP = 10, PAD_X = 14, LINE_H = 17, MAX_TEXT_W = 240, TOGGLE_R = 11;
const svg = document.getElementById("svg");
const viewport = document.getElementById("viewport");
const linkLayer = document.getElementById("links");
const nodeLayer = document.getElementById("nodes");
const card = document.getElementById("card");
const NS = "http://www.w3.org/2000/svg";
const css = getComputedStyle(document.documentElement);
const v = name => css.getPropertyValue(name).trim();

const measureCtx = document.createElement("canvas").getContext("2d");
function textWidth(s, weight) {
  measureCtx.font = `${weight} 13px Inter, system-ui, sans-serif`;
  return measureCtx.measureText(s).width;
}

// Word-wrap a label into at most two lines that fit MAX_TEXT_W.
function wrap(label, weight) {
  const words = label.split(/\s+/);
  const lines = [""];
  for (const w of words) {
    const cur = lines[lines.length - 1];
    const next = cur ? cur + " " + w : w;
    if (textWidth(next, weight) <= MAX_TEXT_W || !cur) lines[lines.length - 1] = next;
    else if (lines.length < 2) lines.push(w);
    else { lines[1] += " " + w; break; }
  }
  if (lines.length === 2 && textWidth(lines[1], weight) > MAX_TEXT_W) {
    let s = lines[1];
    while (s.length > 1 && textWidth(s + "…", weight) > MAX_TEXT_W) s = s.slice(0, -1);
    lines[1] = s.trimEnd() + "…";
  }
  return lines;
}

// Prepare nodes: ids, parents, colours, initial open state.
let uid = 0;
function prepare(node, parent, depth, direction) {
  node.uid = uid++;
  node.parent = parent;
  node.depth = depth;
  node.direction = node.direction || direction;
  node.children = node.children || [];
  // Start with just the case and its two branches showing, like NotebookLM.
  node.open = depth <= 1;
  for (const c of node.children) prepare(c, node, depth + 1, node.direction);
}
prepare(TREE, null, 0, null);

function palette(node) {
  if (node.kind === "case" && node.depth === 0) return { fill: v("--root-bg"), stroke: v("--root-bg"), text: v("--root-text"), line: v("--border"), accent: v("--text") };
  const dir = node.direction === "cited_by" ? "citedby" : "cites";
  const strong = v(`--${dir}`), soft = v(`--${dir}-soft`), line = v(`--${dir}-line`);
  if (node.kind === "branch") return { fill: strong, stroke: strong, text: "#FFFFFF", line, accent: strong };
  if (node.kind === "group") return { fill: soft, stroke: line, text: strong, line, accent: strong };
  if (node.kind === "more") return { fill: "transparent", stroke: v("--border"), text: v("--muted"), line, dashed: true };
  return { fill: v("--bg"), stroke: line, text: v("--text"), line, accent: v("--muted") };
}

function labelOf(node) {
  // Groups show their size on the toggle; a branch's total can exceed what is listed.
  if (node.kind === "branch") return `${node.label} (${node.count})`;
  return node.label;
}

function size(node) {
  if (node.size) return node.size;
  const weight = node.kind === "case" && node.depth > 0 ? 400 : 600;
  const lines = wrap(labelOf(node), weight);
  const w = Math.max(...lines.map(l => textWidth(l, weight))) + PAD_X * 2;
  const h = lines.length * LINE_H + 14;
  node.lines = lines; node.weight = weight;
  node.size = { w: Math.ceil(w), h };
  return node.size;
}

// Tidy horizontal tree: leaves stack top to bottom, parents centre on their children.
function layout() {
  const visible = [];
  let cursor = 0;
  (function place(node, x) {
    const { w, h } = size(node);
    node.x = x;
    visible.push(node);
    const kids = node.open ? node.children : [];
    if (!kids.length) {
      node.y = cursor + h / 2;
      cursor += h + V_GAP;
      return;
    }
    const cx = x + w + H_GAP + (node.children.length ? TOGGLE_R : 0);
    for (const c of kids) place(c, cx);
    node.y = (kids[0].y + kids[kids.length - 1].y) / 2;
  })(TREE, 0);
  return visible;
}

let transform = { x: 0, y: 0, k: 1 };
function applyTransform() {
  viewport.setAttribute("transform", `translate(${transform.x},${transform.y}) scale(${transform.k})`);
}

function linkPath(p, c) {
  const x1 = p.x + p.size.w + (p.children.length ? TOGGLE_R * 2 : 0), y1 = p.y;
  const x2 = c.x, y2 = c.y, mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

let selected = null;
let visibleNodes = [];
let prevPos = new Map();
let animFrame = null;

function render(anchor) {
  const visible = visibleNodes = layout();

  // Keep the clicked node where it was on screen, so the map doesn't jump.
  if (anchor && prevPos.has(anchor.uid)) {
    const old = prevPos.get(anchor.uid);
    transform.x += (old.x - anchor.x) * transform.k;
    transform.y += (old.y - anchor.y) * transform.k;
    // Then pan left if the newly opened children would land off screen.
    if (anchor.open && anchor.children.length) {
      const right = (anchor.children[0].x + Math.max(...anchor.children.map(c => c.size.w)) + 24) * transform.k + transform.x;
      const left = anchor.x * transform.k + transform.x;
      transform.x -= Math.max(0, Math.min(right - svg.clientWidth, left - 24));
    }
    applyTransform();
  }

  nodeLayer.textContent = "";
  linkLayer.textContent = "";
  const items = [];
  for (const n of visible) {
    const pal = palette(n);
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "node" + (n === selected ? " selected" : ""));
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("class", "box");
    rect.setAttribute("width", n.size.w);
    rect.setAttribute("height", n.size.h);
    rect.setAttribute("y", -n.size.h / 2);
    rect.setAttribute("rx", n.kind === "case" ? 8 : n.size.h / 2);
    rect.setAttribute("fill", pal.fill);
    rect.setAttribute("stroke", n === selected ? pal.text : pal.stroke);
    if (pal.dashed) rect.setAttribute("stroke-dasharray", "4 3");
    g.appendChild(rect);
    n.lines.forEach((line, i) => {
      const t = document.createElementNS(NS, "text");
      t.setAttribute("x", PAD_X);
      t.setAttribute("y", (i - (n.lines.length - 1) / 2) * LINE_H + 1);
      t.setAttribute("fill", pal.text);
      t.setAttribute("font-weight", n.weight);
      if (n.kind === "more") t.setAttribute("font-style", "italic");
      t.textContent = line;
      g.appendChild(t);
    });
    const title = document.createElementNS(NS, "title");
    title.textContent = labelOf(n) + (n.date ? ` · ${n.date}` : "");
    g.appendChild(title);

    if (n.children.length) {
      const tg = document.createElementNS(NS, "g");
      tg.setAttribute("class", "toggle");
      tg.setAttribute("transform", `translate(${n.size.w + TOGGLE_R},0)`);
      const c = document.createElementNS(NS, "circle");
      c.setAttribute("r", TOGGLE_R - 1);
      c.setAttribute("fill", v("--bg"));
      c.setAttribute("stroke", n.depth === 0 ? v("--muted") : pal.line);
      const t = document.createElementNS(NS, "text");
      t.setAttribute("y", 1);
      t.setAttribute("fill", pal.accent);
      t.textContent = n.open ? "‹" : (n.children.length > 99 ? "99+" : n.children.length);
      if (n.open) t.setAttribute("font-size", "15px");
      tg.append(c, t);
      g.appendChild(tg);
    }

    g.addEventListener("click", ev => { ev.stopPropagation(); if (!dragged) onNodeClick(n); });
    nodeLayer.appendChild(g);

    let path = null;
    if (n.parent) {
      path = document.createElementNS(NS, "path");
      path.setAttribute("class", "link");
      path.setAttribute("stroke", palette(n).line);
      linkLayer.appendChild(path);
    }
    // New nodes grow out of their parent's old position.
    const entering = !prevPos.has(n.uid) && prevPos.size > 0;
    const from = prevPos.get(n.uid) || (n.parent && prevPos.get(n.parent.uid)) || { x: n.x, y: n.y };
    items.push({ n, g, path, from, entering });
  }

  prevPos = new Map(visible.map(n => [n.uid, { x: n.x, y: n.y }]));
  animate(items);
}

function animate(items) {
  cancelAnimationFrame(animFrame);
  const start = performance.now(), dur = 260;
  const ease = t => 1 - Math.pow(1 - t, 3);
  function frame(now) {
    const t = ease(Math.min(1, (now - start) / dur));
    const pos = new Map();
    for (const it of items) {
      const x = it.from.x + (it.n.x - it.from.x) * t, y = it.from.y + (it.n.y - it.from.y) * t;
      pos.set(it.n.uid, { x, y });
      it.g.setAttribute("transform", `translate(${x},${y})`);
      it.g.style.opacity = it.entering ? t : 1;
    }
    for (const it of items) {
      if (!it.path) continue;
      const p = it.n.parent, a = pos.get(p.uid), b = pos.get(it.n.uid);
      it.path.setAttribute("d", linkPath({ x: a.x, y: a.y, size: p.size, children: p.children }, { x: b.x, y: b.y }));
      it.path.style.opacity = it.g.style.opacity;
    }
    if (t < 1) animFrame = requestAnimationFrame(frame);
  }
  animFrame = requestAnimationFrame(frame);
}

function onNodeClick(n) {
  if (n.kind === "case") showCard(n);
  if (n.children.length) n.open = !n.open;
  render(n);
}

function showCard(n) {
  selected = n;
  const meta = [n.court, n.date, n.relationship && n.depth > 0 ? n.relationship : null].filter(Boolean).join(" · ");
  const esc = s => String(s).replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
  card.innerHTML = `<h4>${esc(n.label)}</h4><div class="meta">#${n.id}${meta ? " · " + esc(meta) : ""}</div>` +
    `<a href="Document?type=judgment&id=${n.id}" target="_blank" rel="noopener">Open judgment ↗</a>` +
    (n.depth > 0 ? `<a href="Citation_Graph?focus=${n.id}" target="_blank" rel="noopener">Map this case ↗</a>` : "");
  card.style.display = "block";
}

function setAll(node, open) {
  if (node.children.length) node.open = open || node.depth === 0;
  for (const c of node.children) setAll(c, open);
}
function resetOpen(node) {
  node.open = node.depth <= 1;
  for (const c of node.children) resetOpen(c);
}

// Fit to the laid-out positions (not the DOM, which may be mid-animation).
// Below minK the labels are unreadable, so instead zoom no further and
// show the top of the map.
function fit(minK = 0) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of visibleNodes) {
    x0 = Math.min(x0, n.x); x1 = Math.max(x1, n.x + n.size.w + TOGGLE_R * 2);
    y0 = Math.min(y0, n.y - n.size.h / 2); y1 = Math.max(y1, n.y + n.size.h / 2);
  }
  const W = svg.clientWidth, H = svg.clientHeight, padX = 32, padTop = 48, padBottom = 32;
  const k = Math.max(minK, Math.min(1.1, (W - padX * 2) / (x1 - x0), (H - padTop - padBottom) / (y1 - y0)));
  const x = Math.max(padX, (W - (x1 - x0) * k) / 2) - x0 * k;
  const y = (y1 - y0) * k > H - padTop - padBottom
    ? padTop - y0 * k
    : padTop + (H - padTop - padBottom - (y1 - y0) * k) / 2 - y0 * k;
  transform = { k, x, y };
  applyTransform();
}

document.getElementById("expand").onclick = () => { setAll(TREE, true); render(null); fit(0.7); };
document.getElementById("collapse").onclick = () => { resetOpen(TREE); render(null); fit(); };
document.getElementById("fit").onclick = fit;

// Pan and zoom.
let drag = null, dragged = false;
svg.addEventListener("pointerdown", e => { drag = { x: e.clientX, y: e.clientY, tx: transform.x, ty: transform.y }; dragged = false; });
window.addEventListener("pointermove", e => {
  if (!drag) return;
  if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) { dragged = true; svg.classList.add("dragging"); }
  transform.x = drag.tx + e.clientX - drag.x;
  transform.y = drag.ty + e.clientY - drag.y;
  applyTransform();
});
window.addEventListener("pointerup", () => { drag = null; svg.classList.remove("dragging"); });
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;
  const k = Math.min(2.5, Math.max(0.2, transform.k * Math.exp(-e.deltaY * 0.0015)));
  transform.x = mx - (mx - transform.x) * (k / transform.k);
  transform.y = my - (my - transform.y) * (k / transform.k);
  transform.k = k;
  applyTransform();
}, { passive: false });
svg.addEventListener("click", () => {
  if (dragged || !selected) return;
  selected = null;
  card.style.display = "none";
  render(null);
});

(document.fonts ? document.fonts.ready : Promise.resolve()).then(() => {
  render(null);
  fit();
});
</script>
</body>
</html>
"""
