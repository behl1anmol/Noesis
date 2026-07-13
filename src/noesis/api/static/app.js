/* Noesis dashboard — vanilla JS, no dependencies, no external requests. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const PAGE = document.body.dataset.page || "";
  const PID = document.body.dataset.projectId || "";
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- formatting -------------------------------------------------------- */

  function fmtDur(s) {
    if (s == null || s === "" || isNaN(s)) return "—";
    s = Math.max(0, Math.round(+s));
    if (s < 60) return s + "s";
    const m = Math.floor(s / 60), sec = s % 60;
    if (m < 60) return sec ? m + "m " + sec + "s" : m + "m";
    const h = Math.floor(m / 60), mm = m % 60;
    if (h < 24) return mm ? h + "h " + mm + "m" : h + "h";
    return Math.floor(h / 24) + "d " + (h % 24) + "h";
  }

  function fmtAgo(s) {
    if (s == null || s === "" || isNaN(s)) return null;
    s = Math.max(0, +s);
    if (s < 45) return "just now";
    return fmtDur(s).split(" ")[0] + " ago";
  }

  function fmtNum(v) {
    return v >= 10000 ? Math.round(v / 1000) + "k"
         : v >= 1000 ? (Math.round(v / 100) / 10) + "k"
         : String(Math.round(v));
  }

  function renderAges(root = document) {
    $$("[data-age]", root).forEach((el) => {
      const ago = fmtAgo(el.dataset.age);
      el.textContent = ago === null
        ? (el.dataset.never || "never indexed")
        : (el.dataset.prefix || "") + ago;
    });
  }

  function renderTimes(root = document) {
    const now = Date.now();
    $$("[data-rel]", root).forEach((el) => {
      const t = Date.parse(el.dataset.ts);
      el.textContent = isNaN(t) ? "—" : fmtAgo((now - t) / 1000) || "just now";
    });
    $$("[data-dur]", root).forEach((el) => {
      const a = Date.parse(el.dataset.start), b = Date.parse(el.dataset.end);
      el.textContent = isNaN(a) || isNaN(b) ? "—" : fmtDur((b - a) / 1000);
    });
    $$("[data-secs]", root).forEach((el) => { el.textContent = fmtDur(el.dataset.secs); });
  }

  function animNum(el, to) {
    if (!el) return;
    to = +to || 0;
    const from = "v" in el.dataset ? +el.dataset.v : +String(el.textContent).replace(/[^\d.-]/g, "") || 0;
    el.dataset.v = to;
    if (REDUCED || from === to) { el.textContent = to.toLocaleString(); return; }
    const t0 = performance.now(), dur = 450;
    (function step(t) {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.textContent = Math.round(from + (to - from) * e).toLocaleString();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  /* ---- toast + fetch ------------------------------------------------------ */

  function toast(msg, kind) {
    const box = $("#toasts");
    if (!box) return;
    const t = document.createElement("div");
    t.className = "toast" + (kind === "ok" ? " ok" : "");
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => { t.classList.add("leaving"); setTimeout(() => t.remove(), 350); }, 4200);
  }

  async function post(url, body) {
    const opts = { method: "POST" };
    if (body !== undefined) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = "Request failed (" + res.status + ")";
      try { const j = await res.json(); if (j && j.detail) detail = String(j.detail); } catch (e) { /* no body */ }
      throw new Error(detail);
    }
    try { return await res.json(); } catch (e) { return null; }
  }

  /* ---- live project state ------------------------------------------------- */

  const CHIP_TEXT = { idle: "never run", running: "running", done: "done", failed: "failed" };

  function setChip(el, status, errTitle) {
    if (!el) return;
    el.className = "chip chip-" + status;
    el.textContent = CHIP_TEXT[status] || status;
    if (errTitle) el.title = errTitle; else el.removeAttribute("title");
  }

  function applyProject(p) {
    const root = document.querySelector('[data-project][data-id="' + CSS.escape(p.id) + '"]');
    if (!root) return;
    const f = (n) => root.querySelector('[data-f="' + n + '"]');

    animNum(f("files"), p.file_count);
    animNum(f("chunks"), p.chunk_count);

    const badge = f("pendingbadge");
    if (badge) { badge.hidden = !(p.pending_count > 0); animNum(f("pending"), p.pending_count); }

    const fresh = f("fresh");
    if (fresh) fresh.dataset.age = p.index_age_s == null ? "" : p.index_age_s;

    const running = !!p.progress || (p.last_run && p.last_run.status === "running");
    setChip(f("chip"), running ? "running" : p.last_run ? p.last_run.status : "idle",
            p.last_run && p.last_run.error ? p.last_run.error : "");

    const prog = f("progress");
    if (prog) {
      prog.hidden = !p.progress;
      if (p.progress) {
        const pr = p.progress, bar = f("bar");
        if (pr.percent == null) { bar.classList.add("indet"); bar.style.width = "100%"; f("pct").textContent = ""; }
        else { bar.classList.remove("indet"); bar.style.width = Math.min(100, pr.percent) + "%"; f("pct").textContent = Math.round(pr.percent) + "%"; }
        f("pfiles").textContent = pr.files_done + " / " + pr.files_to_index + " files";
        f("pchunks").textContent = pr.chunks_written + " chunks";
        f("eta").textContent = pr.eta_s != null ? "ETA ~" + fmtDur(pr.eta_s) : "";
      }
    }

    const watching = f("watching");
    if (watching) watching.hidden = !p.watching;

    const watchmode = f("watchmode");
    if (watchmode) watchmode.hidden = p.watch_mode !== "polling";

    const w = root.querySelector('input[data-toggle="watch"]');
    const a = root.querySelector('input[data-toggle="auto"]');
    if (w) w.checked = !!p.watch_enabled;
    if (a) {
      a.checked = !!p.auto_reindex;
      a.disabled = !p.watch_enabled;
      const lbl = a.closest(".switch");
      lbl.classList.toggle("is-disabled", !p.watch_enabled);
      lbl.title = p.watch_enabled ? "Reindex automatically when watched files change" : "Enable Watch to use auto-reindex";
    }
    const pb = root.querySelector('button[data-act="pending"]');
    if (pb && !pb.classList.contains("busy")) pb.hidden = !(p.pending_count > 0);
  }

  function applyTotals(t) {
    for (const k in t) animNum(document.querySelector('[data-total="' + k + '"]'), t[k]);
    const pend = $('[data-stat="pending"]'), run = $('[data-stat="running"]');
    if (pend) pend.classList.toggle("is-warn", t.pending > 0);
    if (run) run.classList.toggle("is-live", t.running > 0);
  }

  function applyDevice(d) {
    if (!d || !("setting" in d)) return;
    const cur = $("[data-device-current]");
    if (cur) cur.textContent = d.resolved || d.setting;
    $$("button[data-device]").forEach((b) =>
      b.classList.toggle("active", b.dataset.device === (d.config_pin || d.setting)));
  }

  /* ---- project-page tables ------------------------------------------------ */

  function cell(row, cls, text) {
    const td = document.createElement("td");
    if (cls) td.className = cls;
    if (text !== undefined) td.textContent = text;
    row.appendChild(td);
    return td;
  }
  function chipIn(td, cls, text) {
    const s = document.createElement("span");
    s.className = "chip " + cls;
    s.textContent = text;
    td.appendChild(s);
  }

  function renderProjectTables(p) {
    const now = Date.now();
    const pb = $("#pending-body");
    if (pb && p.pending_files) {
      pb.textContent = "";
      p.pending_files.forEach((fl) => {
        const tr = document.createElement("tr");
        cell(tr, "mono", fl.path);
        chipIn(cell(tr), "chip-ev-" + fl.event_type, fl.event_type);
        cell(tr, "mono muted", fmtAgo((now - Date.parse(fl.detected_at)) / 1000) || "just now");
        pb.appendChild(tr);
      });
      const empty = $("#pending-empty");
      if (empty) empty.hidden = p.pending_files.length > 0;
    }

    const fs = $("#failed-section"), fb = $("#failed-body");
    if (fs && fb && p.failed_files) {
      fs.hidden = p.failed_files.length === 0;
      fb.textContent = "";
      p.failed_files.forEach((fl) => {
        const tr = document.createElement("tr");
        cell(tr, "mono", fl.path);
        const td = cell(tr);
        const e = document.createElement("span");
        e.className = "err-text";
        e.textContent = fl.error;
        td.appendChild(e);
        fb.appendChild(tr);
      });
    }

    const rb = $("#runs-body");
    if (rb && p.recent_runs) {
      rb.textContent = "";
      p.recent_runs.forEach((r) => {
        const tr = document.createElement("tr");
        const st = cell(tr);
        chipIn(st, "chip-" + r.status, r.status);
        if (r.error) st.firstChild.title = r.error;
        const trig = cell(tr, "", r.triggered_by);
        if (r.fast_path_used) {
          const t = document.createElement("span");
          t.className = "tag"; t.textContent = "fast path";
          trig.appendChild(t);
        }
        cell(tr, "num", r.files_changed);
        cell(tr, "num" + (r.files_failed ? " age-old" : ""), r.files_failed);
        cell(tr, "num", r.chunks_written);
        const a = Date.parse(r.started_at), b = Date.parse(r.finished_at);
        cell(tr, "num", isNaN(a) || isNaN(b) ? "—" : fmtDur((b - a) / 1000));
        cell(tr, "muted", fmtAgo((now - a) / 1000) || "just now");
        rb.appendChild(tr);
      });
      const empty = $("#runs-empty");
      if (empty) empty.hidden = p.recent_runs.length > 0;
    }
  }

  /* ---- polling ------------------------------------------------------------ */

  let pollTimer = null;
  let pollInFlight = false;
  function schedule(fast) {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(poll, fast ? 2000 : 8000);
  }

  async function poll() {
    // Single chain only: never let two polls overlap (a stray schedule()
    // during an in-flight fetch would otherwise fork the chain and the
    // request rate would compound into a flood).
    if (pollInFlight) return;
    // A hidden/background tab does not need live data — skip the fetch and
    // re-check later. visibilitychange resumes immediately when shown.
    if (document.hidden) { schedule(false); return; }
    pollInFlight = true;
    try {
      if (PAGE === "index") {
        const o = await (await fetch("/api/state")).json();
        (o.projects || []).forEach(applyProject);
        applyTotals(o.totals || {});
        applyDevice(o.device);
        renderAges();
        schedule(o.totals.running > 0 || (o.projects || []).some((p) => p.progress));
      } else if (PAGE === "project") {
        const res = await fetch("/api/projects/" + encodeURIComponent(PID) + "/state");
        if (!res.ok) { schedule(false); return; }
        const p = await res.json();
        applyProject(p);
        renderProjectTables(p);
        renderAges();
        schedule(!!p.progress || (p.last_run && p.last_run.status === "running"));
      }
    } catch (e) {
      schedule(false); // server briefly away — keep trying slowly
    } finally {
      pollInFlight = false;
    }
  }

  // Resume promptly when the tab is brought to the foreground.
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && (PAGE === "index" || PAGE === "project")) schedule(true);
  });

  /* ---- actions ------------------------------------------------------------ */

  document.addEventListener("click", async (e) => {
    const pill = e.target.closest("button[data-device]");
    if (pill && !pill.disabled) {
      try {
        const d = await post("/api/settings/device", { device: pill.dataset.device });
        applyDevice(d && d.setting !== undefined ? d : d && d.device);
        toast("Compute device set to " + pill.dataset.device, "ok");
      } catch (err) { toast(err.message); }
      return;
    }

    const btn = e.target.closest("button[data-act]");
    if (btn && btn.dataset.act === "delete") {
      openDeleteConfirm(btn.closest("[data-project]"));
      return;
    }
    if (btn) {
      const card = btn.closest("[data-project]");
      const id = encodeURIComponent(card.dataset.id);
      const url = btn.dataset.act === "reindex"
        ? "/projects/" + id + "/reindex"
        : "/api/projects/" + id + "/reindex-pending";
      btn.disabled = true;
      btn.classList.add("busy");
      try {
        await post(url);
        toast(btn.dataset.act === "reindex" ? "Reindex started" : "Indexing pending changes", "ok");
        schedule(true);
      } catch (err) { toast(err.message); }
      setTimeout(() => { btn.disabled = false; btn.classList.remove("busy"); }, 700);
      return;
    }

    const card = e.target.closest("[data-project][data-href]");
    if (card && !e.target.closest("a, button, input, label, .switch")) {
      location.href = card.dataset.href;
    }
  });

  document.addEventListener("change", async (e) => {
    const t = e.target.closest("input[data-toggle]");
    if (!t) return;
    const card = t.closest("[data-project]");
    const key = t.dataset.toggle === "watch" ? "watch_enabled" : "auto_reindex";
    const prev = !t.checked;
    t.disabled = true;
    try {
      const p = await post("/api/projects/" + encodeURIComponent(card.dataset.id) + "/flags", { [key]: t.checked });
      if (p && typeof p === "object" && "watch_enabled" in p) { applyProject(p); renderAges(); }
    } catch (err) {
      t.checked = prev;
      toast(err.message);
    }
    if (t.dataset.toggle === "watch") t.disabled = false;
    else t.disabled = !card.querySelector('input[data-toggle="watch"]').checked;
  });

  /* ---- charts (usage page) — hand-rolled inline SVG ------------------------ */

  const NS = "http://www.w3.org/2000/svg";
  function sv(tag, attrs, parent) {
    const el = document.createElementNS(NS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(el);
    return el;
  }
  function svTitle(el, text) {
    const t = document.createElementNS(NS, "title");
    t.textContent = text;
    el.appendChild(t);
  }

  function dayKey(d) {
    const p = (n) => String(n).padStart(2, "0");
    return d.getUTCFullYear() + "-" + p(d.getUTCMonth() + 1) + "-" + p(d.getUTCDate());
  }
  function dayLabel(key) {
    // Parse the key as UTC and format in UTC so the label names the same
    // calendar day as the (UTC) key, for viewers on any offset.
    const d = new Date(key + "T00:00:00Z");
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
  }

  // Continuous last-N-days axis; missing days become zero rows. Everything is
  // in UTC to match the server's UTC day buckets (substr(ts,1,10)); mixing a
  // local base/step with UTC keys shifted the axis off by a day east of UTC
  // and could dup/skip a day across DST boundaries.
  function fillDays(perDay, days) {
    const map = {};
    let endT = new Date().setUTCHours(0, 0, 0, 0);
    (perDay || []).forEach((r) => {
      map[r.day] = r;
      const t = new Date(r.day + "T00:00:00Z").getTime();
      if (t > endT) endT = t;
    });
    const out = [];
    const d = new Date(endT);
    const keys = [];
    for (let i = 0; i < days; i++) {
      keys.push(dayKey(d));
      d.setUTCDate(d.getUTCDate() - 1);
    }
    keys.reverse();
    for (const key of keys) {
      out.push(map[key] || { day: key });
    }
    return out;
  }

  const GEO = { W: 720, H: 220, L: 38, R: 14, T: 12, B: 24 };

  function frame(fig, maxV) {
    const body = fig.querySelector("[data-body]");
    body.textContent = "";
    const svg = sv("svg", { viewBox: "0 0 " + GEO.W + " " + GEO.H, role: "img" }, body);
    const { W, H, L, R, T, B } = GEO;
    for (let i = 0; i <= 4; i++) {
      const y = H - B - (H - B - T) * (i / 4);
      if (i > 0) sv("line", { x1: L, x2: W - R, y1: y, y2: y, class: "gridline" }, svg);
      const t = sv("text", { x: L - 7, y: y + 3.5, class: "tick", "text-anchor": "end" }, svg);
      t.textContent = fmtNum(maxV * i / 4);
    }
    sv("line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "axisline" }, svg);
    return svg;
  }

  function niceMax(v) {
    if (!v || v <= 0) return 4;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 4, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  function xLabels(svg, rows, xOf) {
    const step = Math.ceil(rows.length / 8);
    rows.forEach((r, i) => {
      if (i % step) return;
      const t = sv("text", { x: xOf(i), y: GEO.H - 8, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = dayLabel(r.day);
    });
  }

  function legend(fig, items) {
    const box = fig.querySelector("[data-legend]");
    box.textContent = "";
    items.forEach(([name, cls]) => {
      const s = document.createElement("span");
      s.className = "legend-item";
      const sw = document.createElement("i");
      sw.className = "swatch " + cls;
      s.appendChild(sw);
      s.appendChild(document.createTextNode(name));
      box.appendChild(s);
    });
  }

  function emptyChart(fig, msg) {
    const body = fig.querySelector("[data-body]");
    body.textContent = "";
    const d = document.createElement("div");
    d.className = "chart-empty";
    d.textContent = msg;
    body.appendChild(d);
  }

  function chartRuns(fig, ia, days) {
    const rows = fillDays(ia.per_day, days);
    if (!rows.some((r) => (r.runs || 0) > 0)) return emptyChart(fig, "No index runs recorded yet");
    const maxV = niceMax(Math.max(...rows.map((r) => r.runs || 0)));
    const svg = frame(fig, maxV);
    const { W, H, L, R, T, B } = GEO;
    const plotW = W - L - R, plotH = H - T - B, n = rows.length;
    const bw = Math.min(26, (plotW / n) * 0.62);
    const xOf = (i) => L + plotW * (i + 0.5) / n;
    const hOf = (v) => (v / maxV) * plotH;
    rows.forEach((r, i) => {
      const watcher = Math.min(r.watcher_runs || 0, r.runs || 0);
      const manual = Math.max(0, (r.runs || 0) - watcher);
      const x = xOf(i) - bw / 2;
      let y = H - B;
      const tip = dayLabel(r.day) + " — " + (r.runs || 0) + " runs (" + watcher + " watcher, "
        + manual + " manual)" + ((r.failed || 0) ? ", " + r.failed + " failed" : "");
      if (watcher) {
        const rc = sv("rect", { x, y: y - hOf(watcher), width: bw, height: hOf(watcher), rx: 1.5, class: "bar fill-s1" }, svg);
        svTitle(rc, tip);
        y -= hOf(watcher) + 2;
      }
      if (manual) {
        const rc = sv("rect", { x, y: y - hOf(manual), width: bw, height: hOf(manual), rx: 1.5, class: "bar fill-s2" }, svg);
        svTitle(rc, tip);
      }
      if (r.failed) { // slim status-red bar in front, 2px surface ring
        const fw = Math.max(4, bw * 0.34);
        const rc = sv("rect", { x: xOf(i) - fw / 2, y: H - B - hOf(r.failed), width: fw, height: hOf(r.failed), rx: 2, class: "bar fill-crit" }, svg);
        svTitle(rc, tip);
      }
    });
    xLabels(svg, rows, xOf);
    legend(fig, [["Watcher", "s1"], ["Manual", "s2"], ["Failed", "crit"]]);
  }

  function chartQueries(fig, su, days) {
    const rows = fillDays(su.per_day, days);
    if (!rows.some((r) => (r.queries || 0) > 0)) return emptyChart(fig, "No queries recorded yet");
    const maxV = niceMax(Math.max(...rows.map((r) => Math.max(r.mcp || 0, r.rest || 0))));
    const svg = frame(fig, maxV);
    const { W, H, L, R, T, B } = GEO;
    const plotW = W - L - R, plotH = H - T - B, n = rows.length;
    const xOf = (i) => n === 1 ? L + plotW / 2 : L + plotW * (i / (n - 1));
    const yOf = (v) => H - B - (v / maxV) * plotH;
    [["mcp", "1", "MCP"], ["rest", "2", "REST"]].forEach(([key, slot, name]) => {
      const pts = rows.map((r, i) => xOf(i).toFixed(1) + "," + yOf(r[key] || 0).toFixed(1));
      sv("path", {
        d: "M" + L + "," + (H - B) + " L" + pts.join(" L") + " L" + (W - R) + "," + (H - B) + " Z",
        class: "area-s" + slot,
      }, svg);
      sv("polyline", { points: pts.join(" "), class: "series-line line-s" + slot }, svg);
      const last = rows[n - 1];
      sv("circle", { cx: xOf(n - 1), cy: yOf(last[key] || 0), r: 3.5, class: "dot-s" + slot }, svg);
      const t = sv("text", {
        x: W - R, y: yOf(last[key] || 0) - 7, class: "end-label lbl-s" + slot, "text-anchor": "end",
      }, svg);
      t.textContent = name;
    });
    rows.forEach((r, i) => { // hover columns with native tooltips
      const col = sv("rect", {
        x: L + plotW * i / n, y: T, width: plotW / n, height: plotH, class: "hover-col",
      }, svg);
      svTitle(col, dayLabel(r.day) + " — " + (r.queries || 0) + " queries · " + (r.mcp || 0)
        + " MCP · " + (r.rest || 0) + " REST · " + (r.structural || 0) + " structural · "
        + (r.reranked || 0) + " reranked");
    });
    xLabels(svg, rows, xOf);
    legend(fig, [["MCP", "s1"], ["REST", "s2"]]);
  }

  function chartChannels(fig, su) {
    const data = (su.channel_mix || []).filter((c) => c.queries > 0);
    if (!data.length) return emptyChart(fig, "No queries recorded yet");
    data.sort((a, b) => b.queries - a.queries);
    const SLOTS = { mcp: "s1", rest: "s2" }; // color follows the entity, matching the queries chart
    let next = 3;
    const rowH = 34, W = GEO.W, L = 96, R = 52;
    const H = data.length * rowH + 14;
    const body = fig.querySelector("[data-body]");
    body.textContent = "";
    const svg = sv("svg", { viewBox: "0 0 " + W + " " + H, role: "img" }, body);
    const maxV = Math.max(...data.map((c) => c.queries));
    data.forEach((c, i) => {
      const slot = SLOTS[c.channel] || (SLOTS[c.channel] = "s" + Math.min(4, next++));
      const y = 8 + i * rowH;
      const w = Math.max(3, (c.queries / maxV) * (W - L - R));
      const lab = sv("text", { x: L - 10, y: y + 14, class: "row-label", "text-anchor": "end" }, svg);
      lab.textContent = c.channel;
      const bar = sv("rect", { x: L, y, width: w, height: 20, rx: 3, class: "hbar fill-" + slot }, svg);
      svTitle(bar, c.channel + " — " + c.queries + " queries");
      const val = sv("text", { x: L + w + 8, y: y + 14, class: "val-label" }, svg);
      val.textContent = c.queries.toLocaleString();
    });
    legend(fig, []);
  }

  function chartWatcher(fig, wa, days) {
    const rows = fillDays(wa.per_day, days);
    if (!rows.some((r) => (r.events_seen || 0) > 0 || (r.events_coalesced || 0) > 0)) {
      return emptyChart(fig, "No watcher activity yet");
    }
    const maxV = niceMax(Math.max(...rows.map((r) => Math.max(r.events_seen || 0, r.events_coalesced || 0))));
    const svg = frame(fig, maxV);
    const { W, H, L, R, T, B } = GEO;
    const plotW = W - L - R, plotH = H - T - B, n = rows.length;
    const gw = Math.min(26, (plotW / n) * 0.66), bw = gw / 2 - 1;
    const xOf = (i) => L + plotW * (i + 0.5) / n;
    rows.forEach((r, i) => {
      const tip = dayLabel(r.day) + " — " + (r.events_seen || 0) + " seen, "
        + (r.events_coalesced || 0) + " coalesced, " + (r.auto_runs || 0) + " auto runs";
      [["events_seen", "s1", xOf(i) - gw / 2], ["events_coalesced", "s2", xOf(i) + 1]].forEach(([key, slot, x]) => {
        const h = ((r[key] || 0) / maxV) * plotH;
        if (!h) return;
        const rc = sv("rect", { x, y: H - B - h, width: bw, height: h, rx: 1.5, class: "bar fill-" + slot }, svg);
        svTitle(rc, tip);
      });
    });
    xLabels(svg, rows, xOf);
    legend(fig, [["Events seen", "s1"], ["Coalesced", "s2"]]);
  }

  /* ---- init ---------------------------------------------------------------- */

  renderAges();
  renderTimes();
  if (!REDUCED) $$("[data-count]").forEach((el) => {
    const v = +el.dataset.count || 0;
    el.dataset.v = 0;
    el.textContent = "0";
    animNum(el, v);
  });

  if (PAGE === "usage") {
    const raw = $("#usage-data");
    if (raw) {
      const usage = JSON.parse(raw.textContent);
      chartRuns($('[data-chart="runs"]'), usage.index_activity, usage.days);
      chartQueries($('[data-chart="queries"]'), usage.search_usage, usage.days);
      chartChannels($('[data-chart="channels"]'), usage.search_usage);
      chartWatcher($('[data-chart="watcher"]'), usage.watcher_activity, usage.days);
    }
  }

  /* ---- delete confirm (ADR-43) -------------------------------------------- */

  let deleteTarget = null; // the card element pending confirmation

  function openDeleteConfirm(card) {
    const modal = $("[data-confirm-modal]");
    if (!modal || !card) return;
    deleteTarget = card;
    $("[data-confirm-name]", modal).textContent =
      (card.querySelector(".card-title") || {}).textContent || "this project";
    modal.hidden = false;
  }

  function initDeleteConfirm() {
    const modal = $("[data-confirm-modal]");
    if (!modal) return;
    const closeConfirm = () => { modal.hidden = true; deleteTarget = null; };
    $$("[data-confirm-cancel]", modal).forEach((b) => b.addEventListener("click", closeConfirm));
    modal.addEventListener("click", (e) => { if (e.target === modal) closeConfirm(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modal.hidden) closeConfirm(); });
    $("[data-confirm-yes]", modal).addEventListener("click", async () => {
      if (!deleteTarget) return;
      const card = deleteTarget;
      const yes = $("[data-confirm-yes]", modal);
      yes.disabled = true;
      try {
        const res = await fetch("/api/projects/" + encodeURIComponent(card.dataset.id), { method: "DELETE" });
        if (!res.ok) {
          let detail = "Delete failed (" + res.status + ")";
          try { const j = await res.json(); if (j && j.detail) detail = String(j.detail); } catch (e2) { /* no body */ }
          throw new Error(detail);
        }
        card.remove();
        toast("Project deleted", "ok");
        closeConfirm();
        schedule(true); // refresh totals promptly
      } catch (err) { toast(err.message); }
      yes.disabled = false;
    });
  }

  /* ---- register modal (ADR-42) ------------------------------------------- */

  function initRegister() {
    const backdrop = $("[data-register-modal]");
    if (!backdrop) return;
    const form = $("[data-reg-form]", backdrop);
    const pathEl = $("[data-reg-path]", backdrop);
    const pathHint = $("[data-reg-path-hint]", backdrop);
    const errEl = $("[data-reg-error]", backdrop);
    const langsBox = $("[data-reg-langs]", backdrop);
    const previewBox = $("[data-reg-preview]", backdrop);
    const previewTotal = $("[data-reg-preview-total]", backdrop);
    const previewBars = $("[data-reg-preview-bars]", backdrop);
    const watchEl = $("[data-reg-watch]", backdrop);
    const autoEl = $("[data-reg-auto]", backdrop);
    const autoLabel = $("[data-reg-autolabel]", backdrop);
    const picker = $("[data-reg-picker]", backdrop);
    let submitting = false;

    function setErr(msg) { if (errEl) { errEl.textContent = msg || ""; errEl.classList.toggle("is-error", !!msg); } }

    // Language chips are server-rendered into the template — no runtime
    // fetch, nothing to load, the modal is complete the moment it opens.
    function open() {
      backdrop.hidden = false;
      document.body.classList.add("modal-open");
      setErr("");
      setTimeout(() => pathEl && pathEl.focus(), 30);
    }
    function close() {
      backdrop.hidden = true;
      document.body.classList.remove("modal-open");
      if (picker) picker.hidden = true;
    }

    function selectedLangs() {
      return $$('[data-lang="1"]', langsBox).filter((c) => c.checked).map((c) => c.value);
    }
    function scope() {
      const langs = selectedLangs();
      const mb = parseFloat($("[data-reg-maxmb]", backdrop).value);
      const ignores = ($("[data-reg-ignores]", backdrop).value || "")
        .split(",").map((s) => s.trim()).filter(Boolean);
      const body = {
        root_path: (pathEl.value || "").trim(),
        follow_symlinks: $("[data-reg-symlinks]", backdrop).checked,
      };
      if (langs.length) body.index_languages = langs;
      if (!isNaN(mb) && mb > 0) body.max_file_bytes = Math.round(mb * 1048576);
      if (ignores.length) body.extra_ignores = ignores;
      return body;
    }

    async function runPreview() {
      const body = scope();
      if (!body.root_path) { setErr("Enter a project folder first."); return; }
      setErr("");
      previewTotal.textContent = "…";
      previewBox.hidden = false;
      try {
        const p = await post("/api/register/preview", body);
        previewTotal.textContent = p.total_files.toLocaleString();
        const max = Math.max(1, ...p.by_language.map((b) => b.files));
        previewBars.textContent = "";
        if (!p.by_language.length) {
          const d = document.createElement("div"); d.className = "muted";
          d.textContent = "No files match — nothing would be indexed."; previewBars.appendChild(d);
        }
        p.by_language.forEach((b) => {
          const row = document.createElement("div"); row.className = "pv-row";
          const name = document.createElement("span"); name.className = "pv-name"; name.textContent = b.language;
          const track = document.createElement("span"); track.className = "pv-track";
          const fill = document.createElement("span"); fill.className = "pv-fill";
          fill.style.width = Math.round((b.files / max) * 100) + "%";
          track.appendChild(fill);
          const n = document.createElement("span"); n.className = "pv-n"; n.textContent = b.files;
          row.appendChild(name); row.appendChild(track); row.appendChild(n);
          previewBars.appendChild(row);
        });
      } catch (e) { previewBox.hidden = true; setErr(e.message); }
    }

    async function submit(indexNow) {
      if (submitting) return;
      const body = scope();
      if (!body.root_path) { setErr("Enter a project folder first."); return; }
      body.watch = watchEl.checked;
      body.auto_reindex = watchEl.checked && autoEl.checked;
      body.index_now = indexNow;
      submitting = true;
      form.classList.add("busy");
      setErr("");
      try {
        await post("/api/register", body);
        toast(indexNow ? "Project added — indexing started" : "Project added", "ok");
        close();
        form.reset();
        previewBox.hidden = true;
        location.reload();
      } catch (e) { setErr(e.message); }
      submitting = false;
      form.classList.remove("busy");
    }

    /* ---- folder picker ---- */
    async function browseTo(path) {
      try {
        const url = "/api/browse" + (path ? "?path=" + encodeURIComponent(path) : "");
        const data = await (await fetch(url)).json();
        if (data.detail) { setErr(String(data.detail)); return; }
        picker.hidden = false;
        picker.dataset.cwd = data.path;
        picker.dataset.parent = data.parent || "";
        $("[data-picker-cwd]", picker).textContent = data.path;
        $("[data-picker-up]", picker).disabled = !data.parent;
        const list = $("[data-picker-list]", picker);
        list.textContent = "";
        if (!data.entries.length) {
          const li = document.createElement("li"); li.className = "muted"; li.textContent = "(no sub-folders)";
          list.appendChild(li);
        }
        data.entries.forEach((e) => {
          const li = document.createElement("li");
          const b = document.createElement("button");
          b.type = "button"; b.className = "picker-item"; b.textContent = "📁 " + e.name;
          b.dataset.path = e.path;
          li.appendChild(b); list.appendChild(li);
        });
      } catch (e) { setErr("browse failed: " + e.message); }
    }

    // wiring
    $$("[data-open-register]").forEach((b) => b.addEventListener("click", open));
    $("[data-reg-close]", backdrop).addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !backdrop.hidden) close(); });

    pathEl.addEventListener("input", () => { setErr(""); pathHint.textContent = ""; });
    watchEl.addEventListener("change", () => {
      autoEl.disabled = !watchEl.checked;
      autoLabel.classList.toggle("is-disabled", !watchEl.checked);
      if (!watchEl.checked) autoEl.checked = false;
    });

    $("[data-reg-preview-btn]", backdrop).addEventListener("click", runPreview);
    $("[data-reg-refresh]", backdrop).addEventListener("click", runPreview);
    $("[data-reg-browse]", backdrop).addEventListener("click", () => browseTo(pathEl.value.trim() || null));
    $("[data-picker-up]", picker).addEventListener("click", () => browseTo(picker.dataset.parent || null));
    $("[data-picker-cancel]", picker).addEventListener("click", () => { picker.hidden = true; });
    $("[data-picker-choose]", picker).addEventListener("click", () => {
      pathEl.value = picker.dataset.cwd || ""; picker.hidden = true; setErr("");
    });
    $("[data-picker-list]", picker).addEventListener("click", (e) => {
      const b = e.target.closest(".picker-item");
      if (b) browseTo(b.dataset.path);
    });

    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const btn = e.submitter;
      submit(btn && btn.dataset.regSubmit === "index");
    });
  }

  if (PAGE === "index" || PAGE === "project") {
    const busyNow = $$('[data-f="progress"]').some((el) => !el.hidden);
    schedule(busyNow);
    // keep relative times fresh between polls
    setInterval(renderTimes, 30000);
  }

  if (PAGE === "index") { initRegister(); initDeleteConfirm(); }
})();
