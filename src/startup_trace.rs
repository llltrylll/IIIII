//! Opt-in startup timing. Set `MP_STARTUP_TRACE=1` (report on stderr) or
//! `MP_STARTUP_TRACE=<file>` to record spans, per-thumbnail rows, per-scan-file
//! phases and per-frame work times from process start. Everything is a no-op
//! (one cached check per call) unless the variable is set. Times are ms since
//! the first `init()` call, which `main` makes before anything else.
//!
//! The report is written when the first `MP_STARTUP_FRAMES` frames (default
//! 300) have run, and again when the shelf thumbnails finish; the later write
//! replaces the earlier one with a superset.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct Rec {
    name: &'static str,
    detail: String,
    thread: String,
    at: f64,
    dur: f64,
}

/// One shelf thumbnail, all durations in ms.
pub struct Thumb {
    pub label: String,
    pub kind: &'static str,
    pub at: f64,
    pub layers: f64,
    pub composite: f64,
    pub render: f64,
    pub gpu_wait: f64,
    pub copy_map: f64,
    pub total: f64,
}

struct Frame {
    at: f64,
    /// Wall time since the previous frame started.
    delta: f64,
    /// Time spent inside `ui()`.
    work: f64,
}

struct Trace {
    t0: Instant,
    recs: Vec<Rec>,
    thumbs: Vec<Thumb>,
    events: Vec<(&'static str, f64)>,
    counters: Vec<(&'static str, u64)>,
    frames: Vec<Frame>,
    last_frame_start: Option<Instant>,
    smart_files: Vec<(String, Vec<String>)>,
    requests: HashMap<String, (String, Instant)>,
    decoded: HashSet<String>,
}

static ON: OnceLock<Option<String>> = OnceLock::new();
static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

thread_local! {
    static DETAIL: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn target() -> &'static Option<String> {
    ON.get_or_init(|| std::env::var("MP_STARTUP_TRACE").ok().filter(|v| !v.is_empty() && v != "0"))
}

fn frames_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("MP_STARTUP_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(300))
}

fn with<R>(f: impl FnOnce(&mut Trace) -> R) -> Option<R> {
    target().as_ref()?;
    let mut g = TRACE.lock().ok()?;
    let t = g.get_or_insert_with(|| Trace {
        t0: Instant::now(),
        recs: Vec::new(),
        thumbs: Vec::new(),
        events: Vec::new(),
        counters: Vec::new(),
        frames: Vec::new(),
        last_frame_start: None,
        smart_files: Vec::new(),
        requests: HashMap::new(),
        decoded: HashSet::new(),
    });
    Some(f(t))
}

fn thread_name() -> String {
    let t = std::thread::current();
    t.name().unwrap_or("unnamed").to_string()
}

pub fn enabled() -> bool {
    target().is_some()
}

/// Milliseconds since the trace started (0 when disabled).
pub fn now_ms() -> f64 {
    with(|t| t.t0.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0)
}

/// Starts the clock; call first thing in `main`.
pub fn init() {
    with(|_| ());
}

/// `dir/file` for a scan image, the key the worker and the shelf agree on.
pub fn file_key(path: &Path) -> String {
    let dir = path.parent().and_then(|d| d.file_name()).map_or(String::new(), |d| format!("{}/", d.to_string_lossy()));
    format!("{dir}{}", path.file_name().map_or("image".into(), |n| n.to_string_lossy()))
}

/// A span records its duration when dropped. Without an explicit detail it
/// takes the thread's current `detail()` scope, if any.
pub struct Span(Option<(&'static str, Instant, Option<String>)>);

pub fn span(name: &'static str) -> Span {
    Span(target().as_ref().map(|_| (name, Instant::now(), None)))
}

pub fn span_d(name: &'static str, detail: String) -> Span {
    Span(target().as_ref().map(|_| (name, Instant::now(), Some(detail))))
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((name, start, detail)) = self.0.take() {
            let dur = start.elapsed().as_secs_f64() * 1e3;
            let detail = detail.or_else(|| DETAIL.with(|d| d.borrow().clone())).unwrap_or_default();
            let thread = thread_name();
            with(|t| {
                let at = start.duration_since(t.t0).as_secs_f64() * 1e3;
                t.recs.push(Rec { name, detail, thread, at, dur });
            });
        }
    }
}

/// Spans opened on this thread while the guard lives carry `detail`.
pub struct DetailGuard(Option<Option<String>>);

pub fn detail(detail: String) -> DetailGuard {
    if !enabled() {
        return DetailGuard(None);
    }
    DetailGuard(Some(DETAIL.with(|d| d.borrow_mut().replace(detail))))
}

impl Drop for DetailGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.0.take() {
            DETAIL.with(|d| *d.borrow_mut() = prev);
        }
    }
}

/// A named instant. Repeated names form a series (cumulative counts in the report).
pub fn event(name: &'static str) {
    with(|t| {
        let at = t.t0.elapsed().as_secs_f64() * 1e3;
        t.events.push((name, at));
    });
}

pub fn count(name: &'static str, n: u64) {
    with(|t| match t.counters.iter_mut().find(|c| c.0 == name) {
        Some(c) => c.1 += n,
        None => t.counters.push((name, n)),
    });
}

pub fn thumb(row: Thumb) {
    with(|t| t.thumbs.push(row));
}

/// Remembers which scan files a smart material needs (once per label).
pub fn smart_files(label: &str, files: Vec<String>) {
    with(|t| {
        if !t.smart_files.iter().any(|s| s.0 == label) {
            t.smart_files.push((label.to_string(), files));
        }
    });
}

fn request_key(paths: &[impl AsRef<Path>]) -> String {
    let mut keys: Vec<String> = paths.iter().map(|p| file_key(p.as_ref())).collect();
    keys.sort();
    keys.join("|")
}

/// The shelf asked the scan worker for these files on behalf of `label`.
pub fn scan_request(label: &str, paths: &[impl AsRef<Path>]) {
    let key = request_key(paths);
    with(|t| {
        t.requests.insert(key, (label.to_string(), Instant::now()));
    });
    count("scan: batches requested", 1);
    count("scan: files requested from worker", paths.len() as u64);
}

/// The label a request was made for (empty when unknown).
pub fn scan_label(paths: &[impl AsRef<Path>]) -> String {
    let key = request_key(paths);
    with(|t| t.requests.get(&key).map(|r| r.0.clone())).flatten().unwrap_or_default()
}

/// The UI thread got a finished batch: records the request-to-receive wait.
pub fn scan_received(paths: &[impl AsRef<Path>]) {
    let key = request_key(paths);
    let found = with(|t| t.requests.remove(&key)).flatten();
    if let Some((label, sent)) = found {
        let dur = sent.elapsed().as_secs_f64() * 1e3;
        with(|t| {
            let at = sent.duration_since(t.t0).as_secs_f64() * 1e3;
            t.recs.push(Rec { name: "scan: request -> received on UI thread", detail: label, thread: "ui".into(), at, dur });
        });
    }
}

/// Notes that the worker is decoding `key`; counts repeats of the same file.
pub fn note_decode(key: &str) {
    let again = with(|t| !t.decoded.insert(key.to_string())).unwrap_or(false);
    count(if again { "scan: files decoded AGAIN" } else { "scan: files decoded (first time)" }, 1);
}

/// Per-frame guard: records wall delta and the time spent in `ui()`.
pub struct FrameGuard(Option<Instant>);

pub fn frame() -> FrameGuard {
    if !enabled() {
        return FrameGuard(None);
    }
    let now = Instant::now();
    with(|t| {
        if t.frames.is_empty() {
            let at = t.t0.elapsed().as_secs_f64() * 1e3;
            t.events.push(("first ui() frame", at));
        }
        let _ = t.last_frame_start;
    });
    FrameGuard(Some(now))
}

impl Drop for FrameGuard {
    fn drop(&mut self) {
        let Some(start) = self.0.take() else { return };
        let work = start.elapsed().as_secs_f64() * 1e3;
        let done = with(|t| {
            let delta = t.last_frame_start.map_or(0.0, |p| start.duration_since(p).as_secs_f64() * 1e3);
            t.last_frame_start = Some(start);
            let at = start.duration_since(t.t0).as_secs_f64() * 1e3;
            if t.frames.len() < frames_limit() {
                t.frames.push(Frame { at, delta, work });
            }
            t.frames.len() == frames_limit()
        })
        .unwrap_or(false);
        if done {
            write_report();
        }
    }
}

/// The shelf thumbnails are all in place: writes the report so far.
pub fn shelf_done() {
    event("shelf thumbnails done");
    write_report();
}

fn write_report() {
    let Some(text) = with(|t| render(t)) else { return };
    match target() {
        Some(p) if p != "1" => {
            let _ = std::fs::write(p, &text);
        }
        _ => eprintln!("{text}"),
    }
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() as f64 - 1.0) * p) as usize]
}

fn render(t: &Trace) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    let _ = writeln!(o, "== startup trace, all times in ms since process start unless noted ==");
    let _ = writeln!(o, "(written at {:.0} ms)", t.t0.elapsed().as_secs_f64() * 1e3);

    // 1. Milestones.
    let _ = writeln!(o, "\n-- 1. milestones --");
    let mut names: Vec<&str> = Vec::new();
    for e in &t.events {
        if !names.contains(&e.0) {
            names.push(e.0);
        }
    }
    let mut rows: Vec<(f64, String)> = Vec::new();
    for n in &names {
        let ts: Vec<f64> = t.events.iter().filter(|e| e.0 == *n).map(|e| e.1).collect();
        if ts.len() == 1 {
            rows.push((ts[0], format!("{:>9.1}  {n}", ts[0])));
        } else {
            rows.push((ts[0], format!("{:>9.1}  {n}: first of {}", ts[0], ts.len())));
            rows.push((ts[ts.len() - 1], format!("{:>9.1}  {n}: last of {}", ts[ts.len() - 1], ts.len())));
        }
    }
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    for r in rows {
        let _ = writeln!(o, "  {}", r.1);
    }

    // 2. Span totals by name and thread.
    let _ = writeln!(o, "\n-- 2. spans: count, total, avg, max (ms) by thread --");
    let mut agg: Vec<(&str, &str, usize, f64, f64)> = Vec::new();
    for r in &t.recs {
        match agg.iter_mut().find(|a| a.0 == r.name && a.1 == r.thread) {
            Some(a) => {
                a.2 += 1;
                a.3 += r.dur;
                a.4 = a.4.max(r.dur);
            }
            None => agg.push((r.name, &r.thread, 1, r.dur, r.dur)),
        }
    }
    agg.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(b.1)));
    for (n, th, c, tot, max) in &agg {
        let _ = writeln!(o, "  {n:<46} [{th:<22}] x{c:<4} total {tot:>9.1}  avg {:>8.2}  max {max:>8.1}", tot / *c as f64);
    }

    // 2b. Spans with per-item detail (pipeline labels, named resources, etc.).
    // Scan phases and CPU shelf previews have their own tables below.
    let _ = writeln!(o, "\n-- 2b. detailed spans: count, total, avg, max (ms), top 80 by total --");
    let mut detailed: Vec<(String, String, String, usize, f64, f64)> = Vec::new();
    for r in t.recs.iter().filter(|r| {
        !r.detail.is_empty()
            && !r.name.starts_with("scan:")
            && !r.name.starts_with("cpu preview:")
    }) {
        if let Some(a) = detailed.iter_mut().find(|a| a.0 == r.name && a.1 == r.detail && a.2 == r.thread) {
            a.3 += 1;
            a.4 += r.dur;
            a.5 = a.5.max(r.dur);
        } else {
            detailed.push((r.name.to_string(), r.detail.clone(), r.thread.clone(), 1, r.dur, r.dur));
        }
    }
    detailed.sort_by(|a, b| b.4.total_cmp(&a.4));
    for (name, detail, thread, count, total, max) in detailed.iter().take(80) {
        let _ = writeln!(
            o,
            "  {name:<34} [{thread:<18}] x{count:<3} total {total:>9.1} avg {:>8.2} max {max:>8.1} | {detail}",
            total / *count as f64
        );
    }

    // 3. CPU previews.
    let _ = writeln!(o, "\n-- 3. CPU previews (worker `shelf-cpu-previews`) --");
    for (name, cat) in [("cpu preview: material", "materials"), ("cpu preview: smart", "smart materials"), ("cpu preview: generator", "generators")] {
        let rs: Vec<&Rec> = t.recs.iter().filter(|r| r.name == name).collect();
        let tot: f64 = rs.iter().map(|r| r.dur).sum();
        let _ = writeln!(o, "  {cat}: {} previews, total {tot:.1} ms", rs.len());
        for r in rs {
            let _ = writeln!(o, "    {:>9.1} +{:>8.2}  {}", r.at, r.dur, r.detail);
        }
    }

    // 4. Scan files.
    let _ = writeln!(o, "\n-- 4. scan images, per file (ms): worker = read/decode/convert/resize/mips, UI thread = GPU create/write --");
    let phases = [
        "scan: read file",
        "scan: decode",
        "scan: convert",
        "scan: resize",
        "scan: mips",
        "scan: gpu create_texture",
        "scan: gpu write_texture",
    ];
    let _ = writeln!(o, "  {:<48} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9}", "file", "read", "decode", "convert", "resize", "mips", "gpu-create", "gpu-write");
    let mut files: Vec<&str> = Vec::new();
    for r in &t.recs {
        if phases.contains(&r.name) && !files.contains(&r.detail.as_str()) {
            files.push(&r.detail);
        }
    }
    let phase_sum = |file: &str, phase: &str| -> f64 { t.recs.iter().filter(|r| r.name == phase && r.detail == file).map(|r| r.dur).sum() };
    let mut grand = [0.0f64; 7];
    for f in &files {
        let v: Vec<f64> = phases.iter().map(|p| phase_sum(f, p)).collect();
        for (g, x) in grand.iter_mut().zip(&v) {
            *g += x;
        }
        let _ = writeln!(o, "  {f:<48} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>9.1}", v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    }
    let _ = writeln!(o, "  {:<48} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>9.1}", "ALL FILES", grand[0], grand[1], grand[2], grand[3], grand[4], grand[5], grand[6]);

    let _ = writeln!(o, "\n-- 4b. scan images, per smart material (ms) --");
    let _ = writeln!(
        o,
        "  {:<34} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>11} {:>11}",
        "smart material", "files", "read", "decode", "convert", "resize", "mips", "gpu-create", "gpu-write", "UI-upload", "req->recv"
    );
    for (label, fs) in &t.smart_files {
        let v: Vec<f64> = phases.iter().map(|p| fs.iter().map(|f| phase_sum(f, p)).sum()).collect();
        let upload: f64 = t.recs.iter().filter(|r| r.name == "scan: upload batch (UI thread)" && &r.detail == label).map(|r| r.dur).sum();
        let rt: f64 = t.recs.iter().filter(|r| r.name == "scan: request -> received on UI thread" && &r.detail == label).map(|r| r.dur).sum();
        // Avoid printing negative zero for phases with no recorded work.
        let clean = |x: f64| if x.abs() < 0.05 { 0.0 } else { x };
        let _ = writeln!(
            o,
            "  {label:<34} {:>5} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1} {:>9.1} {:>9.1} {:>11.1} {:>11.1}",
            fs.len(),
            clean(v[0]),
            clean(v[1]),
            clean(v[2]),
            clean(v[3]),
            clean(v[4]),
            clean(v[5]),
            clean(v[6]),
            clean(upload),
            clean(rt)
        );
    }

    // 5. Thumbnails.
    let _ = writeln!(o, "\n-- 5. shelf thumbnails (ms): layers / composite submit / render_3d submit / GPU wait / copy+map / total to ColorImage --");
    let _ = writeln!(o, "  {:>9}  {:<9} {:>8} {:>9} {:>8} {:>8} {:>8} {:>8}  label", "at", "kind", "layers", "composite", "render", "gpu-wait", "copymap", "total");
    let mut sum = [0.0f64; 6];
    for r in &t.thumbs {
        let v = [r.layers, r.composite, r.render, r.gpu_wait, r.copy_map, r.total];
        for (s, x) in sum.iter_mut().zip(&v) {
            *s += x;
        }
        let _ = writeln!(
            o,
            "  {:>9.1}  {:<9} {:>8.2} {:>9.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}  {}",
            r.at, r.kind, r.layers, r.composite, r.render, r.gpu_wait, r.copy_map, r.total, r.label
        );
    }
    let _ = writeln!(
        o,
        "  {:>9}  {:<9} {:>8.1} {:>9.1} {:>8.1} {:>8.1} {:>8.1} {:>8.1}  SUM of {} thumbnails",
        "", "", sum[0], sum[1], sum[2], sum[3], sum[4], sum[5], t.thumbs.len()
    );
    for kind in ["smart", "material", "generator"] {
        let rs: Vec<&Thumb> = t.thumbs.iter().filter(|r| r.kind == kind).collect();
        if rs.is_empty() {
            continue;
        }
        let f = |g: fn(&Thumb) -> f64| rs.iter().map(|r| g(r)).sum::<f64>();
        let _ = writeln!(
            o,
            "  {kind:<9} x{:<3}: layers {:.1}  composite {:.1}  render {:.1}  gpu-wait {:.1}  copy+map {:.1}  total {:.1}",
            rs.len(),
            f(|r| r.layers),
            f(|r| r.composite),
            f(|r| r.render),
            f(|r| r.gpu_wait),
            f(|r| r.copy_map),
            f(|r| r.total)
        );
    }

    // 6. Counters.
    let _ = writeln!(o, "\n-- 6. counters --");
    for (n, c) in &t.counters {
        let _ = writeln!(o, "  {n:<52} {c}");
    }

    // 7. Cumulative series.
    let _ = writeln!(o, "\n-- 7. previews ready over time (cumulative, per second since process start) --");
    let series = ["cpu preview applied", "thumbnail ready"];
    let end = t.events.iter().filter(|e| series.contains(&e.0)).map(|e| e.1).fold(0.0, f64::max);
    let _ = writeln!(o, "  {:>8} {:>20} {:>16}", "t (s)", series[0], series[1]);
    let mut s = 0.0;
    while s <= end + 1000.0 && s < 90_000.0 {
        let c: Vec<usize> = series.iter().map(|n| t.events.iter().filter(|e| e.0 == *n && e.1 <= s).count()).collect();
        let _ = writeln!(o, "  {:>8.1} {:>20} {:>16}", s / 1e3, c[0], c[1]);
        s += 1000.0;
    }

    // 8. Frames.
    let _ = writeln!(o, "\n-- 8. first {} frames (idle gaps are not counted as work) --", t.frames.len());
    let mut work: Vec<f64> = t.frames.iter().map(|f| f.work).collect();
    work.sort_by(|a, b| a.total_cmp(b));
    let _ = writeln!(
        o,
        "  work inside ui(): median {:.2}  p95 {:.2}  p99 {:.2}  max {:.2}  sum {:.0}; frames over 16.7 ms: {}, over 50 ms: {}, over 100 ms: {}",
        pct(&work, 0.5),
        pct(&work, 0.95),
        pct(&work, 0.99),
        work.last().copied().unwrap_or(0.0),
        work.iter().sum::<f64>(),
        work.iter().filter(|&&w| w > 16.7).count(),
        work.iter().filter(|&&w| w > 50.0).count(),
        work.iter().filter(|&&w| w > 100.0).count()
    );
    // A gap beyond the previous frame's work plus 100 ms is the app waiting
    // for an event (egui only repaints on request), not work.
    let mut busy: Vec<f64> = Vec::new();
    let (mut idle_n, mut idle_ms) = (0usize, 0.0f64);
    for (i, f) in t.frames.iter().enumerate().skip(1) {
        let prev_work = t.frames[i - 1].work;
        let gap = f.delta - prev_work;
        if gap > 100.0 {
            idle_n += 1;
            idle_ms += gap;
        } else {
            busy.push(f.delta);
        }
    }
    busy.sort_by(|a, b| a.total_cmp(b));
    let _ = writeln!(
        o,
        "  frame-to-frame (non-idle): median {:.1}  p95 {:.1}  max {:.1} ({} frames); idle gaps >100 ms: {} totalling {:.0} ms (excluded)",
        pct(&busy, 0.5),
        pct(&busy, 0.95),
        busy.last().copied().unwrap_or(0.0),
        busy.len(),
        idle_n,
        idle_ms
    );
    let _ = writeln!(o, "  first 40 frames: at / work / delta");
    for f in t.frames.iter().take(40) {
        let _ = writeln!(o, "    {:>9.1}  {:>8.2}  {:>8.1}", f.at, f.work, f.delta);
    }
    o
}
