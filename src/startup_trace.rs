//! Opt-in startup timing. Set `MP_STARTUP_TRACE=1` (stderr) or
//! `MP_STARTUP_TRACE=<file>` to record named spans from process start and the
//! CPU time of the first `FRAMES` frames; the report is written once.
//! Disabled (the default) it costs one cached bool check per call.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Frames recorded before the report (`MP_STARTUP_FRAMES` overrides, default 300).
fn frames_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("MP_STARTUP_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(300))
}

struct Trace {
    t0: Instant,
    spans: Vec<(&'static str, f64, f64)>,
    frames: Vec<f64>,
    last_frame: Option<Instant>,
    done: bool,
}

static ON: OnceLock<Option<String>> = OnceLock::new();
static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

fn target() -> &'static Option<String> {
    ON.get_or_init(|| std::env::var("MP_STARTUP_TRACE").ok().filter(|v| !v.is_empty() && v != "0"))
}

fn with<R>(f: impl FnOnce(&mut Trace) -> R) -> Option<R> {
    target().as_ref()?;
    let mut g = TRACE.lock().ok()?;
    let t = g.get_or_insert_with(|| Trace { t0: Instant::now(), spans: Vec::new(), frames: Vec::new(), last_frame: None, done: false });
    Some(f(t))
}

/// Starts the clock; call first thing in `main`.
pub fn init() {
    with(|_| ());
}

/// A span that records its duration when dropped.
pub struct Span(Option<(&'static str, Instant)>);

pub fn span(name: &'static str) -> Span {
    Span(target().as_ref().map(|_| (name, Instant::now())))
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((name, start)) = self.0.take() {
            let dur = start.elapsed().as_secs_f64() * 1e3;
            with(|t| {
                let at = start.duration_since(t.t0).as_secs_f64() * 1e3;
                t.spans.push((name, at, dur));
            });
        }
    }
}

/// Call once at the top of every frame. `wall` frame deltas are measured
/// between consecutive calls, so they include vsync and GPU waits.
pub fn frame() {
    let report = with(|t| {
        if t.done {
            return None;
        }
        let now = Instant::now();
        if let Some(prev) = t.last_frame {
            t.frames.push(now.duration_since(prev).as_secs_f64() * 1e3);
        }
        t.last_frame = Some(now);
        if t.frames.len() >= frames_limit() {
            t.done = true;
            return Some(render(t));
        }
        None
    })
    .flatten();
    if let Some(text) = report {
        match target() {
            Some(p) if p != "1" => {
                let _ = std::fs::write(p, &text);
            }
            _ => eprintln!("{text}"),
        }
    }
}

/// Writes the report now (for harnesses that stop before the frame limit).
pub fn flush() {
    let text = with(|t| {
        t.done = true;
        if t.frames.is_empty() {
            t.frames.push(0.0);
        }
        render(t)
    });
    if let Some(text) = text {
        match target() {
            Some(p) if p != "1" => {
                let _ = std::fs::write(p, &text);
            }
            _ => eprintln!("{text}"),
        }
    }
}

fn render(t: &Trace) -> String {
    let mut out = String::from("== startup trace (ms since process start) ==\n");
    let mut spans = t.spans.clone();
    spans.sort_by(|a, b| a.1.total_cmp(&b.1));
    // Spans repeated per item (thumbnails) are summed by name below.
    let mut sums: Vec<(&str, usize, f64, f64)> = Vec::new();
    for (n, at, d) in &spans {
        match sums.iter_mut().find(|s| s.0 == *n) {
            Some(s) => {
                s.1 += 1;
                s.2 += d;
                s.3 = s.3.max(*d);
            }
            None => sums.push((n, 1, *d, *d)),
        }
        if sums.iter().find(|s| s.0 == *n).is_some_and(|s| s.1 <= 3) {
            out += &format!("  at {at:9.1}  {d:9.2}  {n}\n");
        }
    }
    out += "-- totals per span name: count, total ms, max ms --\n";
    for (n, c, tot, max) in &sums {
        out += &format!("  {n:<40} x{c:<4} total {tot:9.1}  max {max:8.2}\n");
    }
    let mut sorted = t.frames.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let pct = |p: f64| sorted[((n as f64 - 1.0) * p) as usize];
    let slow = t.frames.iter().filter(|&&f| f > 50.0).count();
    out += &format!(
        "-- first {n} frames (wall, ms): median {:.1}, p95 {:.1}, p99 {:.1}, max {:.1}; frames over 50 ms: {slow}; sum {:.0}\n",
        pct(0.5),
        pct(0.95),
        pct(0.99),
        sorted[n - 1],
        t.frames.iter().sum::<f64>()
    );
    let first: Vec<String> = t.frames.iter().take(20).map(|f| format!("{f:.0}")).collect();
    out += &format!("-- first 20 frame times: {}\n", first.join(" "));
    out
}
