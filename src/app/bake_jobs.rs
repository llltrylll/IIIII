//! Starting, queueing and collecting mesh-map bakes.
use super::*;

impl PainterApp {
    /// Builds the GPU bake pipeline only when a bake is requested.
    pub(super) fn ensure_baker(&mut self) {
        if self.baker.is_none() {
            let _s = crate::startup_trace::span("baker: lazy GpuBaker::new");
            self.baker = Some(GpuBaker::new(&self.rs.device));
        }
    }

    /// Bakes the texture set being edited.
    pub(super) fn start_bake(&mut self, ctx: &egui::Context) {
        self.sets.bake_queue.clear();
        self.start_bake_set(self.sets.active, ctx);
    }

    /// Bakes every texture set, the edited one first.
    /// The settings a bake runs with: a "vertex colors only" bake skips the slow
    /// ray-traced maps (AO, cavity, thickness, sky...) and keeps just the ID map.
    pub(super) fn job_bake_settings(&self) -> bake::BakeSettings {
        let mut settings = self.bake_settings.clone();
        if self.bake_id_only {
            settings.maps = 0;
            settings.custom.clear();
        }
        settings
    }

    /// Bakes the ID map alone from the vertex colors and leaves every other baked map as it is.
    pub(super) fn start_bake_id_only(&mut self, ctx: &egui::Context) {
        self.bake_id_only = true;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.start_bake_set(active, ctx);
    }

    pub(super) fn start_bake_all(&mut self, ctx: &egui::Context) {
        self.bake_id_only = false;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.start_bake_set(active, ctx);
    }

    pub(super) fn start_bake_set(&mut self, set: usize, ctx: &egui::Context) {
        self.ensure_baker();
        if set == self.sets.active {
            self.preview_full_bake_started();
        }
        let Some(res) = self.set_engine_mut(set).map(|e| e.res.min(bake::MAX_BAKE_RES)) else { return };
        self.sets.baking = set;
        let mesh = self.mesh.clone();
        let settings = self.job_bake_settings();
        self.baked_id_source = settings.id_source;
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let key = bake_cache::key(&mesh, &settings, res, set);
            let prepared = match bake_cache::load(&key) {
                Some(entry) => Ok(Prepared::Cached(entry)),
                None => bake::prepare(&mesh, &settings, set).map(|scene| Prepared::Scene(scene, key)),
            };
            let _ = tx.send(prepared);
            ctx.request_repaint();
        });
        // A newer job replaces any running one; its results are dropped.
        self.bake_job = Some(BakeJob::Preparing(rx));
        self.status = if self.mesh.sets.len() > 1 {
            format!("Preparing bake of {}…", self.mesh.sets[set].name)
        } else {
            "Preparing bake…".into()
        };
    }

    /// A bake finished (or failed): the next queued set starts.
    pub(super) fn bake_next(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        if let Some(set) = self.sets.bake_queue.pop_front() {
            self.start_bake_set(set, ctx);
        }
    }

    pub(super) fn poll_bake(&mut self, ctx: &egui::Context) {
        match &mut self.bake_job {
            None => {}
            Some(BakeJob::Preparing(rx)) => match rx.try_recv() {
                Ok(Ok(Prepared::Cached(entry))) => {
                    let rs = &self.rs;
                    match out_of_memory_guard(&rs.device, || entry.upload(&rs.device, &rs.queue)) {
                        Ok((maps, report)) => {
                            self.status = format!("Loaded {} cached mesh maps ({}², from {}).", report.baked.count_ones(), report.resolution, report.source);
                            let id_only = self.bake_id_only;
                            if let Some(e) = self.set_engine_mut(self.sets.baking) {
                                if id_only { e.replace_id_map(maps, Arc::new(report)) } else { e.set_maps(maps, Arc::new(report)) }
                            }
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                        }
                    }
                    self.bake_next(ctx);
                }
                Ok(Ok(Prepared::Scene(scene, key))) => {
                    let res = self.set_engines().get(self.sets.baking).map_or(self.engine.res, |e| e.res);
                    let settings = self.job_bake_settings();
                    let (baker, rs, mesh) = (
                        self.baker.as_ref().expect("a bake job initializes the GPU baker"),
                        &self.rs,
                        &self.gpu_mesh,
                    );
                    match out_of_memory_guard(&rs.device, || baker.start(&rs.device, &rs.queue, mesh, &scene, &settings, res)) {
                        Ok(run) => {
                            self.bake_job = Some(BakeJob::Running(run, key));
                            ctx.request_repaint();
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                            self.bake_job = None;
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.fail_mcp_bake(&e);
                    self.status = e;
                    self.bake_job = None;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.fail_mcp_bake("The bake preparation stopped unexpectedly.");
                    self.bake_job = None;
                }
            },
            Some(BakeJob::Running(run, key)) => {
                let (baker, rs) = (
                    self.baker.as_ref().expect("a running bake owns an initialized GPU baker"),
                    &self.rs,
                );
                let pumped = match out_of_memory_guard(&rs.device, || baker.pump(run, &rs.device, &rs.queue, ctx)) {
                    Ok(maps) => maps,
                    Err(e) => {
                        self.fail_mcp_bake(&e);
                        self.status = e;
                        self.bake_job = None;
                        return;
                    }
                };
                if let Some(maps) = pumped {
                    let report = run.report.clone();
                    let matched = match report.matched_names {
                        0 => String::new(),
                        n => format!(", {n} parts matched by name"),
                    };
                    let set_res = texture_sets::engines(&self.engine, &self.sets, self.mesh.sets.len()).get(self.sets.baking).map_or(self.engine.res, |e| e.res);
                    let upscaled = if report.resolution < set_res { format!(", upscaled to {set_res}²") } else { String::new() };
                    let set_name = match self.mesh.sets.get(self.sets.baking) {
                        Some(s) if self.mesh.sets.len() > 1 => format!(" for {}", s.name),
                        _ => String::new(),
                    };
                    self.status = format!(
                        "Baked {} mesh maps{set_name} from {} ({} tris, {} parts{matched}{}) at {}²{upscaled}.",
                        report.baked.count_ones(),
                        report.source,
                        report.triangles,
                        report.parts,
                        if report.used_cage { ", cage" } else { "" },
                        report.resolution
                    );
                    bake_cache::store(key.clone(), &maps, report.clone(), &self.rs.device, &self.rs.queue);
                    let id_only = self.bake_id_only;
                    if let Some(e) = self.set_engine_mut(self.sets.baking) {
                        if id_only { e.replace_id_map(maps, report) } else { e.set_maps(maps, report) }
                    }
                    if id_only {
                        self.status = "Baked the ID map from the vertex colors.".into();
                    }
                    self.bake_next(ctx);
                }
            }
        }
    }

    pub(super) fn bake_progress(&self) -> Option<(f32, &'static str)> {
        match &self.bake_job {
            None => None,
            Some(BakeJob::Preparing(_)) => Some((0.0, "Preparing")),
            Some(BakeJob::Running(run, _)) => Some((run.progress(), "Baking")),
        }
    }
}
