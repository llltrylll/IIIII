//! Starting, queueing and collecting mesh-map bakes.
use super::*;

impl PainterApp {
    /// Builds the GPU bake pipeline only when a bake is requested.
    pub(super) fn ensure_baker(&mut self) {
        if self.baker.is_none() {
            let _s = crate::startup_trace::span("baker: lazy GpuBaker::new");
            self.baker = Some(GpuBaker::new_for_adapter(&self.rs.device, &self.rs.adapter));
        }
    }

    /// Settings for the next bake. ID-only jobs retain only the categorical ID map.
    pub(super) fn job_bake_settings(&self) -> bake::BakeSettings {
        let mut settings = self.bake_settings.clone();
        if self.bake_id_only {
            settings.maps = 0;
            settings.custom.clear();
            // ID masks and the eyedropper always read the loaded low-poly mesh.
            settings.high_poly = None;
            settings.cage = None;
            settings.cage_paint = None;
        }
        settings
    }

    /// Starts a full bake of the active Texture Set.
    pub(super) fn start_bake(&mut self, ctx: &egui::Context) {
        if self.bake_job.is_some() {
            self.status = "A bake is already running; the new request was not started.".into();
            return;
        }
        self.bake_id_only = false;
        self.sets.bake_queue.clear();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.start_bake_set(self.sets.active, ctx);
    }

    /// Starts an ID-only batch. If another bake is active, defer this request
    /// until its entire queue finishes; never drop an in-flight GPU run.
    pub(super) fn start_bake_id_only(&mut self, ctx: &egui::Context) {
        self.bake_settings.id_source = IdSource::VertexColors;
        if self.bake_job.is_some() {
            self.deferred_id_rebake = true;
            self.status = "Vertex ID rebake queued; the current bake will finish first.".into();
            return;
        }
        self.bake_id_only = true;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.start_bake_set(active, ctx);
    }

    /// Bakes every Texture Set, the edited one first.
    pub(super) fn start_bake_all(&mut self, ctx: &egui::Context) {
        if self.bake_job.is_some() {
            self.status = "A bake is already running; the new request was not started.".into();
            return;
        }
        self.bake_id_only = false;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.start_bake_set(active, ctx);
    }

    fn start_deferred_id_rebake(&mut self, ctx: &egui::Context) {
        if !self.deferred_id_rebake || self.bake_job.is_some() {
            return;
        }
        self.deferred_id_rebake = false;
        self.bake_settings.id_source = IdSource::VertexColors;
        self.bake_id_only = true;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.start_bake_set(active, ctx);
    }

    /// Cancels the remaining sets after a failed batch member, preserving the
    /// per-set source metadata of any sets that did finish successfully.
    fn abort_bake_batch(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        self.sets.bake_queue.clear();
        self.bake_batch_settings = None;
        self.bake_id_only = false;
        self.start_deferred_id_rebake(ctx);
    }

    pub(super) fn start_bake_set(&mut self, set: usize, ctx: &egui::Context) {
        self.ensure_baker();
        if set == self.sets.active {
            self.preview_full_bake_started();
        }
        let Some(res) = self.set_engine_mut(set).map(|e| e.res.min(bake::MAX_BAKE_RES)) else {
            self.abort_bake_batch(ctx);
            return;
        };
        self.sets.baking = set;

        // This snapshot is retained by the BakeJob and reused for every later
        // Texture Set in the batch. Cache key, CPU preparation and GPU uniforms
        // therefore always describe the same settings.
        let settings = self.bake_batch_settings.clone().unwrap_or_else(|| self.job_bake_settings());
        let worker_settings = settings.clone();
        let id_only = self.bake_id_only;
        let mesh = Arc::clone(&self.mesh);
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let worker_cancel = Arc::clone(&cancel);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let prepared = match bake::validate_settings(&worker_settings) {
                Err(error) => Err(error),
                Ok(()) if worker_cancel.load(std::sync::atomic::Ordering::Relaxed) => {
                    Err("Bake preparation cancelled".into())
                }
                Ok(()) => {
                    let key = bake_cache::key(&mesh, &worker_settings, res, set);
                    match bake_cache::load(&key) {
                        Some(entry) if !worker_cancel.load(std::sync::atomic::Ordering::Relaxed) => Ok(Prepared::Cached(entry)),
                        Some(_) => Err("Bake preparation cancelled".into()),
                        None => bake::prepare_cancellable(&mesh, &worker_settings, set, &worker_cancel)
                            .map(|scene| Prepared::Scene(scene, key)),
                    }
                }
            };
            let _ = tx.send(prepared);
            ctx.request_repaint();
        });
        self.bake_job = Some(BakeJob::Preparing { rx, cancel, settings, id_only, set });
        self.status = if self.mesh.sets.len() > 1 {
            format!("Preparing bake of {}…", self.mesh.sets[set].name)
        } else {
            "Preparing bake…".into()
        };
    }

    /// Installs one completed result and updates the persisted source metadata
    /// for exactly the Texture Set that owns those maps.
    fn install_baked_maps(
        &mut self,
        set: usize,
        id_only: bool,
        id_source: IdSource,
        maps: bake::BakedMaps,
        report: Arc<bake::BakeReport>,
    ) -> bool {
        {
            let Some(engine) = self.set_engine_mut(set) else { return false };
            if id_only {
                engine.replace_id_map(maps, report);
            } else {
                engine.set_maps(maps, report);
            }
        }
        self.sets.baked_id_sources.resize(self.mesh.sets.len(), self.baked_id_source);
        self.sets.baked_id_sources[set] = id_source;
        if set == self.sets.active {
            self.baked_id_source = id_source;
        }
        // Mesh maps and their source metadata are serialized project state.
        self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        true
    }

    /// A bake finished successfully: start the next set or the deferred ID batch.
    pub(super) fn bake_next(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        if let Some(set) = self.sets.bake_queue.pop_front() {
            self.start_bake_set(set, ctx);
        } else {
            self.bake_batch_settings = None;
            self.bake_id_only = false;
            self.start_deferred_id_rebake(ctx);
        }
    }

    pub(super) fn poll_bake(&mut self, ctx: &egui::Context) {
        // Take the job out while advancing it. This permits failure handling to
        // clear a whole batch without overlapping borrows of PainterApp.
        let Some(job) = self.bake_job.take() else { return };
        match job {
            BakeJob::Preparing { rx, cancel, settings, id_only, set } => match rx.try_recv() {
                Ok(Ok(Prepared::Cached(entry))) => {
                    let uploaded = {
                        let rs = &self.rs;
                        out_of_memory_guard(&rs.device, || entry.upload(&rs.device, &rs.queue))
                    };
                    match uploaded {
                        Ok((maps, report)) => {
                            let message = format!(
                                "Loaded {} cached mesh maps ({}², from {}).",
                                report.baked.count_ones(), report.resolution, report.source
                            );
                            if self.install_baked_maps(set, id_only, settings.id_source, maps, Arc::new(report)) {
                                self.status = message;
                                self.bake_next(ctx);
                            } else {
                                self.fail_mcp_bake("The Texture Set for this cached bake no longer exists.");
                                self.status = "Cached bake could not be assigned to its Texture Set.".into();
                                self.abort_bake_batch(ctx);
                            }
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                            self.abort_bake_batch(ctx);
                        }
                    }
                }
                Ok(Ok(Prepared::Scene(mut scene, key))) => {
                    let res = self.set_engines().get(set).map_or(self.engine.res, |e| e.res);
                    let started = {
                        let (baker, rs, mesh) = (
                            self.baker.as_ref().expect("a bake job initializes the GPU baker"),
                            &self.rs,
                            &self.gpu_mesh,
                        );
                        out_of_memory_guard(&rs.device, || baker.start_owned(&rs.device, &rs.queue, mesh, &mut scene, &settings, res))
                    };
                    match started {
                        Ok(run) => {
                            self.bake_job = Some(BakeJob::Running { run, key, settings, id_only, set });
                            ctx.request_repaint();
                        }
                        Err(e) => {
                            self.fail_mcp_bake(&e);
                            self.status = e;
                            self.abort_bake_batch(ctx);
                        }
                    }
                }
                Ok(Err(e)) => {
                    self.fail_mcp_bake(&e);
                    self.status = e;
                    self.abort_bake_batch(ctx);
                }
                Err(mpsc::TryRecvError::Empty) => {
                    self.bake_job = Some(BakeJob::Preparing { rx, cancel, settings, id_only, set });
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.fail_mcp_bake("The bake preparation stopped unexpectedly.");
                    self.status = "Bake preparation stopped unexpectedly.".into();
                    self.abort_bake_batch(ctx);
                }
            },
            BakeJob::Running { mut run, key, settings, id_only, set } => {
                let pumped = {
                    let (baker, rs) = (
                        self.baker.as_ref().expect("a running bake owns an initialized GPU baker"),
                        &self.rs,
                    );
                    out_of_memory_guard(&rs.device, || baker.pump(&mut run, &rs.device, &rs.queue, ctx))
                };
                match pumped {
                    Err(e) => {
                        self.fail_mcp_bake(&e);
                        self.status = e;
                        self.abort_bake_batch(ctx);
                    }
                    Ok(None) => {
                        self.bake_job = Some(BakeJob::Running { run, key, settings, id_only, set });
                    }
                    Ok(Some(maps)) => {
                        let report = run.report.clone();
                        let matched = match report.matched_names {
                            0 => String::new(),
                            n => format!(", {n} parts matched by name"),
                        };
                        let set_res = self.set_engines().get(set).map_or(self.engine.res, |e| e.res);
                        let upscaled = if report.resolution < set_res { format!(", upscaled to {set_res}²") } else { String::new() };
                        let set_name = match self.mesh.sets.get(set) {
                            Some(s) if self.mesh.sets.len() > 1 => format!(" for {}", s.name),
                            _ => String::new(),
                        };
                        let message = format!(
                            "Baked {} mesh maps{set_name} from {} ({} tris, {} parts{matched}{}) at {}²{upscaled}.",
                            report.baked.count_ones(),
                            report.source,
                            report.triangles,
                            report.parts,
                            if report.used_cage { ", cage" } else { "" },
                            report.resolution
                        );
                        // Cache the exact maps under the exact settings snapshot
                        // before moving their textures into the destination Engine.
                        bake_cache::store(key, &maps, report.clone(), &self.rs.device, &self.rs.queue);
                        if self.install_baked_maps(set, id_only, settings.id_source, maps, report) {
                            self.status = if id_only {
                                "Baked the ID map from the vertex colors.".into()
                            } else {
                                message
                            };
                            self.bake_next(ctx);
                        } else {
                            self.fail_mcp_bake("The Texture Set for this bake no longer exists.");
                            self.status = "Bake result could not be assigned to its Texture Set.".into();
                            self.abort_bake_batch(ctx);
                        }
                    }
                }
            }
        }
    }

    pub(super) fn bake_progress(&self) -> Option<(f32, &'static str)> {
        match &self.bake_job {
            None => None,
            Some(BakeJob::Preparing { .. }) => Some((0.0, "Preparing")),
            Some(BakeJob::Running { run, .. }) => Some((run.progress(), "Baking")),
        }
    }
}
