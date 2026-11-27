//! Starting, queueing and collecting mesh-map bakes.
/// Plans the full-map and low-poly ID-only stages needed to restore saved sources.
/// Vertex Color IDs cannot be projected through a high-poly mesh, so those sets
/// use a valid interim source for the full pass and get a separate ID-only pass.
fn project_bake_source_plan(
    saved: &[IdSource],
    set_count: usize,
    active: usize,
    high_poly: bool,
    has_vertex_colors: bool,
    fallback: IdSource,
) -> (Vec<IdSource>, std::collections::VecDeque<usize>) {
    let mut sources = saved.iter().copied().take(set_count).collect::<Vec<_>>();
    sources.resize(set_count, fallback);
    let mut followups = std::collections::VecDeque::new();
    if high_poly {
        let ordered_sets = std::iter::once(active)
            .chain((0..set_count).filter(|&k| k != active));
        for set in ordered_sets {
            if sources[set] == IdSource::VertexColors {
                sources[set] = fallback;
                if has_vertex_colors {
                    followups.push_back(set);
                }
            }
        }
    } else if !has_vertex_colors {
        // Legacy projects may remember Vertex Colors without having serialized
        // that attribute. Keep the rest of the maps bakeable until the user
        // reimports the original FBX/OBJ through the Vertex ID panel.
        sources.iter_mut().for_each(|source| {
            if *source == IdSource::VertexColors {
                *source = fallback;
            }
        });
    }
    (sources, followups)
}

use super::*;

/// The regular bake source must remain valid for the selected projection.
/// Vertex Colors is applied only as the ID-only job's per-set override when a
/// high-poly source is selected; after that job, the next full bake still uses
/// Mesh Parts or Materials.
fn next_full_bake_id_source(current: IdSource, high_poly: bool, has_vertex_colors: bool) -> IdSource {
    if high_poly || !has_vertex_colors {
        if current == IdSource::VertexColors { IdSource::MeshParts } else { current }
    } else {
        IdSource::VertexColors
    }
}

/// Applies an explicit source selection to all Texture Sets, filling any
/// missing legacy entries from the supplied active-set fallback.
fn set_id_source_preferences(
    preferences: &mut Vec<IdSource>,
    set_count: usize,
    fallback: IdSource,
    source: IdSource,
) -> bool {
    preferences.resize(set_count, fallback);
    let mut changed = false;
    for preference in preferences {
        if *preference != source {
            *preference = source;
            changed = true;
        }
    }
    changed
}

/// Adds a deferred per-set ID pass once. Requests for the set already running
/// or already queued in the current ID-only batch do not need another pass.
fn queue_deferred_id_rebake(
    queue: &mut std::collections::VecDeque<usize>,
    set: usize,
    already_scheduled: bool,
) -> bool {
    if already_scheduled || queue.contains(&set) {
        return false;
    }
    queue.push_back(set);
    true
}

impl PainterApp {
    /// Keep the full-bake source valid for the selected projection.
    /// Vertex Colors is applied later as a per-set ID-only override.
    fn ensure_valid_full_bake_id_source(&mut self) {
        if (self.bake_settings.high_poly.is_some() || !vertex_id::has_colors(&self.mesh))
            && self.bake_settings.id_source == IdSource::VertexColors
        {
            self.bake_settings.id_source = IdSource::MeshParts;
            self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        }
    }

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
        self.ensure_valid_full_bake_id_source();
        self.bake_id_only = false;
        self.sets.bake_queue.clear();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.bake_batch_id_sources = None;
        self.bake_batch_followup_id_sets.clear();
        self.start_bake_set(self.sets.active, ctx);
    }

    /// An explicit ID source selection in the Bake window applies to every
    /// Texture Set for subsequent full bakes. The source actually baked stays
    /// per-set and is updated only when a bake completes.
    pub(super) fn prefer_id_source_for_all_sets(&mut self, source: IdSource) -> bool {
        set_id_source_preferences(
            &mut self.sets.id_source_preferences,
            self.mesh.sets.len(),
            self.baked_id_source,
            source,
        )
    }

    /// Records the requested Vertex Color source for a set list. A currently
    /// running full bake can finish after a request and write its older source
    /// back, so deferred targets are reasserted immediately before their batch.
    fn prefer_vertex_colors_for_id_sets(&mut self, sets: &[usize]) -> bool {
        self.sets.id_source_preferences.resize(self.mesh.sets.len(), self.baked_id_source);
        let mut changed = false;
        for &set in sets {
            if let Some(source) = self.sets.id_source_preferences.get_mut(set) {
                if *source != IdSource::VertexColors {
                    *source = IdSource::VertexColors;
                    changed = true;
                }
            }
        }
        changed
    }

    /// Rebuilds only the active Texture Set's ID map from the loaded low-poly
    /// vertex colors. Other sets keep their own saved ID source and map.
    /// If another bake is active, remember this set's index and run it afterward.
    pub(super) fn start_bake_id_only(&mut self, ctx: &egui::Context) {
        let set = self.sets.active;
        if !vertex_id::has_colors(&self.mesh) {
            self.status = "Vertex ID rebake requires per-corner vertex colors. Import them from the original FBX/OBJ first.".into();
            return;
        }
        let mut project_settings_changed = false;
        let next_source = next_full_bake_id_source(
            self.bake_settings.id_source,
            self.bake_settings.high_poly.is_some(),
            vertex_id::has_colors(&self.mesh),
        );
        if self.bake_settings.id_source != next_source {
            self.bake_settings.id_source = next_source;
            project_settings_changed = true;
        }
        project_settings_changed |= self.prefer_vertex_colors_for_id_sets(&[set]);
        if project_settings_changed {
            self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        }

        if let Some(job) = &self.bake_job {
            let already_running = match job {
                BakeJob::Preparing { set: running_set, id_only, settings, .. }
                | BakeJob::Running { set: running_set, id_only, settings, .. } => {
                    *running_set == set && *id_only && settings.id_source == IdSource::VertexColors
                }
            };
            let already_queued_in_id_batch = self.bake_id_only && self.sets.bake_queue.contains(&set);
            let already_planned_as_high_poly_followup = self.bake_batch_followup_id_sets.contains(&set);
            if already_queued_in_id_batch {
                if let Some(source) = self.bake_batch_id_sources.as_mut().and_then(|sources| sources.get_mut(set)) {
                    *source = IdSource::VertexColors;
                }
            }
            let already_scheduled = already_running
                || already_queued_in_id_batch
                || already_planned_as_high_poly_followup;
            let queued = queue_deferred_id_rebake(
                &mut self.deferred_id_rebake_sets,
                set,
                already_scheduled,
            );
            self.status = if already_scheduled || !queued {
                "This Texture Set already has a Vertex Color ID bake running or queued.".into()
            } else {
                "Vertex ID rebake queued for this Texture Set; the current bake will finish first.".into()
            };
            return;
        }

        self.bake_id_only = true;
        let mut queued = std::mem::take(&mut self.deferred_id_rebake_sets);
        queued.retain(|&queued_set| queued_set != set);
        self.sets.bake_queue = queued;
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.bake_batch_id_sources = Some(self.sets.id_source_preferences.clone());
        self.bake_batch_followup_id_sets.clear();
        self.start_bake_set(set, ctx);
    }

    /// Bakes every Texture Set, the edited one first.
    pub(super) fn start_bake_all(&mut self, ctx: &egui::Context) {
        if self.bake_job.is_some() {
            self.status = "A bake is already running; the new request was not started.".into();
            return;
        }
        self.ensure_valid_full_bake_id_source();
        self.bake_id_only = false;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.bake_batch_id_sources = None;
        self.bake_batch_followup_id_sets.clear();
        self.start_bake_set(active, ctx);
    }

    /// Restores each Texture Set's saved ID source when opening a project.
    /// The baked maps are rebuilt from the cache or rebaked, since .mpp stores
    /// mesh geometry and layer data rather than duplicating the map textures.
    pub(super) fn start_bake_all_preserving_sources(&mut self, ctx: &egui::Context) {
        if self.bake_job.is_some() {
            self.status = "A bake is already running; the saved-source bake was not started.".into();
            return;
        }
        self.ensure_valid_full_bake_id_source();
        self.bake_id_only = false;
        let active = self.sets.active;
        self.sets.bake_queue = (0..self.mesh.sets.len()).filter(|&k| k != active).collect();
        let high_poly = self.bake_settings.high_poly.is_some();
        let fallback = if self.bake_settings.id_source == IdSource::VertexColors {
            IdSource::MeshParts
        } else {
            self.bake_settings.id_source
        };
        let has_vertex_colors = vertex_id::has_colors(&self.mesh);
        self.bake_batch_settings = Some(self.job_bake_settings());
        let (sources, followups) = project_bake_source_plan(
            &self.sets.id_source_preferences,
            self.mesh.sets.len(),
            active,
            high_poly,
            has_vertex_colors,
            fallback,
        );
        self.bake_batch_id_sources = Some(sources);
        self.bake_batch_followup_id_sets = followups;
        self.start_bake_set(active, ctx);
    }

    fn start_deferred_id_rebake(&mut self, ctx: &egui::Context) {
        if self.bake_job.is_some() {
            return;
        }
        let Some(set) = self.deferred_id_rebake_sets.pop_front() else { return };
        let mut requested_sets = Vec::with_capacity(self.deferred_id_rebake_sets.len() + 1);
        requested_sets.push(set);
        requested_sets.extend(self.deferred_id_rebake_sets.iter().copied());
        let mut project_settings_changed = self.prefer_vertex_colors_for_id_sets(&requested_sets);
        let next_source = next_full_bake_id_source(
            self.bake_settings.id_source,
            self.bake_settings.high_poly.is_some(),
            vertex_id::has_colors(&self.mesh),
        );
        if self.bake_settings.id_source != next_source {
            self.bake_settings.id_source = next_source;
            project_settings_changed = true;
        }
        if project_settings_changed {
            self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        }
        self.bake_id_only = true;
        self.sets.bake_queue = std::mem::take(&mut self.deferred_id_rebake_sets);
        self.bake_batch_settings = Some(self.job_bake_settings());
        self.bake_batch_id_sources = Some(self.sets.id_source_preferences.clone());
        self.bake_batch_followup_id_sets.clear();
        self.start_bake_set(set, ctx);
    }

    /// Cancels the remaining sets after a failed batch member, preserving the
    /// per-set source metadata of any sets that did finish successfully.
    fn abort_bake_batch(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        self.sets.bake_queue.clear();
        self.bake_batch_settings = None;
        self.bake_batch_id_sources = None;
        self.bake_batch_followup_id_sets.clear();
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
        let mut settings = self.bake_batch_settings.clone().unwrap_or_else(|| self.job_bake_settings());
        if let Some(sources) = &self.bake_batch_id_sources {
            if let Some(source) = sources.get(set) {
                settings.id_source = *source;
            }
        }
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
        let mut metadata_changed = self.sets.baked_id_sources[set] != id_source;
        self.sets.baked_id_sources[set] = id_source;
        // A project-restore batch can temporarily use Mesh Parts/Materials
        // for the full high-poly pass, then replace only the ID map from the
        // low-poly Vertex Colors. Do not overwrite the saved preferred source
        // until an ordinary user-requested bake commits a new choice.
        if self.bake_batch_id_sources.is_none() {
            self.sets.id_source_preferences.resize(self.mesh.sets.len(), self.baked_id_source);
            metadata_changed |= self.sets.id_source_preferences[set] != id_source;
            self.sets.id_source_preferences[set] = id_source;
        }
        if set == self.sets.active {
            self.baked_id_source = id_source;
        }
        // Only the per-set source metadata is serialized; the transient GPU
        // textures are regenerated from the project bake cache when reopened.
        if metadata_changed {
            self.project_settings_serial = self.project_settings_serial.wrapping_add(1);
        }
        true
    }

    /// A bake finished successfully: start the next set or the deferred ID batch.
    pub(super) fn bake_next(&mut self, ctx: &egui::Context) {
        self.bake_job = None;
        if let Some(set) = self.sets.bake_queue.pop_front() {
            self.start_bake_set(set, ctx);
        } else if let Some(set) = self.bake_batch_followup_id_sets.pop_front() {
            // The full high-poly pass is done. Rebuild only the ID map for the
            // saved Vertex Color sets, keeping AO/normal/height and all other
            // high-poly maps from the just-completed pass.
            self.bake_id_only = true;
            self.bake_batch_settings = Some(self.job_bake_settings());
            self.bake_batch_id_sources = Some(vec![IdSource::VertexColors; self.mesh.sets.len()]);
            self.sets.bake_queue = std::mem::take(&mut self.bake_batch_followup_id_sets);
            self.start_bake_set(set, ctx);
        } else {
            self.bake_batch_settings = None;
            self.bake_batch_id_sources = None;
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

#[cfg(test)]
mod project_bake_source_plan_tests {
    use super::{next_full_bake_id_source, project_bake_source_plan, queue_deferred_id_rebake, set_id_source_preferences};
    use crate::bake::IdSource;
    use std::collections::VecDeque;

    #[test]
    fn explicit_id_source_selection_updates_all_texture_set_preferences_once() {
        let mut preferences = vec![
            IdSource::Materials,
            IdSource::VertexColors,
            IdSource::MeshParts,
        ];
        assert!(set_id_source_preferences(
            &mut preferences,
            4,
            IdSource::Materials,
            IdSource::VertexColors,
        ));
        assert_eq!(preferences, vec![IdSource::VertexColors; 4]);
        assert!(!set_id_source_preferences(
            &mut preferences,
            4,
            IdSource::Materials,
            IdSource::VertexColors,
        ));
    }

    #[test]
    fn id_only_vertex_color_pass_keeps_the_next_full_bake_source_valid() {
        assert_eq!(
            next_full_bake_id_source(IdSource::MeshParts, false, true),
            IdSource::VertexColors,
        );
        assert_eq!(
            next_full_bake_id_source(IdSource::Materials, false, true),
            IdSource::VertexColors,
        );
        assert_eq!(
            next_full_bake_id_source(IdSource::MeshParts, true, true),
            IdSource::MeshParts,
        );
        assert_eq!(
            next_full_bake_id_source(IdSource::Materials, true, true),
            IdSource::Materials,
        );
        assert_eq!(
            next_full_bake_id_source(IdSource::VertexColors, true, true),
            IdSource::MeshParts,
        );
        assert_eq!(
            next_full_bake_id_source(IdSource::VertexColors, false, false),
            IdSource::MeshParts,
        );
    }

    #[test]
    fn deferred_vertex_id_queue_deduplicates_per_set_without_merging_targets() {
        let mut queue = VecDeque::from([2, 0]);
        assert!(!queue_deferred_id_rebake(&mut queue, 2, false));
        assert!(!queue_deferred_id_rebake(&mut queue, 1, true));
        assert!(queue_deferred_id_rebake(&mut queue, 3, false));
        assert_eq!(queue.into_iter().collect::<Vec<_>>(), vec![2, 0, 3]);
    }


    #[test]
    fn low_poly_project_restores_each_saved_id_source_without_extra_passes() {
        let saved = [IdSource::VertexColors, IdSource::Materials, IdSource::MeshParts];
        let (sources, followups) = project_bake_source_plan(
            &saved, 3, 1, false, true, IdSource::MeshParts,
        );
        assert_eq!(sources, saved.to_vec());
        assert!(followups.is_empty());
    }

    #[test]
    fn high_poly_project_defers_only_vertex_color_id_maps_in_active_first_order() {
        let saved = [
            IdSource::VertexColors,
            IdSource::Materials,
            IdSource::VertexColors,
            IdSource::MeshParts,
        ];
        let (sources, followups) = project_bake_source_plan(
            &saved, 4, 2, true, true, IdSource::Materials,
        );
        assert_eq!(
            sources,
            vec![
                IdSource::Materials,
                IdSource::Materials,
                IdSource::Materials,
                IdSource::MeshParts,
            ]
        );
        assert_eq!(followups.into_iter().collect::<Vec<_>>(), vec![2, 0]);
    }

    #[test]
    fn missing_legacy_vertex_colors_fall_back_without_aborting_other_texture_sets() {
        let (sources, followups) = project_bake_source_plan(
            &[IdSource::VertexColors, IdSource::Materials],
            2,
            0,
            false,
            false,
            IdSource::MeshParts,
        );
        assert_eq!(sources, vec![IdSource::MeshParts, IdSource::Materials]);
        assert!(followups.is_empty());
    }

    #[test]
    fn high_poly_project_without_vertex_colors_keeps_maps_bakeable_without_followup() {
        let saved = [
            IdSource::VertexColors,
            IdSource::Materials,
            IdSource::VertexColors,
        ];
        let (sources, followups) = project_bake_source_plan(
            &saved,
            3,
            1,
            true,
            false,
            IdSource::MeshParts,
        );
        assert_eq!(
            sources,
            vec![
                IdSource::MeshParts,
                IdSource::Materials,
                IdSource::MeshParts,
            ]
        );
        assert!(followups.is_empty());
    }

    #[test]
    fn legacy_short_source_lists_are_padded_to_the_texture_set_count() {
        let (sources, followups) = project_bake_source_plan(
            &[IdSource::VertexColors],
            3,
            0,
            true,
            true,
            IdSource::MeshParts,
        );
        assert_eq!(sources, vec![IdSource::MeshParts; 3]);
        assert_eq!(followups.into_iter().collect::<Vec<_>>(), vec![0]);
    }
}
