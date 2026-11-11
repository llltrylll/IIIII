//! The colour pickers: the swatch button with its popup, the HUD behind the toolbar's
//! Color button (two colours, HSV/RGB, temperature, harmonies, tray) and the HSV helpers.
use super::*;

pub(super) fn color_edit(ui: &mut egui::Ui, c: &mut [f32; 3]) -> bool { color_button(ui, c, None) }
pub(super) fn color_button(ui: &mut egui::Ui, c: &mut [f32; 3], tip: Option<&str>) -> bool {
    let rgb = c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8);
    let response = ui.add(egui::Button::new(egui::RichText::new("   ").background_color(Color32::from_rgb(rgb[0], rgb[1], rgb[2])))
        .fill(Color32::from_rgb(rgb[0], rgb[1], rgb[2])).min_size(vec2(28.0, 18.0)));
    let response = if let Some(tip) = tip { response.on_hover_text(tip) } else { response };
    let mut changed = false;
    egui::Popup::from_toggle_button_response(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(|ui| {
            ui.set_width(color_picker::WIDTH + 28.0);
            egui::ScrollArea::vertical().max_height(600.0).show(ui, |ui| {
                // The hue is remembered between frames so it survives passing through gray or black.
                let memory = response.id.with("picker_hsv");
                let mut hsv = match ui.data(|d| d.get_temp::<[f32; 3]>(memory)) {
                    Some(previous) => hsv_keeping(previous, *c),
                    None => rgb_to_hsv(*c),
                };
                let picked = color_picker::picker(ui, &mut hsv);
                if picked { *c = hsv_to_rgb(hsv); changed = true; }
                let mut hsv_values = [hsv[0] * 360.0, hsv[1] * 100.0, hsv[2] * 100.0];
                for (label, index, range) in [("H", 0usize, 0.0..=360.0), ("S", 1usize, 0.0..=100.0), ("V", 2usize, 0.0..=100.0)] {
                    changed |= labeled(ui, label, |ui| ui.add(egui::Slider::new(&mut hsv_values[index], range).max_decimals(2)).changed());
                }
                if changed && !picked {
                    hsv = [hsv_values[0] / 360.0, hsv_values[1] / 100.0, hsv_values[2] / 100.0];
                    *c = hsv_to_rgb(hsv);
                }
                ui.separator();
                let mut channels = c.map(|v| v * 255.0);
                for (label, index) in [("R", 0usize), ("G", 1usize), ("B", 2usize)] {
                    changed |= labeled(ui, label, |ui| ui.add(egui::Slider::new(&mut channels[index], 0.0..=255.0).max_decimals(0)).changed());
                }
                if changed { *c = channels.map(|v| (v / 255.0).clamp(0.0, 1.0)); }
                let mut hex = color_picker::hex_string(*c);
                ui.horizontal(|ui| {
                    ui.label("HEX");
                    if ui.add(egui::TextEdit::singleline(&mut hex).desired_width(90.0)).changed()
                        && let Some(parsed) = parse_color_hex(&hex) { *c = parsed; changed = true; }
                });
                ui.data_mut(|d| d.insert_temp(memory, if changed { hsv_keeping(hsv, *c) } else { hsv }));
            });
        });
    changed
}
pub(super) fn swap_button(ui: &mut egui::Ui) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(vec2(20.0, 20.0), egui::Sense::click());
    let stroke = egui::Stroke::new(1.5, if response.hovered() { Color32::WHITE } else { theme::TEXT });
    let center = rect.center();
    ui.painter().arrow(center + vec2(5.0, -3.5), vec2(-10.0, 0.0), stroke);
    ui.painter().arrow(center + vec2(-5.0, 3.5), vec2(10.0, 0.0), stroke);
    response
}
#[derive(Default)]
pub(super) struct ColorSliders {
    pub(super) hsv: [f32; 3],
    pub(super) out: [f32; 3],
    pub(super) temp: f32,
    pub(super) base: [f32; 3],
    pub(super) shifting: bool,
    pub(super) hsv_mode: bool,
    pub(super) hex_edit: String,
    pub(super) hex_dirty: bool,
    /// Color when the panel was opened, for the old | new swatch.
    pub(super) opened: [f32; 3],
}
pub(super) fn parse_color_hex(value: &str) -> Option<[f32; 3]> {
    let raw = value.trim().strip_prefix('#').unwrap_or(value.trim());
    if raw.len() != 6 { return None; }
    let packed = u32::from_str_radix(raw, 16).ok()?;
    Some([
        ((packed >> 16) & 0xff) as f32 / 255.0,
        ((packed >> 8) & 0xff) as f32 / 255.0,
        (packed & 0xff) as f32 / 255.0,
    ])
}
pub(super) fn rgb_to_hsv(c: [f32; 3]) -> [f32; 3] {
    let max = c[0].max(c[1]).max(c[2]);
    let delta = max - c[0].min(c[1]).min(c[2]);
    let h = if delta <= 0.0 { 0.0 }
    else if max == c[0] { ((c[1] - c[2]) / delta).rem_euclid(6.0) }
    else if max == c[1] { (c[2] - c[0]) / delta + 2.0 }
    else { (c[0] - c[1]) / delta + 4.0 };
    [h / 6.0, if max > 0.0 { delta / max } else { 0.0 }, max]
}
pub(super) fn hsv_to_rgb([h, s, v]: [f32; 3]) -> [f32; 3] {
    let channel = |n: f32| { let k = (n + h * 6.0) % 6.0; v - v * s * k.min(4.0 - k).clamp(0.0, 1.0) };
    [channel(5.0), channel(3.0), channel(1.0)]
}
pub(super) fn hsv_keeping(prev: [f32; 3], color: [f32; 3]) -> [f32; 3] {
    let next = rgb_to_hsv(color);
    [if next[1] > 0.0 && next[2] > 0.0 { next[0] } else { prev[0] },
     if next[2] > 0.0 { next[1] } else { prev[1] }, next[2]]
}
pub(super) fn shift_temperature(color: [f32; 3], amount: f32) -> [f32; 3] {
    let gain = [1.0 + 0.30 * amount, 1.0 + 0.05 * amount, 1.0 - 0.30 * amount];
    std::array::from_fn(|i| (color[i] * gain[i]).clamp(0.0, 1.0))
}


/// The Color HUD below its two swatches: picker, RGB/HSV values, HEX, temperature, mix strip
/// and harmonies. Takes the active colour and returns the one to apply; `state` remembers the
/// hue and the half-typed hex between frames.
pub(super) fn hud_body(ui:&mut egui::Ui,state:&mut ColorSliders,active:[f32;3],second:&mut [f32;3])->[f32;3] {
    let mut color=active;
        if state.out.iter().zip(color).any(|(a,b)|(a-b).abs()>0.002) { state.hsv=hsv_keeping(state.hsv,color); state.temp=0.0; state.shifting=false; }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut state.hsv_mode,false,"RGB");
            ui.selectable_value(&mut state.hsv_mode,true,"HSV");
            ui.separator();
        });
        let picked=color_picker::picker(ui,&mut state.hsv);
        let mut numeric_changed=false;
        if state.hsv_mode {
            let (mut h,mut s,mut v)=(state.hsv[0]*360.0,state.hsv[1]*100.0,state.hsv[2]*100.0);
            numeric_changed |= labeled(ui,"H",|ui|ui.add(egui::Slider::new(&mut h,0.0..=360.0).max_decimals(2)).changed());
            numeric_changed |= labeled(ui,"S",|ui|ui.add(egui::Slider::new(&mut s,0.0..=100.0).max_decimals(2)).changed());
            numeric_changed |= labeled(ui,"V",|ui|ui.add(egui::Slider::new(&mut v,0.0..=100.0).max_decimals(2)).changed());
            if numeric_changed { state.hsv=[h/360.0,s/100.0,v/100.0]; }
        } else {
            let mut rgb=color.map(|v|v*255.0);
            numeric_changed |= labeled(ui,"R",|ui|ui.add(egui::Slider::new(&mut rgb[0],0.0..=255.0).max_decimals(0)).changed());
            numeric_changed |= labeled(ui,"G",|ui|ui.add(egui::Slider::new(&mut rgb[1],0.0..=255.0).max_decimals(0)).changed());
            numeric_changed |= labeled(ui,"B",|ui|ui.add(egui::Slider::new(&mut rgb[2],0.0..=255.0).max_decimals(0)).changed());
            if numeric_changed { color=rgb.map(|v|v/255.0); state.hsv=hsv_keeping(state.hsv,color); }
        }
        if picked||numeric_changed { color=hsv_to_rgb(state.hsv); }
        if !state.hex_dirty {
            state.hex_edit = color_picker::hex_string(color);
        }
        ui.horizontal(|ui| {
            if color_picker::compare_swatch(ui,state.opened,color) { color=state.opened; state.hsv=hsv_keeping(state.hsv,color); }
            ui.label(egui::RichText::new("HEX").small().color(theme::TEXT_DIM));
            let response = ui.add(egui::TextEdit::singleline(&mut state.hex_edit).desired_width(94.0).font(egui::TextStyle::Monospace));
            if response.changed() {
                state.hex_dirty = true;
                if let Some(parsed) = parse_color_hex(&state.hex_edit) {
                    color = parsed;
                    state.hsv = hsv_keeping(state.hsv, color);
                }
            }
            if response.lost_focus() {
                state.hex_dirty = false;
                state.hex_edit = color_picker::hex_string(color);
            }
            if ui.button("Copy").on_hover_text("Copy the hex code").clicked() { ui.ctx().copy_text(color_picker::hex_string(color)); }
        });
        let temp=labeled(ui,"Temperature",|ui|ui.add(egui::Slider::new(&mut state.temp,-100.0..=100.0).max_decimals(0)));
        if temp.changed() {
            if !state.shifting { state.base=color; state.shifting=true; }
            color=shift_temperature(state.base,state.temp/100.0);
            state.hsv=hsv_keeping(state.hsv,color);
        }
        if temp.drag_stopped()||temp.lost_focus() { state.temp=0.0; state.shifting=false; }
        ui.add_space(2.0);
        let from_mix=color_picker::mix_strip(ui,color,*second);
        let from_harmony=color_picker::harmony_row(ui,color);
        if let Some((picked,as_second))=from_mix.or(from_harmony) {
            if as_second { *second=picked; } else { color=picked; state.hsv=hsv_keeping(state.hsv,color); }
        }
    color
}

impl PainterApp {
    pub(super) fn active_color(&self) -> [f32; 3] {
        if self.target == StrokeTarget::Content
            && let Some(layer) = self.engine.layers.get(self.selected)
                && let Content::Fill(fill) = &layer.content { return fill.mat.color; }
        self.brush.mat.color
    }
    pub(super) fn set_active_color(&mut self, color: [f32; 3]) {
        if self.target == StrokeTarget::Content
            && let Some(layer) = self.engine.layers.get_mut(self.selected)
                && let Content::Fill(fill) = &mut layer.content {
                    fill.mat.color = color;
                    self.defer_preview_update();
                    return;
                }
        self.brush.mat.color = color;
    }
    pub(super) fn swap_colors(&mut self) { let current=self.active_color(); self.set_active_color(self.color2); self.color2=current; }
    pub(super) fn color_slots(&mut self,ui:&mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x=3.0;
            let mut active=self.active_color();
            if color_button(ui,&mut active,Some("Active layer / brush color")){self.set_active_color(active);}
            if theme::icon_button(ui,icons::EYEDROPPER,false,20.0).on_hover_text("Pick material color from the mesh (I)").clicked(){self.set_tool(Tool::Picker);}
            if swap_button(ui).on_hover_text("Swap the active color and the second color (S)").clicked(){self.swap_colors();}
            color_button(ui,&mut self.color2,Some("Second color"));
        });
    }
    pub(super) fn color_hud(&mut self,ui:&mut egui::Ui) {
        let fill_target=self.target==StrokeTarget::Content && self.engine.layers.get(self.selected).is_some_and(|layer|matches!(&layer.content,Content::Fill(_)));
        let initial_active_color=self.active_color();
        ui.set_width(420.0);
        egui::ScrollArea::vertical().max_height(620.0).show(ui, |ui| {
        self.color_slots(ui); ui.add_space(4.0);
        let mut state=std::mem::take(&mut self.color_sliders);
        let mut second=self.color2;
        let color=hud_body(ui,&mut state,self.active_color(),&mut second);
        self.color2=second;
        if color != self.active_color() {
            self.set_active_color(color);
            // The next frame (sliders, preview) follows right away, not on the next mouse move.
            ui.ctx().request_repaint();
            if self.active_color().iter().zip(color).any(|(a,b)|(a-b).abs()>0.002) {
                self.status="The color could not be applied to this layer.".into();
            }
        }
        state.out=color; self.color_sliders=state;
        ui.separator();
        let mut material_changed=false;
        if fill_target {
            if let Some(layer)=self.engine.layers.get_mut(self.selected)
                && let Content::Fill(fill)=&mut layer.content {
                    let material=&mut fill.mat;
                    material_changed |= slider(ui,"Opacity",&mut material.opacity,0.0..=1.0);
                    material_changed |= channel_toggles(ui,&mut material.enabled);
                    if material.enabled[3] { material_changed |= slider(ui,"Height",&mut material.height,-1.0..=1.0); }
                    if material.enabled[1] { material_changed |= slider(ui,"Roughness",&mut material.rough,0.0..=1.0); }
                    if material.enabled[2] { material_changed |= slider(ui,"Metallic",&mut material.metal,0.0..=1.0); }
                }
        } else {
            let material=&mut self.brush.mat;
            slider(ui,"Opacity",&mut material.opacity,0.0..=1.0);
            channel_toggles(ui,&mut material.enabled);
            if material.enabled[3] { slider(ui,"Height",&mut material.height,-1.0..=1.0); }
            if material.enabled[1] { slider(ui,"Roughness",&mut material.rough,0.0..=1.0); }
            if material.enabled[2] { slider(ui,"Metallic",&mut material.metal,0.0..=1.0); }
        }
        ui.separator(); self.swatch_tray(ui);
        if fill_target && (material_changed || self.active_color()!=initial_active_color) {
            self.schedule_preview_update(ui.ctx());
        }
        });
    }
    pub(super) fn swatch_tray(&mut self,ui:&mut egui::Ui) {
        let current=self.active_color().map(|v|(v.clamp(0.0,1.0)*255.0).round() as u8);
        let shift=ui.input(|i|i.modifiers.shift); let (mut pick,mut remove,mut add)=(None,None,false);
        ui.horizontal_wrapped(|ui|{
            ui.spacing_mut().item_spacing=vec2(3.0,3.0);
            for (index,color) in self.prefs.swatches.iter().enumerate() {
                let (rect,response)=ui.allocate_exact_size(vec2(18.0,18.0),egui::Sense::click());
                let selected=*color==current;
                ui.painter().rect_filled(rect,2.0,Color32::from_rgb(color[0],color[1],color[2]));
                ui.painter().rect_stroke(rect,2.0,egui::Stroke::new(if selected{2.0}else{1.0},if selected{Color32::WHITE}else{Color32::from_gray(80)}),egui::StrokeKind::Inside);
                let hint=format!("#{:02x}{:02x}{:02x}\nClick: brush color\nShift+click: second color\nRight-click: remove",color[0],color[1],color[2]);
                let response=response.on_hover_text(hint);
                if response.clicked(){pick=Some((index,shift));}
                if response.secondary_clicked(){remove=Some(index);}
            }
            let can_add=self.prefs.swatches.len()<prefs::MAX_SWATCHES&&!self.prefs.swatches.contains(&current);
            let plus=egui::Button::new(icons::PLUS).min_size(vec2(18.0,18.0));
            if ui.add_enabled(can_add,plus).on_hover_text("Save the brush color in the tray").clicked(){add=true;}
        });
        if self.prefs.swatches.is_empty(){ui.label(egui::RichText::new("Press + to keep the brush color here.").small().color(theme::TEXT_DIM));}
        if let Some((index,second))=pick {
            let c=self.prefs.swatches[index].map(|v|v as f32/255.0);
            if second{self.color2=c;}else{self.set_active_color(c);}
        }
        if let Some(index)=remove{self.prefs.swatches.remove(index);}
        if add{self.prefs.swatches.push(current);}
    }

}

#[cfg(test)]
mod tests{
use super::*;
fn close(a:[f32;3],b:[f32;3])->bool{a.iter().zip(b).all(|(x,y)|(x-y).abs()<1e-4)}
#[test]fn hsv_round_trips(){for c in [[0.8,0.2,0.1],[0.0,0.0,0.0],[1.0,1.0,1.0],[0.2,0.9,0.4],[0.1,0.3,0.95],[0.5,0.5,0.5],[1.0,0.0,0.0]]{assert!(close(hsv_to_rgb(rgb_to_hsv(c)),c),"{c:?}");}}
#[test]fn gray_keeps_previous_hue(){let p=[0.6,0.7,0.8];assert!(close(hsv_keeping(p,[0.4,0.4,0.4]),[0.6,0.0,0.4]));assert!(close(hsv_keeping(p,[0.0,0.0,0.0]),[0.6,0.7,0.0]));}
#[test]fn temperature_warms_and_cools(){let c=[0.5,0.5,0.5];assert_eq!(shift_temperature(c,0.0),c);let w=shift_temperature(c,1.0);let k=shift_temperature(c,-1.0);assert!(w[0]>c[0]&&w[2]<c[2]);assert!(k[0]<c[0]&&k[2]>c[2]);}

    struct Hud {
        state: ColorSliders,
        active: [f32; 3],
        second: [f32; 3],
    }
    fn hud() -> egui_kittest::Harness<'static, Hud> {
        let start = Hud { state: ColorSliders::default(), active: [0.8, 0.2, 0.1], second: [1.0; 3] };
        let mut harness = egui_kittest::Harness::builder()
            .with_size(egui::vec2(480.0, 800.0))
            .with_accessibility_check(false)
            .build_ui_state(|ui, hud: &mut Hud| hud.active = hud_body(ui, &mut hud.state, hud.active, &mut hud.second), start);
        harness.run();
        harness
    }

    /// Typing a number into a field of the Color HUD keeps it (no jump back to the old value).
    #[test]
    fn color_hud_keeps_typed_red() {
        use egui_kittest::kittest::Queryable;
        let mut harness = hud();
        let fields = harness.query_all_by_role(egui::accesskit::Role::SpinButton).count();
        assert!(fields >= 3, "expected R, G, B fields, found {fields}");
        harness.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().click();
        harness.run();
        harness.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().focus();
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().type_text("128");
        harness.key_press(egui::Key::Enter);
        harness.run();
        harness.run();
        let red = harness.state().active[0] * 255.0;
        assert!((red - 128.0).abs() < 1.0, "typed 128 into R, the colour has {red}");
        // And it stays on later frames.
        harness.run();
        assert!((harness.state().active[0] * 255.0 - 128.0).abs() < 1.0);
    }

    /// Typing a hex code applies it, and the hex box keeps what is typed until it loses focus.
    #[test]
    fn color_hud_applies_typed_hex() {
        use egui_kittest::kittest::Queryable;
        let mut harness = hud();
        let hex = harness.get_by_role(egui::accesskit::Role::TextInput);
        hex.click();
        harness.run();
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.get_by_role(egui::accesskit::Role::TextInput).type_text("336699");
        harness.run();
        harness.run();
        let c = harness.state().active.map(|v| (v * 255.0).round() as u8);
        assert_eq!(c, [0x33, 0x66, 0x99]);
    }

    /// Typing a hue in HSV mode keeps it.
    #[test]
    fn color_hud_keeps_typed_hue() {
        use egui_kittest::kittest::Queryable;
        let mut harness = hud();
        harness.get_by_label("HSV").click();
        harness.run();
        harness.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().click();
        harness.run();
        harness.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        harness.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().type_text("62");
        harness.key_press(egui::Key::Enter);
        for _ in 0..4 {
            harness.run();
        }
        let hue = harness.state().state.hsv[0] * 360.0;
        assert!((hue - 62.0).abs() < 0.5, "typed 62 in H, the hue is {hue}");
        let from_color = rgb_to_hsv(harness.state().active)[0] * 360.0;
        assert!((from_color - 62.0).abs() < 0.5, "the applied colour has hue {from_color}");
    }
}
