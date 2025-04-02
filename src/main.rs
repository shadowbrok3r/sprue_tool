use three_d::*;
use transform_gizmo_egui::{
    math::{DMat4, DQuat, DVec3, Transform},
    *,
};

// Entry point for non-WASM platforms
#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() {
    run().await;
}

pub async fn run() {
    let window = Window::new(WindowSettings {
        title: "Jewelry Sprue Tool".to_string(),
        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    let mut gui = three_d::GUI::new(&context);

    let target = vec3(0.0, 2.0, 0.0);
    let scene_radius = 6.0;
    let mut camera = Camera::new_perspective(
        window.viewport(),
        target + scene_radius * vec3(0.6, 0.3, 1.0).normalize(),
        target,
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(camera.target(), 0.1 * scene_radius, 1000.0 * scene_radius);

    let (initial_model, min_y) = load_model(&context, "assets/suzanne.obj").await;
    let mut jewelry_model = JewelryModel {
        base: initial_model,
        sprue: None,
        selected: None,
    };

    let ambient = AmbientLight::new(&context, 0.7, Srgba::WHITE);
    let directional = DirectionalLight::new(&context, 2.0, Srgba::WHITE, vec3(-1.0, -1.0, -1.0));

    let mut file_path: Option<String> = None;
    let mut generate_sprue = false;
    let mut pending_model: Option<(Gm<Mesh, PhysicalMaterial>, f32)> = None;
    let mut attachment_y = min_y;

    let mut sprue_radius = 0.1;
    let mut sprue_height = 1.0;
    let mut sprue_offset = vec3(0.0, 0.5, 0.0);
    let mut sprue_changed = false;

    let mut gizmo = Gizmo::default();
    let gizmo_modes = GizmoMode::all();
    let mut gizmo_orientation = GizmoOrientation::Local;
    let mut gizmo_transform = Transform::from_scale_rotation_translation(
        DVec3::ONE,
        DQuat::IDENTITY,
        DVec3::new(sprue_offset.x as f64, (attachment_y + sprue_offset.y) as f64, sprue_offset.z as f64).into(),
    );

    window.render_loop(move |mut frame_input| {
        let mut redraw = frame_input.first_frame;
        redraw |= camera.set_viewport(frame_input.viewport);

        let mut egui_consumes_input = false;
        let mut gizmo_interacted = false;

        // Simplified picking
        for event in &frame_input.events {
            if let Event::MousePress { button, position, .. } = event {
                if *button == MouseButton::Left && !egui_consumes_input {
                    let objects: Vec<Gm<Mesh, PhysicalMaterial>> = if let Some(ref sprue) = jewelry_model.sprue {
                        vec![jewelry_model.base.clone(), sprue.clone()]
                    } else {
                        vec![jewelry_model.base.clone()]
                    };
                    let mut id_texture = Texture2D::new_empty::<u8>(
                        &context,
                        frame_input.viewport.width,
                        frame_input.viewport.height,
                        Interpolation::Nearest,
                        Interpolation::Nearest,
                        None,
                        Wrapping::ClampToEdge,
                        Wrapping::ClampToEdge,
                    );
                    let mut render_target = RenderTarget::new(
                        ColorTarget::new_texture_2d(&context, &mut id_texture),
                        DepthTarget::new(&context, frame_input.viewport.width, frame_input.viewport.height),
                    );
                    render_target.clear(ClearState::color_and_depth(0.0, 0.0, 0.0, 1.0, 1.0));
                    for (i, obj) in objects.iter().enumerate() {
                        let id_material = PhysicalMaterial::new_opaque(
                            &context,
                            &CpuMaterial {
                                albedo: Srgba::new_opaque((i + 1) as u8, 0, 0), // ID in red channel
                                ..Default::default()
                            },
                        );
                        render_target.render(&camera, &[Gm::new(obj.geometry.clone(), id_material)], &[]);
                    }
                    let x = (position.x as f32).clamp(0.0, frame_input.viewport.width as f32 - 1.0) as u32;
                    let y = (position.y as f32).clamp(0.0, frame_input.viewport.height as f32 - 1.0) as u32;
                    // Read pixel from texture (simplified, requires OpenGL access)
                    let mut pixel = vec![0u8; 4];
                    unsafe {
                        id_texture.bind();
                        gl::ReadPixels(x as i32, y as i32, 1, 1, gl::RGBA, gl::UNSIGNED_BYTE, pixel.as_mut_ptr() as *mut _);
                    }
                    if pixel[0] > 0 {
                        let id = pixel[0] as u32 - 1; // Adjust for 0-based indexing
                        jewelry_model.selected = Some(id);
                        let pos = if id == 0 {
                            vec3(0.0, 2.0, 0.0)
                        } else {
                            vec3(0.0, attachment_y, 0.0) + sprue_offset
                        };
                        gizmo_transform.translation = DVec3::new(pos.x as f64, pos.y as f64, pos.z as f64).into();
                        redraw = true;
                    }
                }
            }
        }

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                egui::SidePanel::left("tools_panel").show(gui_context, |ui| {
                    ui.heading("Tools");

                    ui.collapsing("File Operations", |ui| {
                        if ui.button("Load OBJ File").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("OBJ", &["obj"])
                                .pick_file()
                            {
                                file_path = Some(path.display().to_string());
                            }
                        }
                        if let Some(ref path) = file_path {
                            ui.label(format!("Loaded: {}", path));
                        } else {
                            ui.label("No file loaded yet.");
                        }
                    });

                    ui.collapsing("Sprue Settings", |ui| {
                        if ui.button("Generate Sprue").clicked() {
                            generate_sprue = true;
                        }
                        if ui.add(egui::Slider::new(&mut sprue_radius, 0.05..=0.5).text("Radius")).changed() {
                            sprue_changed = true;
                        }
                        if ui.add(egui::Slider::new(&mut sprue_height, 0.1..=2.0).text("Height")).changed() {
                            sprue_changed = true;
                        }
                        if ui.add(egui::Slider::new(&mut sprue_offset.x, -1.0..=1.0).text("X Offset")).changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.x = sprue_offset.x as f64;
                        }
                        if ui.add(egui::Slider::new(&mut sprue_offset.y, 0.0..=1.0).text("Y Offset")).changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.y = (attachment_y + sprue_offset.y) as f64;
                        }
                        if ui.add(egui::Slider::new(&mut sprue_offset.z, -1.0..=1.0).text("Z Offset")).changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.z = sprue_offset.z as f64;
                        }
                    });

                    ui.collapsing("Gizmo Settings", |ui| {
                        egui::ComboBox::from_id_salt("orientation_cb")
                            .selected_text(format!("{:?}", gizmo_orientation))
                            .show_ui(ui, |ui| {
                                for orientation in [GizmoOrientation::Global, GizmoOrientation::Local] {
                                    ui.selectable_value(&mut gizmo_orientation, orientation, format!("{:?}", orientation));
                                }
                            });
                    });

                    if jewelry_model.selected.is_some() {
                        let viewport = frame_input.viewport.to_egui();
                        let view_matrix = camera.view().as_dmat4();
                        let projection_matrix = camera.projection().as_dmat4();

                        gizmo.update_config(GizmoConfig {
                            view_matrix: view_matrix.into(),
                            projection_matrix: projection_matrix.into(),
                            viewport,
                            modes: gizmo_modes,
                            orientation: gizmo_orientation,
                            snapping: ui.input(|i| i.modifiers.ctrl),
                            ..Default::default()
                        });

                        if let Some((_, new_transforms)) = gizmo.interact(ui, &[gizmo_transform]) {
                            for new_transform in new_transforms.iter() {
                                gizmo_transform = *new_transform;
                                if jewelry_model.selected == Some(1) {
                                    sprue_offset.x = new_transform.translation.x as f32;
                                    sprue_offset.y = (new_transform.translation.y as f32) - attachment_y;
                                    sprue_offset.z = new_transform.translation.z as f32;
                                    sprue_changed = true;
                                }
                                gizmo_interacted = true;
                            }
                        }
                    }

                    egui_consumes_input = ui.ctx().wants_pointer_input();
                });

                egui::SidePanel::right("object_info_panel").show(gui_context, |ui| {
                    ui.heading("Objects");

                    ui.label("Objects in Scene:");
                    ui.group(|ui| {
                        ui.label(format!("Base (ID: 0) {}", if jewelry_model.selected == Some(0) { "[Selected]" } else { "" }));
                        if jewelry_model.sprue.is_some() {
                            ui.label(format!("Sprue (ID: 1) {}", if jewelry_model.selected == Some(1) { "[Selected]" } else { "" }));
                        }
                    });

                    ui.label("Object Properties:");
                    ui.group(|ui| {
                        if let Some(selected) = jewelry_model.selected {
                            if selected == 0 {
                                let mut material = jewelry_model.base.material.clone();
                                let mut color = [material.albedo.r as f32 / 255.0, material.albedo.g as f32 / 255.0, material.albedo.b as f32 / 255.0];
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    material.albedo = Srgba::new_opaque((color[0] * 255.0) as u8, (color[1] * 255.0) as u8, (color[2] * 255.0) as u8);
                                    jewelry_model.base.material = material;
                                    redraw = true;
                                }
                            } else if selected == 1 {
                                if let Some(sprue) = jewelry_model.sprue.as_mut() {
                                    let mut material = sprue.material.clone();
                                    let mut color = [material.albedo.r as f32 / 255.0, material.albedo.g as f32 / 255.0, material.albedo.b as f32 / 255.0];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        material.albedo = Srgba::new_opaque((color[0] * 255.0) as u8, (color[1] * 255.0) as u8, (color[2] * 255.0) as u8);
                                        sprue.material = material;
                                        redraw = true;
                                    }
                                }
                            }
                        } else {
                            ui.label("No object selected.");
                        }
                    });
                });
            },
        );

        if !egui_consumes_input {
            redraw |= control.handle_events(&mut camera, &mut frame_input.events);
        }

        if let Some(path) = file_path.take() {
            let context = context.clone();
            let new_model = std::thread::spawn(move || {
                tokio::runtime::Handle::current().block_on(load_model(&context, &path))
            })
            .join()
            .unwrap();
            pending_model = Some(new_model);
        }

        if let Some((new_model, min_y)) = pending_model.take() {
            jewelry_model.base = new_model;
            jewelry_model.sprue = None;
            jewelry_model.selected = None;
            attachment_y = min_y;
            gizmo_transform.translation.y = (attachment_y + sprue_offset.y) as f64;
            redraw = true;
        }

        if generate_sprue || (sprue_changed && jewelry_model.sprue.is_some()) {
            let mut sprue_mesh = create_cylinder(10, sprue_radius, sprue_height);
            sprue_mesh
                .transform(Mat4::from_translation(vec3(0.0, attachment_y, 0.0) + sprue_offset))
                .unwrap();
            jewelry_model.sprue = Some(Gm::new(
                Mesh::new(&context, &sprue_mesh),
                PhysicalMaterial::new_opaque(
                    &context,
                    &CpuMaterial {
                        albedo: Srgba::new_opaque(220, 50, 50),
                        roughness: 0.7,
                        metallic: 0.8,
                        ..Default::default()
                    },
                ),
            ));
            jewelry_model.selected = Some(1);
            gizmo_transform.translation = DVec3::new(sprue_offset.x as f64, (attachment_y + sprue_offset.y) as f64, sprue_offset.z as f64).into();
            generate_sprue = false;
            sprue_changed = false;
            redraw = true;
        }

        redraw |= gizmo_interacted;

        if redraw {
            let mut screen = frame_input.screen();
            screen.clear(ClearState::color_and_depth(12.0 / 255.0, 12.0 / 255.0, 16.0 / 255.0, 1.0, 1.0));
            let objects: Vec<&dyn Object> = if let Some(ref sprue) = jewelry_model.sprue {
                vec![&jewelry_model.base, sprue]
            } else {
                vec![&jewelry_model.base]
            };
            screen.render(&camera, &objects, &[&ambient, &directional]);
            let base_highlight = jewelry_model.selected == Some(0);
            let sprue_highlight = jewelry_model.selected == Some(1);
            if base_highlight {
                screen.render(&camera, &[&jewelry_model.base], &[&AmbientLight::new(&context, 0.3, Srgba::new_opaque(255, 255, 0))]);
            } else if sprue_highlight && jewelry_model.sprue.is_some() {
                screen.render(&camera, &[jewelry_model.sprue.as_ref().unwrap()], &[&AmbientLight::new(&context, 0.3, Srgba::new_opaque(255, 255, 0))]);
            }
            screen.write(|| gui.render());
        }

        FrameOutput {
            swap_buffers: redraw,
            ..Default::default()
        }
    });
}

struct JewelryModel {
    base: Gm<Mesh, PhysicalMaterial>,
    sprue: Option<Gm<Mesh, PhysicalMaterial>>,
    selected: Option<u32>,
}

async fn load_model(context: &Context, path: &str) -> (Gm<Mesh, PhysicalMaterial>, f32) {
    let mut loaded = three_d_asset::io::load_async(&[path])
        .await
        .expect("Failed to load OBJ file");
    let mut cpu_mesh: CpuMesh = loaded
        .deserialize(path)
        .expect("Failed to deserialize OBJ file");
    cpu_mesh
        .transform(Mat4::from_translation(vec3(0.0, 2.0, 0.0)))
        .expect("Failed to transform mesh");

    let min_y = cpu_mesh
        .positions
        .to_f32()
        .iter()
        .map(|v| v.y)
        .fold(f32::INFINITY, f32::min);

    let model = Gm::new(
        Mesh::new(context, &cpu_mesh),
        PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba::new_opaque(100, 100, 100),
                roughness: 0.7,
                metallic: 0.8,
                ..Default::default()
            },
        ),
    );
    (model, min_y)
}

fn create_cylinder(sides: u32, radius: f32, height: f32) -> CpuMesh {
    let mut positions = Vec::new();
    let mut indices = Vec::new();

    positions.push(vec3(0.0, 0.0, 0.0));
    for i in 0..sides {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / sides as f32;
        positions.push(vec3(radius * angle.cos(), 0.0, radius * angle.sin()));
    }

    positions.push(vec3(0.0, height, 0.0));
    for i in 0..sides {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / sides as f32;
        positions.push(vec3(radius * angle.cos(), height, radius * angle.sin()));
    }

    for i in 1..sides {
        indices.push(0);
        indices.push(i + 1);
        indices.push(i);
    }
    indices.push(0);
    indices.push(1);
    indices.push(sides);

    let top_start = sides + 1;
    for i in 1..sides {
        indices.push(top_start);
        indices.push(top_start + i);
        indices.push(top_start + i + 1);
    }
    indices.push(top_start);
    indices.push(top_start + sides);
    indices.push(top_start + 1);

    for i in 0..sides {
        let next = (i + 1) % sides;
        indices.push(i + 1);
        indices.push(next + 1);
        indices.push(top_start + next + 1);
        indices.push(top_start + next + 1);
        indices.push(top_start + i + 1);
        indices.push(i + 1);
    }

    CpuMesh {
        positions: Positions::F32(positions),
        indices: Indices::U32(indices),
        ..Default::default()
    }
}

trait ToEguiRect {
    fn to_egui(self) -> egui::Rect;
}

impl ToEguiRect for Viewport {
    fn to_egui(self) -> egui::Rect {
        egui::Rect::from_min_size(
            egui::pos2(self.x as f32, self.y as f32),
            egui::vec2(self.width as f32, self.height as f32),
        )
    }
}

trait AsDMat4 {
    fn as_dmat4(self) -> DMat4;
}

impl AsDMat4 for Mat4 {
    fn as_dmat4(self) -> DMat4 {
        DMat4::from_cols_array(&[
            self.x.x as f64, self.x.y as f64, self.x.z as f64, self.x.w as f64,
            self.y.x as f64, self.y.y as f64, self.y.z as f64, self.y.w as f64,
            self.z.x as f64, self.z.y as f64, self.z.z as f64, self.z.w as f64,
            self.w.x as f64, self.w.y as f64, self.w.z as f64, self.w.w as f64,
        ])
    }
}