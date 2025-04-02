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

    let (initial_model, initial_cpu_mesh, min_y) = load_model(&context, "assets/suzanne.obj").await;
    let mut jewelry_model = JewelryModel {
        base: initial_model,
        base_cpu_mesh: initial_cpu_mesh,
        sprue: None,
        sprue_cpu_mesh: None,
        selected: None,
    };

    let ambient = AmbientLight::new(&context, 0.7, Srgba::WHITE);
    let directional = DirectionalLight::new(&context, 2.0, Srgba::WHITE, vec3(-1.0, -1.0, -1.0));

    let mut file_path: Option<String> = None;
    let mut generate_sprue = false;
    let mut pending_model: Option<(Gm<Mesh, PhysicalMaterial>, CpuMesh, f32)> = None;
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
        DVec3::new(sprue_offset.x as f64, (attachment_y + sprue_offset.y) as f64, sprue_offset.z as f64),
    );

    window.render_loop(move |mut frame_input| {
        let redraw = true; // Always redraw for UI responsiveness

        camera.set_viewport(frame_input.viewport);

        let mut egui_consumes_input = false;
        let mut gizmo_interacted = false;

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
                            println!("Generate Sprue clicked"); // Debug
                        }

                        let radius_response = ui.add(egui::Slider::new(&mut sprue_radius, 0.05..=0.5).text("Radius"));
                        let height_response = ui.add(egui::Slider::new(&mut sprue_height, 0.1..=2.0).text("Height"));
                        let x_offset_response = ui.add(egui::Slider::new(&mut sprue_offset.x, -1.0..=1.0).text("X Offset"));
                        let y_offset_response = ui.add(egui::Slider::new(&mut sprue_offset.y, 0.0..=1.0).text("Y Offset"));
                        let z_offset_response = ui.add(egui::Slider::new(&mut sprue_offset.z, -1.0..=1.0).text("Z Offset"));

                        if radius_response.drag_stopped() && radius_response.changed() {
                            sprue_changed = true;
                            println!("Radius changed to {}", sprue_radius); // Debug
                        }
                        if height_response.drag_stopped() && height_response.changed() {
                            sprue_changed = true;
                            println!("Height changed to {}", sprue_height); // Debug
                        }
                        if x_offset_response.drag_stopped() && x_offset_response.changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.x = sprue_offset.x as f64;
                            println!("X Offset changed to {}", sprue_offset.x); // Debug
                        }
                        if y_offset_response.drag_stopped() && y_offset_response.changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.y = (attachment_y + sprue_offset.y) as f64;
                            println!("Y Offset changed to {}", sprue_offset.y); // Debug
                        }
                        if z_offset_response.drag_stopped() && z_offset_response.changed() {
                            sprue_changed = true;
                            gizmo_transform.translation.z = sprue_offset.z as f64;
                            println!("Z Offset changed to {}", sprue_offset.z); // Debug
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

                        println!("Gizmo transform: {:?}", gizmo_transform.translation); // Debug
                        if let Some((_, new_transforms)) = gizmo.interact(ui, &[gizmo_transform]) {
                            for new_transform in new_transforms.iter() {
                                gizmo_transform = *new_transform;
                                if jewelry_model.selected == Some(1) {
                                    sprue_offset.x = new_transform.translation.x as f32;
                                    sprue_offset.y = (new_transform.translation.y as f32) - attachment_y;
                                    sprue_offset.z = new_transform.translation.z as f32;
                                    sprue_changed = true;
                                    println!("Gizmo moved sprue to {:?}", sprue_offset); // Debug
                                }
                                gizmo_interacted = true;
                            }
                        }
                    }

                    egui_consumes_input = ui.ctx().wants_pointer_input() || ui.ctx().is_pointer_over_area();
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
                                }
                            } else if selected == 1 {
                                if let Some(sprue) = jewelry_model.sprue.as_mut() {
                                    let mut material = sprue.material.clone();
                                    let mut color = [material.albedo.r as f32 / 255.0, material.albedo.g as f32 / 255.0, material.albedo.b as f32 / 255.0];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        material.albedo = Srgba::new_opaque((color[0] * 255.0) as u8, (color[1] * 255.0) as u8, (color[2] * 255.0) as u8);
                                        sprue.material = material;
                                    }
                                }
                            }
                        } else {
                            ui.label("No object selected.");
                        }
                    });
                });

                // Get the gizmo world position from its transform.
                let gizmo_world = vec3(
                    gizmo_transform.translation.x as f32,
                    gizmo_transform.translation.y as f32,
                    gizmo_transform.translation.z as f32,
                );

                // Compute the view-projection matrix.
                let view_proj = camera.projection() * camera.view();

                // Project the world position to clip space.
                let gizmo_clip = view_proj * vec4(gizmo_world.x, gizmo_world.y, gizmo_world.z, 1.0);
                // Convert to normalized device coordinates (NDC).
                let gizmo_ndc = vec2(gizmo_clip.x / gizmo_clip.w, gizmo_clip.y / gizmo_clip.w);

                // Convert NDC (which is in [-1,1]) to screen coordinates.
                let viewport = frame_input.viewport;
                let gizmo_screen_x = ((gizmo_ndc.x + 1.0) / 2.0) * viewport.width as f32;
                let gizmo_screen_y = ((1.0 - gizmo_ndc.y) / 2.0) * viewport.height as f32;


                let gizmo_area_rect = egui::Rect::from_min_size(
                    egui::pos2(gizmo_screen_x - 150.0, gizmo_screen_y - 150.0),
                    egui::vec2(300.0, 300.0)
                );
                

                
                egui::Area::new("gizmo_area".into())
                .order(egui::Order::Foreground)
                .constrain_to(gizmo_area_rect)
                .show(gui_context, |ui| {
                    // Update gizmo configuration.
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
                                println!("Gizmo moved sprue to {:?}", sprue_offset);
                            }
                        }
                    }
                });
            
            },
        );

        // Picking logic
        for event in &frame_input.events {
            if let Event::MousePress { button, position, .. } = event {
                if *button == MouseButton::Left && !egui_consumes_input {
                    let x = (2.0 * position.x as f32 / frame_input.viewport.width as f32) - 1.0;
                    let y = 1.0 - (2.0 * position.y as f32 / frame_input.viewport.height as f32);
                    let near_ndc = vec4(x, y, -1.0, 1.0);
                    let far_ndc = vec4(x, y, 1.0, 1.0);

                    let view_proj = camera.projection() * camera.view();
                    let inv_vp = view_proj.invert().expect("Failed to invert view-projection matrix");
                    let near_world = inv_vp * near_ndc;
                    let far_world = inv_vp * far_ndc;
                    let near = vec3(near_world.x / near_world.w, near_world.y / near_world.w, near_world.z / near_world.w);
                    let far = vec3(far_world.x / far_world.w, far_world.y / far_world.w, far_world.z / far_world.w);
                    let direction = (far - near).normalize();
                    let ray = Ray { origin: near, direction };

                    let mut closest = None;
                    let mut min_dist = f32::INFINITY;

                    if let Some(dist) = raycast_mesh(
                        &jewelry_model.base_cpu_mesh,
                        &ray,
                        &Mat4::identity()
                    ) {
                        if dist < min_dist {
                            min_dist = dist;
                            closest = Some(0);
                        }
                        println!("Base hit at distance: {}", dist); // Debug
                    }
                    
                    if let (Some(sprue), Some(sprue_cpu_mesh)) = (&jewelry_model.sprue, &jewelry_model.sprue_cpu_mesh) {
                        if let Some(dist) = raycast_mesh(
                            sprue_cpu_mesh,
                            &ray,
                            &Mat4::identity()
                        ) {
                            if dist < min_dist {
                                min_dist = dist;
                                closest = Some(1);
                            }
                            println!("Sprue hit at distance: {}", dist); // Debug
                        }
                    }
                    
                    
                    if let Some(id) = closest {
                        jewelry_model.selected = Some(id);
                        let pos = if id == 0 {
                            // Compute the center of the base mesh
                            mesh_center(&jewelry_model.base_cpu_mesh)
                        } else {
                            // For the sprue, use your calculated attachment position.
                            vec3(0.0, attachment_y, 0.0) + sprue_offset
                        };
                        gizmo_transform.translation = DVec3::new(pos.x as f64, pos.y as f64, pos.z as f64).into();
                        println!("Selected object ID: {} at position {:?}", id, pos); // Debug
                    }
                    
                }
            }
        }

        if !egui_consumes_input {
            control.handle_events(&mut camera, &mut frame_input.events);
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

        if let Some((new_model, new_cpu_mesh, min_y)) = pending_model.take() {
            jewelry_model.base = new_model;
            jewelry_model.base_cpu_mesh = new_cpu_mesh;
            jewelry_model.sprue = None;
            jewelry_model.sprue_cpu_mesh = None;
            jewelry_model.selected = None;
            attachment_y = min_y;
            gizmo_transform.translation.y = (attachment_y + sprue_offset.y) as f64;
        }

        if generate_sprue || (sprue_changed && jewelry_model.sprue.is_some()) {
            let mut sprue_mesh = create_cylinder(10, sprue_radius, sprue_height);
            sprue_mesh
                .transform(Mat4::from_translation(vec3(0.0, attachment_y, 0.0) + sprue_offset))
                .unwrap();
            let sprue_cpu_mesh = sprue_mesh.clone();
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
            jewelry_model.sprue_cpu_mesh = Some(sprue_cpu_mesh);
            jewelry_model.selected = Some(1);
            gizmo_transform.translation = DVec3::new(sprue_offset.x as f64, (attachment_y + sprue_offset.y) as f64, sprue_offset.z as f64).into();
            generate_sprue = false;
            sprue_changed = false;
            println!("Sprue generated at {:?}", sprue_offset); // Debug
        }

        // Render every frame
        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(12.0 / 255.0, 12.0 / 255.0, 16.0 / 255.0, 1.0, 1.0));
        let objects: Vec<&dyn Object> = if let Some(ref sprue) = jewelry_model.sprue {
            vec![&jewelry_model.base, sprue]
        } else {
            vec![&jewelry_model.base]
        };
        screen.render(&camera, &objects, &[&ambient, &directional]);

        if let Some(selected) = jewelry_model.selected {
            let highlight_objects: Vec<&dyn Object> = if selected == 0 {
                vec![&jewelry_model.base]
            } else if selected == 1 && jewelry_model.sprue.is_some() {
                vec![jewelry_model.sprue.as_ref().unwrap()]
            } else {
                vec![]
            };
            if !highlight_objects.is_empty() {
                screen.render(
                    &camera,
                    &highlight_objects,
                    &[&AmbientLight::new(&context, 0.5, Srgba::new_opaque(255, 255, 0))],
                );
            }
        }

        screen.write(|| gui.render());

        FrameOutput {
            swap_buffers: redraw,
            ..Default::default()
        }
    });
}

struct JewelryModel {
    base: Gm<Mesh, PhysicalMaterial>,
    base_cpu_mesh: CpuMesh,
    sprue: Option<Gm<Mesh, PhysicalMaterial>>,
    sprue_cpu_mesh: Option<CpuMesh>,
    selected: Option<u32>,
}

async fn load_model(context: &Context, path: &str) -> (Gm<Mesh, PhysicalMaterial>, CpuMesh, f32) {
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
    (model, cpu_mesh, min_y)
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

struct Ray {
    origin: Vec3,
    direction: Vec3,
}

fn raycast_mesh(cpu_mesh: &CpuMesh, ray: &Ray, model_matrix: &Mat4) -> Option<f32> {
    let positions = match &cpu_mesh.positions {
        Positions::F32(pos) => pos,
        _ => return None,
    };
    let indices = match &cpu_mesh.indices {
        Indices::U32(ind) => ind,
        _ => return None,
    };

    let mut min_dist = f32::INFINITY;
    let mut hit = false;

    for i in (0..indices.len()).step_by(3) {
        let v0 = model_matrix * vec4(positions[indices[i] as usize].x, positions[indices[i] as usize].y, positions[indices[i] as usize].z, 1.0);
        let v1 = model_matrix * vec4(positions[indices[i + 1] as usize].x, positions[indices[i + 1] as usize].y, positions[indices[i + 1] as usize].z, 1.0);
        let v2 = model_matrix * vec4(positions[indices[i + 2] as usize].x, positions[indices[i + 2] as usize].y, positions[indices[i + 2] as usize].z, 1.0);

        if let Some(dist) = ray_triangle_intersection(ray, vec3(v0.x, v0.y, v0.z), vec3(v1.x, v1.y, v1.z), vec3(v2.x, v2.y, v2.z)) {
            if dist < min_dist {
                min_dist = dist;
                hit = true;
            }
        }
    }

    if hit { Some(min_dist) } else { None }
}

fn ray_triangle_intersection(ray: &Ray, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<f32> {
    const EPSILON: f32 = 0.000001;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = ray.direction.cross(edge2);
    let a = edge1.dot(h);

    if a > -EPSILON && a < EPSILON {
        return None;
    }

    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * s.dot(h);

    if u < 0.0 || u > 1.0 {
        return None;
    }

    let q = s.cross(edge1);
    let v = f * ray.direction.dot(q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(q);
    if t > EPSILON {
        Some(t)
    } else {
        None
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


fn mesh_center(cpu_mesh: &CpuMesh) -> Vec3 {
    match &cpu_mesh.positions {
        Positions::F32(positions) => {
            let mut min = positions[0];
            let mut max = positions[0];
            for pos in positions {
                min = vec3(min.x.min(pos.x), min.y.min(pos.y), min.z.min(pos.z));
                max = vec3(max.x.max(pos.x), max.y.max(pos.y), max.z.max(pos.z));
            }
            (min + max) / 2.0
        }
        _ => vec3(0.0, 0.0, 0.0),
    }
}
