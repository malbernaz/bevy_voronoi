//! This example demonstrates how to write a custom phase
//!
//! Render phases in bevy are used whenever you need to draw a group of meshes in a specific way.
//! For example, bevy's main pass has an opaque phase, a transparent phase for both 2d and 3d.
//! Sometimes, you may want to only draw a subset of meshes before or after the builtin phase. In
//! those situations you need to write your own phase.
//!
//! This example showcases how writing a custom phase to draw a stencil of a bevy mesh could look
//! like. Some shortcuts have been used for simplicity.
//!
//! This example was made for 3d, but a 2d equivalent would be almost identical.

mod composite;
mod flood;
mod flood_init;
mod plugin;

use bevy::{color::palettes::tailwind::GRAY_900, prelude::*};
use plugin::*;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::NONE))
        .add_plugins((DefaultPlugins, FloodPlugin))
        .add_systems(Startup, setup)
        .run();
}

const X_EXTENT: f32 = 800.;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);

    let shapes = [
        meshes.add(Circle::new(50.0)),
        meshes.add(Annulus::new(25.0, 50.0)),
        meshes.add(Capsule2d::new(25.0, 50.0)),
        meshes.add(Rhombus::new(75.0, 100.0)),
        meshes.add(Rectangle::new(50.0, 100.0)),
        meshes.add(RegularPolygon::new(50.0, 6)),
        meshes.add(Triangle2d::new(
            Vec2::Y * 50.0,
            Vec2::new(-50.0, -50.0),
            Vec2::new(50.0, -50.0),
        )),
    ];
    let num_shapes = shapes.len();

    for (i, shape) in shapes.into_iter().enumerate() {
        // Distribute colors evenly across the rainbow.
        let color = Color::from(GRAY_900);

        commands.spawn((
            Mesh2d(shape),
            MeshMaterial2d(materials.add(color)),
            FloodComponent,
            Transform::from_xyz(
                // Distribute shapes from -X_EXTENT/2 to +X_EXTENT/2.
                -X_EXTENT / 2. + i as f32 / (num_shapes - 1) as f32 * X_EXTENT,
                0.0,
                0.0,
            ),
        ));
    }
}
