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

mod plugin;

use bevy::prelude::*;
use plugin::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, FloodPlugin))
        .add_systems(Startup, setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    // cube
    // This cube will be rendered by the main pass, but it will also be rendered by our custom
    // pass. This should result in an unlit red cube
    commands.spawn((
        Mesh2d(meshes.add(Circle::new(100.))),
        //MeshMaterial2d(materials.add(Color::srgb_u8(124, 144, 255))),
        FloodComponent,
    ));

    // camera
    commands.spawn(Camera2d);
}
