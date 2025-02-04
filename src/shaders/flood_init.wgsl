#import bevy_sprite::{
    mesh2d_vertex_output::VertexOutput,
    mesh2d_view_bindings::view,
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let viewport_dimensions = view.viewport.zw;
    let viewport_uv = in.position.xy / viewport_dimensions;
    return vec4(viewport_uv, 0., 1.);
}
