#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(1.);
}
