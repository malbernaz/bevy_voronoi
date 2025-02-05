#import bevy_sprite::mesh2d_vertex_output::VertexOutput

@group(2) @binding(0) var alpha_texture: texture_2d<f32>;
@group(2) @binding(1) var alpha_sampler: sampler;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let alpha_mask = textureSample(alpha_texture, alpha_sampler, in.uv).a;

    if alpha_mask <= 0. {
        discard;
    }

    return vec4(1.);
}
