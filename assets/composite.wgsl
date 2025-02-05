#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var flood_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let screen_frag = textureSample(screen_texture, texture_sampler, in.uv);
    let flood_frag = textureSample(flood_texture, texture_sampler, in.uv);

    // SDF
    let dist = length(flood_frag.xy - in.uv);

    return vec4(dist);
}
