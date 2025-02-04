#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var flood_texture: texture_2d<f32>;
@group(0) @binding(2) var texture_sampler: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(screen_texture));

    let screen_frag = textureSample(screen_texture, texture_sampler, in.uv);
    let flood_frag = textureSample(flood_texture, texture_sampler, in.uv);

    let frag_uv = flood_frag.xy / dims;
    let dist = distance(in.uv, frag_uv);

    return mix(screen_frag, vec4(vec3(dist), 1.), 0.5);
}
