#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var flood_texture: texture_2d<f32>;
@group(0) @binding(1) var flood_sampler: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let s = textureSample(flood_texture, flood_sampler, in.uv);

    // TODO: understand what why this step is necessary
    // using the uv directly from the mask shader ads an artifact 
    // in the top right corner of the screen
    // this check gets rid of it
    if s.r != 1. && s.g != 1. && s.b != 1. && s.a != 1. {
        return vec4(-1.);
    }

    return vec4(in.uv, 0., 1.);
}
