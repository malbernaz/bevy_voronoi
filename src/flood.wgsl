#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var flood_texture: texture_2d<f32>;
@group(0) @binding(1) var flood_sampler: sampler;
@group(0) @binding(2) var<uniform> step: u32;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(flood_texture));
    var current_dist = 999999.;
    var current_uv = vec2<f32>(-1.);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let coord = in.position.xy + vec2<f32>(vec2<i32>(x, y) * i32(step));
            let n_uv = textureSample(flood_texture, flood_sampler, coord / dims).xy;
            let dist = length(in.uv - n_uv);

            if n_uv.x >= 0. && n_uv.y >= 0. && dist < current_dist {
                current_dist = dist;
                current_uv = n_uv;
            }
        }
    }

    return vec4<f32>(current_uv, 0., 1.);
}

@fragment
fn init(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let s = textureSample(flood_texture, flood_sampler, in.uv);

    if s.r != 1. && s.g != 1. && s.b != 1. && s.a != 1. {
        return vec4(-1.);
    }

    return vec4(in.uv, 0., 1.);
}
