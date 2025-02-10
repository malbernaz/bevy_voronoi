#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var mask_texture: texture_2d<f32>;
@group(0) @binding(1) var sampler_obj: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let screen_size = vec2<f32>(textureDimensions(mask_texture));
    let mask = textureSample(mask_texture, sampler_obj, in.uv).r;
    let null_seed = vec2(-1.);

    if mask != 1. {
        return vec4(null_seed, 0., 1.);
    }

    let offsets = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0), vec2<f32>(-1.0, 0.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(0.0, -1.0)
    );

    for (var i = 0; i < 4; i++) {
        let neighbor_uv = (in.position.xy + offsets[i]) / screen_size;
        let neighbor_mask = textureSample(mask_texture, sampler_obj, neighbor_uv).r;

        // Mark edge pixels as seeds, alpha channel means original seed;
        if neighbor_mask != mask {
            return vec4(in.position.xy, 1., 2.);
        }
    }

    // The blue channel denotes that the fragment is inside the mask
    return vec4(null_seed, 1., 1.);
}
