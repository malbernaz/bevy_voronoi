#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var seed_texture: texture_2d<f32>;
@group(0) @binding(1) var sampler_obj: sampler;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let signed_dist = sdf(in);

    if signed_dist <= 0. {
        return vec4<f32>(vec3(abs(signed_dist)), 1.);
    }

    return vec4<f32>(vec3(signed_dist), 1.);
}

fn sdf(in: FullscreenVertexOutput) -> f32 {
    // Get nearest seed position from the Jump Flood Algorithm output
    let nearest_seed = seed_uv(in);
    // Compute unsigned Euclidean distance
    let dist = length(in.uv - nearest_seed.xy);
    // Determine if the pixel is inside or outside the shape
    let is_inside = nearest_seed.z == 1.;
    // Signed distance: negative if inside, positive if outside
    return select(dist, -dist, is_inside);
}

fn seed_uv(in: FullscreenVertexOutput) -> vec3<f32> {
    let screen_size = vec2<f32>(textureDimensions(seed_texture));
    let seed = textureSample(seed_texture, sampler_obj, in.uv);
    let uv = seed.xy / screen_size;
    return vec3(uv, seed.z);
}
