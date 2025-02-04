#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;
@group(0) @binding(2) var<uniform> step: u32;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(screen_texture));

    var current = textureSample(screen_texture, texture_sampler, in.uv);
    var closest_dist = distance(in.position.xy, current.xy);

    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if x == 0 && y == 0 {
                continue;
            }

            let neighbour_coord = in.position.xy + vec2<f32>(vec2<i32>(x, y) * i32(step));
            let neighbour = textureSample(screen_texture, texture_sampler, neighbour_coord / dims);
            let dist = distance(in.position.xy, neighbour.xy);

            if dist < closest_dist {
                closest_dist = dist;
                current = neighbour;
            }
        }
    }

    return current;
}
