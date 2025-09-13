use bevy::{
    core_pipeline::FullscreenShader,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::{
            binding_types::{sampler, texture_2d, uniform_buffer},
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FragmentState, MultisampleState, Operations,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages,
            TextureFormat, TextureSampleType, UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use crate::plugin::VoronoiTexture;

#[derive(Resource)]
pub struct FloodPipeline {
    pub seed_layout: BindGroupLayout,
    pub seed_pipeline: CachedRenderPipelineId,
    pub layout: BindGroupLayout,
    pub pipeline: CachedRenderPipelineId,
}

pub fn init_flood_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    fullscreen_shader: Res<FullscreenShader>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    let seed_layout = render_device.create_bind_group_layout(
        "flood_seed_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
            ),
        ),
    );

    let fullscreen_vertex_state = fullscreen_shader.to_vertex_state();

    let seed_pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("flood_seed_pipeline".into()),
        layout: vec![seed_layout.clone()],
        vertex: fullscreen_vertex_state.clone(),
        fragment: Some(FragmentState {
            shader: asset_server.load("embedded://bevy_voronoi/flood_seed.wgsl"),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        push_constant_ranges: vec![],
        primitive: Default::default(),
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        zero_initialize_workgroup_memory: false,
    });

    let layout = render_device.create_bind_group_layout(
        "flood_bind_group_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::FRAGMENT,
            (
                texture_2d(TextureSampleType::Float { filterable: true }),
                sampler(SamplerBindingType::Filtering),
                uniform_buffer::<UVec2>(false),
            ),
        ),
    );

    let pipeline = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
        label: Some("flood_pipeline".into()),
        layout: vec![layout.clone()],
        vertex: fullscreen_vertex_state,
        fragment: Some(FragmentState {
            shader: asset_server.load("embedded://bevy_voronoi/flood.wgsl"),
            shader_defs: vec![],
            entry_point: Some("fragment".into()),
            targets: vec![Some(ColorTargetState {
                format: TextureFormat::Rgba16Float,
                blend: None,
                write_mask: ColorWrites::ALL,
            })],
        }),
        push_constant_ranges: vec![],
        primitive: Default::default(),
        depth_stencil: None,
        multisample: MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        zero_initialize_workgroup_memory: false,
    });

    commands.insert_resource(FloodPipeline {
        seed_pipeline,
        seed_layout,
        layout,
        pipeline,
    });
}

pub fn run_flood_seed_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    camera: &ExtractedCamera,
    voronoi_texture: &mut VoronoiTexture,
) {
    let flood_pipeline = world.resource::<FloodPipeline>();

    let Some(pipeline) = world
        .resource::<PipelineCache>()
        .get_render_pipeline(flood_pipeline.seed_pipeline)
    else {
        return;
    };

    let sampler = render_context
        .render_device()
        .create_sampler(&SamplerDescriptor::default());

    let bind_group = render_context.render_device().create_bind_group(
        "flood_seed_bind_group",
        &flood_pipeline.seed_layout,
        &BindGroupEntries::sequential((&voronoi_texture.input().default_view, &sampler)),
    );

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("flood_seed_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &voronoi_texture.output().default_view,
            resolve_target: None,
            ops: Operations::default(),
            depth_slice: None,
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);

    voronoi_texture.flip();
}

pub fn run_flood_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    camera: &ExtractedCamera,
    voronoi_texture: &mut VoronoiTexture,
    step: UVec2,
) {
    let flood_pipeline = world.resource::<FloodPipeline>();

    let mut step = UniformBuffer::from(step);

    step.write_buffer(
        render_context.render_device(),
        world.resource::<RenderQueue>(),
    );

    let (Some(pipeline), Some(step)) = (
        world
            .resource::<PipelineCache>()
            .get_render_pipeline(flood_pipeline.pipeline),
        step.binding(),
    ) else {
        return;
    };

    let sampler = render_context
        .render_device()
        .create_sampler(&SamplerDescriptor::default());

    let bind_group = render_context.render_device().create_bind_group(
        "flood_bind_group",
        &flood_pipeline.layout,
        &BindGroupEntries::sequential((&voronoi_texture.input().default_view, &sampler, step)),
    );

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("flood_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &voronoi_texture.output().default_view,
            resolve_target: None,
            ops: Operations::default(),
            depth_slice: None,
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);

    voronoi_texture.flip();
}
