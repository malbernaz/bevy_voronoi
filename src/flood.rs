use bevy::{
    asset::weak_handle,
    core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state,
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
        texture::CachedTexture,
    },
};

pub const FLOOD_SEED_SHADER: Handle<Shader> = weak_handle!("75d5b3e1-ce23-4fcf-8a24-91e9bd6eeed8");
pub const FLOOD_SHADER: Handle<Shader> = weak_handle!("7ac1a62e-dd67-404f-a5f6-3633b8bdcbef");

#[derive(Resource)]
pub struct FloodPipeline {
    pub seed_layout: BindGroupLayout,
    pub seed_pipeline: CachedRenderPipelineId,
    pub layout: BindGroupLayout,
    pub pipeline: CachedRenderPipelineId,
}

impl FromWorld for FloodPipeline {
    fn from_world(world: &mut World) -> Self {
        let seed_layout = world.resource::<RenderDevice>().create_bind_group_layout(
            "flood_seed_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        );

        let seed_pipeline =
            world
                .resource::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("flood_seed_pipeline".into()),
                    layout: vec![seed_layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: FLOOD_SEED_SHADER,
                        shader_defs: vec![],
                        entry_point: "fragment".into(),
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

        let layout = world.resource::<RenderDevice>().create_bind_group_layout(
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

        let pipeline =
            world
                .resource::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("flood_pipeline".into()),
                    layout: vec![layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: FLOOD_SHADER,
                        shader_defs: vec![],
                        entry_point: "fragment".into(),
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

        Self {
            seed_pipeline,
            seed_layout,
            layout,
            pipeline,
        }
    }
}

pub fn run_flood_seed_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    camera: &ExtractedCamera,
    input: &CachedTexture,
    output: &CachedTexture,
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
        &BindGroupEntries::sequential((&input.default_view, &sampler)),
    );

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("flood_seed_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output.default_view,
            resolve_target: None,
            ops: Operations::default(),
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

pub fn run_flood_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    camera: &ExtractedCamera,
    input: &CachedTexture,
    output: &CachedTexture,
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
        &BindGroupEntries::sequential((&input.default_view, &sampler, step)),
    );

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("flood_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output.default_view,
            resolve_target: None,
            ops: Operations::default(),
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}
