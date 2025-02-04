use bevy::{
    core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FragmentState, MultisampleState, Operations,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages,
            SpecializedRenderPipeline, TextureFormat, TextureSampleType,
            binding_types::{sampler, texture_2d},
        },
        renderer::{RenderContext, RenderDevice},
        texture::CachedTexture,
        view::ViewTarget,
    },
};

pub const COMPOSITE_SHADER: Handle<Shader> =
    Handle::weak_from_u128(65429060464563916594756042659367504274);

#[derive(Component)]
pub struct ViewCompositePipelineId(pub CachedRenderPipelineId);

#[derive(Resource)]
pub struct CompositePipeline {
    pub layout: BindGroupLayout,
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            layout: world.resource::<RenderDevice>().create_bind_group_layout(
                "composite_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                    ),
                ),
            ),
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct CompositePipelineKey {
    pub hdr: bool,
}

impl SpecializedRenderPipeline for CompositePipeline {
    type Key = CompositePipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        RenderPipelineDescriptor {
            label: Some("composite_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: COMPOSITE_SHADER,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.hdr {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
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
        }
    }
}

pub fn composite_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    pipeline_id: &CachedRenderPipelineId,
    camera: &ExtractedCamera,
    input: &CachedTexture,
    target: &ViewTarget,
) {
    let composite_pipeline = world.resource::<CompositePipeline>();

    let Some(pipeline) = world
        .resource::<PipelineCache>()
        .get_render_pipeline(*pipeline_id)
    else {
        return;
    };

    let post_process = target.post_process_write();
    let sampler = render_context
        .render_device()
        .create_sampler(&SamplerDescriptor::default());

    let bind_group = render_context.render_device().create_bind_group(
        "composite_bind_group",
        &composite_pipeline.layout,
        &BindGroupEntries::sequential((post_process.source, &input.default_view, &sampler)),
    );

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("composite_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &post_process.destination,
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
