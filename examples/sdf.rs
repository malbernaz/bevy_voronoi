use bevy::{
    core_pipeline::{
        core_2d::graph::{Core2d, Node2d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    },
    ecs::{query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    render::{
        Render, RenderApp, RenderSet,
        camera::ExtractedCamera,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, FragmentState, MultisampleState, Operations,
            PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, SamplerBindingType, SamplerDescriptor, ShaderStages,
            SpecializedRenderPipeline, SpecializedRenderPipelines, TextureFormat,
            TextureSampleType,
            binding_types::{sampler, texture_2d},
        },
        renderer::{RenderContext, RenderDevice},
        view::{ExtractedView, ViewTarget},
    },
};
use bevy_voronoi::prelude::*;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::NONE))
        .add_plugins((DefaultPlugins, SdfPlugin))
        .add_systems(Startup, setup)
        .run();
}

const X_EXTENT: f32 = 800.;

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands.spawn(Camera2d);

    let shapes = [
        meshes.add(Circle::new(50.0)),
        meshes.add(Annulus::new(25.0, 50.0)),
        meshes.add(Capsule2d::new(25.0, 50.0)),
        meshes.add(Rhombus::new(75.0, 100.0)),
        meshes.add(Rectangle::new(50.0, 100.0)),
        meshes.add(RegularPolygon::new(50.0, 6)),
        meshes.add(Triangle2d::new(
            Vec2::Y * 50.0,
            Vec2::new(-50.0, -50.0),
            Vec2::new(50.0, -50.0),
        )),
    ];
    let num_shapes = shapes.len();

    for (i, shape) in shapes.into_iter().enumerate() {
        commands.spawn((
            Mesh2d(shape),
            VoronoiMaterial::default(),
            Transform::from_xyz(
                -X_EXTENT / 2. + i as f32 / (num_shapes - 1) as f32 * X_EXTENT,
                0.0,
                0.0,
            ),
        ));
    }
}

struct SdfPlugin;

impl Plugin for SdfPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(Voronoi2dPlugin);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<CompositePipeline>>()
            .add_systems(
                Render,
                prepare_composite_pipeline.in_set(RenderSet::Prepare),
            )
            .add_render_graph_node::<ViewNodeRunner<CompositeNode>>(Core2d, CompositePassLabel)
            .add_render_graph_edges(Core2d, (Node2d::EndMainPass, CompositePassLabel));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<CompositePipeline>();
    }
}

fn prepare_composite_pipeline(
    mut commands: Commands,
    view_query: Query<(Entity, &ExtractedView)>,
    mut composite_pipelines: ResMut<SpecializedRenderPipelines<CompositePipeline>>,
    composite_pipeline: Res<CompositePipeline>,
    mut pipeline_cache: ResMut<PipelineCache>,
) {
    for (entity, view) in &view_query {
        let composite_pipeline_id = composite_pipelines.specialize(
            &mut pipeline_cache,
            &composite_pipeline,
            CompositePipelineKey { hdr: view.hdr },
        );

        commands
            .entity(entity)
            .insert(ViewCompositePipelineId(composite_pipeline_id));
    }
}

#[derive(Component)]
pub struct ViewCompositePipelineId(pub CachedRenderPipelineId);

#[derive(Resource)]
pub struct CompositePipeline {
    pub shader: Handle<Shader>,
    pub layout: BindGroupLayout,
}

impl FromWorld for CompositePipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            shader: world.resource::<AssetServer>().load("composite.wgsl"),
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
            push_constant_ranges: vec![],
            vertex: fullscreen_shader_vertex_state(),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
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
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct CompositePassLabel;

#[derive(Default)]
struct CompositeNode;
impl ViewNode for CompositeNode {
    type ViewQuery = (
        Read<ExtractedCamera>,
        Read<ViewTarget>,
        Read<VoronoiTexture>,
        Read<ViewCompositePipelineId>,
    );

    fn run<'w>(
        &self,
        _: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, target, flood_textures, composite_pipeline_id): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let composite_pipeline = world.resource::<CompositePipeline>();

        let Some(pipeline) = world
            .resource::<PipelineCache>()
            .get_render_pipeline(composite_pipeline_id.0)
        else {
            return Ok(());
        };

        let post_process = target.post_process_write();
        let sampler = render_context
            .render_device()
            .create_sampler(&SamplerDescriptor::default());

        let bind_group = render_context.render_device().create_bind_group(
            "composite_bind_group",
            &composite_pipeline.layout,
            &BindGroupEntries::sequential((
                post_process.source,
                &flood_textures.input().default_view,
                &sampler,
            )),
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

        Ok(())
    }
}
