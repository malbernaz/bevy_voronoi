use std::ops::Range;

use bevy::{
    asset::{UntypedAssetId, load_internal_asset},
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{entity::EntityHashSet, query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSet,
        batching::no_gpu_preprocessing::batch_and_prepare_binned_render_phase,
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{MeshVertexBufferLayoutRef, RenderMesh},
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{
            AddRenderCommand, BinnedPhaseItem, BinnedRenderPhaseType,
            CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions, PhaseItem,
            PhaseItemExtraIndex, SetItemPipeline, ViewBinnedRenderPhases,
        },
        render_resource::{
            CachedRenderPipelineId, FragmentState, PipelineCache, RenderPassDescriptor,
            RenderPipelineDescriptor, SpecializedMeshPipeline, SpecializedMeshPipelineError,
            SpecializedMeshPipelines,
        },
        renderer::RenderContext,
        sync_world::{MainEntity, RenderEntity},
        view::{RenderVisibleEntities, ViewTarget},
    },
    sprite::{
        DrawMesh2d, Mesh2dPipeline, Mesh2dPipelineKey, RenderMesh2dInstances, SetMesh2dBindGroup,
        SetMesh2dViewBindGroup,
    },
};

const FLOOD_INIT_SHADER: Handle<Shader> = Handle::weak_from_u128(32132157492758);

#[derive(Component, ExtractComponent, Clone, Copy, Default)]
pub struct FloodComponent;

pub struct FloodPlugin;
impl Plugin for FloodPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            FLOOD_INIT_SHADER,
            "shaders/flood_init.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins(ExtractComponentPlugin::<FloodComponent>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<FloodPipeline>>()
            .init_resource::<ViewBinnedRenderPhases<FloodPhase>>()
            .init_resource::<DrawFunctions<FloodPhase>>()
            .add_render_command::<FloodPhase, DrawFloodMesh>()
            .add_systems(ExtractSchedule, extract_camera_phases)
            .add_systems(
                Render,
                (
                    queue_custom_meshes.in_set(RenderSet::QueueMeshes),
                    batch_and_prepare_binned_render_phase::<FloodPhase, Mesh2dPipeline>
                        .in_set(RenderSet::PrepareResources),
                ),
            );

        render_app
            .add_render_graph_node::<ViewNodeRunner<FloodDrawNode>>(Core2d, FloodDrawPassLabel)
            .add_render_graph_edges(Core2d, (Node2d::MainOpaquePass, FloodDrawPassLabel));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<FloodPipeline>();
    }
}

#[derive(Resource)]
struct FloodPipeline {
    mesh_pipeline: Mesh2dPipeline,
}

impl FromWorld for FloodPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: Mesh2dPipeline::from_world(world),
        }
    }
}

impl SpecializedMeshPipeline for FloodPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let descriptor = self.mesh_pipeline.specialize(key, &layout)?;

        Ok(RenderPipelineDescriptor {
            fragment: Some(FragmentState {
                shader: FLOOD_INIT_SHADER,
                entry_point: "fragment".into(),
                ..descriptor
                    .fragment
                    .expect("mesh2d pipeline should have a fragment state")
            }),
            depth_stencil: None,
            ..descriptor
        })
    }
}

type DrawFloodMesh = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    DrawMesh2d,
);

pub struct FloodPhase {
    pub key: FloodPhaseBinKey,
    pub representative_entity: (Entity, MainEntity),
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

/// Data that must be identical in order to batch phase items together.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloodPhaseBinKey {
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub asset_id: UntypedAssetId,
}

impl PhaseItem for FloodPhase {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.key.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for FloodPhase {
    type BinKey = FloodPhaseBinKey;

    fn new(
        key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        FloodPhase {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for FloodPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

fn extract_camera_phases(
    cameras: Extract<Query<(RenderEntity, &Camera), With<Camera2d>>>,
    mut flood_phases: ResMut<ViewBinnedRenderPhases<FloodPhase>>,
    mut live_entities: Local<EntityHashSet>,
) {
    live_entities.clear();

    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }
        flood_phases.insert_or_clear(entity);
        live_entities.insert(entity);
    }

    // Clear out all dead views
    flood_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

#[allow(clippy::too_many_arguments)]
fn queue_custom_meshes(
    flood_draw_functions: Res<DrawFunctions<FloodPhase>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<FloodPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    flood_init_pipeline: Res<FloodPipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut custom_render_phases: ResMut<ViewBinnedRenderPhases<FloodPhase>>,
    mut views: Query<(Entity, &RenderVisibleEntities, &Msaa)>,
    has_marker: Query<(), With<FloodComponent>>,
) {
    for (view_entity, visible_entities, msaa) in &mut views {
        let Some(flood_phase) = custom_render_phases.get_mut(&view_entity) else {
            continue;
        };
        let draw_flood_mesh = flood_draw_functions.read().id::<DrawFloodMesh>();

        let view_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples());

        // Since our phase can work on any 2d mesh we can reuse the default mesh 2d filter
        for (render_entity, visible_entity) in visible_entities.iter::<With<Mesh2d>>() {
            // We only want meshes with `FloodComponent` to be queued to our phase
            if has_marker.get(*render_entity).is_err() {
                continue;
            }

            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mesh_key =
                view_key | Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &flood_init_pipeline,
                mesh_key,
                &mesh.layout,
            );

            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            let bin_key = FloodPhaseBinKey {
                pipeline: pipeline_id,
                draw_function: draw_flood_mesh,
                asset_id: mesh_instance.mesh_asset_id.into(),
            };

            flood_phase.add(
                bin_key,
                (*render_entity, *visible_entity),
                BinnedRenderPhaseType::mesh(mesh_instance.automatic_batching),
            );
        }
    }
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct FloodDrawPassLabel;

#[derive(Default)]
struct FloodDrawNode;
impl ViewNode for FloodDrawNode {
    type ViewQuery = (Read<ExtractedCamera>, Read<ViewTarget>);

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, target): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(flood_phases) = world.get_resource::<ViewBinnedRenderPhases<FloodPhase>>() else {
            return Ok(());
        };

        let color_attachments = [Some(target.get_color_attachment())];

        let view_entity = graph.view_entity();

        let Some(flood_phase) = flood_phases.get(&view_entity) else {
            return Ok(());
        };

        let desc = RenderPassDescriptor {
            label: Some("flood_init_pass"),
            color_attachments: &color_attachments,
            ..default()
        };

        let mut pass = render_context.begin_tracked_render_pass(desc);

        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }

        if !flood_phase.is_empty() {
            if let Err(err) = flood_phase.render(&mut pass, world, view_entity) {
                error!("Error encountered while rendering the custom phase {err:?}");
            }
        }

        Ok(())
    }
}
