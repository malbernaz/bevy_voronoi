use std::ops::Range;

use bevy::{
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{
        entity::EntityHashSet,
        query::QueryItem,
        system::{
            SystemParamItem,
            lifetimeless::{Read, SRes},
        },
    },
    math::FloatOrd,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSet,
        batching::{
            GetBatchData, GetFullBatchData,
            gpu_preprocessing::{IndirectParameters, IndirectParametersBuffer},
        },
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{
            MeshVertexBufferLayoutRef, RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator,
        },
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{
            AddRenderCommand, CachedRenderPipelinePhaseItem, DrawFunctionId, DrawFunctions,
            PhaseItem, PhaseItemExtraIndex, SetItemPipeline, SortedPhaseItem,
            SortedRenderPhasePlugin, ViewSortedRenderPhases,
        },
        render_resource::{
            CachedRenderPipelineId, FragmentState, PipelineCache, RenderPassDescriptor,
            RenderPipelineDescriptor, SpecializedMeshPipeline, SpecializedMeshPipelineError,
            SpecializedMeshPipelines,
        },
        renderer::RenderContext,
        sync_world::{MainEntity, RenderEntity},
        view::{ExtractedView, RenderVisibleEntities, ViewTarget},
    },
    sprite::{
        DrawMesh2d, Mesh2dPipeline, Mesh2dPipelineKey, Mesh2dUniform, RenderMesh2dInstances,
        SetMesh2dBindGroup, SetMesh2dViewBindGroup,
    },
};
use nonmax::NonMaxU32;

const SHADER_ASSET_PATH: &str = "shaders/custom_stencil.wgsl";

#[derive(Component, ExtractComponent, Clone, Copy, Default)]
pub struct FloodComponent;

pub struct FloodPlugin;
impl Plugin for FloodPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            ExtractComponentPlugin::<FloodComponent>::default(),
            SortedRenderPhasePlugin::<FloodPhase, FloodPipeline>::default(),
        ));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<FloodPipeline>>()
            .init_resource::<DrawFunctions<FloodPhase>>()
            .add_render_command::<FloodPhase, DrawFloodMesh>()
            .add_systems(ExtractSchedule, extract_camera_phases)
            .add_systems(
                Render,
                (queue_custom_meshes.in_set(RenderSet::QueueMeshes),),
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
    shader_handle: Handle<Shader>,
}

impl FromWorld for FloodPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: Mesh2dPipeline::from_world(world),
            shader_handle: world.resource::<AssetServer>().load(SHADER_ASSET_PATH),
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
                shader: self.shader_handle.clone(),
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

impl GetBatchData for FloodPipeline {
    type Param = (
        SRes<RenderMesh2dInstances>,
        SRes<RenderAssets<RenderMesh>>,
        SRes<MeshAllocator>,
    );
    type CompareData = AssetId<Mesh>;
    type BufferData = Mesh2dUniform;

    fn get_batch_data(
        (mesh_instances, _, _): &SystemParamItem<Self::Param>,
        (_entity, main_entity): (Entity, MainEntity),
    ) -> Option<(Self::BufferData, Option<Self::CompareData>)> {
        let mesh_instance = mesh_instances.get(&main_entity)?;
        Some((
            (&mesh_instance.transforms).into(),
            mesh_instance
                .automatic_batching
                .then_some(mesh_instance.mesh_asset_id),
        ))
    }
}

// copied from mesh2d
impl GetFullBatchData for FloodPipeline {
    type BufferInputData = ();

    fn get_binned_batch_data(
        (mesh_instances, _, _): &SystemParamItem<Self::Param>,
        (_entity, main_entity): (Entity, MainEntity),
    ) -> Option<Self::BufferData> {
        let mesh_instance = mesh_instances.get(&main_entity)?;
        Some((&mesh_instance.transforms).into())
    }

    fn get_index_and_compare_data(
        _: &SystemParamItem<Self::Param>,
        _query_item: (Entity, MainEntity),
    ) -> Option<(NonMaxU32, Option<Self::CompareData>)> {
        error!(
            "`get_index_and_compare_data` is only intended for GPU mesh uniform building, \
            but this is not yet implemented for 2d meshes"
        );
        None
    }

    fn get_binned_index(
        _: &SystemParamItem<Self::Param>,
        _query_item: (Entity, MainEntity),
    ) -> Option<NonMaxU32> {
        error!(
            "`get_binned_index` is only intended for GPU mesh uniform building, \
            but this is not yet implemented for 2d meshes"
        );
        None
    }

    fn get_batch_indirect_parameters_index(
        (mesh_instances, meshes, mesh_allocator): &SystemParamItem<Self::Param>,
        indirect_parameters_buffer: &mut IndirectParametersBuffer,
        (_entity, main_entity): (Entity, MainEntity),
        instance_index: u32,
    ) -> Option<NonMaxU32> {
        let mesh_instance = mesh_instances.get(&main_entity)?;
        let mesh = meshes.get(mesh_instance.mesh_asset_id)?;
        let vertex_buffer_slice = mesh_allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id)?;

        // Note that `IndirectParameters` covers both of these structures, even
        // though they actually have distinct layouts. See the comment above that
        // type for more information.
        let indirect_parameters = match mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                count: index_count, ..
            } => {
                let index_buffer_slice =
                    mesh_allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)?;
                IndirectParameters {
                    vertex_or_index_count: index_count,
                    instance_count: 0,
                    first_vertex_or_first_index: index_buffer_slice.range.start,
                    base_vertex_or_first_instance: vertex_buffer_slice.range.start,
                    first_instance: instance_index,
                }
            }
            RenderMeshBufferInfo::NonIndexed => IndirectParameters {
                vertex_or_index_count: mesh.vertex_count,
                instance_count: 0,
                first_vertex_or_first_index: vertex_buffer_slice.range.start,
                base_vertex_or_first_instance: instance_index,
                first_instance: instance_index,
            },
        };

        (indirect_parameters_buffer.push(indirect_parameters) as u32)
            .try_into()
            .ok()
    }
}

type DrawFloodMesh = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    DrawMesh2d,
);

struct FloodPhase {
    pub sort_key: FloatOrd,
    pub entity: (Entity, MainEntity),
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

// For more information about writing a phase item, please look at the custom_phase_item example
impl PhaseItem for FloodPhase {
    #[inline]
    fn entity(&self) -> Entity {
        self.entity.0
    }

    #[inline]
    fn main_entity(&self) -> MainEntity {
        self.entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.draw_function
    }

    #[inline]
    fn batch_range(&self) -> &Range<u32> {
        &self.batch_range
    }

    #[inline]
    fn batch_range_mut(&mut self) -> &mut Range<u32> {
        &mut self.batch_range
    }

    #[inline]
    fn extra_index(&self) -> PhaseItemExtraIndex {
        self.extra_index.clone()
    }

    #[inline]
    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl SortedPhaseItem for FloodPhase {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        // bevy normally uses radsort instead of the std slice::sort_by_key
        // radsort is a stable radix sort that performed better than `slice::sort_by_key` or `slice::sort_unstable_by_key`.
        // Siclip_positiopnnce it is not re-exported by bevy, we just use the std sort for the purpose of the example
        items.sort_by_key(SortedPhaseItem::sort_key);
    }
}

impl CachedRenderPipelinePhaseItem for FloodPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

// When defining a custom phase, we need to extract it from the main world and add it to a resource
// that will be used by the render world. We need to give that resource all views that will use
// that phase
fn extract_camera_phases(
    mut custom_phases: ResMut<ViewSortedRenderPhases<FloodPhase>>,
    cameras: Extract<Query<(RenderEntity, &Camera), With<Camera2d>>>,
    mut live_entities: Local<EntityHashSet>,
) {
    live_entities.clear();
    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }
        custom_phases.insert_or_clear(entity);
        live_entities.insert(entity);
    }
    // Clear out all dead views.
    custom_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

// This is a very important step when writing a custom phase.
//
// This system determines which meshes will be added to the phase.
#[allow(clippy::too_many_arguments)]
fn queue_custom_meshes(
    custom_draw_functions: Res<DrawFunctions<FloodPhase>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<FloodPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    custom_draw_pipeline: Res<FloodPipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut custom_render_phases: ResMut<ViewSortedRenderPhases<FloodPhase>>,
    mut views: Query<(Entity, &ExtractedView, &RenderVisibleEntities, &Msaa)>,
    has_marker: Query<(), With<FloodComponent>>,
) {
    for (view_entity, view, visible_entities, msaa) in &mut views {
        let Some(custom_phase) = custom_render_phases.get_mut(&view_entity) else {
            continue;
        };
        let draw_custom = custom_draw_functions.read().id::<DrawFloodMesh>();

        // Create the key based on the view.
        // In this case we only care about MSAA and HDR
        let view_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        let e = visible_entities.iter::<With<Mesh2d>>();

        // Since our phase can work on any 3d mesh we can reuse the default mesh 3d filter
        for (render_entity, visible_entity) in e {
            // We only want meshes with the marker component to be queued to our phase.
            if has_marker.get(*render_entity).is_err() {
                continue;
            }

            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            // Specialize the key for the current mesh entity
            // For this example we only specialize based on the mesh topology
            // but you could have more complex keys and that's where you'd need to create those keys
            let mut mesh_key = view_key;
            mesh_key |= Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &custom_draw_pipeline,
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
            let mesh_z = mesh_instance.transforms.world_from_local.translation.z;
            // At this point we have all the data we need to create a phase item and add it to our
            // phase
            custom_phase.add(FloodPhase {
                // Sort the data based on the distance to the view
                sort_key: FloatOrd(mesh_z),
                entity: (*render_entity, *visible_entity),
                pipeline: pipeline_id,
                draw_function: draw_custom,
                // Sorted phase items aren't batched
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::NONE,
            });
        }
    }
}

// Render label used to order our render graph node that will render our phase
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
        let Some(flood_phases) = world.get_resource::<ViewSortedRenderPhases<FloodPhase>>() else {
            return Ok(());
        };
        let color_attachments = [Some(target.get_color_attachment())];
        let view_entity = graph.view_entity();
        let Some(flood_phase) = flood_phases.get(&view_entity) else {
            return Ok(());
        };
        let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("flood_init_pass"),
            color_attachments: &color_attachments,
            ..default()
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            pass.set_camera_viewport(viewport);
        }
        if !flood_phase.items.is_empty() {
            if let Err(err) = flood_phase.render(&mut pass, world, view_entity) {
                error!("Error encountered while rendering the custom phase {err:?}");
            }
        }

        // This will generate a task to generate the command buffer in parallel
        //render_context.add_command_buffer_generation_task(move |render_device| {
        //    // Command encoder setup
        //    let mut command_encoder =
        //        render_device.create_command_encoder(&CommandEncoderDescriptor {
        //            label: Some("flood pass encoder"),
        //        });
        //
        //    // Render pass setup
        //    let render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
        //        label: Some("flood pass"),
        //        color_attachments: &color_attachments,
        //        // We don't bind any depth buffer for this pass
        //        depth_stencil_attachment: None,
        //        timestamp_writes: None,
        //        occlusion_query_set: None,
        //    });
        //    let mut render_pass = TrackedRenderPass::new(&render_device, render_pass);
        //    let pass_span = diagnostics.pass_span(&mut render_pass, "custom_pass");
        //
        //    if let Some(viewport) = camera.viewport.as_ref() {
        //        render_pass.set_camera_viewport(viewport);
        //    }
        //
        //    // Render the phase
        //    if !flood_phase.items.is_empty() {
        //        if let Err(err) = flood_phase.render(&mut render_pass, world, view_entity) {
        //            error!("Error encountered while rendering the custom phase {err:?}");
        //        }
        //    }
        //
        //    pass_span.end(&mut render_pass);
        //    drop(render_pass);
        //    command_encoder.finish()
        //});

        Ok(())
    }
}
