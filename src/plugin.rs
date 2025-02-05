use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{entity::EntityHashSet, query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSet,
        batching::no_gpu_preprocessing::batch_and_prepare_binned_render_phase,
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::RenderMesh,
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, ViewBinnedRenderPhases,
        },
        render_resource::{
            PipelineCache, SpecializedMeshPipelines, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::RenderEntity,
        texture::{CachedTexture, TextureCache},
        view::{RenderVisibleEntities, ViewTarget},
    },
    sprite::{Mesh2dPipeline, Mesh2dPipelineKey, RenderMesh2dInstances},
};

use crate::{flood::*, flood_mask::*};

#[derive(Component, ExtractComponent, Clone, Copy, Default)]
pub struct FloodComponent;

pub struct FloodPlugin;
impl Plugin for FloodPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, FLOOD_MASK_SHADER, "flood_mask.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_INIT_SHADER, "flood_init.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_SHADER, "flood.wgsl", Shader::from_wgsl);

        app.add_plugins(ExtractComponentPlugin::<FloodComponent>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<FloodMaskPipeline>>()
            .init_resource::<ViewBinnedRenderPhases<FloodMaskPhase>>()
            .init_resource::<DrawFunctions<FloodMaskPhase>>()
            .add_render_command::<FloodMaskPhase, DrawFloodMesh>()
            .add_systems(ExtractSchedule, extract_camera_phases)
            .add_systems(
                Render,
                (
                    queue_custom_meshes.in_set(RenderSet::QueueMeshes),
                    prepare_flood_textures.in_set(RenderSet::Prepare),
                    batch_and_prepare_binned_render_phase::<FloodMaskPhase, Mesh2dPipeline>
                        .in_set(RenderSet::PrepareResources),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<FloodDrawNode>>(Core2d, FloodDrawPassLabel)
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainOpaquePass,
                    FloodDrawPassLabel,
                    Node2d::MainTransparentPass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<FloodMaskPipeline>()
            .init_resource::<FloodPipeline>();
    }
}

fn extract_camera_phases(
    cameras: Extract<Query<(RenderEntity, &Camera), With<Camera2d>>>,
    mut flood_phases: ResMut<ViewBinnedRenderPhases<FloodMaskPhase>>,
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

fn queue_custom_meshes(
    flood_draw_functions: Res<DrawFunctions<FloodMaskPhase>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<FloodMaskPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    flood_init_pipeline: Res<FloodMaskPipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut custom_render_phases: ResMut<ViewBinnedRenderPhases<FloodMaskPhase>>,
    mut views: Query<(Entity, &RenderVisibleEntities, &Msaa)>,
    has_marker: Query<(), With<FloodComponent>>,
) {
    for (view_entity, visible_entities, msaa) in &mut views {
        let Some(flood_phase) = custom_render_phases.get_mut(&view_entity) else {
            continue;
        };
        let draw_flood_mesh = flood_draw_functions.read().id::<DrawFloodMesh>();

        let view_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples());

        for (render_entity, visible_entity) in visible_entities.iter::<With<Mesh2d>>() {
            if has_marker.get(*render_entity).is_err() {
                continue;
            }

            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &flood_init_pipeline,
                view_key | Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology()),
                &mesh.layout,
            );

            let bin_key = match pipeline_id {
                Ok(id) => FloodMaskPhaseBinKey {
                    pipeline: id,
                    draw_function: draw_flood_mesh,
                    asset_id: mesh_instance.mesh_asset_id.into(),
                },
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            flood_phase.add(
                bin_key,
                (*render_entity, *visible_entity),
                BinnedRenderPhaseType::mesh(mesh_instance.automatic_batching),
            );
        }
    }
}

#[derive(Clone, Component)]
pub struct FloodTextures {
    flip: bool,
    texture_a: CachedTexture,
    texture_b: CachedTexture,
}

impl FloodTextures {
    pub fn input(&self) -> &CachedTexture {
        if self.flip {
            &self.texture_b
        } else {
            &self.texture_a
        }
    }

    pub fn output(&self) -> &CachedTexture {
        if self.flip {
            &self.texture_a
        } else {
            &self.texture_b
        }
    }

    pub fn flip(&mut self) {
        self.flip = !self.flip;
    }
}

fn create_aux_texture(
    view_target: &ViewTarget,
    texture_cache: &mut TextureCache,
    render_device: &RenderDevice,
    label: &'static str,
) -> CachedTexture {
    texture_cache.get(render_device, TextureDescriptor {
        label: Some(label),
        size: view_target.main_texture().size(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn prepare_flood_textures(
    mut commands: Commands,
    view_query: Query<(Entity, &ViewTarget)>,
    flood_init_phases: Res<ViewBinnedRenderPhases<FloodMaskPhase>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, view_target) in &view_query {
        if !flood_init_phases.contains_key(&entity) {
            continue;
        }

        commands.entity(entity).insert(FloodTextures {
            flip: false,
            texture_a: create_aux_texture(
                view_target,
                &mut texture_cache,
                &render_device,
                "flood_texture_a",
            ),
            texture_b: create_aux_texture(
                view_target,
                &mut texture_cache,
                &render_device,
                "flood_texture_b",
            ),
        });
    }
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct FloodDrawPassLabel;

#[derive(Default)]
struct FloodDrawNode;
impl ViewNode for FloodDrawNode {
    type ViewQuery = (Read<ExtractedCamera>, Read<ViewTarget>, Read<FloodTextures>);

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, target, flood_textures): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let mut flood_textures = flood_textures.clone();

        flood_mask_pass(
            world,
            render_context,
            &view_entity,
            flood_textures.output(),
            camera,
        );
        flood_textures.flip();

        flood_init_pass(
            world,
            render_context,
            camera,
            flood_textures.input(),
            flood_textures.output(),
        );
        flood_textures.flip();

        let mut step = target
            .main_texture()
            .width()
            .max(target.main_texture().height())
            / 2;

        while step >= 1 {
            flood_pass(
                world,
                render_context,
                camera,
                flood_textures.input(),
                flood_textures.output(),
                step,
            );
            flood_textures.flip();
            step /= 2;
        }

        Ok(())
    }
}
