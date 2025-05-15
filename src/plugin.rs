use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_2d::{
        graph::{Core2d, Node2d},
        BatchSetKey2d,
    },
    ecs::{
        query::QueryItem,
        system::{lifetimeless::Read, SystemChangeTick},
    },
    platform::collections::HashSet,
    prelude::*,
    render::{
        batching::{
            gpu_preprocessing::GpuPreprocessingMode,
            no_gpu_preprocessing::batch_and_prepare_binned_render_phase,
        },
        camera::{extract_cameras, ExtractedCamera},
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::RenderMesh,
        render_asset::{prepare_assets, RenderAssets},
        render_graph::{
            NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{
            AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex,
            ViewBinnedRenderPhases,
        },
        render_resource::{
            PipelineCache, SpecializedMeshPipelines, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::{MainEntity, MainEntityHashMap},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewTarget},
        Extract, Render, RenderApp, RenderSet,
    },
    sprite::{
        EntitiesNeedingSpecialization, EntitySpecializationTicks, Mesh2dPipeline,
        Mesh2dPipelineKey, RenderMesh2dInstances, SpecializedMaterial2dPipelineCache, ViewKeyCache,
        ViewSpecializationTicks,
    },
    utils::Parallel,
};

use crate::{flood::*, mask::*};

pub struct Voronoi2dPlugin;
impl Plugin for Voronoi2dPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, MASK_SHADER, "mask.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_SEED_SHADER, "flood_seed.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_SHADER, "flood.wgsl", Shader::from_wgsl);

        app.add_plugins(ExtractComponentPlugin::<VoronoiMaterial>::default())
            .init_resource::<EntitiesNeedingSpecialization<VoronoiMaterial>>()
            .add_systems(PostUpdate, check_entities_needing_specialization);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<MaskPipeline>>()
            .init_resource::<EntitySpecializationTicks<VoronoiMaterial>>()
            .init_resource::<SpecializedMaterial2dPipelineCache<VoronoiMaterial>>()
            .init_resource::<ViewBinnedRenderPhases<MaskPhase>>()
            .init_resource::<RenderVoronoiMaterials>()
            .init_resource::<MaskMaterialBindGroups>()
            .init_resource::<DrawFunctions<MaskPhase>>()
            .add_render_command::<MaskPhase, DrawMaskMesh>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_camera_phases,
                    extract_entities_needs_specialization.after(extract_cameras),
                    extract_flood_materials,
                ),
            )
            .add_systems(
                Render,
                (
                    specialize_mask_meshes
                        .in_set(RenderSet::PrepareMeshes)
                        .after(prepare_assets::<RenderMesh>),
                    queue_mask_meshes.in_set(RenderSet::QueueMeshes),
                    prepare_flood_textures.in_set(RenderSet::Prepare),
                    batch_and_prepare_binned_render_phase::<MaskPhase, Mesh2dPipeline>
                        .in_set(RenderSet::PrepareResources),
                    prepare_mask_material_bind_groups.in_set(RenderSet::PrepareBindGroups),
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
            .init_resource::<MaskPipeline>()
            .init_resource::<FloodPipeline>();
    }
}

fn check_entities_needing_specialization(
    needs_specialization: Query<
        Entity,
        (
            Or<(
                Changed<Mesh2d>,
                AssetChanged<Mesh2d>,
                Changed<VoronoiMaterial>,
            )>,
            With<VoronoiMaterial>,
        ),
    >,
    mut par_local: Local<Parallel<Vec<Entity>>>,
    mut entities_needing_specialization: ResMut<EntitiesNeedingSpecialization<VoronoiMaterial>>,
) {
    entities_needing_specialization.clear();

    needs_specialization
        .par_iter()
        .for_each(|entity| par_local.borrow_local_mut().push(entity));

    par_local.drain_into(&mut entities_needing_specialization);
}

#[derive(Component, ExtractComponent, Clone, Default)]
pub struct VoronoiMaterial {
    pub alpha_mask: Handle<Image>,
}

impl VoronoiMaterial {
    pub fn new(alpha_mask: Handle<Image>) -> Self {
        Self { alpha_mask }
    }
}

impl From<VoronoiMaterial> for AssetId<Image> {
    fn from(material: VoronoiMaterial) -> Self {
        material.alpha_mask.id()
    }
}

impl From<&VoronoiMaterial> for AssetId<Image> {
    fn from(material: &VoronoiMaterial) -> Self {
        material.alpha_mask.id()
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct RenderVoronoiMaterials(MainEntityHashMap<AssetId<Image>>);

fn extract_camera_phases(
    cameras: Extract<Query<(Entity, &Camera), With<Camera2d>>>,
    mut flood_phases: ResMut<ViewBinnedRenderPhases<MaskPhase>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();

    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        let retained_view_entity = RetainedViewEntity::new(entity.into(), None, 0);

        flood_phases.prepare_for_new_frame(retained_view_entity, GpuPreprocessingMode::None);
        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views
    flood_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

fn extract_entities_needs_specialization(
    entities_needing_specialization: Extract<Res<EntitiesNeedingSpecialization<VoronoiMaterial>>>,
    mut entity_specialization_ticks: ResMut<EntitySpecializationTicks<VoronoiMaterial>>,
    mut removed_mesh_material_components: Extract<RemovedComponents<VoronoiMaterial>>,
    mut specialized_material2d_pipeline_cache: ResMut<
        SpecializedMaterial2dPipelineCache<VoronoiMaterial>,
    >,
    views: Query<&MainEntity, With<ExtractedView>>,
    ticks: SystemChangeTick,
) {
    for entity in removed_mesh_material_components.read() {
        entity_specialization_ticks.remove(&MainEntity::from(entity));
        for view in views {
            if let Some(cache) = specialized_material2d_pipeline_cache.get_mut(view) {
                cache.remove(&MainEntity::from(entity));
            }
        }
    }
    for entity in entities_needing_specialization.iter() {
        // Update the entity's specialization tick with this run's tick
        entity_specialization_ticks.insert((*entity).into(), ticks.this_run());
    }
}

fn extract_flood_materials(
    mut material_instances: ResMut<RenderVoronoiMaterials>,
    query: Extract<Query<(Entity, &ViewVisibility, &VoronoiMaterial), With<Mesh2d>>>,
) {
    material_instances.clear();

    for (entity, view_visibility, material) in &query {
        if view_visibility.get() {
            material_instances.insert(entity.into(), material.into());
        }
    }
}

fn specialize_mask_meshes(
    render_material_instances: Res<RenderVoronoiMaterials>,
    views: Query<(&MainEntity, &ExtractedView, &RenderVisibleEntities)>,
    mask_render_phases: ResMut<ViewBinnedRenderPhases<MaskPhase>>,
    view_key_cache: Res<ViewKeyCache>,
    view_specialization_ticks: Res<ViewSpecializationTicks>,
    mut specialized_material_pipeline_cache: ResMut<
        SpecializedMaterial2dPipelineCache<VoronoiMaterial>,
    >,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    entity_specialization_ticks: Res<EntitySpecializationTicks<VoronoiMaterial>>,
    ticks: SystemChangeTick,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    pipeline_cache: Res<PipelineCache>,
    mask_pipeline: Res<MaskPipeline>,
    mut mask_pipelines: ResMut<SpecializedMeshPipelines<MaskPipeline>>,
) {
    if render_material_instances.is_empty() {
        return;
    }

    for (view_entity, view, visible_entities) in &views {
        if !mask_render_phases.contains_key(&view.retained_view_entity) {
            continue;
        }

        let Some(view_key) = view_key_cache.get(view_entity) else {
            continue;
        };

        let view_tick = view_specialization_ticks.get(view_entity).unwrap();
        let view_specialized_material_pipeline_cache = specialized_material_pipeline_cache
            .entry(*view_entity)
            .or_default();

        for (_, visible_entity) in visible_entities.iter::<Mesh2d>() {
            if render_material_instances.get(visible_entity).is_none() {
                continue;
            };

            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };

            let entity_tick = entity_specialization_ticks.get(visible_entity).unwrap();

            let last_specialized_tick = view_specialized_material_pipeline_cache
                .get(visible_entity)
                .map(|(tick, _)| *tick);
            let needs_specialization = last_specialized_tick.is_none_or(|tick| {
                view_tick.is_newer_than(tick, ticks.this_run())
                    || entity_tick.is_newer_than(tick, ticks.this_run())
            });
            if !needs_specialization {
                continue;
            }

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let pipeline_id = mask_pipelines.specialize(
                &pipeline_cache,
                &mask_pipeline,
                *view_key | Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology()),
                &mesh.layout,
            );

            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            view_specialized_material_pipeline_cache
                .insert(*visible_entity, (ticks.this_run(), pipeline_id));
        }
    }
}

fn queue_mask_meshes(
    flood_draw_functions: Res<DrawFunctions<MaskPhase>>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut mask_render_phase: ResMut<ViewBinnedRenderPhases<MaskPhase>>,
    views: Query<(&MainEntity, &ExtractedView, &RenderVisibleEntities)>,
    specialized_material_pipeline_cache: ResMut<
        SpecializedMaterial2dPipelineCache<VoronoiMaterial>,
    >,
    render_material_instances: Res<RenderVoronoiMaterials>,
) {
    if render_material_instances.is_empty() {
        return;
    }

    for (view_entity, view, visible_entities) in &views {
        let Some(view_specialized_material_pipeline_cache) =
            specialized_material_pipeline_cache.get(view_entity)
        else {
            continue;
        };

        let Some(flood_phase) = mask_render_phase.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let draw_flood_mesh = flood_draw_functions.read().id::<DrawMaskMesh>();

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh2d>() {
            let Some((current_change_tick, pipeline_id)) = view_specialized_material_pipeline_cache
                .get(visible_entity)
                .map(|(current_change_tick, pipeline_id)| (*current_change_tick, *pipeline_id))
            else {
                continue;
            };

            if flood_phase.validate_cached_entity(*visible_entity, current_change_tick) {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            flood_phase.add(
                BatchSetKey2d {
                    indexed: mesh.indexed(),
                },
                MaskPhaseBinKey {
                    pipeline: pipeline_id,
                    draw_function: draw_flood_mesh,
                    mesh_id: mesh_instance.mesh_asset_id.into(),
                },
                (*render_entity, *visible_entity),
                InputUniformIndex::default(),
                if mesh_instance.automatic_batching {
                    BinnedRenderPhaseType::BatchableMesh
                } else {
                    BinnedRenderPhaseType::UnbatchableMesh
                },
                current_change_tick,
            );
        }
    }
}

#[derive(Clone, Component)]
pub struct VoronoiTexture {
    flip: bool,
    texture_a: CachedTexture,
    texture_b: CachedTexture,
}

impl VoronoiTexture {
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
    texture_cache.get(
        render_device,
        TextureDescriptor {
            label: Some(label),
            size: view_target.main_texture().size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

fn prepare_flood_textures(
    mut commands: Commands,
    view_query: Query<(Entity, &ViewTarget, &ExtractedView)>,
    flood_mask_phases: Res<ViewBinnedRenderPhases<MaskPhase>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, view_target, extracted_view) in &view_query {
        if !flood_mask_phases.contains_key(&extracted_view.retained_view_entity) {
            continue;
        }

        commands.entity(entity).insert(VoronoiTexture {
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
    type ViewQuery = (
        Read<ExtractedCamera>,
        Read<ExtractedView>,
        Read<ViewTarget>,
        Read<VoronoiTexture>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view, target, voronoi_textures): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let mut voronoi_textures = voronoi_textures.clone();

        run_mask_pass(
            world,
            render_context,
            &view.retained_view_entity,
            &view_entity,
            voronoi_textures.output(),
            camera,
        );
        voronoi_textures.flip();

        run_flood_seed_pass(
            world,
            render_context,
            camera,
            voronoi_textures.input(),
            voronoi_textures.output(),
        );
        voronoi_textures.flip();

        let width = target.main_texture().width();
        let height = target.main_texture().height();
        let max_dim = width.max(height);
        let mut step = max_dim / 2;

        while step >= 1 {
            let x_step = (step * width) / max_dim;
            let y_step = (step * height) / max_dim;

            run_flood_pass(
                world,
                render_context,
                camera,
                voronoi_textures.input(),
                voronoi_textures.output(),
                UVec2::new(x_step.max(1), y_step.max(1)),
            );

            voronoi_textures.flip();
            step /= 2;
        }

        // Addicional pass with step = 1 to improve accuracy
        run_flood_pass(
            world,
            render_context,
            camera,
            voronoi_textures.input(),
            voronoi_textures.output(),
            UVec2::new(1, 1),
        );
        voronoi_textures.flip();

        Ok(())
    }
}
