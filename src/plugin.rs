use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{entity::EntityHashSet, query::QueryItem, system::lifetimeless::Read},
    prelude::*,
    render::{
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
        sync_world::{MainEntityHashMap, RenderEntity},
        texture::{CachedTexture, TextureCache},
        view::{RenderVisibleEntities, ViewTarget},
        Extract, Render, RenderApp, RenderSet,
    },
    sprite::{Mesh2dPipeline, Mesh2dPipelineKey, RenderMesh2dInstances},
};

use crate::{flood::*, mask::*};

pub struct Voronoi2dPlugin;
impl Plugin for Voronoi2dPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, MASK_SHADER, "mask.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_SEED_SHADER, "flood_seed.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, FLOOD_SHADER, "flood.wgsl", Shader::from_wgsl);

        app.add_plugins(ExtractComponentPlugin::<VoronoiMaterial>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<MaskPipeline>>()
            .init_resource::<ViewBinnedRenderPhases<MaskPhase>>()
            .init_resource::<RenderVoronoiMaterials>()
            .init_resource::<MaskMaterialBindGroups>()
            .init_resource::<DrawFunctions<MaskPhase>>()
            .add_render_command::<MaskPhase, DrawMaskMesh>()
            .add_systems(
                ExtractSchedule,
                (extract_camera_phases, extract_flood_materials),
            )
            .add_systems(
                Render,
                (
                    queue_custom_meshes.in_set(RenderSet::QueueMeshes),
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
    cameras: Extract<Query<(RenderEntity, &Camera), With<Camera2d>>>,
    mut flood_phases: ResMut<ViewBinnedRenderPhases<MaskPhase>>,
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

fn queue_custom_meshes(
    flood_draw_functions: Res<DrawFunctions<MaskPhase>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<MaskPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mask_pipeline: Res<MaskPipeline>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut custom_render_phases: ResMut<ViewBinnedRenderPhases<MaskPhase>>,
    mut views: Query<(Entity, &RenderVisibleEntities, &Msaa)>,
    render_material_instances: Res<RenderVoronoiMaterials>,
) {
    if render_material_instances.is_empty() {
        return;
    }

    for (view_entity, visible_entities, msaa) in &mut views {
        let Some(flood_phase) = custom_render_phases.get_mut(&view_entity) else {
            continue;
        };
        let draw_flood_mesh = flood_draw_functions.read().id::<DrawMaskMesh>();

        let view_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples());

        for (render_entity, visible_entity) in visible_entities.iter::<With<Mesh2d>>() {
            if render_material_instances.get(visible_entity).is_none() {
                continue;
            };

            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };

            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &mask_pipeline,
                view_key | Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology()),
                &mesh.layout,
            );

            let bin_key = match pipeline_id {
                Ok(id) => MaskPhaseBinKey {
                    pipeline: id,
                    draw_function: draw_flood_mesh,
                    mesh_id: mesh_instance.mesh_asset_id.into(),
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
    view_query: Query<(Entity, &ViewTarget)>,
    flood_mask_phases: Res<ViewBinnedRenderPhases<MaskPhase>>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
) {
    for (entity, view_target) in &view_query {
        if !flood_mask_phases.contains_key(&entity) {
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
        Read<ViewTarget>,
        Read<VoronoiTexture>,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, target, voronoi_textures): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let mut voronoi_textures = voronoi_textures.clone();

        run_mask_pass(
            world,
            render_context,
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
