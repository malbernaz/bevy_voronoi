use bevy::{
    asset::{embedded_asset, AsAssetId, AssetEventSystems},
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{
        component::Tick,
        query::QueryItem,
        system::{lifetimeless::Read, SystemChangeTick},
    },
    platform::collections::HashMap,
    prelude::*,
    render::{
        batching::no_gpu_preprocessing::batch_and_prepare_sorted_render_phase,
        camera::{extract_cameras, ExtractedCamera},
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{AddRenderCommand, DrawFunctions, ViewSortedRenderPhases},
        render_resource::{
            Extent3d, SpecializedMeshPipelines, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::{MainEntity, MainEntityHashMap},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, RetainedViewEntity, ViewTarget},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
    sprite_render::{
        init_mesh_2d_pipeline, EntitiesNeedingSpecialization, EntitySpecializationTicks,
        Mesh2dPipeline, SpecializedMaterial2dPipelineCache,
    },
    utils::Parallel,
};

use crate::{flood::*, mask::*};

pub struct Voronoi2dPlugin;
impl Plugin for Voronoi2dPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "mask.wgsl");
        embedded_asset!(app, "flood_seed.wgsl");
        embedded_asset!(app, "flood.wgsl");

        app.add_plugins(ExtractComponentPlugin::<VoronoiView>::default())
            .add_plugins(ExtractComponentPlugin::<VoronoiMaterial>::default())
            .init_resource::<EntitiesNeedingSpecialization<VoronoiView>>()
            .init_resource::<EntitiesNeedingSpecialization<VoronoiMaterial>>()
            .add_systems(
                PostUpdate,
                (
                    check_views_needing_specialization,
                    check_materials_needing_specialization,
                )
                    .after(AssetEventSystems),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<MaskPipeline>>()
            .init_resource::<ViewSortedRenderPhases<MaskPhase>>()
            .init_resource::<RenderVoronoiMaterials>()
            .init_resource::<MaskMaterialBindGroups>()
            .init_resource::<DrawFunctions<MaskPhase>>()
            .init_resource::<VoronoiTextures>()
            .init_resource::<SpecializedMaterial2dPipelineCache<VoronoiMaterial>>()
            .init_resource::<EntitySpecializationTicks<VoronoiView>>()
            .init_resource::<EntitySpecializationTicks<VoronoiMaterial>>()
            .init_resource::<VoronoiViewSpecializationTicks>()
            .add_render_command::<MaskPhase, DrawMaskMesh>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_mask_phases.after(extract_cameras),
                    extract_entities_needs_specialization,
                    extract_views_need_specialization,
                    extract_voronoi_materials,
                ),
            )
            .add_systems(
                RenderStartup,
                (
                    init_mask_pipeline.after(init_mesh_2d_pipeline),
                    init_flood_pipeline,
                ),
            )
            .add_systems(
                Render,
                (
                    queue_mask_meshes.in_set(RenderSystems::QueueMeshes),
                    (
                        prepare_voronoi_textures,
                        batch_and_prepare_sorted_render_phase::<MaskPhase, Mesh2dPipeline>,
                    )
                        .in_set(RenderSystems::PrepareResources),
                    prepare_mask_material_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<VoronoiDrawNode>>(Core2d, VoronoiDrawPassLabel)
            .add_render_graph_edges(Core2d, (VoronoiDrawPassLabel, Node2d::StartMainPass));
    }
}

#[derive(Component, ExtractComponent, Clone, PartialEq)]
pub struct VoronoiView {
    pub scale: f32,
}

impl Default for VoronoiView {
    fn default() -> Self {
        Self { scale: 0.5 }
    }
}

#[derive(Component, ExtractComponent, Clone, Default, Eq, PartialEq)]
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

impl AsAssetId for VoronoiMaterial {
    type Asset = Image;

    fn as_asset_id(&self) -> AssetId<Self::Asset> {
        self.alpha_mask.id()
    }
}

pub fn check_views_needing_specialization(
    needs_specialization: Query<
        Entity,
        (
            Or<(
                Changed<Camera>,
                Changed<VoronoiView>,
                Changed<GlobalTransform>,
            )>,
            With<VoronoiView>,
        ),
    >,
    mut par_local: Local<Parallel<Vec<Entity>>>,
    mut entities_needing_specialization: ResMut<EntitiesNeedingSpecialization<VoronoiView>>,
) {
    entities_needing_specialization.clear();

    needs_specialization
        .par_iter()
        .for_each(|entity| par_local.borrow_local_mut().push(entity));

    par_local.drain_into(&mut entities_needing_specialization);
}

pub fn check_materials_needing_specialization(
    needs_specialization: Query<
        Entity,
        (
            Or<(
                Changed<Mesh2d>,
                AssetChanged<Mesh2d>,
                Changed<VoronoiMaterial>,
                AssetChanged<VoronoiMaterial>,
                Changed<GlobalTransform>,
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

#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoronoiViewSpecializationTicks(MainEntityHashMap<Tick>);

pub fn extract_views_need_specialization(
    entities_needing_specialization: Extract<Res<EntitiesNeedingSpecialization<VoronoiView>>>,
    mut view_specialization_ticks: ResMut<VoronoiViewSpecializationTicks>,
    ticks: SystemChangeTick,
) {
    for entity in entities_needing_specialization.iter() {
        view_specialization_ticks.insert((*entity).into(), ticks.this_run());
    }
}

pub fn extract_entities_needs_specialization(
    entities_needing_specialization: Extract<Res<EntitiesNeedingSpecialization<VoronoiMaterial>>>,
    mut entity_specialization_ticks: ResMut<EntitySpecializationTicks<VoronoiMaterial>>,
    mut removed_components: Extract<RemovedComponents<VoronoiMaterial>>,
    mut specialized_view_pipeline_cache: ResMut<
        SpecializedMaterial2dPipelineCache<VoronoiMaterial>,
    >,
    views: Query<&MainEntity, With<ExtractedView>>,
    ticks: SystemChangeTick,
) {
    for entity in removed_components.read() {
        entity_specialization_ticks.remove(&MainEntity::from(entity));
        for view in views {
            if let Some(cache) = specialized_view_pipeline_cache.get_mut(view) {
                cache.remove(&MainEntity::from(entity));
            }
        }
    }

    for entity in entities_needing_specialization.iter() {
        entity_specialization_ticks.insert((*entity).into(), ticks.this_run());
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct RenderVoronoiMaterials(MainEntityHashMap<AssetId<Image>>);

fn extract_voronoi_materials(
    mut render_voronoi_instances: ResMut<RenderVoronoiMaterials>,
    query: Extract<Query<(Entity, &ViewVisibility, &VoronoiMaterial), With<Mesh2d>>>,
) {
    render_voronoi_instances.clear();

    for (entity, view_visibility, material) in &query {
        if view_visibility.get() {
            render_voronoi_instances.insert(entity.into(), material.into());
        }
    }
}

#[derive(Clone)]
pub struct VoronoiTexture {
    flip: bool,
    texture_a: CachedTexture,
    texture_b: CachedTexture,
}

#[derive(Resource, Deref, DerefMut, Default)]
pub struct VoronoiTextures(pub HashMap<RetainedViewEntity, VoronoiTexture>);

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
    scale: f32,
) -> CachedTexture {
    let size = view_target.main_texture().size();
    let size = Extent3d {
        width: (size.width as f32 * scale) as u32,
        height: (size.height as f32 * scale) as u32,
        depth_or_array_layers: size.depth_or_array_layers,
    };

    texture_cache.get(
        render_device,
        TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

fn prepare_voronoi_textures(
    views: Query<(&ViewTarget, &ExtractedView, &VoronoiView)>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut voronoi_textures: ResMut<VoronoiTextures>,
) {
    for (view_target, extracted_view, voronoi_view) in &views {
        voronoi_textures.insert(
            extracted_view.retained_view_entity,
            VoronoiTexture {
                flip: false,
                texture_a: create_aux_texture(
                    view_target,
                    &mut texture_cache,
                    &render_device,
                    "voronoi_texture_a",
                    voronoi_view.scale,
                ),
                texture_b: create_aux_texture(
                    view_target,
                    &mut texture_cache,
                    &render_device,
                    "voronoi_texture_b",
                    voronoi_view.scale,
                ),
            },
        );
    }
}

#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
struct VoronoiDrawPassLabel;

#[derive(Default)]
struct VoronoiDrawNode;
impl ViewNode for VoronoiDrawNode {
    type ViewQuery = (Read<ExtractedCamera>, Read<ExtractedView>, Read<ViewTarget>);

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view, target): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph.view_entity();

        let Some(mask_phase) = world
            .resource::<ViewSortedRenderPhases<MaskPhase>>()
            .get(&view.retained_view_entity)
        else {
            return Ok(());
        };

        if mask_phase.items.is_empty() {
            return Ok(());
        }

        let mut voronoi_texture = world
            .resource::<VoronoiTextures>()
            .get(&view.retained_view_entity)
            .expect(&format!(
                "Expected the voronoi texture for {:?} exist",
                view.retained_view_entity.main_entity.id()
            ))
            .clone();

        run_mask_pass(
            world,
            render_context,
            mask_phase,
            &view_entity,
            &mut voronoi_texture,
            camera,
        );

        run_flood_seed_pass(world, render_context, camera, &mut voronoi_texture);

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
                &mut voronoi_texture,
                UVec2::new(x_step.max(1), y_step.max(1)),
            );

            step /= 2;
        }

        // Addicional pass with step = 1 to improve accuracy
        run_flood_pass(
            world,
            render_context,
            camera,
            &mut voronoi_texture,
            UVec2::new(1, 1),
        );
        voronoi_texture.flip();

        Ok(())
    }
}
