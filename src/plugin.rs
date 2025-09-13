use std::any::TypeId;

use bevy::{
    asset::embedded_asset,
    camera::visibility::VisibleEntities,
    core_pipeline::core_2d::graph::{Core2d, Node2d},
    ecs::{query::QueryItem, system::lifetimeless::Read},
    math::FloatOrd,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::{
        batching::no_gpu_preprocessing::batch_and_prepare_sorted_render_phase,
        camera::{extract_cameras, ExtractedCamera},
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::RenderMesh,
        render_asset::RenderAssets,
        render_graph::{
            NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
        },
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, ViewSortedRenderPhases,
        },
        render_resource::{
            Extent3d, PipelineCache, SpecializedMeshPipelines, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::{MainEntity, MainEntityHashMap},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, RenderVisibleEntities, RetainedViewEntity, ViewTarget},
        Extract, Render, RenderApp, RenderStartup, RenderSystems,
    },
    sprite_render::{
        init_mesh_2d_pipeline, Mesh2dPipeline, Mesh2dPipelineKey, RenderMesh2dInstances,
        ViewKeyCache,
    },
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
            .add_plugins(ExtractComponentPlugin::<VoronoiViewNeedsUpdate>::default())
            .add_systems(PostUpdate, check_voronoi_views_needing_update);

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
            .add_render_command::<MaskPhase, DrawMaskMesh>()
            .add_systems(
                ExtractSchedule,
                (
                    extract_mask_phases.after(extract_cameras),
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
            .add_render_graph_edges(
                Core2d,
                (
                    Node2d::MainOpaquePass,
                    VoronoiDrawPassLabel,
                    Node2d::MainTransparentPass,
                ),
            );
    }
}

#[derive(Component, ExtractComponent, Clone, PartialEq)]
pub struct VoronoiView {
    pub scale: f32,
    pub target: Handle<Image>,
}

impl Default for VoronoiView {
    fn default() -> Self {
        Self {
            scale: 0.5,
            target: default(),
        }
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

#[derive(Component, Clone, ExtractComponent)]
struct VoronoiViewNeedsUpdate;

fn check_voronoi_views_needing_update(
    mut commands: Commands,
    changed_views: Query<
        (),
        (
            Or<(
                Changed<Camera>,
                Changed<VoronoiView>,
                Changed<GlobalTransform>,
            )>,
            With<VoronoiView>,
        ),
    >,
    changed_materials: Query<
        (),
        (
            Or<(
                Changed<Mesh2d>,
                AssetChanged<Mesh2d>,
                Changed<VoronoiMaterial>,
                Changed<GlobalTransform>,
            )>,
            With<VoronoiMaterial>,
        ),
    >,
    views: Query<(Entity, &VisibleEntities), With<VoronoiView>>,
) {
    for (entity, visible_entities) in &views {
        commands.entity(entity).remove::<VoronoiViewNeedsUpdate>();

        if changed_views.contains(entity) {
            commands.entity(entity).insert(VoronoiViewNeedsUpdate);
            break;
        }

        for visible_entity in visible_entities.iter(TypeId::of::<VoronoiMaterial>()) {
            if changed_materials.contains(*visible_entity) {
                commands.entity(entity).insert(VoronoiViewNeedsUpdate);
                break;
            }
        }
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

fn queue_mask_meshes(
    mask_draw_functions: Res<DrawFunctions<MaskPhase>>,
    render_meshes: Res<RenderAssets<RenderMesh>>,
    pipeline_cache: Res<PipelineCache>,
    mut render_mesh_instances: ResMut<RenderMesh2dInstances>,
    mut mask_render_phase: ResMut<ViewSortedRenderPhases<MaskPhase>>,
    mut mask_pipelines: ResMut<SpecializedMeshPipelines<MaskPipeline>>,
    mask_pipeline: Res<MaskPipeline>,
    view_key_cache: Res<ViewKeyCache>,
    views: Query<
        (&MainEntity, &ExtractedView, &RenderVisibleEntities),
        With<VoronoiViewNeedsUpdate>,
    >,
    render_material_instances: Res<RenderVoronoiMaterials>,
) {
    if render_material_instances.is_empty() {
        return;
    }

    for (view_entity, view, visible_entities) in &views {
        let Some(view_key) = view_key_cache.get(view_entity) else {
            continue;
        };

        let Some(mask_phase) = mask_render_phase.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let draw_mask_mesh = mask_draw_functions.read().id::<DrawMaskMesh>();

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh2d>() {
            let Some(mesh_instance) = render_mesh_instances.get_mut(visible_entity) else {
                continue;
            };
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
            mask_phase.add(MaskPhase {
                sort_key: FloatOrd(mesh_instance.transforms.world_from_local.translation.z),
                pipeline: pipeline_id,
                draw_function: draw_mask_mesh,
                entity: (*render_entity, *visible_entity),
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
            });
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
    views: Query<(
        &ViewTarget,
        &ExtractedView,
        &VoronoiView,
        Option<&VoronoiViewNeedsUpdate>,
    )>,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    mut voronoi_textures: ResMut<VoronoiTextures>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();

    for (view_target, extracted_view, voronoi_view, needs_update) in &views {
        live_entities.insert(extracted_view.retained_view_entity);

        if needs_update.is_none() {
            continue;
        }

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

    voronoi_textures.retain(|entity, _| live_entities.contains(entity));
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
