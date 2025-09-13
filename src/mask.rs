use std::ops::Range;

use bevy::{
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    math::FloatOrd,
    mesh::MeshVertexBufferLayoutRef,
    platform::collections::HashSet,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        render_asset::RenderAssets,
        render_phase::{
            CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem, PhaseItemExtraIndex,
            RenderCommand, RenderCommandResult, SetItemPipeline, SortedPhaseItem,
            SortedRenderPhase, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::{
            binding_types::{sampler, texture_2d},
            BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, Operations,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SamplerBindingType, SamplerDescriptor, ShaderStages, SpecializedMeshPipeline,
            SpecializedMeshPipelineError, TextureFormat, TextureSampleType,
        },
        renderer::{RenderContext, RenderDevice},
        sync_world::{MainEntity, MainEntityHashMap},
        texture::{FallbackImage, GpuImage},
        view::RetainedViewEntity,
        Extract,
    },
    sprite_render::{
        DrawMesh2d, Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dBindGroup, SetMesh2dViewBindGroup,
    },
};

use crate::plugin::{RenderVoronoiMaterials, VoronoiTexture, VoronoiView};

#[derive(Resource)]
pub struct MaskPipeline {
    pub mesh_pipeline: Mesh2dPipeline,
    pub material_layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

pub fn init_mask_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mesh_2d_pipeline: Res<Mesh2dPipeline>,
    asset_server: Res<AssetServer>,
) {
    commands.insert_resource(MaskPipeline {
        mesh_pipeline: mesh_2d_pipeline.clone(),
        shader: asset_server.load("embedded://bevy_voronoi/mask.wgsl"),
        material_layout: render_device.create_bind_group_layout(
            "mask_material_bind_group_layout",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    texture_2d(TextureSampleType::Float { filterable: true }),
                    sampler(SamplerBindingType::Filtering),
                ),
            ),
        ),
    });
}

impl SpecializedMeshPipeline for MaskPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let descriptor = self.mesh_pipeline.specialize(key, &layout)?;

        let mut mesh_layout = descriptor.layout.clone();
        mesh_layout.push(self.material_layout.clone());

        Ok(RenderPipelineDescriptor {
            label: Some("mask_pipeline".into()),
            layout: mesh_layout,
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            depth_stencil: None,
            multisample: Default::default(),
            ..descriptor
        })
    }
}

pub struct MaskPhase {
    pub sort_key: FloatOrd,
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub entity: (Entity, MainEntity),
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
    pub indexed: bool,
}

impl PhaseItem for MaskPhase {
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

impl SortedPhaseItem for MaskPhase {
    type SortKey = FloatOrd;

    #[inline]
    fn sort_key(&self) -> Self::SortKey {
        self.sort_key
    }

    #[inline]
    fn sort(items: &mut [Self]) {
        radsort::sort_by_key(items, |item| item.sort_key().0);
    }

    fn indexed(&self) -> bool {
        self.indexed
    }
}

impl CachedRenderPipelinePhaseItem for MaskPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.pipeline
    }
}

pub fn extract_mask_phases(
    cameras: Extract<Query<(Entity, &Camera), (With<Camera2d>, With<VoronoiView>)>>,
    mut mask_phases: ResMut<ViewSortedRenderPhases<MaskPhase>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();

    for (entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        let retained_view_entity = RetainedViewEntity::new(entity.into(), None, 0);

        mask_phases.insert_or_clear(retained_view_entity);
        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views
    mask_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

pub type DrawMaskMesh = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    SetMaskMaterialBindGroup<2>,
    DrawMesh2d,
);

#[derive(Resource, Deref, DerefMut, Default)]
pub struct MaskMaterialBindGroups(MainEntityHashMap<BindGroup>);

pub fn prepare_mask_material_bind_groups(
    render_device: Res<RenderDevice>,
    pipeline: Res<MaskPipeline>,
    images: Res<RenderAssets<GpuImage>>,
    fallback_image: Res<FallbackImage>,
    voronoi_materials: Res<RenderVoronoiMaterials>,
    mut bind_groups: ResMut<MaskMaterialBindGroups>,
) {
    // Only update bind groups for entities that have changed or are new
    bind_groups.retain(|entity, _| voronoi_materials.contains_key(entity));

    for (entity, alpha_mask) in voronoi_materials.iter() {
        let alpha_mask_image = if let Some(image) = images.get(*alpha_mask) {
            image
        } else {
            &fallback_image.d2
        };
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());
        let bind_group = render_device.create_bind_group(
            "mask_material_bind_group",
            &pipeline.material_layout,
            &BindGroupEntries::sequential((&alpha_mask_image.texture_view, &sampler)),
        );
        bind_groups.insert(*entity, bind_group);
    }
}

pub struct SetMaskMaterialBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMaskMaterialBindGroup<I> {
    type Param = SRes<MaskMaterialBindGroups>;
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        bind_groups: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let bind_groups = bind_groups.into_inner();
        let Some(bind_group) = bind_groups.get(&item.main_entity()) else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, &bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub fn run_mask_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    phase: &SortedRenderPhase<MaskPhase>,
    view_entity: &Entity,
    voronoi_texture: &mut VoronoiTexture,
    camera: &ExtractedCamera,
) {
    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("mask_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &voronoi_texture.output().default_view,
            resolve_target: None,
            ops: Operations::default(),
            depth_slice: None,
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    if let Err(err) = phase.render(&mut pass, world, *view_entity) {
        error!("Error encountered while rendering the voronoi mask phase {err:?}");
    }

    voronoi_texture.flip();
}
