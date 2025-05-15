use std::ops::Range;

use bevy::{
    asset::UntypedAssetId,
    core_pipeline::core_2d::BatchSetKey2d,
    ecs::system::{lifetimeless::SRes, SystemParamItem},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        mesh::MeshVertexBufferLayoutRef,
        render_asset::RenderAssets,
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex, RenderCommand, RenderCommandResult, SetItemPipeline,
            TrackedRenderPass, ViewBinnedRenderPhases,
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
        texture::{CachedTexture, FallbackImage, GpuImage},
        view::RetainedViewEntity,
    },
    sprite::{
        DrawMesh2d, Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dBindGroup, SetMesh2dViewBindGroup,
    },
};

use crate::plugin::RenderVoronoiMaterials;

pub const MASK_SHADER: Handle<Shader> =
    Handle::weak_from_u128(57844709471149694165463051306473017437);

#[derive(Resource)]
pub struct MaskPipeline {
    pub mesh_pipeline: Mesh2dPipeline,
    pub material_layout: BindGroupLayout,
}

impl FromWorld for MaskPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: Mesh2dPipeline::from_world(world),
            material_layout: world.resource::<RenderDevice>().create_bind_group_layout(
                "mask_material_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_2d(TextureSampleType::Float { filterable: true }),
                        sampler(SamplerBindingType::Filtering),
                    ),
                ),
            ),
        }
    }
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
                shader: MASK_SHADER,
                shader_defs: vec![],
                entry_point: "fragment".into(),
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
    pub bin_key: MaskPhaseBinKey,
    pub batch_set_key: BatchSetKey2d,
    pub representative_entity: (Entity, MainEntity),
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

/// Data that must be identical in order to batch phase items together.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskPhaseBinKey {
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub mesh_id: UntypedAssetId,
}

impl PhaseItem for MaskPhase {
    #[inline]
    fn entity(&self) -> Entity {
        self.representative_entity.0
    }

    fn main_entity(&self) -> MainEntity {
        self.representative_entity.1
    }

    #[inline]
    fn draw_function(&self) -> DrawFunctionId {
        self.bin_key.draw_function
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
        self.extra_index.clone()
    }

    fn batch_range_and_extra_index_mut(&mut self) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
        (&mut self.batch_range, &mut self.extra_index)
    }
}

impl BinnedPhaseItem for MaskPhase {
    type BatchSetKey = BatchSetKey2d;

    type BinKey = MaskPhaseBinKey;

    fn new(
        batch_set_key: Self::BatchSetKey,
        bin_key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        MaskPhase {
            bin_key,
            batch_set_key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for MaskPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.bin_key.pipeline
    }
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
    flood_materials: Res<RenderVoronoiMaterials>,
    mut bind_groups: ResMut<MaskMaterialBindGroups>,
) {
    bind_groups.clear();
    for (entity, alpha_mask) in flood_materials.iter() {
        let alpha_mask = if let Some(image) = images.get(*alpha_mask) {
            image
        } else {
            &fallback_image.d2
        };
        let sampler = render_device.create_sampler(&SamplerDescriptor::default());
        let bind_group = render_device.create_bind_group(
            "mask_material_bind_group",
            &pipeline.material_layout,
            &BindGroupEntries::sequential((&alpha_mask.texture_view, &sampler)),
        );
        // TODO: this is creating bind groups for every render. make it more efficient.
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
    retained_view_entity: &RetainedViewEntity,
    view_entity: &Entity,
    output: &CachedTexture,
    camera: &ExtractedCamera,
) {
    let Some(mask_phases) = world.get_resource::<ViewBinnedRenderPhases<MaskPhase>>() else {
        error!("MaskPhase not available");
        return;
    };

    let Some(phase) = mask_phases.get(retained_view_entity) else {
        error!("View MaskPhase not available");
        return;
    };

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("mask_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output.default_view,
            resolve_target: None,
            ops: Operations::default(),
        })],
        ..default()
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    if !phase.is_empty() {
        if let Err(err) = phase.render(&mut pass, world, *view_entity) {
            error!("Error encountered while rendering the mask phase {err:?}");
        }
    }
}
