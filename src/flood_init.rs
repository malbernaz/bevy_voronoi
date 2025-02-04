use std::ops::Range;

use bevy::{
    asset::UntypedAssetId,
    prelude::*,
    render::{
        camera::ExtractedCamera,
        mesh::MeshVertexBufferLayoutRef,
        render_phase::{
            BinnedPhaseItem, CachedRenderPipelinePhaseItem, DrawFunctionId, PhaseItem,
            PhaseItemExtraIndex, SetItemPipeline, ViewBinnedRenderPhases,
        },
        render_resource::{
            CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState, Operations,
            RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
            SpecializedMeshPipeline, SpecializedMeshPipelineError, TextureFormat,
        },
        renderer::RenderContext,
        sync_world::MainEntity,
        texture::CachedTexture,
    },
    sprite::{
        DrawMesh2d, Mesh2dPipeline, Mesh2dPipelineKey, SetMesh2dBindGroup, SetMesh2dViewBindGroup,
    },
};

pub const FLOOD_INIT_SHADER: Handle<Shader> =
    Handle::weak_from_u128(57844709471149694165463051306473017437);

#[derive(Resource)]
pub struct FloodInitPipeline {
    mesh_pipeline: Mesh2dPipeline,
}

impl FromWorld for FloodInitPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh_pipeline: Mesh2dPipeline::from_world(world),
        }
    }
}

impl SpecializedMeshPipeline for FloodInitPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let descriptor = self.mesh_pipeline.specialize(key, &layout)?;

        Ok(RenderPipelineDescriptor {
            label: Some("flood_init_pipeline".into()),
            fragment: Some(FragmentState {
                shader: FLOOD_INIT_SHADER,
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

pub struct FloodInitPhase {
    pub key: FloodInitPhaseBinKey,
    pub representative_entity: (Entity, MainEntity),
    pub batch_range: Range<u32>,
    pub extra_index: PhaseItemExtraIndex,
}

/// Data that must be identical in order to batch phase items together.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FloodInitPhaseBinKey {
    pub pipeline: CachedRenderPipelineId,
    pub draw_function: DrawFunctionId,
    pub asset_id: UntypedAssetId,
}

impl PhaseItem for FloodInitPhase {
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

impl BinnedPhaseItem for FloodInitPhase {
    type BinKey = FloodInitPhaseBinKey;

    fn new(
        key: Self::BinKey,
        representative_entity: (Entity, MainEntity),
        batch_range: Range<u32>,
        extra_index: PhaseItemExtraIndex,
    ) -> Self {
        FloodInitPhase {
            key,
            representative_entity,
            batch_range,
            extra_index,
        }
    }
}

impl CachedRenderPipelinePhaseItem for FloodInitPhase {
    #[inline]
    fn cached_pipeline(&self) -> CachedRenderPipelineId {
        self.key.pipeline
    }
}

pub type DrawFloodMesh = (
    SetItemPipeline,
    SetMesh2dViewBindGroup<0>,
    SetMesh2dBindGroup<1>,
    DrawMesh2d,
);

pub fn flood_init_pass<'w>(
    world: &'w World,
    render_context: &mut RenderContext<'w>,
    view_entity: &Entity,
    output: &CachedTexture,
    camera: &ExtractedCamera,
) {
    let Some(flood_init_phases) = world.get_resource::<ViewBinnedRenderPhases<FloodInitPhase>>()
    else {
        error!("FloodInitPhase not available");
        return;
    };

    let Some(flood_init_phase) = flood_init_phases.get(view_entity) else {
        error!("View FloodInitPhase not available");
        return;
    };

    let mut pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("flood_init_pass"),
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

    if !flood_init_phase.is_empty() {
        if let Err(err) = flood_init_phase.render(&mut pass, world, *view_entity) {
            error!("Error encountered while rendering the flood init phase {err:?}");
        }
    }
}
