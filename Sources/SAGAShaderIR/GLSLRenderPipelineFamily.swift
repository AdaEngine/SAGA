/// Selects which pre-rasterization stages share GLSL resource bindings with a fragment stage.
public enum GLSLRenderPipelineFamily: Sendable, Equatable {
    /// A conventional vertex and fragment pipeline.
    case vertex
    /// A task/mesh and fragment pipeline.
    case mesh
}
