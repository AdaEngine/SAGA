#if canImport(Metal)
import Metal
import SAGATypes

/// Metal resources and the command buffer for the current native pass.
///
/// Use these objects during the pass body. End any encoders before returning and let SAGA submit
/// the command buffer. Commands run in the same order as the surrounding SAGA commands.
public final class MetalNativePassContext: NativePassContext {
    public let commandBuffer: any MTLCommandBuffer
    private let buffers: [ResourceID: any MTLBuffer]
    private let textures: [ResourceID: any MTLTexture]
    private(set) var committedHandlers: [@MainActor @Sendable () -> Void] = []

    init(
        commandBuffer: any MTLCommandBuffer,
        buffers: [ResourceID: any MTLBuffer],
        textures: [ResourceID: any MTLTexture]
    ) {
        self.commandBuffer = commandBuffer
        self.buffers = buffers
        self.textures = textures
    }

    /// The buffer declared in the pass's uses.
    public func buffer(_ id: ResourceID) throws -> any MTLBuffer {
        guard let buffer = buffers[id] else { throw GPUError.validation("The native pass did not name this buffer") }
        return buffer
    }

    /// The texture declared in the pass's uses.
    public func texture(_ id: ResourceID) throws -> any MTLTexture {
        guard let texture = textures[id] else { throw GPUError.validation("The native pass did not name this texture") }
        return texture
    }

    /// Registers a callback at the point Metal accepts the command buffer.
    /// A submission that fails during recording leaves the callback uncalled.
    public func addCommittedHandler(_ handler: @escaping @MainActor @Sendable () -> Void) {
        committedHandlers.append(handler)
    }
}
#endif
