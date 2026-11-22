import Foundation
import SAGABackend
import SAGAShaderIR
import SAGATypes
#if canImport(Metal)
import Metal
#endif
#if canImport(QuartzCore)
import QuartzCore
#endif

extension MetalDevice {
    public func present(_ submission: Submission, source: ResourceID) async throws {
        try await present(submission, source: source, surface: 0)
    }

    #if canImport(Metal) && canImport(QuartzCore)
    /// Native device for configuring a MetalKit view before attaching its layer.
    public var metalDevice: any MTLDevice { native }

    public func addSurface(_ surface: MetalPresentationSurface) throws -> UInt32 {
        try requireActive()
        guard supportsPresentation else { throw GPUError.unsupported("Metal device was created without presentation") }
        guard nextSurfaceID < UInt32.max else { throw GPUError.validation("Metal surface IDs exhausted") }
        try surface.attach(to: native)
        nextSurfaceID += 1
        surfaces[nextSurfaceID] = surface
        return nextSurfaceID
    }

    public func removeSurface(_ id: UInt32) async {
        guard let surface = surfaces.removeValue(forKey: id) else { return }
        if presentingSurface == id {
            await withCheckedContinuation { surfaceRetirement = $0 }
        }
        surface.detach()
    }
    #endif

    public func present(_ submission: Submission, source: ResourceID, surface id: UInt32) async throws {
        try requireActive()
        #if canImport(Metal) && canImport(QuartzCore)
        guard let surface = surfaces[id] else { throw GPUError.unsupported("Metal backend does not have the requested surface") }
        guard presentingSurface == nil else { throw GPUError.validation("Metal device already has a pending frame") }
        guard submission.resources.contains(source), let texture = textures[source],
              texture.descriptor.format.isNormalizedColor, texture.descriptor.sampleCount == 1,
              texture.descriptor.dimension == .d2, texture.descriptor.arrayLayerCount == 1,
              texture.descriptor.usage.contains(.copySource) else {
            throw GPUError.validation("Unpinned or invalid Metal presentation source")
        }
        let descriptor = texture.descriptor
        presentingSurface = id
        defer {
            presentingSurface = nil
            surfaceRetirement?.resume()
            surfaceRetirement = nil
        }
        let pipeline: any MTLRenderPipelineState
        if let surfacePipeline {
            pipeline = surfacePipeline
        } else {
            pipeline = try MetalSurfaceCopy.makePipeline(device: native)
            surfacePipeline = pipeline
        }
        let drawable = try surface.acquire(width: descriptor.width, height: descriptor.height)
        guard drawable.texture.width == Int(descriptor.width), drawable.texture.height == Int(descriptor.height) else {
            throw GPUError.backendFailure("Metal drawable size did not match the presentation source")
        }
        guard let commands = queue.makeCommandBuffer() else { throw GPUError.outOfMemory }
        let clear = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        var initialized: [ResourceID: Set<TextureSubresource>] = [:]
        try initializeTexture(source, level: 0, on: commands, initialized: &initialized)
        let encoder = try renderEncoder(commands, target: drawable.texture, clear: clear)
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(texture.native, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commands.present(drawable)
        try await commit(Batch(commandBuffer: commands, buffers: [], pipelines: [], textures: [texture.native],
                               samplers: [], renderPipelines: [.init(pipeline: pipeline, depth: nil, writesDepth: false,
                                                       usesVertexPulling: false, vertexRuntimeLengths: [], fragmentRuntimeLengths: [],
                                                       vertexExternalTextures: [], fragmentExternalTextures: [],
                                                       fragmentTextureAtomicGuardBufferIndex: nil,
                                                       vertexImmediateBufferIndex: nil, fragmentImmediateBufferIndex: nil)],
                               drawable: drawable), initialized: initialized)
        #else
        throw GPUError.unsupported("Metal presentation is unavailable")
        #endif
    }
}
