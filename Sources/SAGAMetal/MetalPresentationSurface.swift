import SAGATypes
#if canImport(Metal) && canImport(QuartzCore)
import Metal
import QuartzCore
#endif

public struct MetalPresentedFrame: Sendable, Equatable {
    public let width: UInt32
    public let height: UInt32
    public let presentedTime: Double
}

/// Owns configuration of an Apple Metal layer for one backend.
@MainActor
public final class MetalPresentationSurface {
    #if canImport(Metal) && canImport(QuartzCore)
    private let layer: CAMetalLayer
    private let onPresented: (@MainActor @Sendable (MetalPresentedFrame) -> Void)?
    private var attached = false

    public init(layer: CAMetalLayer, onPresented: (@MainActor @Sendable (MetalPresentedFrame) -> Void)? = nil) {
        self.layer = layer
        self.onPresented = onPresented
    }

    func attach(to device: any MTLDevice) throws {
        guard !attached else { throw GPUError.validation("Metal surface already has a backend") }
        attached = true
        layer.device = device
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = true
        layer.allowsNextDrawableTimeout = true
    }

    func detach() {
        attached = false
        layer.device = nil
    }

    func acquire(width: UInt32, height: UInt32) throws -> any CAMetalDrawable {
        let size = CGSize(width: Int(width), height: Int(height))
        if layer.drawableSize != size { layer.drawableSize = size }
        guard let drawable = layer.nextDrawable() else {
            throw GPUError.backendFailure("Metal surface did not supply a drawable")
        }
        #if !targetEnvironment(simulator)
        if let onPresented {
            drawable.addPresentedHandler { drawable in
                let frame = MetalPresentedFrame(width: width, height: height, presentedTime: drawable.presentedTime)
                Task { @MainActor in onPresented(frame) }
            }
        }
        #endif
        return drawable
    }
    #else
    private init() {}
    #endif
}

#if canImport(Metal) && canImport(QuartzCore)
enum MetalSurfaceCopy {
    static func makePipeline(device: any MTLDevice) throws -> any MTLRenderPipelineState {
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        struct SurfaceVertex { float4 position [[position]]; };
        vertex SurfaceVertex saga_surface_vertex(uint id [[vertex_id]]) {
            float2 point = id == 0u ? float2(-1.0, -1.0) :
                (id == 1u ? float2(3.0, -1.0) : float2(-1.0, 3.0));
            return SurfaceVertex{float4(point, 0.0, 1.0)};
        }
        fragment float4 saga_surface_fragment(
            SurfaceVertex input [[stage_in]], texture2d<float, access::read> source [[texture(0)]]) {
            return source.read(uint2(input.position.xy));
        }
        """
        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard let vertex = library.makeFunction(name: "saga_surface_vertex"),
                  let fragment = library.makeFunction(name: "saga_surface_fragment") else {
                throw GPUError.backendFailure("Metal surface copy entry point missing")
            }
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = vertex
            descriptor.fragmentFunction = fragment
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            return try device.makeRenderPipelineState(descriptor: descriptor)
        } catch {
            throw GPUError.backendFailure("Metal surface copy pipeline: \(error)")
        }
    }
}
#endif
