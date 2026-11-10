#if os(Linux)
import Glibc
import Testing
import SAGATypes
import SAGACore
import SAGACompiler
import SAGAVulkan

// Presentation into a real X11 window. Xlib is loaded at run time, so the tests need no X headers at build time and skip
// when no display is reachable.

private final class XlibWindow {
    typealias OpenDisplay = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutableRawPointer?
    typealias DefaultRoot = @convention(c) (UnsafeMutableRawPointer) -> UInt
    typealias CreateWindow = @convention(c) (UnsafeMutableRawPointer, UInt, Int32, Int32, UInt32, UInt32, UInt32, UInt, UInt) -> UInt
    typealias WindowCall = @convention(c) (UnsafeMutableRawPointer, UInt) -> Int32
    typealias ResizeCall = @convention(c) (UnsafeMutableRawPointer, UInt, UInt32, UInt32) -> Int32
    typealias SyncCall = @convention(c) (UnsafeMutableRawPointer, Int32) -> Int32
    typealias CloseCall = @convention(c) (UnsafeMutableRawPointer) -> Int32

    let library: UnsafeMutableRawPointer
    let display: UnsafeMutableRawPointer
    let window: UInt

    init?(width: UInt32, height: UInt32) {
        guard let library = dlopen("libX11.so.6", RTLD_NOW),
              let open = dlsym(library, "XOpenDisplay"), let root = dlsym(library, "XDefaultRootWindow"),
              let create = dlsym(library, "XCreateSimpleWindow"), let map = dlsym(library, "XMapWindow"),
              let sync = dlsym(library, "XSync"),
              let display = unsafeBitCast(open, to: OpenDisplay.self)(nil) else { return nil }
        self.library = library
        self.display = display
        let parent = unsafeBitCast(root, to: DefaultRoot.self)(display)
        window = unsafeBitCast(create, to: CreateWindow.self)(display, parent, 0, 0, width, height, 0, 0, 0)
        _ = unsafeBitCast(map, to: WindowCall.self)(display, window)
        _ = unsafeBitCast(sync, to: SyncCall.self)(display, 0)
    }

    func resize(width: UInt32, height: UInt32) {
        guard let resize = dlsym(library, "XResizeWindow"), let sync = dlsym(library, "XSync") else { return }
        _ = unsafeBitCast(resize, to: ResizeCall.self)(display, window, width, height)
        _ = unsafeBitCast(sync, to: SyncCall.self)(display, 0)
    }

    func close() {
        if let destroy = dlsym(library, "XDestroyWindow") { _ = unsafeBitCast(destroy, to: WindowCall.self)(display, window) }
        if let close = dlsym(library, "XCloseDisplay") { _ = unsafeBitCast(close, to: CloseCall.self)(display) }
        dlclose(library)
    }
}

@Test(.gpuExclusive) func vulkanPresentsToAnXlibWindowAndRebuildsItsSwapchain() async throws {
    guard VulkanBackend.requestAdapter() != nil, let window = XlibWindow(width: 64, height: 64) else { return }
    defer { window.close() }
    let target = VulkanSurfaceTarget.xlib(display: UInt(bitPattern: window.display), window: window.window)
    let backend: VulkanDevice
    do { backend = try VulkanBackend.makeDevice(surface: target) } catch GPUError.unsupported { return }  // no presentation support
    let device = try await Device.make(backend: backend)
    do {
        let compiler = IndependentShaderCompiler()
        let source = """
            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
                if index == 0u { return vec4f(-1.0f, -1.0f, 0.5f, 1.0f); }
                if index == 1u { return vec4f(3.0f, -1.0f, 0.5f, 1.0f); }
                return vec4f(-1.0f, 3.0f, 0.5f, 1.0f);
            }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(0.0f, 1.0f, 0.0f, 1.0f); }
            """
        let vertex = try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv)
        let fragment = try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv)
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: .bgra8Unorm)
        let frame = try await device.createTexture(.init(width: 64, height: 64, format: .bgra8Unorm,
                                                         usage: [.renderAttachment, .copySource]))
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: frame, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3)])
        try await device.present(frame).wait()
        // Resizing forces the swapchain to be rebuilt on the next present.
        window.resize(width: 320, height: 240)
        try await device.present(frame).wait()
        try await device.present(frame).wait()
        #expect(await backend.presentationRebuildsForTesting() >= 2)
        await device.close()
    } catch { await device.close(); throw error }
}
#endif
