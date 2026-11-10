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
        // The X server applies the new size asynchronously, so keep presenting until the swapchain notices it.
        window.resize(width: 320, height: 240)
        for _ in 0..<100 {
            try await device.present(frame).wait()
            if await backend.presentationRebuildsForTesting() >= 1 { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(await backend.presentationRebuildsForTesting() >= 1)
        await device.close()
    } catch { await device.close(); throw error }
}

// MARK: Wayland

/// The compositor global announced by the registry, recorded by the C callback below.
nonisolated(unsafe) private var announcedCompositor: (name: UInt32, version: UInt32)?

private let registryGlobal: @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UInt32, UnsafePointer<CChar>?, UInt32) -> Void = {
    _, _, name, interface, version in
    if let interface, String(cString: interface) == "wl_compositor" { announcedCompositor = (name, version) }
}
private let registryGlobalRemove: @convention(c) (UnsafeMutableRawPointer?, UnsafeMutableRawPointer?, UInt32) -> Void = { _, _, _ in }

/// A bare `wl_surface` on the compositor, built with the non-variadic libwayland entry points.
private final class WaylandSurface {
    typealias Connect = @convention(c) (UnsafePointer<CChar>?) -> UnsafeMutableRawPointer?
    typealias Roundtrip = @convention(c) (UnsafeMutableRawPointer) -> Int32
    typealias Constructor = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UnsafeRawPointer?) -> UnsafeMutableRawPointer?
    typealias VersionedConstructor = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeMutableRawPointer, UnsafeRawPointer?, UInt32) -> UnsafeMutableRawPointer?
    typealias AddListener = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeMutableRawPointer?) -> Int32
    typealias Disconnect = @convention(c) (UnsafeMutableRawPointer) -> Void

    let library: UnsafeMutableRawPointer
    let display: UnsafeMutableRawPointer
    let surface: UnsafeMutableRawPointer

    init?() {
        announcedCompositor = nil
        guard let library = dlopen("libwayland-client.so.0", RTLD_NOW),
              let connect = dlsym(library, "wl_display_connect"), let roundtrip = dlsym(library, "wl_display_roundtrip"),
              let constructor = dlsym(library, "wl_proxy_marshal_array_constructor"),
              let versioned = dlsym(library, "wl_proxy_marshal_array_constructor_versioned"),
              let addListener = dlsym(library, "wl_proxy_add_listener"),
              let registryInterface = dlsym(library, "wl_registry_interface"),
              let compositorInterface = dlsym(library, "wl_compositor_interface"),
              let surfaceInterface = dlsym(library, "wl_surface_interface"),
              let display = unsafeBitCast(connect, to: Connect.self)(nil) else { return nil }
        self.library = library
        self.display = display
        // wl_display.get_registry is request 1 with a single new_id argument.
        var arguments = [UInt](repeating: 0, count: 4)
        guard let registry = arguments.withUnsafeMutableBytes({
            unsafeBitCast(constructor, to: Constructor.self)(display, 1, $0.baseAddress!, registryInterface)
        }) else { return nil }
        let listener = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 2)  // lives for the process; the test is short
        listener[0] = unsafeBitCast(registryGlobal, to: UnsafeRawPointer.self)
        listener[1] = unsafeBitCast(registryGlobalRemove, to: UnsafeRawPointer.self)
        _ = unsafeBitCast(addListener, to: AddListener.self)(registry, UnsafeMutableRawPointer(listener), nil)
        _ = unsafeBitCast(roundtrip, to: Roundtrip.self)(display)
        guard let compositor = announcedCompositor else { return nil }
        // wl_registry.bind (request 0): name, interface name, version, new_id.
        let version = min(compositor.version, 4)
        let interfaceName = compositorInterface.load(as: UnsafePointer<CChar>.self)
        var bind = [UInt](repeating: 0, count: 4)
        bind[0] = UInt(compositor.name)
        bind[1] = UInt(bitPattern: interfaceName)
        bind[2] = UInt(version)
        guard let compositorProxy = bind.withUnsafeMutableBytes({
            unsafeBitCast(versioned, to: VersionedConstructor.self)(registry, 0, $0.baseAddress!, compositorInterface, version)
        }) else { return nil }
        // wl_compositor.create_surface is request 0 with a single new_id argument.
        var create = [UInt](repeating: 0, count: 1)
        guard let surface = create.withUnsafeMutableBytes({
            unsafeBitCast(constructor, to: Constructor.self)(compositorProxy, 0, $0.baseAddress!, surfaceInterface)
        }) else { return nil }
        _ = unsafeBitCast(roundtrip, to: Roundtrip.self)(display)
        self.surface = surface
    }

    func close() {
        if let disconnect = dlsym(library, "wl_display_disconnect") { unsafeBitCast(disconnect, to: Disconnect.self)(display) }
        dlclose(library)
    }
}

@Test(.gpuExclusive) func vulkanPresentsToAWaylandSurface() async throws {
    guard VulkanBackend.requestAdapter() != nil, let wayland = WaylandSurface() else { return }
    defer { wayland.close() }
    let target = VulkanSurfaceTarget.wayland(display: UInt(bitPattern: wayland.display), surface: UInt(bitPattern: wayland.surface))
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
        let pipeline = try await device.createRenderPipeline(
            vertex: try compiler.compileRenderStage(.wgsl(source), entryPoint: "vs", to: .spirv),
            fragment: try compiler.compileRenderStage(.wgsl(source), entryPoint: "fs", to: .spirv), colorFormat: .bgra8Unorm)
        let frame = try await device.createTexture(.init(width: 64, height: 64, format: .bgra8Unorm,
                                                         usage: [.renderAttachment, .copySource]))
        _ = try await device.submitCommands([
            .render(pipeline: pipeline, target: frame, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3)])
        try await device.present(frame).wait()
        // The first swapchain is not a rebuild; presenting without an error is the evidence here.
        #expect(await backend.presentationRebuildsForTesting() == 0)
        await device.close()
    } catch { await device.close(); throw error }
}
#endif
