#if os(macOS) && canImport(AppKit) && canImport(QuartzCore)
import AppKit
import Foundation
import QuartzCore
import SAGACompiler
import SAGACore
import SAGAMetal
import SAGAShaderIR
import SAGATypes

@main
struct MetalWindowDiagnostic {
    @MainActor static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        let delegate = WindowDiagnostic(smoke: CommandLine.arguments.contains("--smoke"))
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}

@MainActor
private final class WindowDiagnostic: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private let smoke: Bool
    private var window: NSWindow?
    private var drawing: Task<Void, Never>?
    private var closing = false
    private var presented: [MetalPresentedFrame] = []

    init(smoke: Bool) { self.smoke = smoke }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 288, height: 288),
            styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.title = "SAGA Metal Presentation"
        window.isReleasedWhenClosed = false
        window.delegate = self
        let view = MetalWindowView(frame: window.contentView?.bounds ?? .zero)
        view.wantsLayer = true
        window.contentView = view
        guard let layer = view.layer as? CAMetalLayer else {
            report("FAIL: AppKit did not install the Metal backing layer")
            exit(1)
        }
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate()
        self.window = window

        let surface = MetalPresentationSurface(layer: layer) { [weak self] frame in
            self?.presented.append(frame)
        }
        drawing = Task {
            do {
                try await drawFrames(surface: surface)
                if smoke { NSApplication.shared.terminate(nil) }
            } catch {
                if !closing {
                    report("FAIL: \(error)")
                    exit(1)
                }
            }
        }
    }

    func windowWillClose(_ notification: Notification) {
        guard !closing else { return }
        closing = true
        drawing?.cancel()
        Task {
            await drawing?.value
            NSApplication.shared.terminate(nil)
        }
    }

    private func drawFrames(surface: MetalPresentationSurface) async throws {
        let source = ShaderInput.wgsl("""
            @vertex fn vertexMain(@builtin(vertex_index) id: u32) -> @builtin(position) vec4f {
                if id == 0u { return vec4f(-1.0, 1.0, 0.25, 1.0); }
                if id == 1u { return vec4f(1.0, 1.0, 0.25, 1.0); }
                return vec4f(-1.0, -1.0, 0.25, 1.0);
            }
            @fragment fn fragmentMain() -> @location(0) vec4f { return vec4f(0.0, 1.0, 0.0, 1.0); }
            """)
        let compiler = IndependentShaderCompiler()
        let vertex = try compiler.compileRenderStage(source, entryPoint: "vertexMain", to: .msl)
        let fragment = try compiler.compileRenderStage(source, entryPoint: "fragmentMain", to: .msl)
        let device = try await Device.make(backend: MetalBackend.makeDevice(surface: surface))
        do {
            try await draw(device, vertex: vertex, fragment: fragment, width: 96, height: 96, format: .rgba8Unorm)
            window?.setContentSize(NSSize(width: 384, height: 288))
            try await draw(device, vertex: vertex, fragment: fragment, width: 128, height: 96, format: .bgra8Unorm)
            await device.close()
            report("PASS: 2 window frames; pixel checks passed; resize passed")
        } catch {
            await device.close()
            throw error
        }
    }

    private func draw(
        _ device: Device, vertex: RenderStageArtifact, fragment: RenderStageArtifact,
        width: UInt32, height: UInt32, format: TextureFormat
    ) async throws {
        let pipeline = try await device.createRenderPipeline(vertex: vertex, fragment: fragment, colorFormat: format)
        let texture = try await device.createTexture(.init(width: width, height: height, format: format,
                                                           usage: [.renderAttachment, .copySource]))
        let stride = (width * 4 + 255) & ~255
        let size = UInt64(height - 1) * UInt64(stride) + UInt64(width) * 4
        let read = try await device.createBuffer(.init(size: size, usage: [.copyDestination, .mapRead]))
        let ticket = try await device.submitCommands([
            .render(pipeline: pipeline, target: texture, clear: .init(red: 0, green: 0, blue: 1, alpha: 1), vertexCount: 3),
            .copyTextureToBuffer(source: texture, destination: read, bytesPerRow: stride),
            .read(buffer: read, byteCount: size),
        ])
        let frame = try await device.present(texture)
        try await device.destroyTexture(texture)
        let readback = try await ticket.wait()
        let blue: UInt32 = format == .rgba8Unorm ? 0xffff0000 : 0xff0000ff
        guard readback.count == 1, readback[0].first == 0xff00ff00, readback[0].last == blue else {
            throw GPUError.backendFailure("Window source pixel checks failed")
        }
        try await frame.wait()
        try await device.destroyBuffer(read)
        let deadline = ContinuousClock.now + .seconds(3)
        while !presented.contains(where: { $0.width == width && $0.height == height && $0.presentedTime > 0 }) {
            guard ContinuousClock.now < deadline else {
                let visible = window?.occlusionState.contains(.visible) == true
                throw GPUError.backendFailure("No displayed \(width)x\(height) drawable (visible=\(visible), callbacks=\(presented))")
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        report("PRESENTED \(width)x\(height) \(format)")
    }

    private func report(_ message: String) {
        print(message)
        guard Bundle.main.bundleURL.pathExtension == "app" else { return }
        let directory = Bundle.main.bundleURL.deletingLastPathComponent()
        let log = directory.appendingPathComponent("saga-metal-window.log")
        let data = Data((message + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: log) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? data.write(to: log)
        }
    }
}

@MainActor
private final class MetalWindowView: NSView {
    override func makeBackingLayer() -> CALayer { CAMetalLayer() }
}
#else
@main
struct MetalWindowDiagnostic {
    static func main() { print("The Metal window diagnostic requires macOS.") }
}
#endif
