#if os(Windows)
import WinSDK
import SAGATypes
import SAGABackend

// Presentation through a DXGI flip-model swap chain bound to a window. The swap chain takes the size and format of the
// presented texture (BGRA8 or RGBA8), so a plain copy fills the back buffer.

final class D3D12Presenter: @unchecked Sendable {
    let factory: COMObject
    let window: UInt
    var swapChain: COMObject?
    var width: UInt32 = 0
    var height: UInt32 = 0
    var format: UInt32 = 0
    private(set) var rebuilds = 0
    var presentationsInFlight = 0
    var resizeWaiters: [CheckedContinuation<Void, Never>] = []

    init(window: UInt) throws {
        guard let module = "dxgi.dll".withCString(encodedAs: UTF16.self, { LoadLibraryW($0) }),
              let create = GetProcAddress(module, "CreateDXGIFactory2") else { throw GPUError.unsupported("dxgi.dll is unavailable") }
        typealias CreateFactory = @convention(c) (UInt32, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var iid = D3DGUID("50c83a1c-e072-4c48-87b0-3630fa36a6d0")  // IDXGIFactory2
        var object: UnsafeMutableRawPointer?
        try d3dCheck(unsafeBitCast(create, to: CreateFactory.self)(0, &iid, &object), "DXGI factory creation")
        factory = COMObject(object!)
        self.window = window
    }

    func needsRebuild(width: UInt32, height: UInt32, format: UInt32) -> Bool {
        swapChain == nil || self.width != width || self.height != height || self.format != format
    }

    func finishPresentation() {
        presentationsInFlight -= 1
        if presentationsInFlight == 0 {
            let waiters = resizeWaiters
            resizeWaiters.removeAll(keepingCapacity: true)
            waiters.forEach { $0.resume() }
        }
    }

    /// Builds the swap chain for the given size and format when they changed.
    func prepare(queue: COMObject, width: UInt32, height: UInt32, format: UInt32) throws {
        guard needsRebuild(width: width, height: height, format: format) else { return }
        swapChain = nil
        typealias CreateForHwnd = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt, UnsafeRawPointer, UnsafeRawPointer?,
                                                  UnsafeMutableRawPointer?, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        // DXGI_SWAP_CHAIN_DESC1: size, format, stereo, sample count and quality, usage, buffers, scaling, effect, alpha, flags.
        let description: [UInt32] = [width, height, format, 0, 1, 0, 0x20, 2, 0, 4, 0, 0]
        var object: UnsafeMutableRawPointer?
        let result: HRESULT = description.withUnsafeBytes {
            factory.slot(15, CreateForHwnd.self)(factory.pointer, queue.pointer, window, $0.baseAddress!, nil, nil, &object)
        }
        try d3dCheck(result, "swap chain creation")
        swapChain = COMObject(object!)
        self.width = width
        self.height = height
        self.format = format
        rebuilds += 1
    }
}

extension D3D12Device: PipelinedPresentationBackendDevice {
    public func present(_ submission: Submission, source: ResourceID) async throws {
        try await present(submission, source: source, surface: 0)
    }

    public func present(_ submission: Submission, source: ResourceID, surface: UInt32) async throws {
        try await present(source: source, surface: surface, onSubmitted: nil)
    }

    public func present(
        _ submission: Submission, source: ResourceID, surface: UInt32, onSubmitted: @escaping @Sendable () -> Void
    ) async throws {
        try await present(source: source, surface: surface, onSubmitted: onSubmitted)
    }

    private struct PendingPresentation {
        let recording: CommandRecording
        var retained: Retained
        let fence: UInt64
        let result: HRESULT
    }

    private func present(source: ResourceID, surface: UInt32, onSubmitted: (@Sendable () -> Void)?) async throws {
        try requireActive()
        guard supportsPresentation else { throw GPUError.unsupported("This Direct3D 12 device has no presentation support") }
        guard let presenter = presenters[surface] else { throw GPUError.validation("Unknown Direct3D 12 presentation surface") }
        guard let texture = textures[source], texture.descriptor.usage.contains(.copySource),
              texture.descriptor.sampleCount == 1, texture.descriptor.dimension == .d2, texture.descriptor.arrayLayerCount == 1,
              texture.format == 87 || texture.format == 28 else {
            throw GPUError.validation("Direct3D 12 presentation needs a single-sampled rgba8unorm or bgra8unorm copy-source texture")
        }
        // DXGI allows one flip-model chain per window. Drop every GPU reference
        // to the previous chain before rebuilding it for the resized drawable.
        while true {
            if presenter.presentationsInFlight > 0,
               presenter.needsRebuild(width: texture.descriptor.width, height: texture.descriptor.height, format: texture.format) {
                await withCheckedContinuation { presenter.resizeWaiters.append($0) }
            } else if onSubmitted != nil, let oldest = try pendingFenceAtCapacity() {
                try await waitForFence(oldest)
            } else {
                break
            }
            try requireActive()
            guard presenters[surface] === presenter else { throw GPUError.validation("Direct3D 12 presentation surface was removed") }
        }
        var pending = try queuePresentation(texture, to: presenter)
        presenter.presentationsInFlight += 1
        defer {
            // Release the swapchain and backbuffer references before resuming a resize.
            pending.retained.objects.removeAll()
            presenter.finishPresentation()
        }
        onSubmitted?()
        do {
            try await waitForFence(pending.fence)
        } catch {
            terminal = true
            quarantine.append(pending.retained)
            throw error
        }
        recycleCommandRecording(pending.recording)
        try d3dCheck(pending.result, "presentation")
    }

    private func queuePresentation(_ texture: Texture, to presenter: D3D12Presenter) throws -> PendingPresentation {
        try presenter.prepare(queue: queue, width: texture.descriptor.width, height: texture.descriptor.height, format: texture.format)
        let chain = presenter.swapChain!

        typealias CurrentIndex = @convention(c) (UnsafeMutableRawPointer) -> UInt32
        typealias GetBuffer = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias QueryInterface = @convention(c) (UnsafeMutableRawPointer, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var chain3IID = D3DGUID("94d99bdb-f1f8-4ab0-b236-7da0170edab1")
        var chain3Object: UnsafeMutableRawPointer?
        try d3dCheck(chain.slot(0, QueryInterface.self)(chain.pointer, &chain3IID, &chain3Object), "swap chain interface")
        let chain3 = COMObject(chain3Object!)
        let index = chain3.slot(36, CurrentIndex.self)(chain3.pointer)
        var resourceIID = IID.resource
        var backObject: UnsafeMutableRawPointer?
        try d3dCheck(chain.slot(9, GetBuffer.self)(chain.pointer, index, &resourceIID, &backObject), "back buffer access")
        let backBuffer = COMObject(backObject!)

        let recording = try beginCommandRecording()
        let list = recording.list
        typealias Barrier = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer) -> Void
        typealias CopyResource = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UnsafeMutableRawPointer) -> Void
        func transition(_ resource: COMObject, _ before: UInt32, _ after: UInt32) {
            var barrier = D3D12ResourceBarrier(type: D3D12.barrierTransition, resource: resource.pointer,
                                               subresource: 0xFFFF_FFFF, stateBefore: before, stateAfter: after)
            list.slot(26, Barrier.self)(list.pointer, 1, &barrier)
        }
        transition(texture.resource, D3D12.stateCommon, D3D12.stateCopySource)
        transition(backBuffer, D3D12.stateCommon, D3D12.stateCopyDestination)
        list.slot(17, CopyResource.self)(list.pointer, backBuffer.pointer, texture.resource.pointer)
        transition(backBuffer, D3D12.stateCopyDestination, D3D12.stateCommon)
        transition(texture.resource, D3D12.stateCopySource, D3D12.stateCommon)
        typealias Close = @convention(c) (UnsafeMutableRawPointer) -> HRESULT
        try d3dCheck(list.slot(9, Close.self)(list.pointer), "command recording")

        typealias Execute = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafePointer<UnsafeMutableRawPointer?>) -> Void
        typealias Present = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32) -> HRESULT
        var lists: [UnsafeMutableRawPointer?] = [list.pointer]
        queue.slot(10, Execute.self)(queue.pointer, 1, &lists)
        let presented = chain.slot(8, Present.self)(chain.pointer, 1, 0)
        // The surface owner may resize or detach after its presentation ticket retires.
        // Signal after Present so the fence covers DXGI's work as well as the copy.
        let retained = Retained(objects: [recording.allocator, list, texture.resource, backBuffer, chain, chain3])
        do {
            return PendingPresentation(recording: recording, retained: retained, fence: try signalSubmissionFence(), result: presented)
        } catch {
            terminal = true
            quarantine.append(retained)
            throw error
        }
    }

    /// Attaches an existing window to a presentation-enabled device.
    public func addSurface(windowHandle: UInt) throws -> UInt32 {
        try requireActive()
        guard supportsPresentation else { throw GPUError.unsupported("This Direct3D 12 device has no presentation support") }
        guard let window = HWND(bitPattern: windowHandle), IsWindow(window),
              !presenters.values.contains(where: { $0.window == windowHandle }) else {
            throw GPUError.validation("Direct3D 12 needs a valid Win32 window without an attached surface")
        }
        guard nextSurfaceID < UInt32.max else { throw GPUError.outOfMemory }
        let id = nextSurfaceID
        presenters[id] = try D3D12Presenter(window: windowHandle)
        nextSurfaceID += 1
        return id
    }

    /// Detaches a surface after the owner has awaited its last presentation ticket.
    public func removeSurface(_ id: UInt32) {
        presenters[id] = nil
    }

    /// Number of swap chain builds for this surface, including the initial presentation.
    public func presentationRebuildsForTesting(surface: UInt32 = 0) -> Int { presenters[surface]?.rebuilds ?? 0 }
}
#endif
