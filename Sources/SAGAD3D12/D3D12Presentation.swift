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

    /// Builds the swap chain for the given size and format when they changed.
    func prepare(queue: COMObject, width: UInt32, height: UInt32, format: UInt32) throws {
        if swapChain != nil, self.width == width, self.height == height, self.format == format { return }
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

extension D3D12Device: PresentationBackendDevice {
    public func present(_ submission: Submission, source: ResourceID) async throws {
        try requireActive()
        guard let presenter else { throw GPUError.unsupported("This Direct3D 12 device has no presentation window") }
        guard let texture = textures[source], texture.descriptor.usage.contains(.copySource),
              texture.descriptor.sampleCount == 1, texture.descriptor.dimension == .d2, texture.descriptor.arrayLayerCount == 1,
              texture.format == 87 || texture.format == 28 else {
            throw GPUError.validation("Direct3D 12 presentation needs a single-sampled rgba8unorm or bgra8unorm copy-source texture")
        }
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

        typealias CreateAllocator = @convention(c) (UnsafeMutableRawPointer, UInt32, UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        typealias CreateList = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32, UnsafeMutableRawPointer, UnsafeMutableRawPointer?,
                                               UnsafeRawPointer, UnsafeMutablePointer<UnsafeMutableRawPointer?>) -> HRESULT
        var allocatorIID = IID.commandAllocator, listIID = IID.commandList
        var allocatorObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(9, CreateAllocator.self)(device.pointer, D3D12.commandListDirect, &allocatorIID, &allocatorObject),
                     "command allocator creation")
        let allocator = COMObject(allocatorObject!)
        var listObject: UnsafeMutableRawPointer?
        try d3dCheck(device.slot(12, CreateList.self)(device.pointer, 0, D3D12.commandListDirect, allocator.pointer, nil, &listIID,
                                                      &listObject), "command list creation")
        let list = COMObject(listObject!)
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
        typealias Signal = @convention(c) (UnsafeMutableRawPointer, UnsafeMutableRawPointer, UInt64) -> HRESULT
        var lists: [UnsafeMutableRawPointer?] = [list.pointer]
        queue.slot(10, Execute.self)(queue.pointer, 1, &lists)
        fenceValue += 1
        let target = fenceValue
        try d3dCheck(queue.slot(14, Signal.self)(queue.pointer, fence.pointer, target), "fence signal")
        do { try await waitForFence(target) } catch {
            terminal = true
            quarantine.append(Retained(objects: [allocator, list, backBuffer, chain3]))
            throw error
        }
        typealias Present = @convention(c) (UnsafeMutableRawPointer, UInt32, UInt32) -> HRESULT
        try d3dCheck(chain.slot(8, Present.self)(chain.pointer, 1, 0), "presentation")
    }

    /// Number of swap chains built so far; the first present builds one and each size or format change builds another.
    public func presentationRebuildsForTesting() -> Int { presenter?.rebuilds ?? 0 }
}
#endif
