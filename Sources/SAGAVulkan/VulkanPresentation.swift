import SAGATypes
import SAGABackend
#if os(Windows)
import WinSDK
#endif

// Presentation: a swapchain per attached window, fed by blitting an offscreen color texture.
//
// The instance enables the surface extensions that the loader offers, and the device enables VK_KHR_swapchain
// only when a surface is requested, so compute-only devices keep working on drivers without them.

/// A native window to present into. Handles are the platform pointers or identifiers, widened to `UInt`.
public enum VulkanSurfaceTarget: Sendable {
    /// `HINSTANCE` and `HWND`.
    case win32(hinstance: UInt, hwnd: UInt)
    /// `Display*` and `Window`.
    case xlib(display: UInt, window: UInt)
    /// `wl_display*` and `wl_surface*`.
    case wayland(display: UInt, surface: UInt)
    /// An `ANativeWindow*` retained by the caller until the Vulkan surface is removed.
    case android(window: UInt)

    var instanceExtension: String {
        switch self {
        case .win32: "VK_KHR_win32_surface"
        case .xlib: "VK_KHR_xlib_surface"
        case .wayland: "VK_KHR_wayland_surface"
        case .android: "VK_KHR_android_surface"
        }
    }
}

/// Runs `body` with a C array of NUL-terminated names.
func withCStringArray<R>(_ names: [String], _ body: (UnsafePointer<UnsafePointer<CChar>?>?) throws -> R) rethrows -> R {
    let pointers = names.map { name -> UnsafeMutablePointer<CChar> in
        let bytes = Array(name.utf8CString)
        let pointer = UnsafeMutablePointer<CChar>.allocate(capacity: bytes.count)
        pointer.initialize(from: bytes, count: bytes.count)
        return pointer
    }
    defer { pointers.forEach { $0.deallocate() } }
    let constant: [UnsafePointer<CChar>?] = pointers.map { UnsafePointer($0) }
    return try constant.withUnsafeBufferPointer { try body($0.baseAddress) }
}

enum VKPresent {
    static let swapchainExtension = "VK_KHR_swapchain"
    static let surfaceExtension = "VK_KHR_surface"
    static let outOfDate: VkResult = -1000001004
    static let suboptimal: VkResult = 1000001003
    static let layoutTransferDestination: UInt32 = 7
    static let layoutPresentSource: UInt32 = 1000001002
    static let immediate: UInt32 = 0
    static let fifo: UInt32 = 2
    static let mailbox: UInt32 = 1
    static let fifoLatestReady: UInt32 = 1_000_361_000
    static let fifoLatestReadyFeatureType: UInt32 = 1_000_361_000
    static let usageTransferDestination: UInt32 = 0x2
    static let accessTransferWrite: UInt32 = 0x1000
    static let stageTransfer: UInt32 = 0x1000

    static func fifoLatestReadyExtension(in available: Set<String>) -> String? {
        // The KHR promotion shares the feature structure and mode value with EXT.
        ["VK_KHR_present_mode_fifo_latest_ready", "VK_EXT_present_mode_fifo_latest_ready"].first(where: available.contains)
    }

    static func preferredMode(_ modes: [UInt32], fifoLatestReadyEnabled: Bool) -> UInt32 {
        if fifoLatestReadyEnabled, modes.contains(fifoLatestReady) { return fifoLatestReady }
        if modes.contains(mailbox) { return mailbox }
        return fifo
    }

    static func uncappedMode(_ modes: [UInt32]) -> UInt32 {
        modes.contains(immediate) ? immediate : fifo
    }
}

struct VkSwapchainCreateInfoKHR {
    var sType: UInt32 = 1000001000
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var surface: UInt64 = 0
    var minImageCount: UInt32 = 0
    var imageFormat: UInt32 = 0
    var imageColorSpace: UInt32 = 0
    var imageExtentWidth: UInt32 = 0
    var imageExtentHeight: UInt32 = 0
    var imageArrayLayers: UInt32 = 1
    var imageUsage: UInt32 = 0
    var imageSharingMode: UInt32 = 0
    var queueFamilyIndexCount: UInt32 = 0
    var pQueueFamilyIndices: UnsafePointer<UInt32>?
    var preTransform: UInt32 = 1
    var compositeAlpha: UInt32 = 1
    var presentMode: UInt32 = VKPresent.fifo
    var clipped: UInt32 = 1
    var oldSwapchain: UInt64 = 0

    mutating func configurePresentation(mode: UInt32, minimumImages: UInt32, maximumImages: UInt32) {
        presentMode = mode
        let desired: UInt32 = presentMode == VKPresent.fifo ? 2 : 3
        minImageCount = max(minimumImages, maximumImages == 0 ? desired : min(maximumImages, desired))
    }
}

struct VkPresentInfoKHR {
    var sType: UInt32 = 1000001001
    var pNext: UnsafeRawPointer?
    var waitSemaphoreCount: UInt32 = 0
    var pWaitSemaphores: UnsafePointer<UInt64>?
    var swapchainCount: UInt32 = 0
    var pSwapchains: UnsafePointer<UInt64>?
    var pImageIndices: UnsafePointer<UInt32>?
    var pResults: UnsafeMutablePointer<VkResult>?
}

struct VkSemaphoreCreateInfo {
    var sType: UInt32 = 9
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
}

struct VkImageBlit {
    var srcAspect: UInt32 = VK.Image.aspectColor
    var srcMipLevel: UInt32 = 0
    var srcBaseArrayLayer: UInt32 = 0
    var srcLayerCount: UInt32 = 1
    var srcOffsets: (VkOffset3D, VkOffset3D) = (VkOffset3D(), VkOffset3D())
    var dstAspect: UInt32 = VK.Image.aspectColor
    var dstMipLevel: UInt32 = 0
    var dstBaseArrayLayer: UInt32 = 0
    var dstLayerCount: UInt32 = 1
    var dstOffsets: (VkOffset3D, VkOffset3D) = (VkOffset3D(), VkOffset3D())
}

struct VkWin32SurfaceCreateInfoKHR {
    var sType: UInt32 = 1000009000
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var hinstance: UInt = 0
    var hwnd: UInt = 0
}

struct VkXlibSurfaceCreateInfoKHR {
    var sType: UInt32 = 1000004000
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var display: UInt = 0
    var window: UInt = 0
}

struct VkAndroidSurfaceCreateInfoKHR {
    var sType: UInt32 = 1000008000
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var window: UInt = 0
}

struct VkWaylandSurfaceCreateInfoKHR {
    var sType: UInt32 = 1000006000
    var pNext: UnsafeRawPointer?
    var flags: UInt32 = 0
    var display: UInt = 0
    var surface: UInt = 0
}

private typealias CreateSurface = @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult

/// Entry points that exist only when the surface and swapchain extensions are present.
struct VulkanPresentationFunctions: Sendable {
    let destroySurface: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let getSurfaceSupport: @convention(c) (UInt, UInt32, UInt64, UnsafeMutablePointer<UInt32>) -> VkResult
    let getSurfaceCapabilities: @convention(c) (UInt, UInt64, UnsafeMutableRawPointer) -> VkResult
    let getSurfacePresentModes: @convention(c) (UInt, UInt64, UnsafeMutablePointer<UInt32>, UnsafeMutablePointer<UInt32>?) -> VkResult
    let getSurfaceFormats: @convention(c) (UInt, UInt64, UnsafeMutablePointer<UInt32>, UnsafeMutableRawPointer?) -> VkResult
    let createSwapchain: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroySwapchain: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let getSwapchainImages: @convention(c) (UInt, UInt64, UnsafeMutablePointer<UInt32>, UnsafeMutablePointer<UInt64>?) -> VkResult
    let acquireNextImage: @convention(c) (UInt, UInt64, UInt64, UInt64, UInt64, UnsafeMutablePointer<UInt32>) -> VkResult
    let queuePresent: @convention(c) (UInt, UnsafeRawPointer) -> VkResult
    let createSemaphore: @convention(c) (UInt, UnsafeRawPointer, UnsafeRawPointer?, UnsafeMutablePointer<UInt64>) -> VkResult
    let destroySemaphore: @convention(c) (UInt, UInt64, UnsafeRawPointer?) -> Void
    let cmdBlitImage: @convention(c) (UInt, UInt64, UInt32, UInt64, UInt32, UInt32, UnsafeRawPointer, UInt32) -> Void
    fileprivate let createSurface: CreateSurface

    init(instance: VulkanInstance, target: VulkanSurfaceTarget) throws {
        guard let library = VulkanLibrary.open(), let address = library.symbol("vkGetInstanceProcAddr") else {
            throw GPUError.adapterUnavailable
        }
        let getInstanceProcAddr = unsafeBitCast(address, to: GetInstanceProcAddr.self)
        let handle = instance.handle
        func load<T>(_ name: String, as type: T.Type = T.self) throws -> T {
            guard let address = getInstanceProcAddr(handle, name) else {
                throw GPUError.unsupported("Vulkan loader does not export \(name)")
            }
            return unsafeBitCast(address, to: T.self)
        }
        destroySurface = try load("vkDestroySurfaceKHR")
        getSurfaceSupport = try load("vkGetPhysicalDeviceSurfaceSupportKHR")
        getSurfaceCapabilities = try load("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")
        getSurfacePresentModes = try load("vkGetPhysicalDeviceSurfacePresentModesKHR")
        getSurfaceFormats = try load("vkGetPhysicalDeviceSurfaceFormatsKHR")
        createSwapchain = try load("vkCreateSwapchainKHR")
        destroySwapchain = try load("vkDestroySwapchainKHR")
        getSwapchainImages = try load("vkGetSwapchainImagesKHR")
        acquireNextImage = try load("vkAcquireNextImageKHR")
        queuePresent = try load("vkQueuePresentKHR")
        createSemaphore = try load("vkCreateSemaphore")
        destroySemaphore = try load("vkDestroySemaphore")
        cmdBlitImage = try load("vkCmdBlitImage")
        let name: String = switch target {
        case .win32: "vkCreateWin32SurfaceKHR"
        case .xlib: "vkCreateXlibSurfaceKHR"
        case .wayland: "vkCreateWaylandSurfaceKHR"
        case .android: "vkCreateAndroidSurfaceKHR"
        }
        createSurface = try load(name)
    }
}

/// A swapchain bound to one window. Used only from inside `VulkanDevice`.
final class VulkanPresenter {
    let functions: VulkanPresentationFunctions
    let surface: UInt64
    private let target: VulkanSurfaceTarget
    private let onPresentationRetryChanged: (@Sendable (Bool) -> Void)?
    private(set) var swapchain: UInt64 = 0
    private(set) var rebuilds = 0
    private(set) var images: [UInt64] = []
    private(set) var extent = (width: UInt32(0), height: UInt32(0))
    private(set) var acquireSemaphores: [UInt64] = []
    private(set) var renderSemaphores: [UInt64] = []
    private var frame = 0
    private let presentMode: UInt32
    var usesMailbox: Bool { presentMode == VKPresent.mailbox }
    var usesImmediate: Bool { presentMode == VKPresent.immediate }
    private var wantSRGB = false
    private var formatCandidates: [(format: UInt32, colorSpace: UInt32)] = []

    init(
        adapter: VulkanAdapter, target: VulkanSurfaceTarget, prefersLatestFrame: Bool = false,
        prefersUncapped: Bool = false,
        supportsFIFOLatestReady: Bool = false, onPresentationRetryChanged: (@Sendable (Bool) -> Void)? = nil
    ) throws {
        let functions = try VulkanPresentationFunctions(instance: adapter.instance, target: target)
        self.functions = functions
        self.target = target
        self.onPresentationRetryChanged = onPresentationRetryChanged
        var surfaceHandle: UInt64 = 0
        let instance = adapter.instance.handle
        let create = functions.createSurface
        let result: VkResult
        switch target {
        case .win32(let hinstance, let hwnd):
            var info = VkWin32SurfaceCreateInfoKHR(hinstance: hinstance, hwnd: hwnd)
            result = withUnsafePointer(to: &info) { create(instance, UnsafeRawPointer($0), nil, &surfaceHandle) }
        case .xlib(let display, let window):
            var info = VkXlibSurfaceCreateInfoKHR(display: display, window: window)
            result = withUnsafePointer(to: &info) { create(instance, UnsafeRawPointer($0), nil, &surfaceHandle) }
        case .wayland(let display, let surface):
            var info = VkWaylandSurfaceCreateInfoKHR(display: display, surface: surface)
            result = withUnsafePointer(to: &info) { create(instance, UnsafeRawPointer($0), nil, &surfaceHandle) }
        case .android(let window):
            var info = VkAndroidSurfaceCreateInfoKHR(window: window)
            result = withUnsafePointer(to: &info) { create(instance, UnsafeRawPointer($0), nil, &surfaceHandle) }
        }
        try vulkanCheck(result, "surface creation")
        surface = surfaceHandle
        var supported: UInt32 = 0
        let supportResult = functions.getSurfaceSupport(adapter.physical, adapter.queueFamily, surface, &supported)
        guard supportResult == VK.success, supported != 0 else {
            functions.destroySurface(instance, surface, nil)
            throw GPUError.unsupported("The Vulkan queue family cannot present to this window")
        }
        if prefersLatestFrame || prefersUncapped, case .win32 = target {
            var modeCount: UInt32 = 0
            if functions.getSurfacePresentModes(adapter.physical, surfaceHandle, &modeCount, nil) == VK.success, modeCount > 0 {
                var modes = [UInt32](repeating: 0, count: Int(modeCount))
                let status = modes.withUnsafeMutableBufferPointer {
                    functions.getSurfacePresentModes(adapter.physical, surfaceHandle, &modeCount, $0.baseAddress)
                }
                let supported = status == VK.success ? Array(modes.prefix(Int(modeCount))) : []
                presentMode = prefersUncapped
                    ? VKPresent.uncappedMode(supported)
                    : VKPresent.preferredMode(supported, fifoLatestReadyEnabled: supportsFIFOLatestReady)
            } else {
                presentMode = VKPresent.fifo
            }
        } else {
            presentMode = VKPresent.fifo
        }
        var count: UInt32 = 0
        _ = functions.getSurfaceFormats(adapter.physical, surface, &count, nil)
        var raw = [UInt32](repeating: 0, count: Int(count) * 2)
        raw.withUnsafeMutableBytes { _ = functions.getSurfaceFormats(adapter.physical, surface, &count, $0.baseAddress) }
        formatCandidates = (0..<Int(count)).map { (raw[$0 * 2], raw[$0 * 2 + 1]) }
        guard !formatCandidates.isEmpty else {
            functions.destroySurface(instance, surface, nil)
            throw GPUError.unsupported("The window surface reports no formats")
        }
    }

    /// Rebuilds the swapchain when the window size or the wanted color encoding changed. Returns false while the
    /// window has no drawable area, for example when minimized.
    func prepare(native: VulkanNativeDevice, adapter: VulkanAdapter, srgb: Bool) throws -> Bool {
        var capabilities = [UInt32](repeating: 0, count: 13)
        let queried = capabilities.withUnsafeMutableBytes { functions.getSurfaceCapabilities(adapter.physical, surface, $0.baseAddress!) }
        try vulkanCheck(queried, "surface capability query")
        let current = (width: capabilities[2], height: capabilities[3])
        guard current.width != 0, current.height != 0, current.width != UInt32.max else { return false }
        if swapchain != 0, current == extent, srgb == wantSRGB { return true }
        let device = native.device
        try vulkanCheck(native.f.deviceWaitIdle(device), "wait for swapchain recreation")
        let old = swapchain
        let preferred: [UInt32] = srgb ? [50, 43] : [44, 37]
        let chosen = preferred.lazy.compactMap { wanted in self.formatCandidates.first { $0.format == wanted } }.first
            ?? formatCandidates[0]
        var info = VkSwapchainCreateInfoKHR()
        info.surface = surface
        info.configurePresentation(mode: presentMode, minimumImages: capabilities[0], maximumImages: capabilities[1])
        info.imageFormat = chosen.format
        info.imageColorSpace = chosen.colorSpace
        info.imageExtentWidth = current.width
        info.imageExtentHeight = current.height
        info.imageUsage = VKPresent.usageTransferDestination
        info.preTransform = capabilities[10]
        let alpha = capabilities[11]
        info.compositeAlpha = alpha & 1 != 0 ? 1 : (alpha & 2 != 0 ? 2 : (alpha & 4 != 0 ? 4 : 8))
        info.oldSwapchain = old
        var created: UInt64 = 0
        let result = withUnsafePointer(to: &info) { functions.createSwapchain(device, UnsafeRawPointer($0), nil, &created) }
        // vkCreateSwapchainKHR retires oldSwapchain even when it fails.
        releaseSwapchain(device: device, handle: old)
        swapchain = 0
        images.removeAll()
        try vulkanCheck(result, "swapchain creation")

        func makeSemaphore() throws -> UInt64 {
            var info = VkSemaphoreCreateInfo()
            var handle: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &info) { functions.createSemaphore(device, UnsafeRawPointer($0), nil, &handle) },
                            "semaphore creation")
            return handle
        }

        do {
            var imageCount: UInt32 = 0
            try vulkanCheck(functions.getSwapchainImages(device, created, &imageCount, nil), "swapchain image count")
            guard imageCount > 0 else { throw GPUError.backendFailure("Vulkan swapchain has no images") }
            var newImages = [UInt64](repeating: 0, count: Int(imageCount))
            try vulkanCheck(functions.getSwapchainImages(device, created, &imageCount, &newImages), "swapchain image query")
            guard imageCount > 0 else { throw GPUError.backendFailure("Vulkan swapchain has no images") }
            for _ in 0...imageCount { acquireSemaphores.append(try makeSemaphore()) }
            for _ in 0..<imageCount { renderSemaphores.append(try makeSemaphore()) }
            images = Array(newImages.prefix(Int(imageCount)))
        } catch {
            releaseSwapchain(device: device, handle: created)
            throw error
        }
        swapchain = created
        rebuilds += 1
        extent = current
        wantSRGB = srgb
        return true
    }

    func nextAcquireSemaphore() -> UInt64 {
        frame = (frame + 1) % acquireSemaphores.count
        return acquireSemaphores[frame]
    }

    func acquireTimeout(isOnlySurface: Bool) -> UInt64 {
        #if os(Windows)
        if case let .win32(_, window) = target {
            let handle = HWND(bitPattern: window)
            if GetForegroundWindow() == handle { return 50_000_000 }
            // A standalone visible window can pace itself even when another process has focus.
            // A secondary surface must never stall a different window on the same graphics queue.
            return isOnlySurface && IsWindowVisible(handle) && !IsIconic(handle) ? 16_000_000 : 0
        }
        #endif
        return UInt64.max
    }

    func reportPresentationRetry(_ needed: Bool) { onPresentationRetryChanged?(needed) }

    /// Invalidates the swapchain so the next `prepare` rebuilds it.
    func invalidate() { extent = (0, 0) }

    private func releaseSwapchain(device: UInt, handle: UInt64) {
        acquireSemaphores.forEach { functions.destroySemaphore(device, $0, nil) }
        renderSemaphores.forEach { functions.destroySemaphore(device, $0, nil) }
        acquireSemaphores = []
        renderSemaphores = []
        if handle != 0 { functions.destroySwapchain(device, handle, nil) }
    }

    func destroy(native: VulkanNativeDevice) {
        _ = native.f.deviceWaitIdle(native.device)
        releaseSwapchain(device: native.device, handle: swapchain)
        swapchain = 0
        functions.destroySurface(native.instance.handle, surface, nil)
    }
}


extension VulkanDevice: PresentationBackendDevice {
    public func present(_ submission: Submission, source: ResourceID) async throws {
        guard let presenter else { throw GPUError.unsupported("This Vulkan device has no presentation surface") }
        try await present(submission, source: source, presenter: presenter)
    }

    public func present(_ submission: Submission, source: ResourceID, surface: UInt32) async throws {
        guard supportsPresentation else { throw GPUError.unsupported("This Vulkan device has no presentation surface") }
        guard let presenter = surface == 0 ? presenter : extraPresenters[surface] else {
            throw GPUError.validation("Unknown presentation surface")
        }
        try await present(submission, source: source, presenter: presenter)
    }

    /// Attaches a window to a device created with swapchain support.
    public func addSurface(_ target: VulkanSurfaceTarget) throws -> UInt32 {
        try addSurface(target, onPresentationRetryChanged: nil)
    }

    /// Attaches a surface and reports whether the latest backbuffer needs another present attempt.
    public func addSurface(_ target: VulkanSurfaceTarget, onPresentationRetryChanged: (@Sendable (Bool) -> Void)?) throws -> UInt32 {
        try attachSurface(target, prefersLatestFrame: false, onPresentationRetryChanged: onPresentationRetryChanged).id
    }

    /// Requests the freshest complete image at refresh on Win32. FIFO Latest Ready is preferred
    /// when available, with mailbox and FIFO as fallbacks. Pace production explicitly for mailbox.
    public func attachSurface(
        _ target: VulkanSurfaceTarget, prefersLatestFrame: Bool,
        onPresentationRetryChanged: (@Sendable (Bool) -> Void)?
    ) throws -> (id: UInt32, usesMailbox: Bool) {
        let (id, presenter) = try makeSurface(
            target, prefersLatestFrame: prefersLatestFrame, prefersUncapped: false,
            onPresentationRetryChanged: onPresentationRetryChanged
        )
        return (id, presenter.usesMailbox)
    }

    /// Requests immediate Win32 presentation and reports whether the surface supports it. Falls back to FIFO.
    public func attachUncappedSurface(
        _ target: VulkanSurfaceTarget,
        onPresentationRetryChanged: (@Sendable (Bool) -> Void)?
    ) throws -> (id: UInt32, usesImmediate: Bool) {
        let (id, presenter) = try makeSurface(
            target, prefersLatestFrame: false, prefersUncapped: true,
            onPresentationRetryChanged: onPresentationRetryChanged
        )
        return (id, presenter.usesImmediate)
    }

    private func makeSurface(
        _ target: VulkanSurfaceTarget, prefersLatestFrame: Bool, prefersUncapped: Bool,
        onPresentationRetryChanged: (@Sendable (Bool) -> Void)?
    ) throws -> (UInt32, VulkanPresenter) {
        guard supportsPresentation else { throw GPUError.unsupported("This Vulkan device has no presentation support") }
        let id = nextSurfaceID
        let presenter = try VulkanPresenter(
            adapter: adapter, target: target, prefersLatestFrame: prefersLatestFrame, prefersUncapped: prefersUncapped,
            supportsFIFOLatestReady: native.fifoLatestReady, onPresentationRetryChanged: onPresentationRetryChanged
        )
        extraPresenters[id] = presenter
        nextSurfaceID += 1
        return (id, presenter)
    }

    /// Detaches a surface. The device can still attach replacement surfaces.
    public func removeSurface(_ id: UInt32) {
        if id == 0 {
            presenter?.destroy(native: native)
            presenter = nil
        } else {
            extraPresenters.removeValue(forKey: id)?.destroy(native: native)
        }
    }

    private func present(_ submission: Submission, source: ResourceID, presenter: VulkanPresenter) async throws {
        try requireActive()
        guard let texture = textures[source], texture.descriptor.usage.contains(.copySource),
              texture.descriptor.sampleCount == 1, texture.format.aspect == VK.Image.aspectColor else {
            throw GPUError.validation("Presentation requires a single-sampled copy-source color texture")
        }
        let srgb = texture.descriptor.format == .rgba8UnormSrgb || texture.descriptor.format == .bgra8UnormSrgb
        let f = native.f, device = native.device, queue = native.queue
        let functions = presenter.functions
        let isOnlySurface = extraPresenters.count == (self.presenter == nil ? 1 : 0)

        var index: UInt32 = 0
        var acquire: UInt64 = 0
        var attempts = 0
        while true {
            guard try presenter.prepare(native: native, adapter: adapter, srgb: srgb) else {
                presenter.reportPresentationRetry(true)
                return  // minimized
            }
            acquire = presenter.nextAcquireSemaphore()
            let timeout = presenter.acquireTimeout(isOnlySurface: isOnlySurface)
            let result = functions.acquireNextImage(device, presenter.swapchain, timeout, acquire, 0, &index)
            if result == VK.success || result == VKPresent.suboptimal { break }
            if result == VK.notReady || result == VK.timeout {
                presenter.reportPresentationRetry(true)
                return
            }
            guard result == VKPresent.outOfDate else {
                try vulkanCheck(result, "swapchain image acquisition")
                return
            }
            presenter.invalidate()
            if attempts >= 2 {
                presenter.reportPresentationRetry(true)
                return
            }
            attempts += 1
        }

        let image = presenter.images[Int(index)]
        let renderDone = presenter.renderSemaphores[Int(index)]
        let generation = presenter.rebuilds
        let acquiredSwapchain = presenter.swapchain
        let extent = presenter.extent
        let sourceImage = texture.image
        let sourceExtent = texture.descriptor.mipExtent(0) ?? (width: texture.descriptor.width, height: texture.descriptor.height)
        var batch = PendingBatch()
        do {
            let commandBuffer = try acquireCommandBuffer(batch: &batch)
            var begin = VkCommandBufferBeginInfo()
            try vulkanCheck(withUnsafePointer(to: &begin) { f.beginCommandBuffer(commandBuffer, UnsafeRawPointer($0)) }, "presentation recording")
            func transition(from old: UInt32, to new: UInt32, access: UInt32) {
                var barrier = VkImageMemoryBarrier()
                barrier.dstAccessMask = access
                barrier.oldLayout = old
                barrier.newLayout = new
                barrier.image = image
                withUnsafePointer(to: &barrier) {
                    f.cmdPipelineBarrier(commandBuffer, VK.pipelineStageAllCommands, VK.pipelineStageAllCommands, 0,
                                         0, nil, 0, nil, 1, UnsafeRawPointer($0))
                }
            }
            transition(from: VK.Image.layoutUndefined, to: VKPresent.layoutTransferDestination, access: VKPresent.accessTransferWrite)
            var region = VkImageBlit()
            region.srcOffsets = (VkOffset3D(), VkOffset3D(x: Int32(sourceExtent.width), y: Int32(sourceExtent.height), z: 1))
            region.dstOffsets = (VkOffset3D(), VkOffset3D(x: Int32(extent.width), y: Int32(extent.height), z: 1))
            withUnsafePointer(to: &region) {
                functions.cmdBlitImage(commandBuffer, sourceImage, VK.Image.layoutGeneral, image,
                                       VKPresent.layoutTransferDestination, 1, UnsafeRawPointer($0), 0)
            }
            transition(from: VKPresent.layoutTransferDestination, to: VKPresent.layoutPresentSource, access: 0)
            try vulkanCheck(f.endCommandBuffer(commandBuffer), "presentation recording")
            try submitBatch(&batch, waiting: acquire, signaling: renderDone)
        } catch {
            // The acquired image and semaphore were not consumed. Recreate them before another acquire.
            presenter.invalidate()
            if case GPUError.deviceLost = error { terminal = true }
            releaseBatch(batch)
            throw error
        }

        // Queue the present immediately. Its semaphore orders the copy; the fence still pins the source and command pool.
        var swapchain = acquiredSwapchain
        var imageIndex = index
        var wait = renderDone
        var info = VkPresentInfoKHR()
        info.waitSemaphoreCount = 1
        info.swapchainCount = 1
        let result = withUnsafePointer(to: &wait) { waitPointer in
            withUnsafePointer(to: &swapchain) { swapchainPointer in
                withUnsafePointer(to: &imageIndex) { indexPointer in
                    info.pWaitSemaphores = waitPointer
                    info.pSwapchains = swapchainPointer
                    info.pImageIndices = indexPointer
                    return withUnsafePointer(to: &info) { functions.queuePresent(queue, UnsafeRawPointer($0)) }
                }
            }
        }
        try await waitForBatch(batch)
        releaseBatch(batch, recycle: true)
        if result != VK.success, presenter.rebuilds == generation, presenter.swapchain == acquiredSwapchain {
            // On some failures the presentation wait was never queued and renderDone stays signaled.
            presenter.invalidate()
        }
        if result == VKPresent.outOfDate {
            presenter.reportPresentationRetry(true)
            return
        }
        if result == VK.success || result == VKPresent.suboptimal {
            presenter.reportPresentationRetry(false)
            return
        }
        if result == VK.deviceLost { terminal = true }
        try vulkanCheck(result, "presentation")
    }
}

extension VulkanDevice {
    /// Number of swapchains built so far; the first present builds one and each window resize builds another.
    public func presentationRebuildsForTesting() -> Int { presenter?.rebuilds ?? 0 }

    func presentationRebuildsForTesting(surface id: UInt32) -> Int {
        id == 0 ? presentationRebuildsForTesting() : extraPresenters[id]?.rebuilds ?? 0
    }
}
