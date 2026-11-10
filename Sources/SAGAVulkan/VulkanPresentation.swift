import SAGATypes
import SAGABackend

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

    var instanceExtension: String {
        switch self {
        case .win32: "VK_KHR_win32_surface"
        case .xlib: "VK_KHR_xlib_surface"
        case .wayland: "VK_KHR_wayland_surface"
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
    static let usageTransferDestination: UInt32 = 0x2
    static let accessTransferWrite: UInt32 = 0x1000
    static let stageTransfer: UInt32 = 0x1000
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
    var presentMode: UInt32 = 2
    var clipped: UInt32 = 1
    var oldSwapchain: UInt64 = 0
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
        }
        createSurface = try load(name)
    }
}

/// A swapchain bound to one window. Used only from inside `VulkanDevice`.
final class VulkanPresenter {
    let functions: VulkanPresentationFunctions
    let surface: UInt64
    private(set) var swapchain: UInt64 = 0
    private(set) var rebuilds = 0
    private(set) var images: [UInt64] = []
    private(set) var extent = (width: UInt32(0), height: UInt32(0))
    private(set) var acquireSemaphores: [UInt64] = []
    private(set) var renderSemaphores: [UInt64] = []
    private var frame = 0
    private var wantSRGB = false
    private var formatCandidates: [(format: UInt32, colorSpace: UInt32)] = []

    init(adapter: VulkanAdapter, target: VulkanSurfaceTarget) throws {
        let functions = try VulkanPresentationFunctions(instance: adapter.instance, target: target)
        self.functions = functions
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
        }
        try vulkanCheck(result, "surface creation")
        surface = surfaceHandle
        var supported: UInt32 = 0
        let supportResult = functions.getSurfaceSupport(adapter.physical, adapter.queueFamily, surface, &supported)
        guard supportResult == VK.success, supported != 0 else {
            functions.destroySurface(instance, surface, nil)
            throw GPUError.unsupported("The Vulkan queue family cannot present to this window")
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
        _ = native.f.deviceWaitIdle(device)
        let old = swapchain
        let preferred: [UInt32] = srgb ? [50, 43] : [44, 37]
        let chosen = preferred.lazy.compactMap { wanted in self.formatCandidates.first { $0.format == wanted } }.first
            ?? formatCandidates[0]
        var info = VkSwapchainCreateInfoKHR()
        info.surface = surface
        info.minImageCount = max(capabilities[0], min(2, capabilities[1] == 0 ? 2 : capabilities[1]))
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
        try vulkanCheck(withUnsafePointer(to: &info) { functions.createSwapchain(device, UnsafeRawPointer($0), nil, &created) },
                        "swapchain creation")
        releaseSwapchain(device: device, handle: old)
        swapchain = created
        rebuilds += 1
        extent = current
        wantSRGB = srgb
        var imageCount: UInt32 = 0
        _ = functions.getSwapchainImages(device, created, &imageCount, nil)
        images = [UInt64](repeating: 0, count: Int(imageCount))
        _ = functions.getSwapchainImages(device, created, &imageCount, &images)
        var semaphoreInfo = VkSemaphoreCreateInfo()
        for _ in 0..<(images.count + 1) {
            var semaphore: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &semaphoreInfo) { functions.createSemaphore(device, UnsafeRawPointer($0), nil, &semaphore) },
                            "semaphore creation")
            acquireSemaphores.append(semaphore)
        }
        for _ in images {
            var semaphore: UInt64 = 0
            try vulkanCheck(withUnsafePointer(to: &semaphoreInfo) { functions.createSemaphore(device, UnsafeRawPointer($0), nil, &semaphore) },
                            "semaphore creation")
            renderSemaphores.append(semaphore)
        }
        return true
    }

    func nextAcquireSemaphore() -> UInt64 {
        frame = (frame + 1) % acquireSemaphores.count
        return acquireSemaphores[frame]
    }

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
        try requireActive()
        guard let presenter else { throw GPUError.unsupported("This Vulkan device has no presentation surface") }
        guard let texture = textures[source], texture.descriptor.usage.contains(.copySource),
              texture.descriptor.sampleCount == 1, texture.format.aspect == VK.Image.aspectColor else {
            throw GPUError.validation("Presentation requires a single-sampled copy-source color texture")
        }
        let srgb = texture.descriptor.format == .rgba8UnormSrgb || texture.descriptor.format == .bgra8UnormSrgb
        let f = native.f, device = native.device, queue = native.queue
        let functions = presenter.functions

        var index: UInt32 = 0
        var acquire: UInt64 = 0
        var attempts = 0
        while true {
            guard try presenter.prepare(native: native, adapter: adapter, srgb: srgb) else { return }  // minimized
            acquire = presenter.nextAcquireSemaphore()
            let result = functions.acquireNextImage(device, presenter.swapchain, UInt64.max, acquire, 0, &index)
            if result == VK.success || result == VKPresent.suboptimal { break }
            guard result == VKPresent.outOfDate, attempts < 2 else {
                try vulkanCheck(result, "swapchain image acquisition")
                return
            }
            presenter.invalidate()
            attempts += 1
        }

        let image = presenter.images[Int(index)]
        let renderDone = presenter.renderSemaphores[Int(index)]
        let extent = presenter.extent
        let sourceImage = texture.image
        let sourceExtent = texture.descriptor.mipExtent(0) ?? (width: texture.descriptor.width, height: texture.descriptor.height)
        try await runOneShot(waiting: acquire, signaling: renderDone) { commandBuffer in
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
        }

        var swapchain = presenter.swapchain
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
        if result == VKPresent.outOfDate || result == VKPresent.suboptimal {
            presenter.invalidate()  // the next present rebuilds the swapchain at the new size
        } else {
            try vulkanCheck(result, "presentation")
        }
    }
}

extension VulkanDevice {
    /// Number of swapchains built so far; the first present builds one and each window resize builds another.
    public func presentationRebuildsForTesting() -> Int { presenter?.rebuilds ?? 0 }
}
