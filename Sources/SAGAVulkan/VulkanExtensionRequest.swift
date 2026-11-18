#if os(macOS)
import os
#else
import Synchronization
#endif

/// Extra Vulkan extensions a host needs enabled for libraries that share SAGA's device (for example NVIDIA NGX).
///
/// Call ``set(instance:device:)`` before the first adapter or device is created. SAGA enables only the requested
/// extensions the driver offers, and never changes its own selection when nothing is requested.
public enum VulkanExtensionRequest {
    private struct Names: Sendable {
        var instance: [String] = []
        var device: [String] = []
    }

    #if os(macOS)
    // Keep the macOS 14 deployment target; Synchronization.Mutex requires macOS 15.
    private static let names = OSAllocatedUnfairLock(initialState: Names())
    #else
    private static let names = Mutex(Names())
    #endif

    public static func set(instance: [String], device: [String]) {
        names.withLock { $0 = Names(instance: instance, device: device) }
    }

    static var requestedInstance: [String] { names.withLock { $0.instance } }
    static var requestedDevice: [String] { names.withLock { $0.device } }

    /// The requested names the driver offers (in request order, without duplicates or ones already enabled),
    /// and those it does not.
    static func accepted(
        _ requested: [String], available: Set<String>, alreadyEnabled: Set<String>
    ) -> (enabled: [String], unavailable: [String]) {
        var seen = alreadyEnabled
        var enabled: [String] = []
        var unavailable: [String] = []
        for name in requested where seen.insert(name).inserted {
            if available.contains(name) { enabled.append(name) } else { unavailable.append(name) }
        }
        return (enabled, unavailable)
    }
}
