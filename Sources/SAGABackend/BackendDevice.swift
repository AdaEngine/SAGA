import SAGATypes

/// Trusted boundary: initialize allocations; retire work before successful completion.
/// On terminal timeout, a backend must quarantine native objects until actual GPU retirement.
public protocol BackendDevice: Sendable {
    var capabilities: DeviceCapabilities { get }
    func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws
    func destroyBuffer(id: ResourceID) async
    func submit(_ submission: Submission) async throws
    func close(owner: UInt64) async
}

public extension BackendDevice {
    func close(owner: UInt64) async {}
}
