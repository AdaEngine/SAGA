import SAGATypes
import SAGABackend

/// Future JS bridge owns all browser objects and resolves submit only at queue completion.
public protocol BrowserGPUHost: BackendDevice {}

public actor BrowserBackend: BackendDevice {
    public nonisolated let capabilities: DeviceCapabilities
    private let host: any BrowserGPUHost
    public init(host: any BrowserGPUHost) {
        self.host = host; self.capabilities = host.capabilities
    }
    public func allocateBuffer(id: ResourceID, descriptor: BufferDescriptor) async throws {
        try await host.allocateBuffer(id: id, descriptor: descriptor)
    }
    public func destroyBuffer(id: ResourceID) async { await host.destroyBuffer(id: id) }
    public func close(owner: UInt64) async { await host.close(owner: owner) }
    public func submit(_ submission: Submission) async throws { try await host.submit(submission) }
}
