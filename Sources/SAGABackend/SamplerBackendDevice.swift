import SAGATypes

public protocol SamplerBackendDevice: BackendDevice {
    func allocateSampler(id: ResourceID, descriptor: SamplerDescriptor) async throws
    func destroySampler(id: ResourceID) async
}
