import SAGATypes

public protocol ExternalTextureBackendDevice: BackendDevice {
    var supportsExternalTextures: Bool { get }
    func allocateExternalTexture(id: ResourceID, descriptor: ExternalTextureDescriptor) async throws
    func destroyExternalTexture(id: ResourceID) async
}

extension ExternalTextureBackendDevice {
    public var supportsExternalTextures: Bool { true }
}

public protocol PlanarExternalTextureBackendDevice: ExternalTextureBackendDevice {
    func allocatePlanarExternalTexture(id: ResourceID, descriptor: PlanarExternalTextureDescriptor) async throws
}

extension PlanarExternalTextureBackendDevice {
    public func allocateExternalTexture(id: ResourceID, descriptor: ExternalTextureDescriptor) async throws {
        throw GPUError.unsupported("VideoFrame textures require a browser backend")
    }
}
