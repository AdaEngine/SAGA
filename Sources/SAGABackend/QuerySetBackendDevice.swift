import SAGATypes

public protocol QuerySetBackendDevice: BackendDevice {
    func allocateQuerySet(id: ResourceID, descriptor: QuerySetDescriptor) async throws
    func destroyQuerySet(id: ResourceID) async
}
