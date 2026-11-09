import SAGATypes

/// Optional backend support for bottom-level acceleration structure allocations.
public protocol AccelerationStructureBackendDevice: BackendDevice {
    func allocateAccelerationStructure(id: ResourceID, descriptor: AccelerationStructureDescriptor) async throws
    func destroyAccelerationStructure(id: ResourceID) async
}

/// Backend can bind top-level acceleration structures to compute pipelines.
public protocol ComputeAccelerationStructureBackendDevice: ComputeBackendDevice, AccelerationStructureBackendDevice {}
