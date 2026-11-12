import SAGATypes

/// Optional presentation to the backend's attached surface.
public protocol PresentationBackendDevice: RenderBackendDevice {
    /// Fixed when the backend is created.
    var supportsPresentation: Bool { get }

    /// Submit the surface copy in queue order and hold the source through GPU completion.
    func present(_ submission: Submission, source: ResourceID) async throws

    /// Surface zero is the default surface. Backends with dynamic surfaces assign nonzero IDs.
    func present(_ submission: Submission, source: ResourceID, surface: UInt32) async throws
}

extension PresentationBackendDevice {
    public func present(_ submission: Submission, source: ResourceID, surface: UInt32) async throws {
        guard surface == 0 else { throw GPUError.unsupported("Backend does not have the requested surface") }
        try await present(submission, source: source)
    }
}
