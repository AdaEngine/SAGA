import SAGATypes

/// Optional presentation to the backend's attached surface.
public protocol PresentationBackendDevice: RenderBackendDevice {
    /// Fixed when the backend is created.
    var supportsPresentation: Bool { get }

    /// Submit the surface copy in queue order and hold the source through GPU completion.
    func present(_ submission: Submission, source: ResourceID) async throws
}
