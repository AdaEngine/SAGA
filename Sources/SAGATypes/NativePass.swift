/// The backend-native view of one submission, handed to a ``NativePass`` body while that submission is recorded.
///
/// Each backend that supports native passes provides its own conforming type (for example `VulkanNativePassContext`
/// in `SAGAVulkan`); a body downcasts to the one it was written for. Handles are valid only for the duration of the
/// call and must not be retained: the command stream they belong to is submitted after the body returns.
public protocol NativePassContext {}

/// A pass whose commands are recorded by the caller into the backend's native command stream.
///
/// This is the extension point for vendor technology that needs the native objects of the active backend, such as DLSS
/// and MetalFX. The pass names every resource it touches so the backend can order it against neighbouring commands,
/// keep those resources alive until the GPU retires the submission, and make earlier writes visible. A backend that has
/// no native pass support rejects the command with ``GPUError``.
public struct NativePass: Sendable, Equatable {
    /// How a native pass uses a resource.
    public enum Access: Sendable, Equatable {
        case read
        case write
        case readWrite

        public var reads: Bool { self != .write }
        public var writes: Bool { self != .read }
    }

    /// One resource the pass touches.
    public struct Use: Sendable, Equatable {
        public let resource: ResourceID
        public let access: Access

        public init(_ resource: ResourceID, _ access: Access) {
            self.resource = resource
            self.access = access
        }
    }

    private enum Body: Sendable {
        case generic(@Sendable (any NativePassContext) throws -> Void)
        case mainActor(@MainActor @Sendable (any NativePassContext) throws -> Void)
    }

    private final class Storage: Sendable {
        let body: Body
        init(_ body: Body) { self.body = body }
    }

    public let label: String
    public let uses: [Use]
    private let storage: Storage

    /// - Parameters:
    ///   - label: A debug name for diagnostics.
    ///   - uses: Every buffer and texture the body reads or writes.
    ///   - body: Records native commands. It may throw; the submission then fails and nothing of it executes.
    public init(label: String, uses: [Use], body: @escaping @Sendable (any NativePassContext) throws -> Void) {
        self.init(label: label, uses: uses, body: .generic(body))
    }

    private init(label: String, uses: [Use], body: Body) {
        self.label = label
        self.uses = uses
        self.storage = Storage(body)
    }

    /// Records platform work that is isolated to the main actor, such as Metal compositor presentation.
    /// The backend must support recording on the main actor.
    public static func onMainActor(
        label: String, uses: [Use],
        body: @escaping @MainActor @Sendable (any NativePassContext) throws -> Void
    ) -> NativePass {
        NativePass(label: label, uses: uses, body: .mainActor(body))
    }

    /// Runs the body. Backends call this while they record the submission.
    public func record(into context: any NativePassContext) throws {
        guard case let .generic(body) = storage.body else {
            throw GPUError.unsupported("This backend cannot record a main-actor native pass")
        }
        try body(context)
    }

    /// Records a pass on a backend that already owns its native context on the main actor.
    @MainActor
    public func recordOnMainActor(into context: any NativePassContext) throws {
        switch storage.body {
        case let .generic(body): try body(context)
        case let .mainActor(body): try body(context)
        }
    }

    public static func == (lhs: NativePass, rhs: NativePass) -> Bool {
        lhs.storage === rhs.storage && lhs.label == rhs.label && lhs.uses == rhs.uses
    }
}
