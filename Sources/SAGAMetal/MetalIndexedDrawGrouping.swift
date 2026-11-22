/// Attachment configurations that an indexed draw producer permits Metal to group.
/// Metal still checks the actual attachments, bindings, and state before grouping a draw.
@_spi(IndexedDrawGrouping)
public struct MetalIndexedDrawGrouping: OptionSet, Sendable {
    public let rawValue: UInt8

    public init(rawValue: UInt8) {
        self.rawValue = rawValue
    }

    /// Allows grouping draws with a primary color attachment.
    public static let color = Self(rawValue: 1 << 0)
    /// Allows adding depth and stencil to a color group.
    public static let depthStencil = Self(rawValue: 1 << 1)
    /// Allows adding further color attachments to a color group.
    public static let multipleColorAttachments = Self(rawValue: 1 << 2)

    /// Eligibility by index in the original command list, before splitting query ranges.
    @TaskLocal public static var commands: [Int: Self] = [:]

    func allows(depthStencil: Bool, multipleColorAttachments: Bool) -> Bool {
        var required: Self = .color
        if depthStencil { required.insert(.depthStencil) }
        if multipleColorAttachments { required.insert(.multipleColorAttachments) }
        return contains(required)
    }
}
