/// Carries the declared IR span in a debug name attached to the structure ID.
/// Offsets and array strides remain the authoritative physical layout.
package enum SPIRVStructureSpanName {
    package static let prefix = "saga.struct-span.v1:"

    package static func encode(span: UInt32, name: String?) -> String {
        "\(prefix)\(span):\(name ?? "")"
    }

    package static func decode(_ value: String) -> (span: UInt32, name: String?)? {
        guard value.hasPrefix(prefix) else { return nil }
        let fields = value.dropFirst(prefix.count).split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
        guard fields.count == 2, let span = UInt32(fields[0]), fields[0] == String(span) else { return nil }
        return (span, fields[1].isEmpty ? nil : String(fields[1]))
    }
}
