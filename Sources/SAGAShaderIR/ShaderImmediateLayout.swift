/// The immediate data required by one shader entry point.
public struct ShaderImmediateLayout: Sendable, Equatable {
    public let byteSize: UInt64
    public let metalBufferIndex: UInt32?
    public let hlslBinding: IRResourceBinding?
    public let glslUniformBinding: UInt32?
    package var wordAlignedByteSize: UInt64 { (byteSize + 3) / 4 * 4 }

    package init(byteSize: UInt64, metalBufferIndex: UInt32? = nil,
                 hlslBinding: IRResourceBinding? = nil, glslUniformBinding: UInt32? = nil) {
        self.byteSize = byteSize
        self.metalBufferIndex = metalBufferIndex
        self.hlslBinding = hlslBinding
        self.glslUniformBinding = glslUniformBinding
    }
}
