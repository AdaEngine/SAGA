public enum RenderStencilOperation: UInt32, Sendable, Equatable {
    case keep
    case zero
    case replace
    case invert
    case incrementClamp
    case decrementClamp
    case incrementWrap
    case decrementWrap
}

public struct RenderStencilFaceState: Sendable, Equatable {
    public let compare: CompareFunction
    public let fail: RenderStencilOperation
    public let depthFail: RenderStencilOperation
    public let pass: RenderStencilOperation

    package var writes: Bool { fail != .keep || depthFail != .keep || pass != .keep }

    public init(compare: CompareFunction = .always, fail: RenderStencilOperation = .keep,
                depthFail: RenderStencilOperation = .keep, pass: RenderStencilOperation = .keep) {
        self.compare = compare
        self.fail = fail
        self.depthFail = depthFail
        self.pass = pass
    }
}

public struct RenderStencilState: Sendable, Equatable {
    public let front: RenderStencilFaceState
    public let back: RenderStencilFaceState
    public let readMask: UInt32
    public let writeMask: UInt32

    package func writes(cullMode: RenderPrimitiveState.CullMode) -> Bool {
        writeMask != 0 && ((cullMode != .front && front.writes) || (cullMode != .back && back.writes))
    }

    public init(front: RenderStencilFaceState = .init(), back: RenderStencilFaceState = .init(),
                readMask: UInt32 = .max, writeMask: UInt32 = .max) {
        self.front = front
        self.back = back
        self.readMask = readMask
        self.writeMask = writeMask
    }
}

public struct RenderStencilAttachmentState: Sendable, Equatable {
    public enum LoadAction: Sendable, Equatable {
        case load
        case readOnly
        case clear(UInt32)
    }

    public let loadAction: LoadAction
    public let reference: UInt32

    public init(loadAction: LoadAction = .clear(0), reference: UInt32 = 0) {
        self.loadAction = loadAction
        self.reference = reference
    }
}
