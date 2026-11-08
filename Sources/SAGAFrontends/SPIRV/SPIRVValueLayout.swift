import SAGAShaderIR

struct SPVValueLayout {
    let alignment: UInt64
    let size: UInt64

    func aligned(_ offset: UInt64) -> UInt64 {
        (offset + alignment - 1) / alignment * alignment
    }
}

extension SPVDecoder {
    mutating func addImplicitStruct(_ instruction: SPVInstruction) throws {
        let id = instruction.a[0]
        var fields: [IRStructMember] = []
        var end: UInt64 = 0
        var alignment: UInt64 = 1
        for (index, member) in instruction.a.dropFirst().enumerated() {
            let type = try type(member, instruction)
            let layout = try valueLayout(type, instruction)
            let offset = layout.aligned(end)
            end = offset + layout.size
            guard end <= UInt64(Int32.max) else { throw instruction.fail("Value structure layout exceeds the size limit") }
            alignment = max(alignment, layout.alignment)
            fields.append(.init(name: memberNames[id]?[UInt32(index)], type: type,
                                offset: UInt32(offset), location: instruction.location))
        }
        let span = SPVValueLayout(alignment: alignment, size: 0).aligned(end)
        guard span <= UInt64(Int32.max) else { throw instruction.fail("Value structure span exceeds the size limit") }
        try addType(id, .structure(members: fields, span: UInt32(span)), instruction)
        let handle = try type(id, instruction)
        implicitStructs.insert(handle)
        valueLayouts[handle] = .init(alignment: alignment, size: span)
    }

    mutating func valueLayout(_ type: IRHandle<IRType>, _ instruction: SPVInstruction) throws -> SPVValueLayout {
        try charge()
        if let cached = valueLayouts[type] { return cached }
        try budget.enterDepth()
        defer { try? budget.leaveDepth() }
        let layout: SPVValueLayout
        switch types[type.index].kind {
        case .scalar(let scalar):
            let width = UInt64(scalar.width)
            layout = .init(alignment: width, size: width)
        case .vector(let size, let scalar):
            let lanes = UInt64(size.rawValue)
            let width = UInt64(scalar.width)
            layout = .init(alignment: (lanes == 3 ? 4 : lanes) * width, size: lanes * width)
        case .matrix(let columns, let rows, let scalar):
            let rowsAligned = rows == .three ? 4 : UInt64(rows.rawValue)
            let alignment = rowsAligned * UInt64(scalar.width)
            layout = .init(alignment: alignment, size: alignment * UInt64(columns.rawValue))
        case .array(let element, .constant(let count), let stride):
            let elementLayout = try valueLayout(element, instruction)
            layout = .init(alignment: elementLayout.alignment, size: UInt64(count) * UInt64(stride))
        case .structure(let members, let span):
            var alignment: UInt64 = 1
            for member in members {
                alignment = max(alignment, try valueLayout(member.type, instruction).alignment)
            }
            layout = .init(alignment: alignment, size: UInt64(span))
        default:
            throw instruction.fail("Unsupported member type in a value structure", code: .unsupportedConstruct)
        }
        guard layout.size <= UInt64(Int32.max) else { throw instruction.fail("Value layout exceeds the size limit") }
        valueLayouts[type] = layout
        return layout
    }

    mutating func requireExplicitBufferLayout(_ type: IRHandle<IRType>, _ instruction: SPVInstruction) throws {
        var pending = [type]
        while let next = pending.popLast() {
            try charge()
            if verifiedBufferLayouts.contains(next) { continue }
            guard !implicitStructs.contains(next) else {
                throw instruction.fail("Buffer structures require explicit member offsets")
            }
            guard !implicitArrays.contains(next) else {
                throw instruction.fail("Buffer arrays require explicit ArrayStride")
            }
            verifiedBufferLayouts.insert(next)
            switch types[next.index].kind {
            case .structure(let members, _):
                try charge(members.count)
                pending.append(contentsOf: members.map(\.type))
            case .array(let element, _, _):
                pending.append(element)
            default:
                break
            }
        }
    }
}
