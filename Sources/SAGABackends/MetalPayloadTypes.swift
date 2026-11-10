import SAGAShaderIR

/// Types whose declarations participate in the object_data payload ABI.
func collectMetalPayloadTypes(_ root: IRHandle<IRType>, module: IRModule, budget: inout ShaderBudget) throws -> Set<IRHandle<IRType>> {
    var result: Set<IRHandle<IRType>> = []
    var pending = [root]
    while let type = pending.popLast() {
        try budget.consume(.work)
        guard result.insert(type).inserted else { continue }
        switch try module.types.value(for: type).kind {
        case .structure(let members, _):
            try budget.consume(.work, amount: members.count)
            pending.append(contentsOf: members.map(\.type))
        case .array(let element, _, _): pending.append(element)
        default: break
        }
    }
    return result
}

/// Naga uses a packed three-lane vector when the following field begins at
/// its logical end. Restore WGSL's vector alignment on the member itself.
func metalPayloadPackedMembers(_ members: [IRStructMember], span: UInt32, module: IRModule, budget: inout ShaderBudget) throws -> Set<Int> {
    var result: Set<Int> = []
    for (index, member) in members.enumerated() {
        try budget.consume(.work)
        guard case .vector(let size, let scalar) = try module.types.value(for: member.type).kind,
              size == .three, scalar.width == 2 || scalar.width == 4 else { continue }
        let end = member.offset + UInt32(scalar.width) * 3
        let next = index + 1 < members.count ? members[index + 1].offset : span
        if end == next { result.insert(index) }
    }
    return result
}

/// Native layout of the declarations emitted for an object-data payload.
func metalPayloadLayout(
    _ type: IRHandle<IRType>, module: IRModule,
    cache: inout [IRHandle<IRType>: (size: UInt32, alignment: UInt32)], budget: inout ShaderBudget
) throws -> (size: UInt32, alignment: UInt32) {
    try budget.consume(.work)
    if let cached = cache[type] { return cached }
    let layout = try budget.withDepth { budget in
        try metalPayloadLayoutUncached(type, module: module, cache: &cache, budget: &budget)
    }
    cache[type] = layout
    return layout
}

private func metalPayloadLayoutUncached(
    _ type: IRHandle<IRType>, module: IRModule,
    cache: inout [IRHandle<IRType>: (size: UInt32, alignment: UInt32)], budget: inout ShaderBudget
) throws -> (size: UInt32, alignment: UInt32) {
    func round(_ value: UInt32, _ alignment: UInt32) -> UInt64 {
        (UInt64(value) + UInt64(alignment) - 1) / UInt64(alignment) * UInt64(alignment)
    }
    switch try module.types.value(for: type).kind {
    case .scalar(let scalar), .atomic(let scalar):
        return (UInt32(scalar.width), UInt32(scalar.width))
    case .vector(let lanes, let scalar):
        let width = UInt32(scalar.width)
        let alignment = width * (lanes == .three ? 4 : UInt32(lanes.rawValue))
        return (alignment, alignment)
    case .matrix(let columns, let rows, let scalar):
        let width = UInt32(scalar.width)
        let alignment = width * (rows == .three ? 4 : UInt32(rows.rawValue))
        return (alignment * UInt32(columns.rawValue), alignment)
    case .array(let element, .constant(let count), let stride):
        let inner = try metalPayloadLayout(element, module: module, cache: &cache, budget: &budget)
        guard round(inner.size, inner.alignment) == UInt64(stride) else {
            throw ShaderFailure.unsupported("Metal task payload array stride cannot be represented")
        }
        let (size, overflow) = stride.multipliedReportingOverflow(by: count)
        guard !overflow else { throw ShaderFailure.unsupported("Metal task payload array size overflow") }
        return (size, inner.alignment)
    case .structure(let members, let span):
        var alignment: UInt32 = 1
        for member in members {
            try budget.consume(.work)
            alignment = max(alignment, try metalPayloadLayout(member.type, module: module, cache: &cache, budget: &budget).alignment, member.explicitAlignment ?? 1)
        }
        guard span % alignment == 0 else {
            throw ShaderFailure.unsupported("Metal task payload structure span cannot be represented")
        }
        return (span, alignment)
    default:
        throw ShaderFailure.unsupported("Metal task payload type cannot be represented")
    }
}
