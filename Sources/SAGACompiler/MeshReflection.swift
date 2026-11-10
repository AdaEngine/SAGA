import SAGAShaderIR

func meshWritesPrimitiveIndex(_ handle: IRHandle<IRType>, in module: IRModule) throws -> Bool {
    guard case .structure(let members, _) = try module.types.value(for: handle).kind else {
        throw ShaderFailure.unsupported("Mesh interface structure")
    }
    return members.contains { $0.binding == .builtIn(.primitiveIndex) }
}

// Accept only layouts that match the ordinary Metal declarations emitted for payload types.
func meshPayload(_ handle: IRHandle<IRGlobalVariable>?, in module: IRModule) throws -> MeshPayloadLayout? {
    guard let handle else { return nil }
    let global = try module.globalVariables.value(for: handle)
    guard case .structure(let members, _) = try module.types.value(for: global.type).kind else {
        throw ShaderFailure.unsupported("Mesh payload must be a structure")
    }
    var budget = ShaderBudget()
    let layout = try metalPayloadType(global.type, in: module, depth: 0, budget: &budget)
    let scalars = try members.compactMap { member -> MeshPayloadLayout.Member? in
        guard case .scalar(let scalar) = try module.types.value(for: member.type).kind else { return nil }
        return .init(offset: member.offset, scalar: scalar)
    }
    return .init(size: layout.size, members: scalars, shape: layout.shape)
}

private func metalPayloadType(_ handle: IRHandle<IRType>, in module: IRModule, depth: Int, budget: inout ShaderBudget) throws
    -> (size: UInt32, alignment: UInt32, shape: MeshPayloadLayout.Shape) {
    try budget.consume(.work)
    guard depth < 64 else { throw ShaderFailure.unsupported("Mesh payload nesting") }
    func roundUp(_ value: UInt32, _ alignment: UInt32) throws -> UInt32 {
        let (sum, overflow) = value.addingReportingOverflow(alignment - 1)
        guard !overflow else { throw ShaderFailure.unsupported("Mesh payload size") }
        return sum / alignment * alignment
    }
    switch try module.types.value(for: handle).kind {
    case .scalar(let scalar) where [.f16, .i16, .u16, .u32, .i32, .f32, .i64, .u64].contains(scalar):
        let size = UInt32(scalar.width)
        return (size, size, .scalar(scalar))
    case .vector(let width, let scalar) where [.f16, .i16, .u16, .u32, .i32, .f32, .i64, .u64].contains(scalar):
        let count = width.rawValue
        let size = UInt32(scalar.width) * (count == 2 ? 2 : 4)
        return (size, size, .vector(scalar: scalar, count: count))
    case .array(let element, .constant(let count), let stride):
        let child = try metalPayloadType(element, in: module, depth: depth + 1, budget: &budget)
        let nativeStride = try roundUp(child.size, child.alignment)
        let (size, overflow) = nativeStride.multipliedReportingOverflow(by: count)
        guard count > 0, !overflow, stride == nativeStride else { throw ShaderFailure.unsupported("Mesh payload native array stride") }
        return (size, child.alignment, .array(element: child.shape, count: count, stride: stride))
    case .structure(let members, let span):
        guard !members.isEmpty else { throw ShaderFailure.unsupported("Mesh payload empty structure") }
        var size: UInt32 = 0
        var alignment: UInt32 = 1
        var fields: [MeshPayloadLayout.Field] = []
        try budget.consume(.nodes, amount: members.count)
        for (index, member) in members.enumerated() {
            guard member.binding == nil else { throw ShaderFailure.unsupported("Mesh payload member binding") }
            let child = try metalPayloadType(member.type, in: module, depth: depth + 1, budget: &budget)
            let memberAlignment = max(child.alignment, member.explicitAlignment ?? 1)
            let minimumOffset = try roundUp(size, memberAlignment)
            let offset = member.offset
            guard offset >= minimumOffset, offset % memberAlignment == 0 else {
                throw ShaderFailure.unsupported("Mesh payload native layout")
            }
            var nativeSize = child.size
            if case .vector(let width, let scalar) = try module.types.value(for: member.type).kind,
               width == .three, scalar.width == 2 || scalar.width == 4 {
                let (logicalEnd, overflow) = offset.addingReportingOverflow(UInt32(scalar.width) * 3)
                let next = index + 1 < members.count ? members[index + 1].offset : span
                if !overflow && logicalEnd == next { nativeSize = UInt32(scalar.width) * 3 }
            }
            let (end, overflow) = offset.addingReportingOverflow(nativeSize)
            guard !overflow else { throw ShaderFailure.unsupported("Mesh payload native layout") }
            fields.append(.init(offset: offset, shape: child.shape))
            size = end
            alignment = max(alignment, memberAlignment)
        }
        size = try roundUp(size, alignment)
        guard size <= span, span % alignment == 0 else { throw ShaderFailure.unsupported("Mesh payload native size") }
        return (span, alignment, .structure(fields))
    default: throw ShaderFailure.unsupported("Mesh payload native type")
    }
}

func meshVaryings(_ handle: IRHandle<IRType>, in module: IRModule) throws -> [MeshStageArtifact.Varying] {
    guard case .structure(let members, _) = try module.types.value(for: handle).kind else {
        throw ShaderFailure.unsupported("Mesh interface structure")
    }
    var result: [MeshStageArtifact.Varying] = []
    for member in members {
        guard case .location(let location, let interpolation, let sampling, let blend, _) = member.binding else { continue }
        guard blend == nil else { throw ShaderFailure.unsupported("Mesh varying blend source") }
        let scalar: IRScalar
        let count: UInt8
        switch try module.types.value(for: member.type).kind {
        case .scalar(let value): scalar = value; count = 1
        case .vector(let size, let value): scalar = value; count = size.rawValue
        default: throw ShaderFailure.unsupported("Mesh varying type")
        }
        result.append(.init(location: location, scalar: scalar, componentCount: count, interpolation: interpolation, sampling: sampling))
    }
    return result.sorted { $0.location < $1.location }
}
