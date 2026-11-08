import SAGAShaderIR

struct WGSLSwizzleView {
    let baseType: WType
    let lanes: [Int]
}

extension WGSLParser {
    func vectorSelection(_ value: WGSLValue, lanes: [Int], scalar: IRScalar, at name: WGSLToken) throws -> WGSLValue {
        let mapped = value.swizzle.map { view in lanes.map { view.lanes[$0] } } ?? lanes
        if mapped.count == 1 {
            let type = try intern(.scalar(scalar))
            let result = value.pointer ? try intern(.pointer(base: type, space: pointerSpace(value))) : type
            let lane = try expressionNode(.accessIndex(base: value.handle, index: UInt32(mapped[0])), type: result, at: name)
            return WGSLValue(handle: lane.handle, type: type, pointer: value.pointer, writable: value.writable)
        }
        guard let width = IRVectorSize(rawValue: UInt8(lanes.count)) else { throw ShaderFailure.invalidHandle }
        let type = try intern(.vector(size: width, scalar: scalar))
        if allowsSwizzleAssignment, value.pointer, value.writable {
            let view = WGSLSwizzleView(baseType: value.swizzle?.baseType ?? value.type, lanes: mapped)
            return WGSLValue(handle: value.handle, type: type, pointer: true, writable: true, swizzle: view)
        }
        let source = try loaded(value, at: name)
        return try projectSwizzle(source.handle, lanes: lanes, type: type, at: name)
    }

    func readSwizzleView(_ value: WGSLValue, view: WGSLSwizzleView, at name: WGSLToken) throws -> WGSLValue {
        let vector = try expressionNode(.load(pointer: value.handle), type: view.baseType, at: name)
        return try projectSwizzle(vector.handle, lanes: view.lanes, type: value.type, at: name)
    }

    private func projectSwizzle(_ vector: WExpr, lanes: [Int], type: WType, at name: WGSLToken) throws -> WGSLValue {
        guard let width = IRVectorSize(rawValue: UInt8(lanes.count)) else { throw ShaderFailure.invalidHandle }
        var components = lanes.map { IRSwizzleComponent.allCases[$0] }
        while components.count < 4 { components.append(.x) }
        let pattern = IRSwizzle(x: components[0], y: components[1], z: components[2], w: components[3])
        return try expressionNode(.swizzle(size: width, vector: vector, pattern: pattern), type: type, at: name)
    }

    func indexSwizzleView(
        _ value: WGSLValue, view: WGSLSwizzleView, index: WGSLValue, tokens: Range<Int>, at name: WGSLToken
    ) throws -> WGSLValue {
        guard case .vector(let width, let scalar) = types[view.baseType.index].kind else {
            throw ShaderFailure.invalidHandle
        }
        if let index = try knownIntegerConstant(tokens, at: name) {
            guard index.value >= 0, index.value < Int64(view.lanes.count) else {
                throw wgslError(.invalidType, "Constant swizzle index out of bounds", name)
            }
            return try vectorSelection(value, lanes: [Int(index.value)], scalar: scalar, at: name)
        }
        let u32 = try intern(.scalar(.u32))
        let bool = try intern(.scalar(.bool))
        // An invalid view index maps to a deliberately invalid backing-vector index.
        var lane = try expressionNode(.literal(.u32(UInt32(width.rawValue))), type: u32, at: name, emit: false)
        for (position, offset) in view.lanes.enumerated() {
            let key: IRLiteral = types[index.type.index].kind == .scalar(.i32) ? .i32(Int32(position)) : .u32(UInt32(position))
            let compare = try expressionNode(.literal(key), type: index.type, at: name, emit: false)
            let condition = try expressionNode(.binary(op: .equal, left: index.handle, right: compare.handle),
                                               type: bool, at: name)
            let component = try expressionNode(.literal(.u32(UInt32(offset))), type: u32, at: name, emit: false)
            lane = try expressionNode(.select(condition: condition.handle, accept: component.handle, reject: lane.handle),
                                      type: u32, at: name)
        }
        let type = try intern(.scalar(scalar))
        let pointer = try intern(.pointer(base: type, space: pointerSpace(value)))
        let result = try expressionNode(.access(base: value.handle, index: lane.handle), type: pointer, at: name)
        return WGSLValue(handle: result.handle, type: type, pointer: true, writable: value.writable)
    }

    func writeSwizzleView(_ value: WGSLValue, view: WGSLSwizzleView, update: WGSLValue, at name: WGSLToken) throws {
        guard Set(view.lanes).count == view.lanes.count else {
            throw wgslError(.invalidReference, "Swizzle assignment requires distinct components", name)
        }
        guard case .vector(let width, let scalar) = types[view.baseType.index].kind else {
            throw ShaderFailure.invalidHandle
        }
        // Read after evaluating the update, since calls on the RHS can modify the other components.
        let old = try expressionNode(.load(pointer: value.handle), type: view.baseType, at: name)
        let lane = try intern(.scalar(scalar))
        let components = try (0..<Int(width.rawValue)).map { index -> WExpr in
            let source = view.lanes.firstIndex(of: index)
            let base = source == nil ? old.handle : update.handle
            return try expressionNode(.accessIndex(base: base, index: UInt32(source ?? index)), type: lane, at: name).handle
        }
        let merged = try expressionNode(.compose(type: view.baseType, components: components), type: view.baseType, at: name)
        try statement(.store(pointer: value.handle, value: merged.handle), at: name)
    }
}
