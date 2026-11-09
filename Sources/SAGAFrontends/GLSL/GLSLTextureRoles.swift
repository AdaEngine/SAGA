import SAGAShaderIR

enum GLSLTextureOrigin: Hashable {
    case global(IRHandle<IRGlobalVariable>)
    case parameter(function: Int, index: UInt32)
}

struct GLSLTextureRole {
    let comparison: Bool
    let site: GLSLToken
}

extension GLSLParser {
    func requireGLSLTextureRole(_ value: GLSLValue, comparison: Bool, at token: GLSLToken) throws {
        let context = try requireContext()
        let expression = context.expressions[value.expression.index]
        guard let origin = textureOrigin(expression.kind, in: functions.count) else {
            throw glslError(token, .unsupportedConstruct, "Texture sampling requires a global or parameter resource")
        }
        if let old = textureRoles[origin] {
            guard old.comparison == comparison else { throw mixedTextureRoles(at: token) }
        } else {
            try budget.consume(.nodes)
            textureRoles[origin] = .init(comparison: comparison, site: token)
        }
    }

    func finalizeGLSLComparisonTextures() throws {
        try budget.consume(.work, amount: textureRoles.count)
        guard textureRoles.values.contains(where: \.comparison) else { return }
        let seedCount = textureRoles.count
        try budget.consume(.work, amount: seedCount * (Int.bitWidth - seedCount.leadingZeroBitCount))
        var pending = textureRoles.sorted { $0.value.site.start < $1.value.site.start }.map(\.key)
        var actuals: [GLSLTextureOrigin: [GLSLTextureOrigin]] = [:]
        var formals: [GLSLTextureOrigin: [GLSLTextureOrigin]] = [:]
        for (caller, function) in functions.enumerated() {
            for statement in function.statements.values {
                try budget.consume(.work)
                guard case .call(let target, let arguments, _) = statement.kind else { continue }
                for (index, argument) in arguments.enumerated() {
                    try budget.consume(.work)
                    let expression = try function.expressions.value(for: argument)
                    guard kind(expression.type).isSampledTexture,
                          let actual = textureOrigin(expression.kind, in: caller) else { continue }
                    let formal = GLSLTextureOrigin.parameter(function: target.index, index: UInt32(index))
                    try budget.consume(.nodes, amount: 2)
                    formals[actual, default: []].append(formal)
                    actuals[formal, default: []].append(actual)
                }
            }
        }
        while let origin = pending.popLast() {
            try budget.consume(.work)
            guard let role = textureRoles[origin] else { throw ShaderFailure.invalidHandle }
            for actual in actuals[origin] ?? [] {
                try budget.consume(.work)
                if let other = textureRoles[actual] {
                    guard other.comparison == role.comparison else { throw mixedTextureRoles(at: other.site) }
                } else {
                    try budget.consume(.nodes)
                    textureRoles[actual] = role
                    pending.append(actual)
                }
            }
        }
        // An otherwise unconstrained parameter can use the depth type if every
        // caller supplies it. This does not promote unrelated query-only bindings.
        try budget.consume(.work, amount: textureRoles.count)
        pending = textureRoles.filter { $0.value.comparison }.map(\.key)
        var depthCallers: [GLSLTextureOrigin: Int] = [:]
        while let actual = pending.popLast() {
            try budget.consume(.work)
            guard let role = textureRoles[actual] else { throw ShaderFailure.invalidHandle }
            for formal in formals[actual] ?? [] {
                try budget.consume(.work)
                guard textureRoles[formal] == nil else { continue }
                let count = depthCallers[formal, default: 0] + 1
                depthCallers[formal] = count
                if count == actuals[formal]?.count {
                    try budget.consume(.nodes)
                    textureRoles[formal] = role
                    pending.append(formal)
                }
            }
        }
        for (actual, targets) in formals {
            try budget.consume(.work)
            guard let role = textureRoles[actual], role.comparison else { continue }
            for formal in targets {
                try budget.consume(.work)
                guard textureRoles[formal]?.comparison == true else {
                    throw glslError(role.site, .unsupportedConstruct,
                                    "A GLSL texture parameter cannot receive both ordinary and comparison textures")
                }
            }
        }

        for index in globals.indices {
            try budget.consume(.work)
            let handle = try IRHandle<IRGlobalVariable>(index: index)
            guard let role = textureRoles[.global(handle)], role.comparison else { continue }
            let global = globals[index]
            let type = try depthTextureType(global.type, at: role.site)
            globals[index] = .init(name: global.name, space: global.space, binding: global.binding, type: type,
                                   initializer: global.initializer, memoryDecorations: global.memoryDecorations,
                                   location: global.location)
        }
        for (functionIndex, function) in functions.enumerated() {
            var arguments = function.arguments
            var expressions = function.expressions.values
            var changed = false
            for (index, argument) in function.arguments.enumerated() {
                try budget.consume(.work)
                guard let role = textureRoles[.parameter(function: functionIndex, index: UInt32(index))],
                      role.comparison else { continue }
                arguments[index] = .init(name: argument.name, type: try depthTextureType(argument.type, at: role.site),
                                         binding: argument.binding, location: argument.location)
                changed = true
            }
            for (index, expression) in function.expressions.values.enumerated() {
                try budget.consume(.work)
                guard let origin = textureOrigin(expression.kind, in: functionIndex),
                      textureRoles[origin]?.comparison == true else { continue }
                let type = switch origin {
                case .global(let global): globals[global.index].type
                case .parameter(_, let parameter): arguments[Int(parameter)].type
                }
                expressions[index] = .init(type: type, kind: expression.kind, location: expression.location)
                changed = true
            }
            if changed {
                functions[functionIndex] = .init(name: function.name, arguments: arguments, result: function.result,
                    localVariables: function.localVariables, expressions: .init(expressions), namedExpressions: function.namedExpressions,
                    statements: function.statements, blocks: function.blocks, body: function.body,
                    diagnosticFilter: function.diagnosticFilter, location: function.location)
            }
        }
    }

    private func textureOrigin(_ expression: IRExpressionKind, in function: Int) -> GLSLTextureOrigin? {
        switch expression {
        case .globalVariable(let global): .global(global)
        case .functionArgument(let index): .parameter(function: function, index: index)
        default: nil
        }
    }

    private func depthTextureType(_ type: IRHandle<IRType>, at token: GLSLToken) throws -> IRHandle<IRType> {
        guard case .image(let dimension, let arrayed, .sampled(.float, false)) = kind(type),
              dimension == .d2 || dimension == .cube else {
            throw glslError(token, .unsupportedConstruct, "Comparison sampling needs a supported depth texture dimension")
        }
        return try addType(.image(dimension: dimension, arrayed: arrayed, class: .depth(multisampled: false)))
    }

    private func mixedTextureRoles(at token: GLSLToken) -> ShaderFailure {
        glslError(token, .unsupportedConstruct, "A GLSL texture cannot be imported for both ordinary and comparison sampling")
    }
}
