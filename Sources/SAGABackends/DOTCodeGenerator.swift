import SAGAShaderIR
import SAGAValidation

/// Writes a diagnostic Graphviz graph of the validated module before target lowering.
public struct DOTCodeGenerator: ShaderCodeGenerator {
    public let cfgOnly: Bool
    public let limits: ShaderLimits
    public var outputFormat: ShaderOutputFormat { .dot }

    public init(cfgOnly: Bool = false, limits: ShaderLimits = .default) {
        self.cfgOnly = cfgOnly
        self.limits = limits
    }

    public func generate(_ module: ValidatedModule<IRModule>) throws -> ShaderArtifact {
        var writer = DOTGraphWriter(cfgOnly: cfgOnly, limits: limits)
        return try .text(format: .dot, source: writer.writeModule(module.module))
    }
}

private extension DOTGraphWriter {
    mutating func writeDeclarations(_ module: IRModule) throws {
        for (index, type) in module.types.values.enumerated() {
            let id = "t\(index)"
            let name = type.name.map { " \($0)" } ?? ""
            try node(id, label: "Type\(name)\n\(describe(type.kind))", shape: "box")
            switch type.kind {
            case .pointer(let base, _):
                try edge(id, "t\(base.index)", label: "pointee", style: "dotted")
            case .array(let base, let size, _), .bindingArray(let base, let size):
                try edge(id, "t\(base.index)", label: "element", style: "dotted")
                if case .override(let count) = size { try edge(id, "o\(count.index)", label: "count", style: "dotted") }
            case .structure(let fields, _):
                for (slot, field) in fields.enumerated() {
                    try edge(id, "t\(field.type.index)", label: "\(field.name ?? "member \(slot)") at \(field.offset)", style: "dotted")
                }
            default: break
            }
        }
        for (index, constant) in module.constants.values.enumerated() {
            let id = "c\(index)"
            try typedNode(id, label: "Constant \(constant.name ?? "\(index)")", type: constant.type)
            try edge(id, "ge\(constant.initializer.index)", label: "initializer", style: "dotted")
        }
        for (index, override) in module.overrides.values.enumerated() {
            let id = "o\(index)"
            let annotation = override.id.map { " @id(\($0))" } ?? ""
            try typedNode(id, label: "Override \(override.name ?? "\(index)")\(annotation)", type: override.type)
            if let initializer = override.initializer {
                try edge(id, "ge\(initializer.index)", label: "initializer", style: "dotted")
            }
        }
        for (index, global) in module.globalVariables.values.enumerated() {
            let id = "g\(index)"
            let binding = global.binding.map { "\n@group(\($0.group)) @binding(\($0.binding))" } ?? ""
            try typedNode(id, label: "Global \(global.name ?? "\(index)")\n\(describe(global.space))\(binding)", type: global.type)
            if let initializer = global.initializer {
                try edge(id, "ge\(initializer.index)", label: "initializer", style: "dotted")
            }
        }
    }

    mutating func writeFunctionDeclarations(_ function: IRFunction, id: String) throws {
        for (index, argument) in function.arguments.enumerated() {
            let binding = argument.binding.map { "\n\($0)" } ?? ""
            try typedNode("\(id)_arg\(index)", label: "Argument \(argument.name ?? "\(index)")\(binding)", type: argument.type)
        }
        for (index, local) in function.localVariables.values.enumerated() {
            let variable = "\(id)_local\(index)"
            try typedNode(variable, label: "Local \(local.name ?? "\(index)")", type: local.type)
            if let initializer = local.initializer {
                try edge(variable, "\(id)_e\(initializer.index)", label: "initializer", style: "dotted")
            }
        }
        if let result = function.result { try edge(id, "t\(result.type.index)", label: "result type", style: "dotted") }
    }

    mutating func typedNode(_ id: String, label: String, type: IRHandle<IRType>) throws {
        try node(id, label: label, shape: "hexagon")
        try edge(id, "t\(type.index)", label: "type", style: "dotted")
    }

    func describe(_ kind: IRTypeKind) -> String {
        switch kind {
        case .scalar(let scalar): describe(scalar)
        case .vector(let size, let scalar): "vec\(size.rawValue)<\(describe(scalar))>"
        case .matrix(let columns, let rows, let scalar): "mat\(columns.rawValue)x\(rows.rawValue)<\(describe(scalar))>"
        case .atomic(let scalar): "atomic<\(describe(scalar))>"
        case .pointer(_, let space): "pointer in \(describe(space))"
        case .valuePointer(let width, let scalar, let space): "value pointer \(width.map { "vec\($0.rawValue) " } ?? "")\(describe(scalar)) in \(describe(space))"
        case .array(_, let size, let stride): "array \(describe(size)), stride \(stride)"
        case .bindingArray(_, let size): "binding array \(describe(size))"
        case .structure(_, let span): "struct, span \(span)"
        case .sampler(let comparison): comparison ? "comparison sampler" : "sampler"
        default: String(describing: kind)
        }
    }

    func describe(_ size: IRArraySize) -> String {
        switch size {
        case .constant(let count): "\(count)"
        case .dynamic: "dynamic"
        case .override: "override count"
        }
    }

    func describe(_ scalar: IRScalar) -> String {
        let bits = Int(scalar.width) * 8
        return switch scalar.kind {
        case .sint: "i\(bits)"
        case .uint: "u\(bits)"
        case .float: "f\(bits)"
        case .bool, .abstractInt, .abstractFloat: scalar.kind.rawValue
        }
    }

    func describe(_ space: IRAddressSpace) -> String {
        guard case .storage(let access) = space else { return String(describing: space) }
        let modes = [(IRStorageAccess.load, "read"), (.store, "write"), (.atomic, "atomic")]
        return "storage " + modes.compactMap { access.contains($0.0) ? $0.1 : nil }.joined(separator: "/")
    }
}

struct DOTGraphWriter {
    let cfgOnly: Bool
    private var budget: ShaderBudget
    private var declarations = ""
    private var edges = ""
    private var indentation = 0

    init(cfgOnly: Bool, limits: ShaderLimits) {
        self.cfgOnly = cfgOnly
        self.budget = ShaderBudget(limits: limits)
    }

    mutating func charge(_ amount: Int = 1) throws { try budget.consume(.work, amount: amount) }

    mutating func node(_ id: String, label: String, shape: String? = nil) throws {
        try budget.consume(.nodes)
        let geometry = shape.map { ", shape=\($0)" } ?? ""
        try append("\(id) [label=\(quote(label))\(geometry)];")
    }

    mutating func edge(_ from: String, _ to: String, label: String? = nil, style: String? = nil) throws {
        try charge()
        var attributes: [String] = []
        if let label { attributes.append("label=\(try quote(label))") }
        if let style { attributes.append("style=\(style)") }
        let options = attributes.isEmpty ? "" : " [\(attributes.joined(separator: ", "))]"
        let line = "  \(from) -> \(to)\(options);\n"
        try budget.consume(.sourceBytes, amount: line.utf8.count)
        // Edges belong to the graph root, so references cannot move nodes between clusters.
        edges += line
    }

    mutating func writeModule(_ module: IRModule) throws -> String {
        try append("digraph Module {")
        indentation = 1
        if !cfgOnly {
            try beginCluster("module_data", label: "Module declarations")
            try writeDeclarations(module)
            try writeGlobalExpressions(module.globalExpressions)
            try endCluster()
        }
        for (index, entry) in module.entryPoints.enumerated() {
            let id = "ep\(index)"
            try node(id, label: "@\(entry.stage.rawValue) \(entry.name)", shape: "oval")
            try edge(id, "f\(entry.function.index)", label: "entry point")
            if !cfgOnly, let dimensions = entry.workgroupOverrides {
                for (axis, value) in [("x", dimensions.x), ("y", dimensions.y), ("z", dimensions.z)] {
                    if let value { try edge(id, "ge\(value.index)", label: "workgroup \(axis)", style: "dotted") }
                }
            }
        }
        for (index, function) in module.functions.values.enumerated() {
            let id = "f\(index)"
            let name = "Function \(function.name ?? "\(index)")"
            try beginCluster(id, label: name)
            try node(id, label: name, shape: "component")
            if !cfgOnly {
                try writeFunctionDeclarations(function, id: id)
                try writeFunctionExpressions(function, functionIndex: index)
            }
            try writeControlFlow(function, functionIndex: index)
            try endCluster()
        }
        try budget.consume(.sourceBytes, amount: 2)
        return declarations + edges + "}\n"
    }

    private mutating func append(_ line: String) throws {
        try charge()
        try budget.consume(.sourceBytes, amount: line.utf8.count + indentation * 2 + 1)
        declarations += String(repeating: "  ", count: indentation) + line + "\n"
    }

    private mutating func beginCluster(_ id: String, label: String) throws {
        try append("subgraph cluster_\(id) {")
        indentation += 1
        try append("label=\(quote(label));")
    }

    private mutating func endCluster() throws {
        indentation -= 1
        try append("}")
    }

    private mutating func quote(_ value: String) throws -> String {
        try charge(value.utf8.count)
        var result = "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 34: result += "\\\""
            case 92: result += "\\\\"
            case 10, 13: result += "\\n"
            case 0...31, 127: result += " "
            default: result.unicodeScalars.append(scalar)
            }
        }
        return result + "\""
    }
}
