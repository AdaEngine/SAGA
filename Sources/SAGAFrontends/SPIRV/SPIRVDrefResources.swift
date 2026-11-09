import SAGAShaderIR

extension SPVDecoder {
    mutating func prepareDrefResources() throws {
        var formals: [UInt32: [(position: Int, id: UInt32)]] = [:]
        var parameterIDs: Set<UInt32> = []
        for function in functions {
            for (position, parameter) in function.body.enumerated() {
                try charge()
                guard parameter.op == 55 else { break }
                guard parameter.a.count == 2, let type = typeMap[parameter.a[0]] else { continue }
                let kind = types[type.index].kind
                if kind.isSampler || kind.isSampledTexture {
                    let id = parameter.a[1]
                    parameterIDs.insert(id)
                    formals[function.header.a[1], default: []].append((position, id))
                }
            }
        }

        var links: [UInt32: [UInt32]] = [:]
        var unresolvedArguments: [UInt32: SPVInstruction] = [:]
        var regularSamplers: Set<UInt32> = []
        var comparisonSites: [UInt32: SPVInstruction] = [:]
        var depthSitesWithoutPairedDepth: [UInt32: SPVInstruction] = [:]
        var requested: [UInt32] = []

        for function in functions {
            var sources: [UInt32: UInt32] = [:]
            var extractions: [(id: UInt32, pair: UInt32)] = []
            var pairs: [UInt32: (type: UInt32, image: UInt32, sampler: UInt32)] = [:]
            var consumers: [SPVInstruction] = []
            for instruction in function.body {
                try charge()
                let a = instruction.a
                switch (instruction.op, a.count) {
                case (61, 3), (83, 3): sources[a[1]] = a[2] // OpLoad, OpCopyObject
                case (65, 3), (66, 3): sources[a[1]] = a[2] // Empty access chain
                case (65, 4), (66, 4):
                    if let raw = globalMap[a[2]]?.1 ?? functionValueTypes[a[2]], let pointer = pointers[raw],
                       let element = descriptorArrayElements[pointer.base], pointers[a[0]]?.base == element {
                        sources[a[1]] = a[2]
                    }
                case (86, 4): pairs[a[1]] = (a[0], a[2], a[3]) // OpSampledImage
                case (100, 3): extractions.append((a[1], a[2])) // OpImage
                case (57, 3...), (87...94, 3...), (96...97, 3...): consumers.append(instruction)
                default: break
                }
            }
            for (id, pair) in extractions {
                try charge()
                sources[id] = pairs[pair]?.image
            }

            // Index the whole function so textual block order cannot hide an origin.
            var origins: [UInt32: UInt32] = [:]
            for instruction in consumers {
                try charge()
                let a = instruction.a
                if instruction.op == 57 {
                    for formal in formals[a[2]] ?? [] where formal.position < a.count - 3 {
                        try charge()
                        let argument = a[formal.position + 3]
                        let actualType = globalMap[argument]?.1 ?? functionValueTypes[argument]
                        guard let formalType = functionValueTypes[formal.id], actualType == formalType else {
                            throw instruction.fail("Opaque function argument type mismatch", code: .typeMismatch)
                        }
                        if let actual = try opaqueResourceOrigin(argument, sources: sources,
                                                               parameters: parameterIDs, cache: &origins) {
                            links[formal.id, default: []].append(actual)
                            links[actual, default: []].append(formal.id)
                        } else {
                            unresolvedArguments[formal.id] = instruction
                        }
                    }
                    continue
                }

                let comparison = SPVImageSampleOperation(rawValue: instruction.op)?.comparison == true
                let pair = pairs[a[2]]
                let sampler = try pair.flatMap {
                    try opaqueResourceOrigin($0.sampler, sources: sources, parameters: parameterIDs, cache: &origins)
                }
                guard let sampler, try drefResourceKind(sampler, at: instruction).isSampler else {
                    if comparison {
                        throw instruction.fail("Cannot resolve the comparison sampler to a descriptor or function parameter",
                                               code: .unsupportedConstruct)
                    }
                    continue
                }
                if !comparison {
                    regularSamplers.insert(sampler)
                    continue
                }
                if comparisonSites[sampler] == nil {
                    comparisonSites[sampler] = instruction
                    requested.append(sampler)
                }
                guard let pair, let image = try opaqueResourceOrigin(pair.image, sources: sources,
                                                                    parameters: parameterIDs, cache: &origins) else {
                    throw instruction.fail("Cannot resolve the depth image to a descriptor or function parameter",
                                           code: .unsupportedConstruct)
                }
                if comparisonSites[image] == nil {
                    comparisonSites[image] = instruction
                    requested.append(image)
                }
                // Vulkan permits the image and combined image type to disagree on Depth.
                if sampledImageTypes[pair.type].flatMap({ imageDepths[$0] }) != 1 {
                    depthSitesWithoutPairedDepth[image] = instruction
                }
            }
        }

        // A formal has one IR type, so every call site and transitive forwarder must agree.
        var visited: Set<UInt32> = []
        for root in requested {
            guard visited.insert(root).inserted, let site = comparisonSites[root] else { continue }
            var component = [root]
            var cursor = 0
            var needsDeclaredDepth: SPVInstruction?
            let sampler = try drefResourceKind(root, at: site).isSampler
            while cursor < component.count {
                let id = component[cursor]
                cursor += 1
                let neighbors = links[id] ?? []
                try charge(1 + neighbors.count)
                for neighbor in neighbors where visited.insert(neighbor).inserted { component.append(neighbor) }
                if let call = unresolvedArguments[id] {
                    throw call.fail("Cannot resolve an inferred depth image or comparison sampler argument",
                                    code: .unsupportedConstruct)
                }
                if sampler, regularSamplers.contains(id) {
                    throw site.fail("A sampler cannot be used for both regular and comparison sampling",
                                    code: .unsupportedConstruct)
                }
                if needsDeclaredDepth == nil { needsDeclaredDepth = depthSitesWithoutPairedDepth[id] }
            }

            let kind = try drefQualifiedKind(root, sampler: sampler, needsDeclaredDepth: needsDeclaredDepth, at: site)
            let qualified = try internType(kind)
            for id in component {
                try charge()
                let memberKind = try drefQualifiedKind(id, sampler: sampler, needsDeclaredDepth: needsDeclaredDepth, at: site)
                guard memberKind == kind else {
                    throw site.fail("Incompatible resource types passed to a comparison function", code: .typeMismatch)
                }
                if let (handle, _) = globalMap[id] {
                    let global = globals[handle.index]
                    let current = types[global.type.index].kind
                    let target: IRHandle<IRType>
                    if case .bindingArray(let element, let size) = current {
                        if types[element.index].kind == kind { continue }
                        target = try internType(.bindingArray(base: qualified, size: size))
                    } else {
                        if current == kind { continue }
                        target = qualified
                    }
                    globals[handle.index] = .init(name: global.name, space: global.space, binding: global.binding,
                                                  type: target, initializer: global.initializer,
                                                  memoryDecorations: global.memoryDecorations, location: global.location)
                } else {
                    drefParameterTypes[id] = qualified
                }
            }
        }
    }

    mutating func opaqueResourceOrigin(
        _ id: UInt32, sources: [UInt32: UInt32], parameters: Set<UInt32>, cache: inout [UInt32: UInt32]
    ) throws -> UInt32? {
        var current = id
        var visited: Set<UInt32> = []
        var origin: UInt32 = 0
        while true {
            try charge()
            if let cached = cache[current] {
                origin = cached
                break
            }
            if parameters.contains(current) {
                origin = current
                break
            }
            if let (global, _) = globalMap[current], globals[global.index].space == .handle {
                origin = current
                break
            }
            guard let next = sources[current], visited.insert(current).inserted else { break }
            current = next
        }
        for id in visited { cache[id] = origin }
        return origin == 0 ? nil : origin
    }

    private func drefRawResourceType(_ id: UInt32, at instruction: SPVInstruction) throws -> UInt32 {
        guard let raw = globalMap[id]?.1 ?? functionValueTypes[id] else {
            throw instruction.fail("Unknown comparison resource type")
        }
        let pointee = pointers[raw]?.base ?? raw
        return descriptorArrayElements[pointee] ?? pointee
    }

    private func drefResourceKind(_ id: UInt32, at instruction: SPVInstruction) throws -> IRTypeKind {
        try kind(drefRawResourceType(id, at: instruction), instruction)
    }

    private func drefQualifiedKind(
        _ id: UInt32, sampler: Bool, needsDeclaredDepth: SPVInstruction?, at instruction: SPVInstruction
    ) throws -> IRTypeKind {
        let raw = try drefRawResourceType(id, at: instruction)
        let kind = try kind(raw, instruction)
        if sampler {
            guard kind.isSampler else {
                throw instruction.fail("Comparison parameter requires a sampler", code: .typeMismatch)
            }
            return .sampler(comparison: true)
        }
        guard let depth = imageDepths[raw], case .image(let dimension, let arrayed, _) = kind,
              kind.sampledTextureScalar == .f32, !kind.isMultisampled2D else {
            throw instruction.fail("Dref requires a depth-capable f32 image", code: .typeMismatch)
        }
        if depth == 0, let site = needsDeclaredDepth {
            throw site.fail("Dref requires compatible depth metadata", code: .typeMismatch)
        }
        let qualified = IRTypeKind.image(dimension: dimension, arrayed: arrayed, class: .depth(multisampled: false))
        guard qualified.isSampledTexture else {
            throw instruction.fail("Unsupported depth image dimensionality", code: .unsupportedConstruct)
        }
        return qualified
    }
}
