// Raw immutable cross-format IR, not a validation witness. See design-wgsl-frontend ADR.
public enum IRStage: String, Sendable, Equatable, CaseIterable {
    case vertex
    case task
    case mesh
    case fragment
    case compute
    case rayGeneration
    case miss
    case anyHit
    case closestHit
}

public enum IRScalarKind: String, Sendable, Equatable, CaseIterable {
    case sint
    case uint
    case float
    case bool
    case abstractInt
    case abstractFloat
}

public struct IRScalar: Sendable, Hashable {
    public let kind: IRScalarKind
    public let width: UInt8
    public init(kind: IRScalarKind, width: UInt8) {
        self.kind = kind
        self.width = width
    }
}

extension IRScalar {
 public static let u16 = IRScalar(kind: .uint, width: 2)
 public static let u32 = IRScalar(kind: .uint, width: 4)
 public static let u64 = IRScalar(kind: .uint, width: 8)
 public static let i64 = IRScalar(kind: .sint, width: 8)
 public static let i32 = IRScalar(kind: .sint, width: 4)
 public static let i16 = IRScalar(kind: .sint, width: 2)
 public static let f32 = IRScalar(kind: .float, width: 4)
 public static let f16 = IRScalar(kind: .float, width: 2)
 public static let bool = IRScalar(kind: .bool, width: 1)

 package var isWGSLConversionScalar: Bool {
  switch self {
  case .bool, .i16, .u16, .i32, .u32, .i64, .u64, .f32, .f16: return true
  default: return false
  }
 }
 public static let abstractInt = IRScalar(kind: .abstractInt, width: 8)
 public static let abstractFloat = IRScalar(kind: .abstractFloat, width: 8)
}
public enum IRVectorSize: UInt8, Sendable, Equatable, CaseIterable { case two = 2, three = 3, four = 4 }
public enum IRCooperativeSize: UInt8, Sendable, Equatable { case eight = 8, sixteen = 16 }

public enum IRCooperativeRole: String, Sendable, Equatable, CaseIterable {
    case a
    case b
    case c
}

public enum IRAddressSpace: Sendable, Equatable {
    case function
    case `private`
    case workgroup
    case uniform
    case storage(access: IRStorageAccess)
    case handle
    case immediate
    case taskPayload
    case rayPayload
    case incomingRayPayload
}

public struct IRStorageAccess: OptionSet, Sendable, Equatable {
 public let rawValue: UInt8
 public init(rawValue: UInt8) { self.rawValue = rawValue }
 public static let load = IRStorageAccess(rawValue: 1)
 public static let store = IRStorageAccess(rawValue: 2)
 public static let atomic = IRStorageAccess(rawValue: 4)
}

public struct IRMemoryDecorations: OptionSet, Sendable, Equatable {
 public let rawValue: UInt8
 public init(rawValue: UInt8) { self.rawValue = rawValue }
 public static let coherent = IRMemoryDecorations(rawValue: 1)
 public static let volatile = IRMemoryDecorations(rawValue: 2)
}

public struct IRBarrier: OptionSet, Sendable, Equatable {
 public let rawValue: UInt8
 public init(rawValue: UInt8) { self.rawValue = rawValue }
 public static let storage = IRBarrier(rawValue: 1)
 public static let workgroup = IRBarrier(rawValue: 2)
 public static let subgroup = IRBarrier(rawValue: 4)
 public static let texture = IRBarrier(rawValue: 8)
}

public enum IRInterpolation: String, Sendable, Equatable, CaseIterable {
    case perspective
    case linear
    case flat
    case perVertex
}

public enum IRSampling: String, Sendable, Equatable, CaseIterable {
    case center
    case centroid
    case sample
    case first
    case either
}

public enum IRBuiltIn: Sendable, Hashable {
    case primitiveIndex
    case position(invariant: Bool)
    case viewIndex
    case baseInstance
    case baseVertex
    case clipDistances
    case cullDistance
    case instanceIndex
    case pointSize
    case vertexIndex
    case drawIndex
    case fragDepth
    case pointCoord
    case frontFacing
    case barycentric(perspective: Bool)
    case sampleIndex
    case sampleMask
    case globalInvocationId
    case globalInvocationIndex
    case localInvocationId
    case localInvocationIndex
    case workgroupId
    case workgroupIndex
    case workgroupSize
    case numWorkgroups
    case numSubgroups
    case subgroupId
    case subgroupSize
    case subgroupInvocationId
    case meshTaskSize
    case cullPrimitive
    case pointIndex
    case lineIndices
    case triangleIndices
    case vertexCount
    case vertices
    case primitiveCount
    case primitives
    case rayInvocationId
    case numRayInvocations
    case instanceCustomData
    case geometryIndex
    case worldRayOrigin
    case worldRayDirection
    case objectRayOrigin
    case objectRayDirection
    case rayTmin
    case rayTCurrentMax
    case objectToWorld
    case worldToObject
    case hitKind
    case hitBarycentrics
}

public enum IRBinding: Sendable, Equatable {
    case builtIn(IRBuiltIn)
    case location(index: UInt32, interpolation: IRInterpolation?, sampling: IRSampling?, blendSource: UInt32?, perPrimitive: Bool)
}

public struct IRResourceBinding: Sendable, Hashable {
    public let group: UInt32
    public let binding: UInt32
    public init(group: UInt32, binding: UInt32) {
        self.group = group
        self.binding = binding
    }
}

public enum IRImageDimension: String, Sendable, Equatable, CaseIterable {
    case d1
    case d2
    case d3
    case cube
}

public enum IRStorageFormat: String, Sendable, Equatable, CaseIterable {
    case r8Unorm
    case r8Snorm
    case r8Uint
    case r8Sint
    case r16Uint
    case r16Sint
    case r16Float
    case rg8Unorm
    case rg8Snorm
    case rg8Uint
    case rg8Sint
    case r32Uint
    case r32Sint
    case r32Float
    case rg16Uint
    case rg16Sint
    case rg16Float
    case rgba8Unorm
    case rgba8Snorm
    case rgba8Uint
    case rgba8Sint
    case bgra8Unorm
    case rgb10a2Uint
    case rgb10a2Unorm
    case rg11b10Ufloat
    case r64Uint
    case rg32Uint
    case rg32Sint
    case rg32Float
    case rgba16Uint
    case rgba16Sint
    case rgba16Float
    case rgba32Uint
    case rgba32Sint
    case rgba32Float
    case r16Unorm
    case r16Snorm
    case rg16Unorm
    case rg16Snorm
    case rgba16Unorm
    case rgba16Snorm

    package var isNormalized16Bit: Bool {
        switch self {
        case .r16Unorm, .r16Snorm, .rg16Unorm, .rg16Snorm, .rgba16Unorm, .rgba16Snorm: true
        default: false
        }
    }

    package var isPackedColor: Bool { self == .rgb10a2Unorm || self == .rgb10a2Uint || self == .rg11b10Ufloat }

    package var isNarrow8Bit: Bool {
        switch self {
        case .r8Unorm, .r8Snorm, .r8Uint, .r8Sint, .rg8Unorm, .rg8Snorm, .rg8Uint, .rg8Sint: true
        default: false
        }
    }

    package var requires8BitReadWriteFeature: Bool {
        switch self {
        case .r8Unorm, .r8Uint, .r8Sint, .rgba8Unorm, .rgba8Uint, .rgba8Sint: true
        default: false
        }
    }

    package var requiresWideRGBAReadWriteFeature: Bool {
        switch self {
        case .rgba16Float, .rgba32Float, .rgba32Uint, .rgba32Sint: true
        default: false
        }
    }

    package var requiresTextureFormatsTier1: Bool {
        switch self {
        case .r16Float, .rg16Float, .r16Uint, .r16Sint, .rg16Uint, .rg16Sint: true
        default: isNormalized16Bit || isPackedColor || isNarrow8Bit
        }
    }

    package var storageTexel: (scalar: IRScalar, components: Int) {
        switch self {
        case .r8Uint, .r16Uint, .r32Uint: (.u32, 1)
        case .r64Uint: (.u64, 1)
        case .r8Sint, .r16Sint, .r32Sint: (.i32, 1)
        case .r8Unorm, .r8Snorm, .r16Float, .r32Float, .r16Unorm, .r16Snorm: (.f32, 1)
        case .rg8Uint, .rg16Uint, .rg32Uint: (.u32, 2)
        case .rg8Sint, .rg16Sint, .rg32Sint: (.i32, 2)
        case .rg8Unorm, .rg8Snorm, .rg16Float, .rg32Float, .rg16Unorm, .rg16Snorm: (.f32, 2)
        case .rgba8Uint, .rgba16Uint, .rgba32Uint, .rgb10a2Uint: (.u32, 4)
        case .rgba8Sint, .rgba16Sint, .rgba32Sint: (.i32, 4)
        case .rgba8Unorm, .rgba8Snorm, .bgra8Unorm, .rgba16Float, .rgba32Float, .rgba16Unorm, .rgba16Snorm,
             .rgb10a2Unorm: (.f32, 4)
        case .rg11b10Ufloat: (.f32, 3)
        }
    }
}

public enum IRImageClass: Sendable, Equatable {
    case sampled(kind: IRScalarKind, multisampled: Bool)
    case depth(multisampled: Bool)
    case external
    case storage(format: IRStorageFormat, access: IRStorageAccess)
}

public enum IRArraySize: Sendable, Equatable {
    case constant(UInt32)
    case override(IRHandle<IROverride>)
    case dynamic
}

public struct IRStructMember: Sendable, Equatable {
    public let name: String?
    public let type: IRHandle<IRType>
    public let binding: IRBinding?
    public let offset: UInt32
    public let location: ShaderLocation?
    public init(name: String? = nil, type: IRHandle<IRType>, binding: IRBinding? = nil, offset: UInt32, location: ShaderLocation? = nil) {
        self.name = name
        self.type = type
        self.binding = binding
        self.offset = offset
        self.location = location
    }
}

public enum IRTypeKind: Sendable, Equatable {
    case scalar(IRScalar)
    case vector(size: IRVectorSize, scalar: IRScalar)
    case matrix(columns: IRVectorSize, rows: IRVectorSize, scalar: IRScalar)
    case cooperativeMatrix(columns: IRCooperativeSize, rows: IRCooperativeSize, scalar: IRScalar, role: IRCooperativeRole)
    case atomic(IRScalar)
    case pointer(base: IRHandle<IRType>, space: IRAddressSpace)
    case valuePointer(size: IRVectorSize?, scalar: IRScalar, space: IRAddressSpace)
    case array(base: IRHandle<IRType>, size: IRArraySize, stride: UInt32)
    case structure(members: [IRStructMember], span: UInt32)
    case image(dimension: IRImageDimension, arrayed: Bool, class: IRImageClass)
    case sampler(comparison: Bool)
    case accelerationStructure(vertexReturn: Bool)
    case rayQuery(vertexReturn: Bool)
    case bindingArray(base: IRHandle<IRType>, size: IRArraySize)
}

extension IRTypeKind {
    package static let sampledFloat2D = Self.image(dimension: .d2, arrayed: false,
                                                   class: .sampled(kind: .float, multisampled: false))
    package static let sampledDepth2D = Self.image(dimension: .d2, arrayed: false,
                                                   class: .depth(multisampled: false))

    package static let multisampledDepth2D = Self.image(dimension: .d2, arrayed: false,
                                                        class: .depth(multisampled: true))

    package var isMultisampled2D: Bool {
        switch self {
        case .image(.d2, false, .sampled(_, true)), .multisampledDepth2D: true
        default: false
        }
    }

    package var isDepthTexture: Bool {
        if case .image(_, _, .depth) = self { return true }
        return false
    }

    package var isArrayedTexture: Bool {
        if case .image(_, true, _) = self { return true }
        return false
    }

    package var isCubeTexture: Bool {
        if case .image(.cube, _, _) = self { return true }
        return false
    }

    package var isVolumeTexture: Bool {
        if case .image(.d3, _, _) = self { return true }
        return false
    }

    package var isLineTexture: Bool {
        if case .image(.d1, _, _) = self { return true }
        return false
    }

    package func texelCoordinateType(_ scalar: IRScalar) -> IRTypeKind {
        isLineTexture ? .scalar(scalar) : .vector(size: isVolumeTexture ? .three : .two, scalar: scalar)
    }

    package var isSampledTexture: Bool { sampledTextureScalar != nil }

    package var sampledTextureScalar: IRScalar? {
        switch self {
        case .image(let dimension, let arrayed, .sampled(let kind, let multisampled))
            where ((dimension == .d2 || dimension == .cube) && (!multisampled || (dimension == .d2 && !arrayed))) ||
                  ((dimension == .d1 || dimension == .d3) && !arrayed && !multisampled):
            switch kind {
            case .sint: .i32
            case .uint: .u32
            case .float: .f32
            default: nil
            }
        case .image(let dimension, let arrayed, .depth(let multisampled))
            where (dimension == .d2 || dimension == .cube) && (!multisampled || (dimension == .d2 && !arrayed)): .f32
        default: nil
        }
    }

    package var isSampler: Bool {
        if case .sampler = self { return true }
        return false
    }

    package var isExternalTexture: Bool {
        if case .image(.d2, false, .external) = self { return true }
        return false
    }

    package var isTextureOrSampler: Bool { isSampledTexture || isExternalTexture || isSampler || storageTexture != nil }

    package var storageTexture: (format: IRStorageFormat, access: IRStorageAccess, scalar: IRScalar, components: Int)? {
        guard case .image(let dimension, let arrayed, .storage(let format, let access)) = self,
              dimension == .d2 || ((dimension == .d1 || dimension == .d3) && !arrayed) else { return nil }
        let texel = format.storageTexel
        guard format != .bgra8Unorm || access == .store,
              access == .load || access == .store ||
                  (access == [.load, .store, .atomic] && (format == .r32Uint || format == .r32Sint || format == .r64Uint)) ||
                  (access == [.load, .store] && !format.isNormalized16Bit && format != .r8Snorm &&
                      (texel.components == 1 || format.requires8BitReadWriteFeature || format.requiresWideRGBAReadWriteFeature ||
                       format == .rgba16Uint || format == .rgba16Sint))
        else { return nil }
        return (format, access, texel.scalar, texel.components)
    }
}

public struct IRType: Sendable, Equatable {
    public let name: String?
    public let kind: IRTypeKind
    public let location: ShaderLocation?
    public init(name: String? = nil, kind: IRTypeKind, location: ShaderLocation? = nil) {
        self.name = name
        self.kind = kind
        self.location = location
    }
}

public enum IRLiteral: Sendable, Equatable {
    case f64(Double)
    case f32(Float)
    case f16Bits(UInt16)
    case u16(UInt16)
    case i16(Int16)
    case u32(UInt32)
    case i32(Int32)
    case u64(UInt64)
    case i64(Int64)
    case bool(Bool)
    case abstractInt(Int64)
    case abstractFloat(Double)
}

public enum IRUnaryOperator: String, Sendable, Equatable, CaseIterable {
    case negate
    case logicalNot
    case bitwiseNot
}

public enum IRBinaryOperator: String, Sendable, Hashable, CaseIterable {
    case add
    case subtract
    case multiply
    case divide
    case modulo
    case equal
    case notEqual
    case less
    case lessEqual
    case greater
    case greaterEqual
    case and
    case exclusiveOr
    case inclusiveOr
    case logicalAnd
    case logicalOr
    case shiftLeft
    case shiftRight
}

public enum IRAtomicFunction: Sendable, Equatable {
    case add
    // Add the f32 values encoded by an atomic u32 and its operand, returning the old bits.
    case addFloat32Bits
    case subtract
    case and
    case exclusiveOr
    case inclusiveOr
    case min
    case max
    // SPIR-V can specify the comparison order independently of the integer storage type.
    case minWithOppositeSignedness
    case maxWithOppositeSignedness
    case exchange(compare: IRHandle<IRExpression>?, strong: Bool = false)

    package static let integerRMWBuiltins: [String: Self] = [
        "atomicAdd": .add, "atomicSub": .subtract,
        "atomicMin": .min, "atomicMax": .max,
        "atomicAnd": .and, "atomicOr": .inclusiveOr, "atomicXor": .exclusiveOr,
        "atomicExchange": .exchange(compare: nil),
    ]

    package var integerRMWBuiltin: String? {
        Self.integerRMWBuiltins.first { $0.value == self }?.key
    }
}

public enum IRDerivativeControl: String, Sendable, Equatable, CaseIterable {
    case coarse
    case fine
    case none

    package var nameSuffix: String {
        switch self {
        case .none: ""
        case .fine: "Fine"
        case .coarse: "Coarse"
        }
    }
}

public enum IRDerivativeAxis: String, Sendable, Equatable, CaseIterable {
    case x
    case y
    case width

    package func wgslName(control: IRDerivativeControl) -> String {
        let name = switch self {
        case .x: "dpdx"
        case .y: "dpdy"
        case .width: "fwidth"
        }
        return name + control.nameSuffix
    }

    package static let wgslBuiltins: [String: (axis: IRDerivativeAxis, control: IRDerivativeControl)] =
        Dictionary(uniqueKeysWithValues: allCases.flatMap { axis in
            IRDerivativeControl.allCases.map { control in (axis.wgslName(control: control), (axis, control)) }
        })
}

public enum IRRelationalFunction: String, Sendable, Equatable, CaseIterable {
    case all
    case any
    case isNan
    case isInf

    package var isBooleanReduction: Bool { self == .all || self == .any }

    package func accepts(_ operand: IRTypeKind) -> Bool {
        guard isBooleanReduction else { return false }
        switch operand {
        case .scalar(.bool), .vector(_, .bool): return true
        default: return false
        }
    }
}

public enum IRMathFunction: String, Sendable, Equatable, CaseIterable {
    case abs
    case min
    case max
    case clamp
    case saturate
    case cos
    case cosh
    case sin
    case sinh
    case tan
    case tanh
    case acos
    case asin
    case atan
    case atan2
    case asinh
    case acosh
    case atanh
    case radians
    case degrees
    case ceil
    case floor
    case round
    case fract
    case trunc
    case modf
    case frexp
    case ldexp
    case exp
    case exp2
    case log
    case log2
    case pow
    case dot
    case dot4I8Packed
    case dot4U8Packed
    case outer
    case cross
    case distance
    case length
    case normalize
    case faceForward
    case reflect
    case refract
    case sign
    case fma
    case mix
    case step
    case smoothStep
    case sqrt
    case inverseSqrt
    case inverse
    case transpose
    case determinant
    case quantizeToF16
    case countTrailingZeros
    case countLeadingZeros
    case countOneBits
    case reverseBits
    case extractBits
    case insertBits
    case firstTrailingBit
    case firstLeadingBit
    case pack4x8snorm
    case pack4x8unorm
    case pack2x16snorm
    case pack2x16unorm
    case pack2x16float
    case pack4xI8
    case pack4xU8
    case pack4xI8Clamp
    case pack4xU8Clamp
    case unpack4x8snorm
    case unpack4x8unorm
    case unpack2x16snorm
    case unpack2x16unorm
    case unpack2x16float
    case unpack4xI8
    case unpack4xU8
}

public enum IRSwizzleComponent: String, Sendable, Equatable, CaseIterable {
    case x
    case y
    case z
    case w
}

public struct IRSwizzle: Sendable, Equatable {
    public let x: IRSwizzleComponent
    public let y: IRSwizzleComponent
    public let z: IRSwizzleComponent
    public let w: IRSwizzleComponent
    public init(x: IRSwizzleComponent, y: IRSwizzleComponent, z: IRSwizzleComponent, w: IRSwizzleComponent) {
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }
}

extension IRMathFunction {
    package var integerBitArity: Int? {
        switch self {
        case .countLeadingZeros, .countTrailingZeros, .countOneBits, .reverseBits, .firstLeadingBit, .firstTrailingBit: 1
        case .extractBits: 3
        case .insertBits: 4
        default: nil
        }
    }
}

public enum IRSampleLevel: Sendable, Equatable {
    case auto
    case zero
    case exact(IRHandle<IRExpression>)
    case bias(IRHandle<IRExpression>)
    case gradient(x: IRHandle<IRExpression>, y: IRHandle<IRExpression>)

    package var operands: [IRHandle<IRExpression>] {
        switch self {
        case .auto, .zero: []
        case .exact(let level), .bias(let level): [level]
        case .gradient(let x, let y): [x, y]
        }
    }
}

public enum IRImageQuery: Sendable, Equatable {
    case size(level: IRHandle<IRExpression>?)
    case numLevels
    case numLayers
    case numSamples
}

public struct IRImageSample: Sendable, Equatable {
    public let image: IRHandle<IRExpression>
    public let sampler: IRHandle<IRExpression>
    public let gather: IRSwizzleComponent?
    public let coordinate: IRHandle<IRExpression>
    public let arrayIndex: IRHandle<IRExpression>?
    public let offset: IRHandle<IRExpression>?
    public let level: IRSampleLevel
    public let depthReference: IRHandle<IRExpression>?
    public let clampToEdge: Bool
    public init(image: IRHandle<IRExpression>, sampler: IRHandle<IRExpression>, gather: IRSwizzleComponent? = nil, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>? = nil, offset: IRHandle<IRExpression>? = nil, level: IRSampleLevel = .auto, depthReference: IRHandle<IRExpression>? = nil, clampToEdge: Bool = false) {
        self.image = image
        self.sampler = sampler
        self.gather = gather
        self.coordinate = coordinate
        self.arrayIndex = arrayIndex
        self.offset = offset
        self.level = level
        self.depthReference = depthReference
        self.clampToEdge = clampToEdge
    }
}

public struct IRImageLoad: Sendable, Equatable {
    public let image: IRHandle<IRExpression>
    public let coordinate: IRHandle<IRExpression>
    public let arrayIndex: IRHandle<IRExpression>?
    public let sample: IRHandle<IRExpression>?
    public let level: IRHandle<IRExpression>?
    public init(image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>? = nil, sample: IRHandle<IRExpression>? = nil, level: IRHandle<IRExpression>? = nil) {
        self.image = image
        self.coordinate = coordinate
        self.arrayIndex = arrayIndex
        self.sample = sample
        self.level = level
    }
}

public struct IRCooperativeData: Sendable, Equatable {
    public let pointer: IRHandle<IRExpression>
    public let stride: IRHandle<IRExpression>
    public let rowMajor: Bool
    public init(pointer: IRHandle<IRExpression>, stride: IRHandle<IRExpression>, rowMajor: Bool) {
        self.pointer = pointer
        self.stride = stride
        self.rowMajor = rowMajor
    }
}

public enum IRExpressionKind: Sendable, Equatable {
    case literal(IRLiteral)
    case constant(IRHandle<IRConstant>)
    case override(IRHandle<IROverride>)
    /// Captures an immutable runtime value without changing its uniformity.
    case runtimeValue(IRHandle<IRExpression>)
    case zeroValue(IRHandle<IRType>)
    case compose(type: IRHandle<IRType>, components: [IRHandle<IRExpression>])
    case access(base: IRHandle<IRExpression>, index: IRHandle<IRExpression>)
    case accessIndex(base: IRHandle<IRExpression>, index: UInt32)
    case splat(size: IRVectorSize, value: IRHandle<IRExpression>)
    case swizzle(size: IRVectorSize, vector: IRHandle<IRExpression>, pattern: IRSwizzle)
    case functionArgument(UInt32)
    case globalVariable(IRHandle<IRGlobalVariable>)
    case localVariable(IRHandle<IRLocalVariable>)
    case load(pointer: IRHandle<IRExpression>)
    case imageSample(IRImageSample)
    case imageLoad(IRImageLoad)
    case imageQuery(image: IRHandle<IRExpression>, query: IRImageQuery)
    case unary(op: IRUnaryOperator, expression: IRHandle<IRExpression>)
    case binary(op: IRBinaryOperator, left: IRHandle<IRExpression>, right: IRHandle<IRExpression>)
    case select(condition: IRHandle<IRExpression>, accept: IRHandle<IRExpression>, reject: IRHandle<IRExpression>)
    case derivative(axis: IRDerivativeAxis, control: IRDerivativeControl, expression: IRHandle<IRExpression>)
    case relational(function: IRRelationalFunction, argument: IRHandle<IRExpression>)
    case math(function: IRMathFunction, argument: IRHandle<IRExpression>, argument1: IRHandle<IRExpression>?, argument2: IRHandle<IRExpression>?, argument3: IRHandle<IRExpression>?)
    case cast(expression: IRHandle<IRExpression>, target: IRScalar)
    case bitcast(expression: IRHandle<IRExpression>, type: IRHandle<IRType>)
    case callResult(IRHandle<IRFunction>)
    case atomicResult(type: IRHandle<IRType>, comparison: Bool)
    case workgroupUniformLoadResult(type: IRHandle<IRType>)
    case arrayLength(IRHandle<IRExpression>)
    case rayQueryVertexPositions(query: IRHandle<IRExpression>, committed: Bool)
    case rayQueryProceedResult
    case rayQueryGetIntersection(query: IRHandle<IRExpression>, committed: Bool)
    case subgroupBallotResult
    case subgroupOperationResult(type: IRHandle<IRType>)
    case cooperativeLoad(columns: IRCooperativeSize, rows: IRCooperativeSize, role: IRCooperativeRole, data: IRCooperativeData)
    case cooperativeMultiplyAdd(a: IRHandle<IRExpression>, b: IRHandle<IRExpression>, c: IRHandle<IRExpression>)
}

public struct IRExpression: Sendable, Equatable {
    public let type: IRHandle<IRType>
    public let kind: IRExpressionKind
    public let location: ShaderLocation?
    public init(type: IRHandle<IRType>, kind: IRExpressionKind, location: ShaderLocation? = nil) {
        self.type = type
        self.kind = kind
        self.location = location
    }
}

public enum IRSwitchValue: Sendable, Hashable {
    case i32(Int32)
    case u32(UInt32)
    case `default`
}

public struct IRSwitchCase: Sendable, Equatable {
    public let value: IRSwitchValue
    public let body: IRHandle<IRBlock>
    public let fallThrough: Bool
    public init(value: IRSwitchValue, body: IRHandle<IRBlock>, fallThrough: Bool = false) {
        self.value = value
        self.body = body
        self.fallThrough = fallThrough
    }
}

public enum IRRayQueryFunction: Sendable, Equatable {
    case initialize(accelerationStructure: IRHandle<IRExpression>, descriptor: IRHandle<IRExpression>)
    case proceed(result: IRHandle<IRExpression>)
    case generateIntersection(hitT: IRHandle<IRExpression>)
    case confirmIntersection
    case terminate
    case begin
}

public enum IRDirection: String, Sendable, Equatable, CaseIterable {
    case x
    case y
    case diagonal
}

public enum IRGatherMode: Sendable, Equatable {
    case broadcastFirst
    case broadcast(IRHandle<IRExpression>)
    case shuffle(IRHandle<IRExpression>)
    case shuffleDown(IRHandle<IRExpression>)
    case shuffleUp(IRHandle<IRExpression>)
    case shuffleXor(IRHandle<IRExpression>)
    case quadBroadcast(IRHandle<IRExpression>)
    /// The source requires a dynamically uniform selector in 0..<4.
    case quadBroadcastDynamic(IRHandle<IRExpression>)
    case quadSwap(IRDirection)
}

extension IRGatherMode {
    package var index: IRHandle<IRExpression>? {
        switch self {
        case .broadcast(let index), .shuffle(let index), .shuffleDown(let index),
             .shuffleUp(let index), .shuffleXor(let index), .quadBroadcast(let index), .quadBroadcastDynamic(let index): index
        case .broadcastFirst, .quadSwap: nil
        }
    }

    package func remappingIndex(_ map: (IRHandle<IRExpression>) throws -> IRHandle<IRExpression>) rethrows -> Self {
        switch self {
        case .broadcast(let index): try .broadcast(map(index))
        case .shuffle(let index): try .shuffle(map(index))
        case .shuffleDown(let index): try .shuffleDown(map(index))
        case .shuffleUp(let index): try .shuffleUp(map(index))
        case .shuffleXor(let index): try .shuffleXor(map(index))
        case .quadBroadcast(let index): try .quadBroadcast(map(index))
        case .quadBroadcastDynamic(let index): try .quadBroadcastDynamic(map(index))
        case .broadcastFirst, .quadSwap: self
        }
    }
}

public enum IRSubgroupOperation: String, Sendable, Equatable, CaseIterable {
    case all
    case any
    case add
    case mul
    case min
    case max
    case and
    case or
    case xor
}

extension IRSubgroupOperation {
    package static let wgslBuiltins: [String: (IRSubgroupOperation, IRCollectiveOperation)] = {
        var entries: [String: (IRSubgroupOperation, IRCollectiveOperation)] = [:]
        for operation in allCases {
            for collective in IRCollectiveOperation.allCases {
                if let name = operation.wgslName(collective) { entries[name] = (operation, collective) }
            }
        }
        return entries
    }()

    package func wgslName(_ collective: IRCollectiveOperation) -> String? {
        guard collective == .reduce || self == .add || self == .mul else { return nil }
        let prefix: String
        switch collective {
        case .reduce: prefix = "subgroup"
        case .inclusiveScan: prefix = "subgroupInclusive"
        case .exclusiveScan: prefix = "subgroupExclusive"
        }
        return prefix + rawValue.prefix(1).uppercased() + rawValue.dropFirst()
    }

    package var isVote: Bool { self == .all || self == .any }
    package var isBitwise: Bool { self == .and || self == .or || self == .xor }

    package func accepts(_ type: IRTypeKind, collective: IRCollectiveOperation) -> Bool {
        guard wgslName(collective) != nil else { return false }
        if isVote { return type == .scalar(.bool) }
        let scalar: IRScalar
        switch type {
        case .scalar(let value), .vector(_, let value): scalar = value
        default: return false
        }
        if isBitwise { return scalar == .i32 || scalar == .u32 }
        return [.i32, .u32, .f32, .f16].contains(scalar)
    }
}

public enum IRCollectiveOperation: String, Sendable, Equatable, CaseIterable {
    case reduce
    case inclusiveScan
    case exclusiveScan
}

public enum IRStatementKind: Sendable, Equatable {
    case emit([IRHandle<IRExpression>])
    case block(IRHandle<IRBlock>)
    case ifCondition(condition: IRHandle<IRExpression>, accept: IRHandle<IRBlock>, reject: IRHandle<IRBlock>)
    case switchValue(selector: IRHandle<IRExpression>, cases: [IRSwitchCase])
    case loop(body: IRHandle<IRBlock>, continuing: IRHandle<IRBlock>, breakIf: IRHandle<IRExpression>?)
    case `break`
    case `continue`
    case returnValue(IRHandle<IRExpression>?)
    case kill
    /// Suppresses fragment output and subsequent resource writes while execution can continue.
    case demoteToHelperInvocation
    case controlBarrier(IRBarrier)
    case memoryBarrier(IRBarrier)
    case store(pointer: IRHandle<IRExpression>, value: IRHandle<IRExpression>)
    case imageStore(image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?, value: IRHandle<IRExpression>)
    case atomic(pointer: IRHandle<IRExpression>, function: IRAtomicFunction, value: IRHandle<IRExpression>, result: IRHandle<IRExpression>?)
    case imageAtomic(image: IRHandle<IRExpression>, coordinate: IRHandle<IRExpression>, arrayIndex: IRHandle<IRExpression>?, function: IRAtomicFunction, value: IRHandle<IRExpression>)
    case workgroupUniformLoad(pointer: IRHandle<IRExpression>, result: IRHandle<IRExpression>)
    case call(function: IRHandle<IRFunction>, arguments: [IRHandle<IRExpression>], result: IRHandle<IRExpression>?)
    case rayQuery(query: IRHandle<IRExpression>, function: IRRayQueryFunction)
    case traceRay(accelerationStructure: IRHandle<IRExpression>, descriptor: IRHandle<IRExpression>, payload: IRHandle<IRExpression>)
    case subgroupBallot(result: IRHandle<IRExpression>, predicate: IRHandle<IRExpression>?)
    case subgroupGather(mode: IRGatherMode, argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>)
    case subgroupCollectiveOperation(op: IRSubgroupOperation, collective: IRCollectiveOperation, argument: IRHandle<IRExpression>, result: IRHandle<IRExpression>)
    case cooperativeStore(target: IRHandle<IRExpression>, data: IRCooperativeData)
    case debugPrintf(format: String, arguments: [IRHandle<IRExpression>])
}

public struct IRStatement: Sendable, Equatable {
    public let kind: IRStatementKind
    public let location: ShaderLocation?
    public init(kind: IRStatementKind, location: ShaderLocation? = nil) {
        self.kind = kind
        self.location = location
    }
}

public struct IRBlock: Sendable, Equatable {
    public let statements: [IRHandle<IRStatement>]
    public let location: ShaderLocation?
    public init(statements: [IRHandle<IRStatement>] = [], location: ShaderLocation? = nil) {
        self.statements = statements
        self.location = location
    }
}

public struct IRConstant: Sendable, Equatable {
    public let name: String?
    public let type: IRHandle<IRType>
    public let initializer: IRHandle<IRExpression>
    public let location: ShaderLocation?
    public init(name: String? = nil, type: IRHandle<IRType>, initializer: IRHandle<IRExpression>, location: ShaderLocation? = nil) {
        self.name = name
        self.type = type
        self.initializer = initializer
        self.location = location
    }
}

public struct IROverride: Sendable, Equatable {
    public let name: String?
    public let id: UInt16?
    public let type: IRHandle<IRType>
    public let initializer: IRHandle<IRExpression>?
    public let location: ShaderLocation?
    public init(name: String? = nil, id: UInt16? = nil, type: IRHandle<IRType>, initializer: IRHandle<IRExpression>? = nil, location: ShaderLocation? = nil) {
        self.name = name
        self.id = id
        self.type = type
        self.initializer = initializer
        self.location = location
    }
}

public struct IRGlobalVariable: Sendable, Equatable {
    public let name: String?
    public let space: IRAddressSpace
    public let binding: IRResourceBinding?
    public let type: IRHandle<IRType>
    public let initializer: IRHandle<IRExpression>?
    public let memoryDecorations: IRMemoryDecorations
    public let location: ShaderLocation?
    public init(name: String? = nil, space: IRAddressSpace, binding: IRResourceBinding? = nil, type: IRHandle<IRType>, initializer: IRHandle<IRExpression>? = nil, memoryDecorations: IRMemoryDecorations = [], location: ShaderLocation? = nil) {
        self.name = name
        self.space = space
        self.binding = binding
        self.type = type
        self.initializer = initializer
        self.memoryDecorations = memoryDecorations
        self.location = location
    }
}

public struct IRLocalVariable: Sendable, Equatable {
    public let name: String?
    public let type: IRHandle<IRType>
    public let initializer: IRHandle<IRExpression>?
    public let location: ShaderLocation?
    public init(name: String? = nil, type: IRHandle<IRType>, initializer: IRHandle<IRExpression>? = nil, location: ShaderLocation? = nil) {
        self.name = name
        self.type = type
        self.initializer = initializer
        self.location = location
    }
}

public struct IRFunctionArgument: Sendable, Equatable {
    public let name: String?
    public let type: IRHandle<IRType>
    public let binding: IRBinding?
    public let location: ShaderLocation?
    public init(name: String? = nil, type: IRHandle<IRType>, binding: IRBinding? = nil, location: ShaderLocation? = nil) {
        self.name = name
        self.type = type
        self.binding = binding
        self.location = location
    }
}

public struct IRFunctionResult: Sendable, Equatable {
    public let type: IRHandle<IRType>
    public let binding: IRBinding?
    public init(type: IRHandle<IRType>, binding: IRBinding? = nil) {
        self.type = type
        self.binding = binding
    }
}

public struct IRNamedExpression: Sendable, Equatable {
    public let expression: IRHandle<IRExpression>
    public let name: String
    public init(expression: IRHandle<IRExpression>, name: String) {
        self.expression = expression
        self.name = name
    }
}

public struct IRFunction: Sendable, Equatable {
    public let name: String?
    public let arguments: [IRFunctionArgument]
    public let result: IRFunctionResult?
    public let localVariables: IRTable<IRLocalVariable>
    public let expressions: IRTable<IRExpression>
    public let namedExpressions: [IRNamedExpression]
    public let statements: IRTable<IRStatement>
    public let blocks: IRTable<IRBlock>
    public let body: IRHandle<IRBlock>
    public let diagnosticFilter: IRHandle<IRDiagnosticFilter>?
    public let location: ShaderLocation?
    public init(name: String? = nil, arguments: [IRFunctionArgument] = [], result: IRFunctionResult? = nil, localVariables: IRTable<IRLocalVariable> = IRTable(), expressions: IRTable<IRExpression> = IRTable(), namedExpressions: [IRNamedExpression] = [], statements: IRTable<IRStatement> = IRTable(), blocks: IRTable<IRBlock> = IRTable(), body: IRHandle<IRBlock>, diagnosticFilter: IRHandle<IRDiagnosticFilter>? = nil, location: ShaderLocation? = nil) {
        self.name = name
        self.arguments = arguments
        self.result = result
        self.localVariables = localVariables
        self.expressions = expressions
        self.namedExpressions = namedExpressions
        self.statements = statements
        self.blocks = blocks
        self.body = body
        self.diagnosticFilter = diagnosticFilter
        self.location = location
    }
}

public enum IRConservativeDepth: String, Sendable, Equatable, CaseIterable {
    case greaterEqual
    case lessEqual
    case unchanged
}

public enum IREarlyDepthTest: Sendable, Equatable {
    case force
    case allow(IRConservativeDepth)
}

public struct IRWorkgroupSize: Sendable, Equatable {
    public let x: UInt32
    public let y: UInt32
    public let z: UInt32
    public init(x: UInt32 = 1, y: UInt32 = 1, z: UInt32 = 1) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public struct IRWorkgroupOverrides: Sendable, Equatable {
    public let x: IRHandle<IRExpression>?
    public let y: IRHandle<IRExpression>?
    public let z: IRHandle<IRExpression>?
    public init(x: IRHandle<IRExpression>? = nil, y: IRHandle<IRExpression>? = nil, z: IRHandle<IRExpression>? = nil) {
        self.x = x
        self.y = y
        self.z = z
    }
}

public enum IRMeshOutputTopology: String, Sendable, Equatable, CaseIterable {
    case points
    case lines
    case triangles
}

public struct IRMeshStageInfo: Sendable, Equatable {
    public let topology: IRMeshOutputTopology
    public let maxVertices: UInt32
    public let maxVerticesOverride: IRHandle<IRExpression>?
    public let maxPrimitives: UInt32
    public let maxPrimitivesOverride: IRHandle<IRExpression>?
    public let vertexOutputType: IRHandle<IRType>
    public let primitiveOutputType: IRHandle<IRType>
    public let outputVariable: IRHandle<IRGlobalVariable>
    public init(topology: IRMeshOutputTopology, maxVertices: UInt32, maxVerticesOverride: IRHandle<IRExpression>? = nil, maxPrimitives: UInt32, maxPrimitivesOverride: IRHandle<IRExpression>? = nil, vertexOutputType: IRHandle<IRType>, primitiveOutputType: IRHandle<IRType>, outputVariable: IRHandle<IRGlobalVariable>) {
        self.topology = topology
        self.maxVertices = maxVertices
        self.maxVerticesOverride = maxVerticesOverride
        self.maxPrimitives = maxPrimitives
        self.maxPrimitivesOverride = maxPrimitivesOverride
        self.vertexOutputType = vertexOutputType
        self.primitiveOutputType = primitiveOutputType
        self.outputVariable = outputVariable
    }
}

public enum IRSubgroupSize: Sendable, Equatable {
    case fixed(UInt32)
    case overrideExpression(IRHandle<IRExpression>)
}

public struct IREntryPoint: Sendable, Equatable {
    public let name: String
    public let stage: IRStage
    public let function: IRHandle<IRFunction>
    public let earlyDepthTest: IREarlyDepthTest?
    public let workgroupSize: IRWorkgroupSize
    public let workgroupOverrides: IRWorkgroupOverrides?
    public let subgroupSize: IRSubgroupSize?
    public let meshInfo: IRMeshStageInfo?
    public let taskPayload: IRHandle<IRGlobalVariable>?
    public let incomingRayPayload: IRHandle<IRGlobalVariable>?
    public let location: ShaderLocation?
    public init(name: String, stage: IRStage, function: IRHandle<IRFunction>, earlyDepthTest: IREarlyDepthTest? = nil, workgroupSize: IRWorkgroupSize = IRWorkgroupSize(), workgroupOverrides: IRWorkgroupOverrides? = nil, subgroupSize: IRSubgroupSize? = nil, meshInfo: IRMeshStageInfo? = nil, taskPayload: IRHandle<IRGlobalVariable>? = nil, incomingRayPayload: IRHandle<IRGlobalVariable>? = nil, location: ShaderLocation? = nil) {
        self.name = name
        self.stage = stage
        self.function = function
        self.earlyDepthTest = earlyDepthTest
        self.workgroupSize = workgroupSize
        self.workgroupOverrides = workgroupOverrides
        self.subgroupSize = subgroupSize
        self.meshInfo = meshInfo
        self.taskPayload = taskPayload
        self.incomingRayPayload = incomingRayPayload
        self.location = location
    }
}

public enum IRPredeclaredType: Sendable, Equatable {
    case atomicCompareExchangeWeakResult(IRScalar)
    case modfResult(size: IRVectorSize?, scalar: IRScalar)
    case frexpResult(size: IRVectorSize?, scalar: IRScalar)
}

public struct IRPredeclaredTypeBinding: Sendable, Equatable {
    public let kind: IRPredeclaredType
    public let type: IRHandle<IRType>
    public init(kind: IRPredeclaredType, type: IRHandle<IRType>) {
        self.kind = kind
        self.type = type
    }
}

public struct IRSpecialTypes: Sendable, Equatable {
    public let rayDescriptor: IRHandle<IRType>?
    public let rayIntersection: IRHandle<IRType>?
    public let rayVertexReturn: IRHandle<IRType>?
    public let externalTextureParameters: IRHandle<IRType>?
    public let externalTextureTransferFunction: IRHandle<IRType>?
    public let predeclared: [IRPredeclaredTypeBinding]
    public init(rayDescriptor: IRHandle<IRType>? = nil, rayIntersection: IRHandle<IRType>? = nil, rayVertexReturn: IRHandle<IRType>? = nil, externalTextureParameters: IRHandle<IRType>? = nil, externalTextureTransferFunction: IRHandle<IRType>? = nil, predeclared: [IRPredeclaredTypeBinding] = []) {
        self.rayDescriptor = rayDescriptor
        self.rayIntersection = rayIntersection
        self.rayVertexReturn = rayVertexReturn
        self.externalTextureParameters = externalTextureParameters
        self.externalTextureTransferFunction = externalTextureTransferFunction
        self.predeclared = predeclared
    }
}

public enum IRDiagnosticSeverity: String, Sendable, Equatable, CaseIterable {
    case off
    case info
    case warning
    case error
}

public struct IRDiagnosticFilter: Sendable, Equatable {
    public let parent: IRHandle<IRDiagnosticFilter>?
    public let rule: IRDiagnosticRule
    public let severity: IRDiagnosticSeverity
    public let location: ShaderLocation?
    public init(parent: IRHandle<IRDiagnosticFilter>? = nil, rule: IRDiagnosticRule, severity: IRDiagnosticSeverity,
                location: ShaderLocation? = nil) {
        self.parent = parent
        self.rule = rule
        self.severity = severity
        self.location = location
    }
}

public enum IRDocumentationTarget: Sendable, Equatable {
    case module
    case type(IRHandle<IRType>)
    case member(type: IRHandle<IRType>, index: UInt32)
    case function(IRHandle<IRFunction>)
    case entryPoint(UInt32)
    case constant(IRHandle<IRConstant>)
    case globalVariable(IRHandle<IRGlobalVariable>)
}

public struct IRDocumentation: Sendable, Equatable {
    public let target: IRDocumentationTarget
    public let lines: [String]
    public init(target: IRDocumentationTarget, lines: [String]) {
        self.target = target
        self.lines = lines
    }
}

public struct IRModule: Sendable, Equatable {
    public let types: IRTable<IRType>
    public let specialTypes: IRSpecialTypes
    public let constants: IRTable<IRConstant>
    public let overrides: IRTable<IROverride>
    public let globalVariables: IRTable<IRGlobalVariable>
    public let globalExpressions: IRTable<IRExpression>
    public let functions: IRTable<IRFunction>
    public let entryPoints: [IREntryPoint]
    public let diagnosticFilters: IRTable<IRDiagnosticFilter>
    public let diagnosticFilter: IRHandle<IRDiagnosticFilter>?
    public let documentation: [IRDocumentation]
    public let allowsPointerSubviews: Bool
    public let uniformBufferStandardLayout: Bool
    public let requiresSubgroupUniformity: Bool
    public init(types: IRTable<IRType> = IRTable(), specialTypes: IRSpecialTypes = IRSpecialTypes(), constants: IRTable<IRConstant> = IRTable(), overrides: IRTable<IROverride> = IRTable(), globalVariables: IRTable<IRGlobalVariable> = IRTable(), globalExpressions: IRTable<IRExpression> = IRTable(), functions: IRTable<IRFunction> = IRTable(), entryPoints: [IREntryPoint] = [], diagnosticFilters: IRTable<IRDiagnosticFilter> = IRTable(), diagnosticFilter: IRHandle<IRDiagnosticFilter>? = nil, documentation: [IRDocumentation] = [], allowsPointerSubviews: Bool = false, uniformBufferStandardLayout: Bool = false, requiresSubgroupUniformity: Bool = false) {
        self.types = types
        self.specialTypes = specialTypes
        self.constants = constants
        self.overrides = overrides
        self.globalVariables = globalVariables
        self.globalExpressions = globalExpressions
        self.functions = functions
        self.entryPoints = entryPoints
        self.diagnosticFilters = diagnosticFilters
        self.diagnosticFilter = diagnosticFilter
        self.documentation = documentation
        self.allowsPointerSubviews = allowsPointerSubviews
        self.uniformBufferStandardLayout = uniformBufferStandardLayout
        self.requiresSubgroupUniformity = requiresSubgroupUniformity
    }
}

extension IRModule: ShaderIRModule { public var schemaVersion: UInt32 { 1 } }

/// Filterable language rule, distinct from emitted SAGA diagnostic codes.
public enum IRDiagnosticRule: Sendable, Hashable {
    case derivativeUniformity
    case subgroupUniformity
    case unknown(String)
    case user(namespace: String, name: String)

    package var wgslName: String {
        switch self {
        case .derivativeUniformity: "derivative_uniformity"
        case .subgroupUniformity: "subgroup_uniformity"
        case .unknown(let name): name
        case .user(let namespace, let name): "\(namespace).\(name)"
        }
    }
}

public enum IRRayQueryIntersection: UInt32, Sendable, Equatable { case none = 0, triangle = 1, generated = 2, aabb = 3 }
public struct IRRayFlags: OptionSet, Sendable, Equatable {
    public let rawValue: UInt32
    public init(rawValue: UInt32) { self.rawValue = rawValue }
    public static let forceOpaque = Self(rawValue: 0x1)
    public static let forceNoOpaque = Self(rawValue: 0x2)
    public static let terminateOnFirstHit = Self(rawValue: 0x4)
    public static let skipClosestHitShader = Self(rawValue: 0x8)
    public static let cullBackFacing = Self(rawValue: 0x10)
    public static let cullFrontFacing = Self(rawValue: 0x20)
    public static let cullOpaque = Self(rawValue: 0x40)
    public static let cullNoOpaque = Self(rawValue: 0x80)
    public static let skipTriangles = Self(rawValue: 0x100)
    public static let skipAabbs = Self(rawValue: 0x200)
}
