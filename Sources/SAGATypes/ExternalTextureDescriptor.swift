public enum ExternalTextureColorSpace: UInt32, Sendable {
    case sRGB
    case displayP3
}

public struct ExternalTextureDescriptor: Sendable, Equatable {
    /// Host-scoped token for a registered VideoFrame.
    public let videoFrame: UInt64
    public let colorSpace: ExternalTextureColorSpace

    public init(videoFrame: UInt64, colorSpace: ExternalTextureColorSpace = .sRGB) {
        self.videoFrame = videoFrame
        self.colorSpace = colorSpace
    }
}

public enum PlanarExternalTextureFormat: Sendable, Equatable {
    case rgba
    case nv12
    case yu12
}

public struct ExternalTextureTransferFunction: Sendable, Equatable {
    public let a: Float
    public let b: Float
    public let gamma: Float
    public let k: Float

    public init(a: Float = 1, b: Float = 0, gamma: Float = 1, k: Float = 1) {
        self.a = a
        self.b = b
        self.gamma = gamma
        self.k = k
    }
}

/// Plane handles and column-major parameters for a native external texture.
public struct PlanarExternalTextureDescriptor: Sendable, Equatable {
    public let format: PlanarExternalTextureFormat
    /// Texture or texture-view handles in RGBA, Y and CbCr, or Y, Cb and Cr order.
    public let planes: [ResourceID]
    /// Zero width and height use the first plane's extent.
    public let width: UInt32
    public let height: UInt32
    /// Column-major YCbCr-to-RGBA matrix. The default is the identity matrix.
    public let yuvConversionMatrix: [Float]
    public let gamutConversionMatrix: [Float]
    public let sourceTransferFunction: ExternalTextureTransferFunction
    public let destinationTransferFunction: ExternalTextureTransferFunction
    public let sampleTransform: [Float]
    public let loadTransform: [Float]

    public init(
        format: PlanarExternalTextureFormat, planes: [ResourceID], width: UInt32 = 0, height: UInt32 = 0,
        yuvConversionMatrix: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        gamutConversionMatrix: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1],
        sourceTransferFunction: ExternalTextureTransferFunction = .init(),
        destinationTransferFunction: ExternalTextureTransferFunction = .init(),
        sampleTransform: [Float] = [1, 0, 0, 1, 0, 0], loadTransform: [Float] = [1, 0, 0, 1, 0, 0]
    ) {
        self.format = format
        self.planes = planes
        self.width = width
        self.height = height
        self.yuvConversionMatrix = yuvConversionMatrix
        self.gamutConversionMatrix = gamutConversionMatrix
        self.sourceTransferFunction = sourceTransferFunction
        self.destinationTransferFunction = destinationTransferFunction
        self.sampleTransform = sampleTransform
        self.loadTransform = loadTransform
    }
}

extension PlanarExternalTextureFormat {
    package var planeComponents: [UInt8] {
        switch self {
        case .rgba: [4]
        case .nv12: [1, 2]
        case .yu12: [1, 1, 1]
        }
    }
}

extension PlanarExternalTextureDescriptor {
    package func planeExtent(
        at index: Int, texture: TextureDescriptor, view: TextureViewResource?, capabilities: DeviceCapabilities
    ) throws -> (width: UInt32, height: UInt32) {
        let dimension = view?.dimension ?? TextureViewDescriptor().resolvedDimension(in: texture)
        guard texture.usage.contains(.textureBinding), texture.sampleCount == 1, dimension == .d2,
              (view?.aspect ?? .all) == .all,
              let extent = texture.mipExtent(view?.levels.lowerBound ?? 0) else {
            throw GPUError.validation("External texture plane requires a sampled two-dimensional view")
        }
        let format = texture.format
        let components: UInt8
        switch format {
        case .bc4RUnorm, .bc4RSnorm, .eacR11Unorm, .eacR11Snorm: components = 1
        case .bc5RGUnorm, .bc5RGSnorm, .eacRG11Unorm, .eacRG11Snorm: components = 2
        case .bc6HRGBUfloat, .bc6HRGBFloat, .etc2RGB8Unorm, .etc2RGB8UnormSrgb: components = 3
        default: components = format.colorAttachmentComponentCount
        }
        guard index < self.format.planeComponents.count, components == self.format.planeComponents[index],
              format.sampleType == .float,
              !format.isFloat32Color || capabilities.features.contains(.float32Filterable),
              !format.isNormalized16Color || capabilities.features.contains(.normalized16Filterable) else {
            throw GPUError.validation("Invalid external texture plane format")
        }
        return extent
    }

    package func parameterWords(planeWidth: UInt32, planeHeight: UInt32) throws -> [UInt32] {
        guard planes.count == format.planeComponents.count, planeWidth > 0, planeHeight > 0,
              (width == 0) == (height == 0) else {
            throw GPUError.validation("Invalid external texture plane count or extent")
        }
        guard yuvConversionMatrix.count == 16, gamutConversionMatrix.count == 9,
              yuvConversionMatrix.allSatisfy(\.isFinite), gamutConversionMatrix.allSatisfy(\.isFinite),
              sourceTransferFunction.valid, destinationTransferFunction.valid else {
            throw GPUError.validation("Invalid external texture color conversion parameters")
        }
        let logicalWidth = Float(width == 0 ? planeWidth : width)
        let logicalHeight = Float(height == 0 ? planeHeight : height)
        let sample = try transformBounds(sampleTransform, width: 1, height: 1)
        let load = try transformBounds(loadTransform, width: logicalWidth - 1, height: logicalHeight - 1)
        let plane = SIMD2(Float(planeWidth), Float(planeHeight))
        let footprint = (sample.high - sample.low) * plane
        guard sample.low.x >= -0.000001, sample.low.y >= -0.000001,
              sample.high.x <= 1.000001, sample.high.y <= 1.000001,
              abs(footprint.x - logicalWidth) <= 0.5, abs(footprint.y - logicalHeight) <= 0.5,
              load.low.x >= -0.5, load.low.y >= -0.5,
              load.high.x <= plane.x - 0.5, load.high.y <= plane.y - 0.5 else {
            throw GPUError.validation("External texture transforms exceed the plane or logical extent")
        }

        var words = yuvConversionMatrix.map(\.bitPattern)
        for column in 0..<3 {
            words += gamutConversionMatrix[(3 * column)..<(3 * column + 3)].map(\.bitPattern)
            words.append(0)
        }
        words += sourceTransferFunction.coefficients.map(\.bitPattern)
        words += destinationTransferFunction.coefficients.map(\.bitPattern)
        words += sampleTransform.map(\.bitPattern)
        words += loadTransform.map(\.bitPattern)
        words += [width, height, UInt32(planes.count), 0]
        return words
    }

    private func transformBounds(
        _ transform: [Float], width: Float, height: Float
    ) throws -> (low: SIMD2<Float>, high: SIMD2<Float>) {
        guard transform.count == 6, transform.allSatisfy(\.isFinite) else {
            throw GPUError.validation("Invalid external texture transform")
        }
        let x = SIMD2(transform[0], transform[1])
        let y = SIMD2(transform[2], transform[3])
        guard (x.x != 0 && x.y == 0 && y.x == 0 && y.y != 0) ||
              (x.x == 0 && x.y != 0 && y.x != 0 && y.y == 0) else {
            throw GPUError.validation("External texture transform must preserve axis alignment")
        }
        let origin = SIMD2(transform[4], transform[5])
        let opposite = origin + x * width + y * height
        return (SIMD2(min(origin.x, opposite.x), min(origin.y, opposite.y)),
                SIMD2(max(origin.x, opposite.x), max(origin.y, opposite.y)))
    }
}

private extension ExternalTextureTransferFunction {
    var coefficients: [Float] { [a, b, gamma, k] }
    var valid: Bool { coefficients.allSatisfy(\.isFinite) && a > 0 && b >= 0 && gamma > 0 && k > 0 }
}
