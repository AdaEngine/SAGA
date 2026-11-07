public struct RenderViewport: Sendable, Equatable {
    public let x: Float
    public let y: Float
    public let width: Float
    public let height: Float
    public let minDepth: Float
    public let maxDepth: Float

    public init(x: Float, y: Float, width: Float, height: Float, minDepth: Float = 0, maxDepth: Float = 1) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.minDepth = minDepth
        self.maxDepth = maxDepth
    }
}

public struct RenderScissor: Sendable, Equatable {
    public let x: UInt32
    public let y: UInt32
    public let width: UInt32
    public let height: UInt32

    public init(x: UInt32, y: UInt32, width: UInt32, height: UInt32) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// Viewport and scissor for a single draw. Omitted values use the full attachment.
public struct RenderRasterState: Sendable, Equatable {
    public let viewport: RenderViewport?
    public let scissor: RenderScissor?

    public init(viewport: RenderViewport? = nil, scissor: RenderScissor? = nil) {
        self.viewport = viewport
        self.scissor = scissor
    }

    package var hasEmptyScissor: Bool {
        scissor.map { $0.width == 0 || $0.height == 0 } ?? false
    }

    package func validate(width: UInt32, height: UInt32, maximumViewportDimension: UInt32) throws {
        if let v = viewport {
            let dimension = Double(maximumViewportDimension)
            let range = 2 * dimension
            guard Double(v.x) >= -range, Double(v.y) >= -range,
                  (0...dimension).contains(Double(v.width)), (0...dimension).contains(Double(v.height)),
                  Double(v.x) + Double(v.width) <= range - 1, Double(v.y) + Double(v.height) <= range - 1,
                  v.minDepth >= 0, v.maxDepth <= 1, v.minDepth <= v.maxDepth else {
                throw GPUError.validation("Invalid render viewport")
            }
        }
        if let s = scissor {
            guard s.x <= width, s.width <= width - s.x, s.y <= height, s.height <= height - s.y else {
                throw GPUError.validation("Scissor rectangle exceeds the render attachment")
            }
        }
    }
}
