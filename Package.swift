// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "SAGA",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "SAGAMetalDiagnostic", targets: ["SAGAMetalDiagnostic"]),
        .executable(name: "SAGAMetalWindowDiagnostic", targets: ["SAGAMetalWindowDiagnostic"]),
        .executable(name: "SAGAWebDiagnostic", targets: ["SAGAWebDiagnostic"]),
        .library(name: "SAGA", targets: ["SAGA"]),
        .library(name: "SAGACompiler", targets: ["SAGACompiler"]),
        .library(name: "SAGAMetal", targets: ["SAGAMetal"]),
        .library(name: "SAGAWebGPU", targets: ["SAGAWebGPU"]),
    ],
    targets: [
        .executableTarget(name: "SAGAMetalDiagnostic", dependencies: ["SAGATypes", "SAGACore", "SAGAMetal"]),
        .executableTarget(name: "SAGAMetalWindowDiagnostic", dependencies: ["SAGATypes", "SAGACore", "SAGACompiler", "SAGAShaderIR", "SAGAMetal"]),
        .executableTarget(name: "SAGAWebDiagnostic", dependencies: ["SAGATypes", "SAGABackend", "SAGACore", "SAGAShaderIR", "SAGACompiler", "SAGAWebGPU"]),
        .target(name: "SAGATypes"),
        .target(name: "SAGABackend", dependencies: ["SAGATypes", "SAGAShaderIR"]),
        .target(name: "SAGACore", dependencies: ["SAGATypes", "SAGABackend", "SAGAShaderIR"]),
        .target(name: "SAGA", dependencies: ["SAGATypes", "SAGACore", "SAGABackend"]),
        .target(name: "SAGAShaderIR"),
        .target(name: "SAGAValidation", dependencies: ["SAGAShaderIR"]),
        .target(name: "SAGAFrontends", dependencies: ["SAGAShaderIR"]),
        .target(name: "SAGABackends", dependencies: ["SAGAShaderIR", "SAGAValidation"]),
        .target(name: "SAGACompiler", dependencies: ["SAGAShaderIR", "SAGAValidation", "SAGAFrontends", "SAGABackends"]),
        .target(name: "SAGAMetal", dependencies: ["SAGATypes", "SAGABackend", "SAGAShaderIR"]),
        .target(name: "SAGAWebGPU", dependencies: ["SAGATypes", "SAGABackend", "SAGAShaderIR"]),
        .testTarget(name: "SAGAFoundationTests", dependencies: ["SAGA", "SAGATypes", "SAGACore", "SAGABackend", "SAGAShaderIR", "SAGAValidation", "SAGAFrontends", "SAGABackends", "SAGACompiler", "SAGAMetal", "SAGAWebGPU"]),
    ],
    swiftLanguageModes: [.v6]
)
