import SAGAShaderIR
import SAGATypes

extension DeviceCapabilities {
    /// Validates the mesh-to-fragment interface and its interpolation mode.
    package func validateMeshInterstageInterfaces(mesh: MeshStageArtifact, fragment: RenderStageArtifact) throws {
        guard mesh.writesPrimitiveIndex == fragment.readsPrimitiveIndex else {
            throw GPUError.validation("Mesh and fragment primitive_index interfaces must match")
        }
        let outputs = mesh.vertexVaryings + mesh.primitiveVaryings
        let limit = UInt64(maxInterStageShaderVariables)
        guard UInt64(outputs.count) <= limit,
              outputs.allSatisfy({ UInt64($0.location) < limit }),
              UInt64(fragment.varyings.count) + UInt64(fragment.fragmentInputBuiltinSlots) <= limit else {
            throw GPUError.validation("Mesh inter-stage interface exceeds the device limit")
        }
        guard outputs.allSatisfy({ [IRScalar.f32, .i32, .u32].contains($0.scalar) && (1...4).contains($0.componentCount) }),
              Set(outputs.map(\.location)).count == outputs.count else {
            throw GPUError.unsupported("Mesh varying type is outside the supported profile")
        }
        guard fragment.varyings.allSatisfy({ input in
            let matches: (MeshStageArtifact.Varying) -> Bool = { output in
                output.location == input.location && output.scalar == input.scalar &&
                output.componentCount == input.componentCount
            }
            return (!input.perPrimitive && mesh.vertexVaryings.contains { output in
                let mode = output.interpolation ?? (output.scalar == .f32 ? .perspective : .flat)
                let sampling = output.sampling ?? (mode == .flat ? .first : .center)
                return matches(output) && mode == input.interpolation && sampling == input.sampling &&
                    ((mode == .flat && (sampling == .first || sampling == .either)) ||
                     (output.scalar == .f32 && (mode == .perspective || mode == .linear) &&
                      (sampling == .center || sampling == .centroid || sampling == .sample)))
            }) || mesh.primitiveVaryings.contains { output in
                matches(output) && input.interpolation == .flat &&
                    (input.sampling == .first || input.sampling == .either) &&
                    (output.interpolation == nil || output.interpolation == .flat) &&
                    (output.sampling == nil || output.sampling == input.sampling)
            }
        }) else {
            throw GPUError.validation("Fragment inputs do not match mesh outputs")
        }
    }
}
