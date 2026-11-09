import SAGAShaderIR

extension WGSLParser {
  func parseModuleDirectives() throws {
    var seen: [IRDiagnosticRule: IRDiagnosticSeverity] = [:]
    while isNext("diagnostic") || isNext("enable") || isNext("requires") {
      if try eat("requires") {
        repeat {
          let feature = try take()
          switch feature.text {
          case "unrestricted_pointer_parameters", "pointer_composite_access": break
          case "swizzle_assignment": allowsSwizzleAssignment = true
          case "uniform_buffer_standard_layout": uniformBufferStandardLayout = true
          case "subgroup_id": subgroupIDEnabled = true
          case "subgroup_uniformity": requiresSubgroupUniformity = true
          case "linear_indexing": linearIndexingEnabled = true
          case "immediate_address_space":
            guard options.capabilities.contains(.immediates) else {
              throw wgslError(.unsupportedCapability, "immediate_address_space requires immediates", feature)
            }
          case "texture_formats_tier1", "readonly_and_readwrite_storage_textures", "texture_and_sampler_let": break
          default: throw unsupported("WGSL language feature '\(feature.text)'", at: feature)
          }
          if !(try eat(",")) { break }
        } while !isNext(";")
        try expect(";")
        continue
      }
      if try eat("enable") {
        repeat {
          let extensionName = try take()
          switch extensionName.text {
          case "f16":
            guard options.capabilities.contains(.shaderFloat16) else {
              throw wgslError(.unsupportedCapability, "f16 requires shaderFloat16", extensionName)
            }
            float16Enabled = true
          case "wgpu_ray_query":
            guard options.capabilities.contains(.rayQuery) else {
              throw wgslError(.unsupportedCapability, "wgpu_ray_query requires rayQuery", extensionName)
            }
            rayQueryEnabled = true
            try installRayQueryTypes(at: extensionName)
          case "wgpu_ray_query_vertex_return":
            guard options.capabilities.contains(.rayHitVertexPosition) else {
              throw wgslError(.unsupportedCapability, "wgpu_ray_query_vertex_return requires rayHitVertexPosition", extensionName)
            }
            rayQueryVertexReturnEnabled = true
          case "wgpu_int16":
            guard options.capabilities.contains(.shaderInt16) else {
              throw wgslError(.unsupportedCapability, "wgpu_int16 requires shaderInt16", extensionName)
            }
            int16Enabled = true
          case "wgpu_cooperative_matrix":
            guard options.capabilities.contains(.cooperativeMatrix) else {
              throw wgslError(.unsupportedCapability, "wgpu_cooperative_matrix requires cooperativeMatrix", extensionName)
            }
            cooperativeMatrixEnabled = true
          case "wgpu_debug_printf":
            guard options.capabilities.contains(.debugPrintf) else {
              throw wgslError(.unsupportedCapability, "wgpu_debug_printf requires debugPrintf", extensionName)
            }
            debugPrintfEnabled = true
          case "clip_distances":
            guard options.capabilities.contains(.clipDistances) else {
              throw wgslError(.unsupportedCapability, "clip_distances requires clipDistances", extensionName)
            }
            clipDistancesEnabled = true
          case "primitive_index":
            guard options.capabilities.contains(.primitiveIndex) else {
              throw wgslError(.unsupportedCapability, "primitive_index requires primitiveIndex", extensionName)
            }
            primitiveIndexEnabled = true
          case "draw_index":
            guard options.capabilities.contains(.drawIndex) else {
              throw wgslError(.unsupportedCapability, "draw_index requires drawIndex", extensionName)
            }
            drawIndexEnabled = true
          case "wgpu_per_vertex":
            guard options.capabilities.contains(.perVertex) else {
              throw wgslError(.unsupportedCapability, "wgpu_per_vertex requires perVertex", extensionName)
            }
            perVertexEnabled = true
          case "subgroups":
            guard options.capabilities.contains(.subgroup) else {
              throw wgslError(.unsupportedCapability, "subgroups require the subgroup capability", extensionName)
            }
            subgroupsEnabled = true
          case "subgroup_size_control":
            guard options.capabilities.contains(.subgroup) else {
              throw wgslError(.unsupportedCapability, "subgroup_size_control requires the subgroup capability", extensionName)
            }
            subgroupSizeControlEnabled = true
            subgroupsEnabled = true
          case "dual_source_blending":
            guard options.capabilities.contains(.dualSourceBlending) else {
              throw wgslError(.unsupportedCapability, "dual_source_blending requires the dual-source blending capability", extensionName)
            }
            dualSourceBlendingEnabled = true
          case "wgpu_binding_array":
            guard !options.capabilities.intersection(.bindingArrays).isEmpty else {
              throw wgslError(.unsupportedCapability, "wgpu_binding_array requires a binding array capability", extensionName)
            }
            bindingArraysEnabled = true
          default:
            throw unsupported("WGSL extension", at: extensionName)
          }
          if !(try eat(",")) { break }
        } while !isNext(";")
        try expect(";")
        continue
      }
      let name = try take()
      try expect("(")
      let arguments = try attributeArguments(at: name)
      try expect(";")
      let filter = try diagnosticFilter(arguments, parent: moduleDiagnosticFilter, at: name)
      if let previous = seen[filter.rule] {
        guard previous == filter.severity else {
          throw wgslError(.duplicateName, "Conflicting module diagnostic filters", name)
        }
        continue
      }
      seen[filter.rule] = filter.severity
      moduleDiagnosticFilter = try appendDiagnosticFilter(filter)
    }
  }

  func functionDiagnosticFilters(_ attributes: [String: Attribute]) throws -> IRHandle<IRDiagnosticFilter>? {
    var seen = Set<IRDiagnosticRule>()
    var leaf: IRHandle<IRDiagnosticFilter>?
    for attribute in attributes.values.sorted(by: { $0.name.start < $1.name.start }) where attribute.name.text == "diagnostic" {
      let filter = try diagnosticFilter(attribute.expressions, parent: leaf ?? moduleDiagnosticFilter, at: attribute.name)
      guard seen.insert(filter.rule).inserted else {
        throw wgslError(.duplicateName, "Duplicate function diagnostic rule", attribute.name)
      }
      leaf = try appendDiagnosticFilter(filter)
    }
    return leaf
  }

  private func diagnosticFilter(_ arguments: [[WGSLToken]], parent: IRHandle<IRDiagnosticFilter>?, at name: WGSLToken)
    throws -> IRDiagnosticFilter
  {
    guard arguments.count == 2, arguments[0].count == 1, let severityToken = arguments[0].first,
          let severity = IRDiagnosticSeverity(rawValue: severityToken.text) else {
      throw wgslError(.unexpectedToken, "Expected diagnostic severity and rule", arguments.first?.first ?? name)
    }
    let ruleTokens = arguments[1]
    let rule: IRDiagnosticRule
    if ruleTokens.count == 1, ruleTokens[0].kind == .word {
      let text = ruleTokens[0].text
      switch text {
      case "derivative_uniformity": rule = .derivativeUniformity
      case "subgroup_uniformity": rule = .subgroupUniformity
      default: rule = .unknown(text)
      }
    } else if ruleTokens.count == 3, ruleTokens[0].kind == .word, ruleTokens[1].text == ".", ruleTokens[2].kind == .word {
      rule = .user(namespace: ruleTokens[0].text, name: ruleTokens[2].text)
    } else {
      throw wgslError(.unexpectedToken, "Invalid diagnostic rule name", ruleTokens.first ?? name)
    }
    return .init(parent: parent, rule: rule, severity: severity, location: ruleTokens[0].location)
  }

  private func appendDiagnosticFilter(_ filter: IRDiagnosticFilter) throws -> IRHandle<IRDiagnosticFilter> {
    try node()
    let handle = try IRHandle<IRDiagnosticFilter>(index: diagnosticFilters.count)
    diagnosticFilters.append(filter)
    return handle
  }
}
