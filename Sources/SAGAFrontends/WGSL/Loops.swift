import SAGAShaderIR

extension WGSLParser {
    func forStatement(at t: WGSLToken) throws {
        try enter(); defer { nesting -= 1 }
        body.scopes.append([:]); body.constScopes.append([:]); defer { body.scopes.removeLast(); body.constScopes.removeLast() }
        try expect("(")
        if try eat(";") {} else if try eat("var") { try local(mutable:true,at:t) }
        else if try eat("let") { try local(mutable:false,at:t) }
        else { try simpleStatement(end:";") }
        let saved = body.current; body.current = []
        if !(try eat(";")) {
            let empty = try saveBlock([],at:t)
            let cond = try loaded(expression(),at:t); try expect(";")
            guard types[cond.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"for condition must be bool",t) }
            let savedCond = body.current; body.current = []
            try statement(.break,at:t); let exit = try saveBlock(body.current,at:t); body.current = savedCond
            try statement(.ifCondition(condition:cond.handle,accept:empty,reject:exit),at:t)
        }
        let condition = body.current; body.current = []
        if !(try eat(")")) { try simpleStatement(end:")") }
        let continuing = try saveBlock(body.current,at:t)
        body.current = condition
        body.loopDepth += 1; defer { body.loopDepth -= 1 }
        let inner = try block(); try statement(.block(inner),at:t)
        let whole = try saveBlock(body.current,at:t); body.current = saved
        try statement(.loop(body:whole,continuing:continuing,breakIf:nil),at:t)
    }
    func loopStatement(at t: WGSLToken) throws {
        try enter(); defer { nesting -= 1 }
        try expect("{")
        body.loopDepth += 1; body.scopes.append([:]); body.constScopes.append([:])
        let declarationScope = body.scopes.count - 1
        body.loopScopes.append(.init(depth: body.loopDepth, declarations: declarationScope))
        defer {
            body.loopScopes.removeLast()
            body.loopDepth -= 1; body.scopes.removeLast(); body.constScopes.removeLast()
        }
        let saved = body.current; body.current = []
        while !isNext("}") && !isNext("continuing") {
            guard token.kind != .end else { throw wgslError(.unexpectedToken, "Unterminated loop", token) }
            try parseStatement()
        }
        let inner = try saveBlock(body.current,at:t); body.current = []
        var breakIf: WExpr?
        if try eat("continuing") {
            try expect("{")
            let wasContinuing = body.continuing
            let previousLoopDepth = body.continuingLoopDepth
            let previousSwitchDepth = body.continuingSwitchDepth
            body.continuing = true
            body.continuingLoopDepth = body.loopDepth
            body.continuingSwitchDepth = body.switchDepth
            body.continuingVisibility[declarationScope] = body.loopScopes.last?.visibleAtFirstContinue
            body.scopes.append([:]); body.constScopes.append([:])
            defer {
                body.scopes.removeLast(); body.constScopes.removeLast()
                body.continuingVisibility.removeValue(forKey: declarationScope)
                body.continuing = wasContinuing
                body.continuingLoopDepth = previousLoopDepth
                body.continuingSwitchDepth = previousSwitchDepth
            }
            while !isNext("}") {
                guard token.kind != .end else { throw wgslError(.unexpectedToken,"Unterminated continuing",token) }
                if try eat("break") {
                    try expect("if"); let v = try loaded(expression(),at:t)
                    guard types[v.type.index].kind == .scalar(.bool) else { throw wgslError(.typeMismatch,"break if needs bool",t) }
                    breakIf = v.handle; try expect(";"); guard isNext("}") else { throw wgslError(.invalidControlFlow,"break if must be final",token) }; break
                }
                if ["return","continue"].contains(token.text) { throw wgslError(.invalidControlFlow,"Invalid continuing exit",token) }
                try parseStatement()
            }
            try expect("}")
        }
        let continuing = try saveBlock(body.current,at:t); try expect("}"); body.current = saved
        try statement(.loop(body:inner,continuing:continuing,breakIf:breakIf),at:t)
    }

    func noteContinue() throws {
        guard let index = body.loopScopes.indices.last,
              body.loopScopes[index].depth == body.loopDepth,
              body.loopScopes[index].visibleAtFirstContinue == nil
        else { return }
        let scope = body.loopScopes[index].declarations
        try budget.consume(.work, amount: body.scopes[scope].count)
        body.loopScopes[index].visibleAtFirstContinue = Set(body.scopes[scope].keys)
    }

    func requireVisibleToContinuing(_ key: String, scope: Int, at token: WGSLToken) throws {
        if let visible = body.continuingVisibility[scope], !visible.contains(key) {
            throw wgslError(.invalidControlFlow, "Continue bypasses declaration used in continuing", token)
        }
    }

    func switchStatement(at t: WGSLToken) throws {
        try enter(); defer { nesting -= 1 }
        let selector = try loaded(expression(), at: t)
        var kind: WGSLIntegerConstant.Kind?
        switch types[selector.type.index].kind {
        case .scalar(.u32): kind = .uint
        case .scalar(.i32): kind = .sint
        case .scalar(.abstractInt): kind = nil
        default: throw wgslError(.typeMismatch,"switch selector must be integer",t)
        }
        try expect("{"); body.switchDepth += 1; defer { body.switchDepth -= 1 }
        var clauses: [(value: WGSLIntegerConstant?, token: WGSLToken, body: IRHandle<IRBlock>)] = []
        var hasDefault = false
        while !isNext("}") {
            let grouped = try eat("case")
            var labels: [(WGSLIntegerConstant?, WGSLToken)] = []
            repeat {
                let start = token
                if try eat("default") {
                    guard !hasDefault else { throw wgslError(.duplicateName,"Duplicate default",start) }
                    hasDefault = true
                    labels.append((nil, start))
                } else {
                    guard grouped else { throw wgslError(.unexpectedToken, "Expected case or default", start) }
                    let label = try switchSelectorConstant(at: start)
                    if label.kind != .abstract {
                        guard !label.kind.isWide, kind == nil || kind == label.kind else {
                            throw wgslError(.typeMismatch, "Switch selectors must have the same integer type", start)
                        }
                        kind = label.kind
                    }
                    labels.append((label, start))
                }
                try node()
            } while try grouped && eat(",") && !isNext(":") && !isNext("{")
            _ = try eat(":")
            let b = try block()
            clauses += labels.map { ($0.0, $0.1, b) }
        }
        try expect("}")
        guard hasDefault else { throw wgslError(.invalidControlFlow,"switch requires default",t) }
        let unsigned = kind == .uint
        let v = try coerce(selector, to: intern(.scalar(unsigned ? .u32 : .i32)), at: t)
        var seen: Set<Int64> = []
        let cases = try clauses.map { clause -> IRSwitchCase in
            try budget.consume(.work)
            guard let label = clause.value else { return .init(value: .default, body: clause.body) }
            guard seen.insert(label.value).inserted else { throw wgslError(.duplicateName,"Duplicate switch selector",clause.token) }
            if unsigned, let value = UInt32(exactly: label.value) { return .init(value: .u32(value), body: clause.body) }
            if !unsigned, let value = Int32(exactly: label.value) { return .init(value: .i32(value), body: clause.body) }
            throw wgslError(.invalidType, "Switch selector is outside its integer type", clause.token)
        }
        try statement(.switchValue(selector:v.handle,cases:cases),at:t)
    }

    private func switchSelectorConstant(at start: WGSLToken) throws -> WGSLIntegerConstant {
        var expression: [WGSLToken] = []
        var delimiters: [String] = []
        while !(delimiters.isEmpty && [",", ":", "{"].contains(token.text)) {
            guard token.kind != .end, !isNext("}") else {
                throw wgslError(.unexpectedToken, "Unterminated switch selector", start)
            }
            if isNext("(") || isNext("[") { delimiters.append(isNext("(") ? ")" : "]") }
            else if isNext(")") || isNext("]") {
                guard delimiters.popLast() == token.text else {
                    throw wgslError(.unexpectedToken, "Unmatched switch selector delimiter", token)
                }
            }
            try node()
            expression.append(try take())
        }
        return try integerConstant(expression, at: start)
    }
}
