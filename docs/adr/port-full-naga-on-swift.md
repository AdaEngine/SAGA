# Перенести полный набор Naga на Swift

- Status: accepted
- Date: 2026-11-06
- Authority: пользователь в основном чате: «весь набор Naga ожидаю, ты ответственный за задачи…».
- Reference: read-only wgpu `a55352911bb26a1b33bc8d5fc03bdebb0a967f5a`.

## Context

[Независимая реализация](implement-independent-webgpu.md) и [Apple/Metal + browser WebGPU](use-metal-and-browser-webgpu.md) сохраняются. WGSL→MSL-only отвергнут как окончательный объём. Это дополнение к принятому baseline SAGA-2; исторический отчёт с открытым выбором не переписывается как будто решение уже было принято раньше.

## Decision

На Swift переносится весь функциональный набор Naga закреплённой ревизии:

- Frontends: WGSL, GLSL, SPIR-V.
- Backends: WGSL, GLSL, SPIR-V, MSL, HLSL; диагностический DOT.
- Shared typed IR/arena/handles и naga-types, spans/diagnostics, типизация/semantic validation, capability/limit checks, reflection/layouts, constant evaluation, override specialization, compaction, bounds policies/robustness, backend lowering и coordinate-space handling.
- Самостоятельные library API и CLI: validation/translation, bulk validation, entry points/options/defines/overrides, target versions/profiles. IR serialization/deserialization и CLI binary/debug interchange включены. Совместимость с upstream binary fixtures требует проверяемого mapping; нельзя молча заменить её Swift JSON и заявить полную parity.
- Позитивные/негативные fixtures и snapshots с provenance, semantic/differential и round-trip проверки допустимых комбинаций. Byte-identical shader text не обязателен, семантика и capability matrix обязательны.

Полнота означает parity с поддерживаемыми upstream возможностями, не произвольный all-to-all перевод. HLSL/MSL input не выдумываются: reference содержит их output, но не frontends. Форматы/features сверены с naga/Cargo.toml, naga/src/front и naga/src/back, CLI — с naga-cli/src/bin/naga.rs.

Без Dawn/wgpu/Rust Naga implementation dependencies и FFI. Rust checkout read-only. Browser-owned engine не является зависимостью проекта. Apple Metal и web остаются runtime paths; SPIR-V/HLSL/GLSL codegen не добавляет автоматически Vulkan/D3D/OpenGL runtime backends. Native wgpu extensions сверх WebGPU не включаются автоматически выбором полного Naga.

## Delivery and verification

SAGA-3: безопасные API/Core/Backend/ShaderIR/Validation/Frontends/Backends interfaces и build/tests; далее существующие SAGA-4..11. SAGA-5 владеет всеми frontends/validation, SAGA-6 — shared compiler/transforms и backend/tooling; SAGA-11 проверяет полную parity. Агент декомпозирует этапы на непересекающиеся child tasks при необходимости, parent нельзя закрывать по WGSL/MSL demo.

[Scope manifest](../naga-scope.json) фиксирует 16 направлений с not-started/not-run. Это не atomic feature/fixture denominator. SAGA-3 расширяет его до измеримого compiler feature inventory, не переписывая frozen WebGPU baseline. Тесты scope: python3 -B -m unittest discover -s docs/tools -p test_naga_scope.py -v.

Swift build/unit tests, emitted shader compilation внешними toolchains, GPU, browser и WebGPU CTS — разные evidence. Недоступность DXC/glslang/SPIR-V tools или GPU даёт явный blocker, не pass. Native Metal conformance и browser-backed поведение не смешиваются. Docs scope tests не доказывают реализацию компилятора.

## Consequences

Полный compiler family существенно больше WGSL→MSL: дополнительные fixtures/tooling/serialization compatibility. Metal-first slice допустим как промежуточный результат, не сокращение scope. Агент отвечает за декомпозицию, приоритеты и статусы; done только по проверенному DoD. Коммиты, публикация и изменения Rust reference по-прежнему не разрешены.
