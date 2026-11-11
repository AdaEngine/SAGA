#include <metal_stdlib>
using namespace metal;
kernel void saga_main(device uint* saga_buffer_0 [[buffer(0)]]) {
    uint saga_local_0 = 0u;
    uint saga_local_1 = 0u;
    uint saga_local_2 = 0u;
    saga_local_0 = 0u;
    uint saga_loop_value_4 = 0u;
    uint saga_loop_value_5 = 0u;
    bool saga_first_20 = true;
    while (true) {
        if (!saga_first_20) {
            saga_buffer_0[(0u) / 4u] = (saga_loop_value_5 + 1u);
            saga_local_0 = (saga_loop_value_4 + 1u);
        }
        saga_first_20 = false;
        uint saga_load_4 = saga_local_0;
        saga_loop_value_4 = saga_load_4;
        uint saga_load_5 = saga_buffer_0[(0u) / 4u];
        saga_loop_value_5 = saga_load_5;
        uint saga_select_10 = ((1u < 32u) ? 1u : 0u);
        uint saga_select_13 = ((1u < 32u) ? (saga_load_5 >> (saga_select_10 & 31u)) : 0u);
        saga_local_1 = saga_select_13;
        saga_local_2 = (saga_load_5 & 1u);
        if (saga_load_4 < 2u) {
            continue;
        } else {
            break;
        }
    }
    uint saga_load_22 = saga_local_0;
    saga_buffer_0[(0u) / 4u] = 0u;
    uint saga_load_25 = saga_local_1;
    uint saga_load_28 = saga_local_2;
    if (saga_load_25 < 2u && saga_load_28 < 2u) { saga_buffer_0[(0u + 4u + (saga_load_25) * 8u + (saga_load_28) * 4u) / 4u] = 99u; }
    return;
}
