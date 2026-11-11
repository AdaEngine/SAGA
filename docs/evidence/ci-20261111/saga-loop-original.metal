#include <metal_stdlib>
using namespace metal;
kernel void saga_main(device uint* saga_buffer_0 [[buffer(0)]]) {
    uint saga_local_0 = 0u;
    uint saga_local_1 = 0u;
    uint saga_local_2 = 0u;
    uint saga_local_3 = 0u;
    bool saga_local_4 = false;
    saga_local_0 = 0u;
    saga_local_1 = 0u;
    uint saga_loop_value_5 = 0u;
    bool saga_first_26 = true;
    while (true) {
        if (!saga_first_26) {
            uint saga_load_15 = saga_local_2;
            saga_local_0 = (saga_loop_value_5 + 1u);
            saga_local_1 = saga_load_15;
        }
        saga_first_26 = false;
        saga_local_4 = false;
        uint saga_load_5 = saga_local_0;
        saga_loop_value_5 = saga_load_5;
        uint saga_load_6 = saga_local_1;
        if (saga_load_5 < 6u) {
        } else {
            saga_local_3 = saga_load_6;
            break;
        }
        switch (saga_load_5) {
            case 1u:
            {
                saga_local_2 = saga_load_6;
                continue;
            }
            case 3u:
            {
                saga_local_3 = saga_load_6;
                saga_local_4 = true;
                break;
            }
            default:
            {
            }
            break;
        }
        bool saga_load_11 = saga_local_4;
        if (saga_load_11) {
            break;
        }
        saga_local_2 = (saga_load_6 + (saga_load_5 + 10u));
        continue;
    }
    uint saga_load_19 = saga_local_0;
    uint saga_load_20 = saga_local_1;
    uint saga_load_21 = saga_local_3;
    saga_buffer_0[0] = saga_load_21;
    return;
}
