// Reactor-only Preview 1 services. The browser exposes no filesystem or environment.
export function browserWASI(memory) {
    const view = () => new DataView(memory().buffer);
    const bytes = (pointer, length) => new Uint8Array(memory().buffer, pointer >>> 0, length >>> 0);
    const badFD = () => 8;
    const unsupported = () => 52;
    const empty = (count, size) => {
        view().setUint32(count, 0, true);
        view().setUint32(size, 0, true);
        return 0;
    };
    return {
        args_get: () => 0,
        args_sizes_get: empty,
        environ_get: () => 0,
        environ_sizes_get: empty,
        clock_res_get(clock, pointer) {
            if (clock !== 0 && clock !== 1) return 28;
            view().setBigUint64(pointer, 1_000_000n, true);
            return 0;
        },
        clock_time_get(clock, _precision, pointer) {
            if (clock !== 0 && clock !== 1) return 28;
            const milliseconds = clock === 0 ? Date.now() : performance.now();
            view().setBigUint64(pointer, BigInt(Math.floor(milliseconds * 1000)) * 1000n, true);
            return 0;
        },
        random_get(pointer, length) {
            const target = bytes(pointer, length);
            for (let i = 0; i < target.length; i += 65536) crypto.getRandomValues(target.subarray(i, i + 65536));
            return 0;
        },
        fd_write(fd, iov, count, written) {
            if (fd !== 1 && fd !== 2) return badFD();
            const decoder = new TextDecoder();
            const data = view();
            let size = 0;
            let output = "";
            for (let i = 0; i < count; i++, iov += 8) {
                const length = data.getUint32(iov + 4, true);
                output += decoder.decode(bytes(data.getUint32(iov, true), length), { stream: true });
                size += length;
            }
            output += decoder.decode();
            if (output) (fd === 1 ? console.log : console.error)(output);
            data.setUint32(written, size, true);
            return 0;
        },
        fd_close: badFD,
        fd_fdstat_get: badFD,
        fd_prestat_get: badFD,
        fd_prestat_dir_name: badFD,
        fd_read: badFD,
        fd_seek: badFD,
        path_open: unsupported,
        poll_oneoff: unsupported,
        proc_exit(code) { throw new Error(`Swift/Wasm exited with status ${code}`); },
    };
}
