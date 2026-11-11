//! Mirror of Sources/SAGABenchmark/Main.swift. Keep sizes, shaders and verification identical.
use std::time::Instant;

const ALU_ELEMENTS: usize = 1 << 22;
const ALU_ROUNDS: u32 = 256;
const STREAM_ELEMENTS: usize = 1 << 24;
const ROUNDTRIP_WORDS: usize = 4 << 20;
const ALU_DISPATCHES: usize = 8;
const STREAM_DISPATCHES: usize = 8;
const BATCH_DISPATCHES: usize = 1000;
const SUBMIT_COUNT: usize = 100;
const COMPILE_VARIANTS: usize = 20;
const WARMUP: usize = 3;

const ALU: &str = include_str!("../../shaders/alu.wgsl");
const STREAM: &str = include_str!("../../shaders/stream.wgsl");
const EMPTY: &str = include_str!("../../shaders/empty.wgsl");

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

struct Row {
    name: &'static str,
    samples: Vec<f64>,
    detail: String,
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure(samples: usize, mut body: impl FnMut()) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 0..WARMUP + samples {
        let start = Instant::now();
        body();
        let ms = start.elapsed().as_secs_f64() * 1e3;
        if i >= WARMUP {
            out.push(ms);
        }
    }
    out
}

fn row_groups(elements: usize) -> (u32, u32) {
    (1024, elements.div_ceil(262_144) as u32)
}

impl Gpu {
    fn pipeline(&self, source: &str) -> wgpu::ComputePipeline {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("run"),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn buffer(&self, words: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (words * 4) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    fn bind_group(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn wait(&self) {
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// Records `count` identical dispatches in one pass, submits, and waits for completion.
    fn run(&self, pipeline: &wgpu::ComputePipeline, group: Option<&wgpu::BindGroup>, count: usize, size: (u32, u32)) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            if let Some(group) = group {
                pass.set_bind_group(0, group, &[]);
            }
            for _ in 0..count {
                pass.dispatch_workgroups(size.0, size.1, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.wait();
    }

    fn read_words(&self, source: &wgpu::Buffer, readback: &wgpu::Buffer, words: usize) -> Vec<u32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, readback, 0, (words * 4) as u64);
        self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..(words * 4) as u64);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.wait();
        let out = bytes_to_words(&slice.get_mapped_range().unwrap());
        readback.unmap();
        out
    }
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn main() {
    let mut only: Vec<String> = Vec::new();
    let mut samples = 15usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--only" => only.push(args.next().expect("--only needs a case")),
            "--samples" => samples = args.next().and_then(|v| v.parse().ok()).expect("--samples needs a number"),
            other => panic!("unknown argument {other}"),
        }
    }
    let wants = |name: &str| only.is_empty() || only.iter().any(|n| n == name);

    let backends = match std::env::var("WGPU_BACKEND").as_deref() {
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("vulkan") | Err(_) => wgpu::Backends::VULKAN,
        Ok(other) => panic!("unsupported WGPU_BACKEND {other}"),
    };
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = backends;
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no adapter");
    let info = adapter.get_info();
    println!("wgpu ({:?}) on {}", info.backend, info.name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
        default_queue: wgpu::QueueDescriptor { label: None },
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .expect("no device");
    let gpu = Gpu { device, queue };
    let mut rows: Vec<Row> = Vec::new();

    if wants("compile") {
        let mut variant = 0;
        let s = measure(samples, || {
            for _ in 0..COMPILE_VARIANTS {
                variant += 1;
                let _ = gpu.pipeline(&ALU.replace("__ITERS__", &format!("{}u", 1000 + variant)));
            }
        });
        rows.push(Row { name: "compile", samples: s.iter().map(|v| v / COMPILE_VARIANTS as f64).collect(), detail: "ms per pipeline".into() });
    }

    if wants("dispatch-batch") || wants("dispatch-submit") {
        let empty = gpu.pipeline(EMPTY);
        if wants("dispatch-batch") {
            let s = measure(samples, || gpu.run(&empty, None, BATCH_DISPATCHES, (1, 1)));
            rows.push(Row { name: "dispatch-batch", samples: s, detail: format!("{BATCH_DISPATCHES} dispatches/submit") });
        }
        if wants("dispatch-submit") {
            let s = measure(samples, || {
                for _ in 0..SUBMIT_COUNT {
                    gpu.run(&empty, None, 1, (1, 1));
                }
            });
            rows.push(Row { name: "dispatch-submit", samples: s.iter().map(|v| v / SUBMIT_COUNT as f64).collect(), detail: "ms per submit+wait".into() });
        }
    }

    if wants("alu") {
        let pipeline = gpu.pipeline(&ALU.replace("__ITERS__", &format!("{ALU_ROUNDS}u")));
        let out = gpu.buffer(ALU_ELEMENTS, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = gpu.buffer(1024, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let group = gpu.bind_group(&pipeline, &[&out]);
        let size = row_groups(ALU_ELEMENTS);
        let s = measure(samples, || gpu.run(&pipeline, Some(&group), ALU_DISPATCHES, size));
        gpu.run(&pipeline, Some(&group), 1, size);
        let r = gpu.read_words(&out, &readback, 1024);
        for (i, got) in r.iter().enumerate() {
            let mut x = i as u32;
            for _ in 0..ALU_ROUNDS {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                x ^= x >> 13;
            }
            assert_eq!(*got, x, "alu mismatch at {i}");
        }
        let grounds = (ALU_ELEMENTS * ALU_ROUNDS as usize * ALU_DISPATCHES) as f64 / 1e9;
        rows.push(Row { name: "alu", detail: format!("{:.3} G rounds/s", grounds / (median(&s) / 1e3)), samples: s });
    }

    if wants("stream") {
        let pipeline = gpu.pipeline(STREAM);
        let usage_in = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let a = gpu.buffer(STREAM_ELEMENTS, usage_in);
        let b = gpu.buffer(STREAM_ELEMENTS, usage_in);
        let out = gpu.buffer(STREAM_ELEMENTS, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback = gpu.buffer(1024, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let chunk = 1 << 20;
        for start in (0..STREAM_ELEMENTS).step_by(chunk) {
            let n = chunk.min(STREAM_ELEMENTS - start);
            let wa: Vec<u32> = (0..n).map(|i| (start + i) as u32).collect();
            let wb: Vec<u32> = (0..n).map(|i| ((start + i) as u32).wrapping_mul(7)).collect();
            gpu.queue.write_buffer(&a, (start * 4) as u64, &words_to_bytes(&wa));
            gpu.queue.write_buffer(&b, (start * 4) as u64, &words_to_bytes(&wb));
        }
        gpu.queue.submit([]);
        gpu.wait();
        let group = gpu.bind_group(&pipeline, &[&a, &b, &out]);
        let size = row_groups(STREAM_ELEMENTS);
        let s = measure(samples, || gpu.run(&pipeline, Some(&group), STREAM_DISPATCHES, size));
        gpu.run(&pipeline, Some(&group), 1, size);
        let r = gpu.read_words(&out, &readback, 1024);
        for (i, got) in r.iter().enumerate() {
            assert_eq!(*got, (i as u32).wrapping_mul(10), "stream mismatch at {i}");
        }
        let gb = (STREAM_ELEMENTS * 12 * STREAM_DISPATCHES) as f64 / 1e9;
        rows.push(Row { name: "stream", detail: format!("{:.3} GB/s", gb / (median(&s) / 1e3)), samples: s });
    }

    if wants("roundtrip") {
        let upload = gpu.buffer(ROUNDTRIP_WORDS, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let readback = gpu.buffer(ROUNDTRIP_WORDS, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
        let data: Vec<u32> = (0..ROUNDTRIP_WORDS as u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let bytes = words_to_bytes(&data);
        let mut last = Vec::new();
        let s = measure(samples, || {
            gpu.queue.write_buffer(&upload, 0, &bytes);
            last = gpu.read_words(&upload, &readback, ROUNDTRIP_WORDS);
        });
        assert!(last == data, "roundtrip mismatch");
        let gbps = (ROUNDTRIP_WORDS * 4 * 2) as f64 / 1e9 / (median(&s) / 1e3);
        rows.push(Row { name: "roundtrip", detail: format!("{gbps:.3} GB/s up+down"), samples: s });
    }

    println!("{:>16} {:>12} {:>12}  detail", "case", "median ms", "min ms");
    for r in &rows {
        let min = r.samples.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("{:>16} {:>12.3} {:>12.3}  {}", r.name, median(&r.samples), min, r.detail);
    }
}
