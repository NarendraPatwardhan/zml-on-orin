# zml-on-arm64

Run [ZML](https://github.com/zml/zml) on an NVIDIA Jetson Orin Nano 8 GB.

The device on the bench is `orin-1` (L4T R39.2.1, Ubuntu 24.04, driver 595.78, CUDA 13.2, compute capability 8.7, nvgpu). Compilation happens on BuildBuddy. The Orin only runs the finished binary.

The product direction is a voice assistant. The first program is a matrix multiply on the CUDA platform: a 2×3 matrix times a 3×2 matrix, checked against `[[58, 64], [139, 154]]`.

## Layout

Rust owns the process. Zig owns ZML. They meet at a C ABI (`engine_init`, `engine_matmul`). One thread calls the engine. The prebuilt `libzml_cuda` plugin for `linux-arm64` carries the XLA compiler and the CUDA 13.3 SBSA userspace; the host supplies the kernel driver. ZML does not publish a Linux aarch64 CPU plugin, so the build enables CUDA and disables CPU.

XLA's default GPU allocator reserves 90% of device memory. On this board the GPU memory is the system RAM, so the engine asks for a non-preallocating allocator capped at 45%.

## Build

Push `master`. `buildbuddy.yaml` starts a BuildBuddy runner that executes:

```bash
bb build --config=buildbuddy --config=release //:zml_on_arm64
```

The workstation does not run Bazel. The runner is the coordinator: Zig compiles there, and C++ and XLA compile on BuildBuddy executors. `.bazelrc` sets the target platform to `@zml//platforms:linux_arm64`.

## Run

```bash
tools/run-on-orin.sh
```

This copies `bazel-bin/zml_on_arm64` and its runfiles to the Orin and runs `./zml_on_arm64 matmul`. A password file can be passed as `ORIN_PASSWORD_FILE` (default `/mnt/workspace/orin-1.key`). The host is `orin-1@orin-1.local`.

Success prints `platform: cuda`, the four result numbers, and `matmul ok`.
