# llama.cpp-m

`llama.cpp-m` packages a pinned, audited llama.cpp checkpoint as a C++23
module for mcpp. The public module is:

```cpp
import llamacpp;
```

The package version IS llama.cpp's build number: `b10069`. CPU is the default
backend. Metal is an additive feature for macOS ARM64.

## Add The Package

After `b10069` is published in mcpp-index:

```bash
mcpp add ggml-org:llamacpp@b10069
```

The equivalent manifest entry is:

```toml
[dependencies.ggml-org]
llamacpp = "b10069"
```

For Metal on macOS ARM64:

```toml
[dependencies.ggml-org]
llamacpp = { version = "b10069", features = ["backend-metal"] }
```

## Minimal CPU Program

```cpp
import std;
import llamacpp;

int main() {
    llama_backend_init();

    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file("model.gguf", params);
    if (!model) {
        llama_backend_free();
        return 1;
    }

    llama_model_free(model);
    llama_backend_free();
}
```

Build and run it with the normal mcpp commands:

```bash
mcpp build
mcpp run
```

Model loading alone does not perform inference. See `tests/cpu_decode.cpp` for
a compact load/decode/sample path and `examples/chat-cpu` for token generation.

## CPU Chat Example

The checked-in example uses a path dependency on this repository:

```bash
cd examples/chat-cpu
mcpp run -- /path/to/model.gguf \
  'User: Introduce yourself in one sentence.\nAssistant:'
```

It sets `n_gpu_layers = 0`, imports `llamacpp`, and performs real
tokenization, decode, and sampling.

## Metal Chat Example

Metal requires macOS ARM64 and the `backend-metal` feature:

```bash
cd examples/chat-metal
mcpp run -- /path/to/model.gguf metal \
  'User: Introduce yourself in one sentence.\nAssistant:'
```

The example requests GPU offload and exits nonzero if Metal cannot be used. A
valid run should report all of the following:

```text
using embedded metal library
GPU name: MTL0 (...)
offloaded N/M layers to GPU
MTL0_Mapped model buffer size = ... MiB
MTL0 compute buffer size = ... MiB
backend=metal ... generated_tokens=32
```

The package test is stricter than successful compilation: it checks Metal
registration, graph execution, positive layer offload, positive Metal-owned
model and compute buffers, decode, and sampling. The Metal example exits
nonzero before reporting success if either Metal buffer is absent.

## Supported Boundary

`b10069` includes:

- the public llama.cpp C API exposed through `import llamacpp;`;
- the CPU backend on Linux x86_64, Linux ARM64, Windows x86_64, and macOS ARM64;
- the Metal backend on macOS ARM64;
- architecture-specific x86_64 and ARM64 CPU source selection.

`b10069` does not include `mtmd`, CUDA, Vulkan, RPC, or other upstream backends.
It also does not claim that every upstream model architecture has been tested.
Deprecated upstream C APIs remain exported for API completeness; code that
calls them may receive the deprecation warnings defined by upstream.

See [the update policy](docs/upstream-update-policy.md) for versioning and
checkpoint selection, and [the b10069 validation record](docs/validation/b10069.md)
for exact evidence and remaining release gates.
