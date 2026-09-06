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

For Vulkan on Linux:

```toml
[dependencies.ggml-org]
llamacpp = { version = "b10069", features = ["backend-vulkan"] }
```

That line is the whole diff. The shader compiler, the Khronos loader, the
SPIR-V headers and the adapter that lets a built binary reach the machine's own
driver are declared by this package under the feature, so a consumer names none
of them -- and a consumer that does not name the feature acquires none of them
either.

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

## Vulkan Chat Example

Vulkan requires Linux and the `backend-vulkan` feature:

```bash
cd examples/chat-vulkan
mcpp run -- /path/to/model.gguf vulkan \
  'User: Introduce yourself in one sentence.\nAssistant:'
```

The example refuses to report a device run that did not happen: it exits
nonzero when Vulkan is requested and no layer was offloaded, because a build
that quietly fell back to the CPU produces output indistinguishable from
success.

### How the shaders are built

The backend needs 134 shader sets, and upstream generates them with a tool it
keeps in its own tree. `build.mcpp` does not run that tool 134 times; it
DECLARES 136 build-graph edges -- one per shader, one to compile the generator
from the vendored source, one for the header they share. Each shader is then
incremental, parallel, and attributable to the edge that failed rather than to
`build.mcpp exited 1`.

The build program does exactly one thing itself: it asks `glslc` which
extensions it accepts. That answer has two readers -- the generator, which
decides which variants to emit, and `ggml-vulkan.cpp`, which decides which to
look for -- and asking twice would make one truth into two.

### A libstdc++ toolchain, on this checkpoint

`backend-vulkan` builds with a libstdc++ toolchain. Under libc++ the compile of
upstream's `ggml-vulkan.cpp` stops at

```
error: invalid application of 'sizeof' to an incomplete type 'vk_memory_logger'
  in instantiation of member function 'std::unique_ptr<vk_memory_logger>::~unique_ptr'
```

`vk_device_struct` holds a `std::unique_ptr<vk_memory_logger>` and its
destructor is instantiated at line 1015, while the class is defined at line
1920. Destroying a `unique_ptr` to an incomplete type is undefined behaviour;
libc++ has a static assertion for it and libstdc++ does not. It is upstream's
source rather than this packaging, and nothing in a build program can repair
it without patching the vendored tree, which this repository does not do.

Everything else is unaffected: the CPU backend builds under both standard
libraries, measured.

### Running without a GPU

ggml keeps only Vulkan devices whose type is not `eCpu`, so a software
implementation is excluded for its type alone. Measured: Mesa's lavapipe
advertises `storageBuffer16BitAccess` and every feature the backend requires,
and is still dropped. Upstream ships the selector for this case, and a machine
with no GPU needs both halves of it:

```bash
VK_DRIVER_FILES=<lavapipe payload>/share/vulkan/icd.d/lvp_icd.x86_64.json \
GGML_VK_VISIBLE_DEVICES=0 \
  mcpp test vulkan_decode --features backend-vulkan
```

### What the artifact depends on

Measured on the built test binary, everything outside the mcpp registry:

```
$ ldd target/*/*/bin/vulkan_decode | grep -v mcpp/registry
	linux-vdso.so.1
	libvulkan.so.1 => .../bin/libvulkan.so.1
```

The Khronos loader is built from source by `compat:vulkan` and travels beside
the binary; the kernel's vDSO is the only thing left. The shader generator is
statically linked for the same reason -- it runs from inside the build, and a
build tool that needs the host's libstdc++ is a host dependency this ecosystem
does not accept.

The package test is stricter than a token in range: it decodes the same prompt
twice from one model, once with every layer on the host and once with every
layer offloaded, and requires greedy sampling to produce the SAME token. A
token inside the vocabulary is produced by a backend that computed nonsense
just as readily as by one that worked. The equality is only evidence about the
device if the device ran, so the offload is asserted from llama.cpp's own log
before the tokens are compared.

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
