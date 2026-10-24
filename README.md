# llama.cpp-m

> Pinned llama.cpp b10069 through C++23 modules; upstream API names are preserved.

**English** · [简体中文](README.zh.md)

Upstream: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

## Add The Package

```sh
mcpp new llama_app --template mcpplibs.mwrap.llamacpp@b10069
cd llama_app
mcpp run
```

```toml
[dependencies]
mcpplibs.mwrap.llamacpp = "b10069"
```

## Minimal CPU Program

```cpp
import mcpplibs.mwrap.llamacpp;
int main() { llama_backend_init(); llama_backend_free(); }
```

The default template checks CPU initialization without downloading a model.

## CPU Chat Example

```sh
cd examples/chat-cpu
mcpp run -- /path/to/model.gguf "Hello"
```

A compatible local GGUF model is required for tokenization, decode and sampling.

## Metal Chat Example

Use `backend-metal` on macOS ARM64; [the example](examples/chat-metal) requires actual offload and Metal buffers.

## Vulkan Chat Example

Use `backend-vulkan` on Linux. [The example](examples/chat-vulkan) rejects silent CPU fallback.
Shader generation, libstdc++ constraints and driver boundaries are in [usage](docs/usage.md).

## Supported Boundary

Pin `b10069` exactly; it is not a semver range. `.macros` exposes mapped constants.
Model quality and unsupported GPU configurations are not implied by the CPU template.
[Upstream policy](docs/upstream-update-policy.md) · [Validation](docs/validation/b10069.md) · [MIT](LICENSE)
