# llama.cpp-m

llama.cpp **b10069** as a C++23 module, preserving the upstream C API through `import llamacpp;`.

## Quick start

Use mcpp **2026.10.24.1**. The candidate revision below requires index publication.

```sh
mcpp new inference --template ggml-org.llamacpp@b10069.3
cd inference
mcpp run
```

The default run initializes the CPU backend without downloading a model.
Use `mcpp run -- /path/to/model.gguf` to decode a small batch and sample a token.

## Existing projects

```toml
[dependencies]
"ggml-org.llamacpp" = "b10069.3"
```

```cpp
import std;
import llamacpp;
int main() {
    llama_backend_init();
    std::println("{}", llama_print_system_info());
    llama_backend_free();
}
```

Pin the exact build key; `b10069.3` is an upstream build identifier with a wrapper revision.

## Scope

CPU: Linux x86_64/arm64, Windows x86_64 and macOS arm64.
`backend-metal` adds Metal on macOS arm64; `backend-vulkan` adds Vulkan on Linux with libstdc++.
Vulkan with libc++ is refused for the documented upstream incomplete-type issue.

CUDA, mtmd, RPC and other backends are outside this checkpoint. Model coverage is documented separately.

## Documentation

[Usage and backends](docs/usage.md) · [CPU chat](examples/chat-cpu) ·
[Update policy](docs/upstream-update-policy.md) · [Validation](docs/validation/b10069.md) · [MIT](LICENSE)
