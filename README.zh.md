# llama.cpp-m

> 通过 C++23 模块使用固定的 llama.cpp b10069，保留上游 API 名称。

[English](README.md) · **简体中文** · [Upstream](https://github.com/ggml-org/llama.cpp) · [Releases](https://github.com/mcpplibs/llama.cpp-m/releases) · [mcpp](https://github.com/mcpp-community/mcpp) · [Index](https://github.com/mcpplibs/mcpp-index)

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![mcpp-community](https://img.shields.io/badge/mcpp--community-mwrap-blue)](https://github.com/mcpp-community)
[![CI](https://github.com/mcpplibs/llama.cpp-m/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mcpplibs/llama.cpp-m/actions/workflows/ci.yml)

## 模块

- `import mcpplibs.mwrap.llamacpp;`: llama.cpp 模型、上下文与推理 API
- `import mcpplibs.mwrap.llamacpp.macros;`: 类型化宏常量

> More Modules API -> [模块与 API](docs/usage.md)

## 添加依赖

```sh
mcpp new llama_app --template mcpplibs.mwrap.llamacpp@b10069
cd llama_app
mcpp run
```

```toml
[dependencies]
mcpplibs.mwrap.llamacpp = "b10069"
```

## 最小 CPU 程序

```cpp
import mcpplibs.mwrap.llamacpp;
int main() { llama_backend_init(); llama_backend_free(); }
```

默认模板验证 CPU 初始化，无需下载模型。

## CPU 对话示例

```sh
cd examples/chat-cpu
mcpp run -- /path/to/model.gguf "Hello"
```

实际分词、解码与采样需要兼容的本地 GGUF 模型。

## Metal 对话示例

macOS ARM64 使用 `backend-metal`；[示例](examples/chat-metal) 要求真实 offload 和 Metal buffer。

## Vulkan 对话示例

Linux 使用 `backend-vulkan`；[示例](examples/chat-vulkan) 拒绝静默 CPU 回退。
shader 生成、libstdc++ 限制及驱动边界见[使用说明](docs/usage.md)。

## 支持边界

精确固定 `b10069`，不使用 semver 范围；`.macros` 导出映射常量。
CPU 模板通过不代表模型质量或未支持 GPU 配置通过。
[上游策略](docs/upstream-update-policy.md) · [验证](docs/validation/b10069.md) · [Apache-2.0](LICENSE)

封装：Apache-2.0，mcpp-community。[上游原始 README 与许可证](upstream/llama.cpp/)，上游许可保持原样。
