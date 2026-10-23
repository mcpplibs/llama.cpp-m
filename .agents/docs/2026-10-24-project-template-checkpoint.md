# Runnable project template checkpoint

The existing ggml-org.llamacpp identity is retained. Package b10069.3 follows the existing upstream b10069 plus one packaging revision; no existing release or index hash is rewritten.

The default cpu template uses the fully qualified self.name and self.version dependency. It initializes the actual CPU backend with no model download. With a local GGUF path it loads the model, creates a context, decodes and samples. Resource cleanup is explicit and scoped.

The template test uses the actual mcpp package renderer and verifies the emitted dependency before substituting the staged candidate source. It then compiles and runs the generated program. Both the no-argument flow and a pinned 19 MB GGUF inference flow passed locally on LLVM 23.1.3 (sampled token 471).

A fresh path consumer revealed missing generated build inputs that earlier install-time materialization had hidden. The four stable generated inputs are now checked in verbatim from generated_files, with tools/generate_build_files.py --check guarding drift. Feature-specific Metal/Vulkan generation still uses the build output. Current libc++ also exposed missing standard includes in upstream source; two scoped wrapper compile-flag rules provide cstdlib/cerrno without patching upstream.

mcpp is pinned to 2026.10.21.1. The root package built under LLVM 23.1.3 and all 116 existing unit/repository/build-helper checks passed with LLAMACPP_REQUIRE_BUILD_HELPER=1. Native CPU CI now additionally executes the real template with the existing checksummed model. No merge or release is performed by this checkpoint.

Native CI exposed an existing exact-version conflict: compat.vulkan@1.4.357.0 requires compat.vulkan-runtime@2026.09.11 while this wrapper directly requested 2026.09.07. The direct pin is aligned to 2026.09.11 rather than linking or mangling two copies of a runtime with no module. Evidence: job 120654792434.

The native ARM64 job decoded correctly, but its old static-file-label check rejected the current engine-managed glibc loader. The check now inspects actual PT_INTERP and the loader-reported complete dependency closure: static remains accepted, dynamic is accepted only when loader and every resolved library belong to the declared isolated registry. Any system-library escape fails. This retains the no-host-runtime invariant and architecture check without changing the engine or skipping inference.

The next Vulkan CI passed compilation but strict mode exposed static adapter/header anchors shared between program and multiple DSOs. The actual provider defaults are fixed in index commit a4130c75a49c6e9df03690743467696ef81d3a19, and only the Vulkan CI job checks out that exact candidate index and routes compat through it. This tests the paired change without merging the index or weakening --strict.
