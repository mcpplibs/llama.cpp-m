module;

#include <llama.h>
#include <ggml-backend.h>
#include <ggml-alloc.h>

#undef LLAMA_DEFAULT_SEED
#undef LLAMA_TOKEN_NULL
#undef LLAMA_FILE_MAGIC_GGLA
#undef LLAMA_FILE_MAGIC_GGSN
#undef LLAMA_FILE_MAGIC_GGSQ
#undef LLAMA_SESSION_MAGIC
#undef LLAMA_SESSION_VERSION
#undef LLAMA_STATE_SEQ_MAGIC
#undef LLAMA_STATE_SEQ_VERSION
#undef LLAMA_STATE_SEQ_FLAGS_NONE
#undef LLAMA_STATE_SEQ_FLAGS_SWA_ONLY
#undef LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY
#undef LLAMA_STATE_SEQ_FLAGS_ON_DEVICE

export module llamacpp;

#include "gen_exports/required_ggml.inc"
#include "gen_exports/llama.inc"
#include "gen_exports/typed_constants.inc"
