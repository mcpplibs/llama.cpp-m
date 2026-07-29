/// Portable CPU smoke: load a pinned GGUF, decode one batch, and sample.
import std;

import llamacpp;

#ifdef LLAMA_H
#error "import llamacpp leaked LLAMA_H"
#endif

#ifdef LLAMA_API
#error "import llamacpp leaked LLAMA_API"
#endif


int main() {
    static_assert(LLAMA_DEFAULT_SEED == 0xFFFFFFFFu);
    static_assert(LLAMA_TOKEN_NULL == llama_token{-1});

    const char * model_path = std::getenv("LLAMACPP_TEST_MODEL");
    if (!model_path || !*model_path) {
        std::cerr << "LLAMACPP_TEST_MODEL not set or empty\n";
        return 1;
    }
    {
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file) {
            std::cerr << "cannot open model file: " << model_path << "\n";
            return 2;
        }
        std::cerr << "model size: " << file.tellg() << " bytes\n";
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::cerr << "failed to load model\n";
        llama_backend_free();
        return 3;
    }
    std::cout << "model loaded: " << llama_model_n_params(model) << " params\n";

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    llama_token tokens[] = {1, 2, 3};
    const int token_count = sizeof(tokens) / sizeof(tokens[0]);
    const int decode_result = llama_decode(
        context, llama_batch_get_one(tokens, token_count)
    );
    if (decode_result != 0) {
        std::cerr << "decode returned " << decode_result << "\n";
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }
    std::cout << "decode OK\n";

    llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    const llama_token sampled = llama_sampler_sample(sampler, context, -1);
    std::cout << "sampled token: " << sampled << "\n";
    const int vocabulary_size = llama_vocab_n_tokens(
        llama_model_get_vocab(model)
    );
    if (sampled < 0 || sampled >= vocabulary_size) {
        std::cerr << "sampled token " << sampled
                  << " out of vocab range [0, " << vocabulary_size << ")\n";
        llama_sampler_free(sampler);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    std::cout << "LLAMACPP_CPU_TEST=PASS\n";
    return 0;
}
