import std;
import llamacpp;

namespace {

void log_callback(enum ggml_log_level, const char * text, void *) {
    if (text) std::cerr << text;
}

int usage(const char * program) {
    std::cerr << "usage: " << program
              << " <model.gguf> [prompt]\n";
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) return usage(argv[0]);

    const std::string prompt = argc == 3
        ? argv[2]
        : "User: Hello! Who are you?\nAssistant:";

    llama_log_set(log_callback, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::cerr << "failed to load model: " << argv[1] << '\n';
        llama_backend_free();
        return 4;
    }

    const llama_vocab * vocabulary = llama_model_get_vocab(model);
    const int prompt_size = -llama_tokenize(
        vocabulary, prompt.data(), prompt.size(), nullptr, 0, true, true
    );
    if (prompt_size <= 0) {
        std::cerr << "failed to measure prompt tokens\n";
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }
    std::vector<llama_token> prompt_tokens(prompt_size);
    if (llama_tokenize(
        vocabulary,
        prompt.data(),
        prompt.size(),
        prompt_tokens.data(),
        prompt_tokens.size(),
        true,
        true
    ) < 0) {
        std::cerr << "failed to tokenize prompt\n";
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    constexpr int max_generated_tokens = 32;
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = prompt_size + max_generated_tokens;
    context_params.n_batch = prompt_size;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9F, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8F));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    llama_batch batch = llama_batch_get_one(
        prompt_tokens.data(), prompt_tokens.size()
    );
    std::cout << prompt;
    std::cout.flush();

    int generated = 0;
    int exit_code = 0;
    llama_token sampled = LLAMA_TOKEN_NULL;
    for (; generated < max_generated_tokens; ++generated) {
        const int decode_result = llama_decode(context, batch);
        if (decode_result != 0) {
            std::cerr << "\ndecode failed: " << decode_result << '\n';
            exit_code = 7;
            break;
        }

        sampled = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocabulary, sampled)) break;

        char piece[256] = {};
        const int piece_size = llama_token_to_piece(
            vocabulary, sampled, piece, sizeof(piece), 0, true
        );
        if (piece_size < 0 || piece_size > static_cast<int>(sizeof(piece))) {
            std::cerr << "\nfailed to render sampled token " << sampled << '\n';
            exit_code = 8;
            break;
        }
        std::cout << std::string_view(piece, piece_size);
        std::cout.flush();
        batch = llama_batch_get_one(&sampled, 1);
    }
    std::cout << '\n';
    std::cerr << "backend=cpu"
              << " params=" << llama_model_n_params(model)
              << " generated_tokens=" << generated << '\n';

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
