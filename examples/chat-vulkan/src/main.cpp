// Generation on a Vulkan device, and the same generation on the host.
//
// The second mode is not a convenience. A Vulkan build that quietly fell back
// to the CPU produces output that looks exactly like success, so this program
// reports which backend ran and how many layers were offloaded, and refuses to
// claim the device when it did not get one.
import std;
import llamacpp;

namespace {

enum class backend_kind { cpu, vulkan };

std::string logs;

void log_callback(enum ggml_log_level, const char * text, void *) {
    if (text) {
        logs += text;
        std::cerr << text;
    }
}

int usage(const char * program) {
    std::cerr << "usage: " << program
              << " <model.gguf> <cpu|vulkan> [prompt]\n";
    return 2;
}

int offloaded_layers(const std::string & text) {
    static const std::regex pattern(
        "offloaded ([0-9]+)/([0-9]+) layers to GPU"
    );
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) return 0;
    return std::stoi(match[1].str());
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return usage(argv[0]);

    const std::string requested = argv[2];
    backend_kind backend{};
    if (requested == "cpu") {
        backend = backend_kind::cpu;
    } else if (requested == "vulkan") {
        backend = backend_kind::vulkan;
    } else {
        return usage(argv[0]);
    }

    const std::string prompt = argc == 4
        ? argv[3]
        : "User: Hello! Who are you?\nAssistant:";

    llama_log_set(log_callback, nullptr);
    llama_backend_init();

    if (backend == backend_kind::vulkan) {
        ggml_backend_reg_t registry = ggml_backend_reg_by_name("Vulkan");
        if (!registry || ggml_backend_reg_dev_count(registry) == 0) {
            std::cerr << "no Vulkan device is reachable. The backend is compiled "
                         "in, so this is the loader finding no ICD; on a machine "
                         "with no GPU, name the lavapipe payload with "
                         "VK_DRIVER_FILES=<payload>/share/vulkan/icd.d/"
                         "lvp_icd.x86_64.json\n";
            llama_backend_free();
            return 3;
        }
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(registry, 0);
        std::cerr << "vulkan device: " << ggml_backend_dev_name(device)
                  << " -- " << ggml_backend_dev_description(device) << '\n';
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = backend == backend_kind::vulkan
        ? std::numeric_limits<int>::max()
        : 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::cerr << "failed to load model: " << argv[1] << '\n';
        llama_backend_free();
        return 4;
    }

    if (backend == backend_kind::vulkan && offloaded_layers(logs) == 0) {
        std::cerr << "vulkan was requested and no layer was offloaded; refusing "
                     "to report a device run that did not happen\n";
        llama_model_free(model);
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
    std::cerr << "backend=" << (backend == backend_kind::vulkan ? "vulkan" : "cpu")
              << " offloaded_layers=" << offloaded_layers(logs)
              << " params=" << llama_model_n_params(model)
              << " generated_tokens=" << generated << '\n';

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return exit_code;
}
