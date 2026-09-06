// The Vulkan backend, measured against the CPU rather than against itself.
//
// WHY THE CRITERION IS AN EQUALITY. A device test that only asserts "a token
// came back, and it is inside the vocabulary" passes on a backend that
// computed nonsense, and passes just as well on a build where no device was
// ever used. Both are failures this tier exists to catch. So this test decodes
// the same prompt twice from one model -- once with every layer on the host,
// once with every layer offloaded -- and requires the greedy sample to be the
// same token.
//
// AND WHY THAT EQUALITY NEEDS A SECOND ASSERTION. Two CPU runs also agree.
// The equality is only evidence about the device if the second run actually
// reached one, so the offload is asserted from llama.cpp's own log before the
// tokens are compared. Without that, a build that silently fell back to the
// CPU would produce the strongest-looking green in this repository.
#if !defined(LLAMACPP_VULKAN_TEST)
#error "LLAMACPP_VULKAN_TEST must be enabled: run with --features backend-vulkan"
#endif

import std;

import llamacpp;

#ifdef LLAMA_H
#error "import llamacpp leaked LLAMA_H"
#endif

#ifdef LLAMA_API
#error "import llamacpp leaked LLAMA_API"
#endif


namespace {

std::string logs;

void capture_log(enum ggml_log_level, const char * text, void *) {
    if (text) {
        logs += text;
        std::cerr << text;
    }
}

int fail(const std::string & message) {
    std::cerr << "Vulkan test failed: " << message << "\n";
    return 1;
}

// A four-element F32 add on the device. It runs before any model is loaded,
// so a backend whose shaders were never generated fails here -- where the
// cause is one graph and one dispatch -- rather than inside a transformer.
bool run_device_add_probe(ggml_backend_dev_t device) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return false;

    ggml_init_params params = {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * context = ggml_init(params);
    if (!context) {
        ggml_backend_free(backend);
        return false;
    }

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_tensor * lhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * rhs = ggml_new_tensor_1d(context, GGML_TYPE_F32, 4);
    ggml_tensor * sum = ggml_add(context, lhs, rhs);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!graph || !lhs || !rhs || !sum || !buffer) {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(context);
        ggml_backend_free(backend);
        return false;
    }
    ggml_build_forward_expand(graph, sum);

    const float lhs_values[] = {1.0F, -2.0F, 3.5F, 10.0F};
    const float rhs_values[] = {4.0F, 5.0F, -1.5F, -3.0F};
    ggml_backend_tensor_set(lhs, lhs_values, 0, sizeof(lhs_values));
    ggml_backend_tensor_set(rhs, rhs_values, 0, sizeof(rhs_values));

    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    float actual[4] = {};
    if (passed) {
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(sum, actual, 0, sizeof(actual));
        const float expected[] = {5.0F, 3.0F, 2.0F, 7.0F};
        for (std::size_t index = 0; index < 4; ++index) {
            if (actual[index] != expected[index]) {
                std::cerr << "device add produced " << actual[index]
                          << " at " << index << ", expected " << expected[index] << "\n";
                passed = false;
                break;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
    return passed;
}

struct decode_result {
    bool ok = false;
    llama_token sampled = -1;
};

decode_result decode_once(const char * model_path, int gpu_layers) {
    decode_result result;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) return result;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        llama_model_free(model);
        return result;
    }

    llama_token tokens[] = {1, 2, 3};
    const int decoded = llama_decode(
        context,
        llama_batch_get_one(tokens, sizeof(tokens) / sizeof(tokens[0]))
    );
    if (decoded != 0) {
        llama_free(context);
        llama_model_free(model);
        return result;
    }

    llama_sampler * sampler = llama_sampler_chain_init(
        llama_sampler_chain_default_params()
    );
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    result.sampled = llama_sampler_sample(sampler, context, -1);

    const int vocabulary_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    result.ok = result.sampled >= 0 && result.sampled < vocabulary_size;

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    return result;
}

}  // namespace


int main() {
    const char * model_path = std::getenv("LLAMACPP_TEST_MODEL");
    if (!model_path || !*model_path) {
        return fail("LLAMACPP_TEST_MODEL is not set");
    }

    llama_log_set(capture_log, nullptr);
    llama_backend_init();

    ggml_backend_reg_t vulkan = ggml_backend_reg_by_name("Vulkan");
    if (!vulkan) {
        llama_backend_free();
        return fail(
            "no Vulkan registry: the backend was not compiled in "
            "(GGML_USE_VULKAN reaches ggml-backend-reg.cpp through the feature)"
        );
    }
    if (ggml_backend_reg_dev_count(vulkan) == 0) {
        llama_backend_free();
        // TWO CAUSES, AND THEY LOOK THE SAME FROM HERE. Either the loader
        // found no ICD, or it found one that upstream excludes: ggml keeps
        // only devices whose type is not `eCpu`, and a software Vulkan
        // implementation reports exactly that. Measured on Mesa's lavapipe,
        // which advertises `storageBuffer16BitAccess` and every feature the
        // backend needs and is still dropped for its type alone.
        //
        // Both are named because a message that blamed only the loader would
        // send a reader looking for a missing file that is present.
        return fail(
            "the Vulkan registry reports no device. Either the loader found no "
            "ICD -- on a machine with no GPU, name the lavapipe payload with "
            "VK_DRIVER_FILES=<payload>/share/vulkan/icd.d/lvp_icd.x86_64.json -- "
            "or the ICD it found is a software device, which ggml drops by "
            "type; upstream's own escape hatch for that is "
            "GGML_VK_VISIBLE_DEVICES=0"
        );
    }

    ggml_backend_dev_t device = ggml_backend_reg_dev_get(vulkan, 0);
    if (!device) {
        llama_backend_free();
        return fail("the Vulkan registry advertised a device it cannot return");
    }
    std::cout << "vulkan device: " << ggml_backend_dev_name(device)
              << " -- " << ggml_backend_dev_description(device) << "\n";

    if (!run_device_add_probe(device)) {
        llama_backend_free();
        return fail("the F32 ADD graph did not execute correctly on the device");
    }
    if (!llama_supports_gpu_offload()) {
        llama_backend_free();
        return fail("llama does not report GPU offload support");
    }

    logs.clear();
    const decode_result on_host = decode_once(model_path, 0);
    if (!on_host.ok) {
        llama_backend_free();
        return fail("the host decode did not produce a token in the vocabulary");
    }

    logs.clear();
    const decode_result on_device = decode_once(model_path, std::numeric_limits<int>::max());
    if (!on_device.ok) {
        llama_backend_free();
        return fail("the device decode did not produce a token in the vocabulary");
    }

    // The equality below is evidence about the device only if the device was
    // used. Two host decodes agree trivially.
    const std::regex offload_pattern(
        "offloaded ([1-9][0-9]*)/([1-9][0-9]*) layers to GPU"
    );
    if (!std::regex_search(logs, offload_pattern)) {
        llama_backend_free();
        return fail(
            "the device run offloaded no layer, so the token equality below "
            "would compare two host decodes"
        );
    }

    std::cout << "host token: " << on_host.sampled
              << ", device token: " << on_device.sampled << "\n";
    if (on_host.sampled != on_device.sampled) {
        llama_backend_free();
        return fail(
            "the device sampled a different token than the host for the same "
            "prompt under greedy sampling"
        );
    }

    llama_backend_free();
    std::cout << "LLAMACPP_VULKAN_TEST=PASS\n";
    return 0;
}
