import mcpplibs.mwrap.llamacpp;
namespace macros = mcpplibs::mwrap::llamacpp::macros;
int main() { static_assert(macros::LLAMA_DEFAULT_SEED == 0xFFFFFFFFu); static_assert(macros::LLAMA_TOKEN_NULL == -1); return 0; }
