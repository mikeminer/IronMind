#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct options {
    std::string model;
    std::string prompt;
    std::string prompt_file;
    std::string save_session;
    int32_t ctx = 4096;
    int32_t batch = 128;
    int32_t threads = 6;
    int32_t predict = 128;
    bool json = false;
};

static std::string read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
}

static std::string json_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size() + 16);
    for (unsigned char ch : text) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out += buf;
                } else {
                    out += static_cast<char>(ch);
                }
        }
    }
    return out;
}

static bool parse_args(int argc, char ** argv, options & opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };

        if ((arg == "-m" || arg == "--model")) opt.model = next();
        else if ((arg == "-p" || arg == "--prompt")) opt.prompt = next();
        else if ((arg == "-f" || arg == "--file")) opt.prompt_file = next();
        else if (arg == "--save-session") opt.save_session = next();
        else if ((arg == "-c" || arg == "--ctx-size")) opt.ctx = std::atoi(next());
        else if ((arg == "-b" || arg == "--batch-size")) opt.batch = std::atoi(next());
        else if ((arg == "-t" || arg == "--threads")) opt.threads = std::atoi(next());
        else if ((arg == "-n" || arg == "--predict")) opt.predict = std::atoi(next());
        else if (arg == "--json") opt.json = true;
        else if (arg == "-h" || arg == "--help") return false;
        else if (opt.model.empty()) opt.model = arg;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    if (opt.model.empty()) return false;
    if (!opt.prompt_file.empty()) opt.prompt = read_file(opt.prompt_file);
    return !opt.prompt.empty();
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model model.gguf (--prompt TEXT|--file prompt.txt) [--predict N] [--ctx-size N] [--batch-size N] [--threads N] [--save-session path] [--json]\n",
        argv0);
}

static std::vector<llama_token> tokenize(const llama_model * model, const std::string & text) {
    int32_t n = llama_tokenize(model, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens((size_t) n);
    const int32_t got = llama_tokenize(model, text.c_str(), (int32_t) text.size(), tokens.data(), n, false, true);
    if (got < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize((size_t) got);
    return tokens;
}

static std::string token_piece(const llama_model * model, llama_token token) {
    char buf[256];
    int32_t n = llama_token_to_piece(model, token, buf, sizeof(buf), 0, false);
    if (n < 0) {
        std::vector<char> big((size_t) -n);
        n = llama_token_to_piece(model, token, big.data(), (int32_t) big.size(), 0, false);
        return n > 0 ? std::string(big.data(), (size_t) n) : std::string();
    }
    return n > 0 ? std::string(buf, (size_t) n) : std::string();
}

static void batch_add(llama_batch & batch, llama_token token, llama_pos pos, bool logits) {
    const int32_t i = batch.n_tokens;
    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits ? 1 : 0;
    batch.n_tokens += 1;
}

static int decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int32_t start_pos, int32_t n_batch) {
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    int32_t pos = start_pos;
    for (size_t offset = 0; offset < tokens.size();) {
        batch.n_tokens = 0;
        const int32_t remaining = (int32_t) std::min<size_t>((size_t) n_batch, tokens.size() - offset);
        for (int32_t i = 0; i < remaining; ++i) {
            const bool logits = (offset + (size_t) i + 1) == tokens.size();
            batch_add(batch, tokens[offset + (size_t) i], pos + i, logits);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return -1;
        }
        offset += (size_t) remaining;
        pos += remaining;
    }
    llama_batch_free(batch);
    return pos;
}

static llama_token sample_greedy(llama_context * ctx, const llama_model * model) {
    const int32_t n_vocab = llama_n_vocab(model);
    float * logits = llama_get_logits(ctx);
    llama_token best_id = 0;
    float best = logits[0];
    for (llama_token id = 1; id < n_vocab; ++id) {
        if (logits[id] > best) {
            best = logits[id];
            best_id = id;
        }
    }
    return best_id;
}

int main(int argc, char ** argv) {
    options opt;
    try {
        if (!parse_args(argc, argv, opt)) {
            usage(argv[0]);
            return 2;
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = true;
    mparams.use_mlock = false;

    llama_model * model = llama_model_load_from_file(opt.model.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", opt.model.c_str());
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) opt.ctx;
    cparams.n_batch = (uint32_t) opt.batch;
    cparams.n_ubatch = (uint32_t) opt.batch;
    cparams.n_threads = (uint32_t) opt.threads;
    cparams.n_threads_batch = (uint32_t) opt.threads;
    cparams.offload_kqv = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    int64_t started = llama_time_us();
    std::string output;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> all_tokens;
    int32_t generated = 0;
    int32_t pos = 0;

    try {
        prompt_tokens = tokenize(model, opt.prompt);
        all_tokens = prompt_tokens;
        if ((int32_t) prompt_tokens.size() + opt.predict + 1 > (int32_t) llama_n_ctx(ctx)) {
            throw std::runtime_error("prompt plus prediction exceeds context");
        }

        pos = decode_tokens(ctx, prompt_tokens, 0, std::max(1, opt.batch));
        if (pos < 0) throw std::runtime_error("prompt decode failed");

        for (int32_t i = 0; i < opt.predict; ++i) {
            llama_token next = sample_greedy(ctx, model);
            if (llama_token_is_eog(model, next)) break;
            output += token_piece(model, next);
            all_tokens.push_back(next);
            std::vector<llama_token> one = { next };
            pos = decode_tokens(ctx, one, pos, 1);
            if (pos < 0) throw std::runtime_error("decode failed");
            generated += 1;
        }

        if (!opt.save_session.empty()) {
            llama_state_save_file(ctx, opt.save_session.c_str(), all_tokens.data(), all_tokens.size());
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    int64_t elapsed = llama_time_us() - started;
    if (opt.json) {
        std::cout
            << "{"
            << "\"model\":\"" << json_escape(opt.model) << "\","
            << "\"text\":\"" << json_escape(output) << "\","
            << "\"promptTokens\":" << prompt_tokens.size() << ","
            << "\"generatedTokens\":" << generated << ","
            << "\"elapsedMs\":" << (elapsed / 1000)
            << "}\n";
    } else {
        std::cout << output;
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
