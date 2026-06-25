#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct options {
    std::string model;
    int32_t ctx = 4096;
    int32_t batch = 128;
    int32_t threads = 6;
};

struct request {
    std::string id;
    std::string prompt;
    std::string prompt_file;
    std::string save_session;
    int32_t predict = 128;
    bool reset = false;
    bool shutdown = false;
};

struct completion_stats {
    std::string text;
    int32_t prompt_tokens = 0;
    int32_t generated_tokens = 0;
    int32_t reused_tokens = 0;
    int32_t decoded_prompt_tokens = 0;
    int32_t kv_tokens = 0;
    int64_t elapsed_ms = 0;
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

static bool json_find_value(const std::string & json, const std::string & key, size_t & pos) {
    const std::string needle = "\"" + key + "\"";
    pos = json.find(needle);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    pos += 1;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos += 1;
    return pos < json.size();
}

static bool json_get_string(const std::string & json, const std::string & key, std::string & out) {
    size_t pos = 0;
    if (!json_find_value(json, key, pos) || json[pos] != '"') return false;
    pos += 1;
    std::string value;
    while (pos < json.size()) {
        char ch = json[pos++];
        if (ch == '"') {
            out = value;
            return true;
        }
        if (ch != '\\') {
            value += ch;
            continue;
        }
        if (pos >= json.size()) break;
        const char esc = json[pos++];
        switch (esc) {
            case '"': value += '"'; break;
            case '\\': value += '\\'; break;
            case '/': value += '/'; break;
            case 'b': value += '\b'; break;
            case 'f': value += '\f'; break;
            case 'n': value += '\n'; break;
            case 'r': value += '\r'; break;
            case 't': value += '\t'; break;
            default:
                value += esc;
                break;
        }
    }
    return false;
}

static bool json_get_int(const std::string & json, const std::string & key, int32_t & out) {
    size_t pos = 0;
    if (!json_find_value(json, key, pos)) return false;
    char * end = nullptr;
    const long value = std::strtol(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;
    out = (int32_t) value;
    return true;
}

static bool json_get_bool(const std::string & json, const std::string & key, bool & out) {
    size_t pos = 0;
    if (!json_find_value(json, key, pos)) return false;
    if (json.compare(pos, 4, "true") == 0) {
        out = true;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        out = false;
        return true;
    }
    return false;
}

static request parse_request(const std::string & line) {
    request req;
    json_get_string(line, "id", req.id);
    json_get_string(line, "prompt", req.prompt);
    json_get_string(line, "promptFile", req.prompt_file);
    json_get_string(line, "saveSession", req.save_session);
    json_get_int(line, "predict", req.predict);
    json_get_bool(line, "reset", req.reset);
    json_get_bool(line, "shutdown", req.shutdown);
    if (!req.prompt_file.empty()) req.prompt = read_file(req.prompt_file);
    if (req.id.empty()) throw std::runtime_error("request id is required");
    if (!req.shutdown && req.prompt.empty()) throw std::runtime_error("prompt or promptFile is required");
    req.predict = std::max(1, req.predict);
    return req;
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
        else if ((arg == "-c" || arg == "--ctx-size")) opt.ctx = std::atoi(next());
        else if ((arg == "-b" || arg == "--batch-size")) opt.batch = std::atoi(next());
        else if ((arg == "-t" || arg == "--threads")) opt.threads = std::atoi(next());
        else if (arg == "-h" || arg == "--help") return false;
        else if (opt.model.empty()) opt.model = arg;
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }
    return !opt.model.empty();
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model model.gguf [--ctx-size N] [--batch-size N] [--threads N]\n"
        "stdin protocol: JSON lines with id, promptFile|prompt, predict, saveSession, reset, shutdown\n",
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

class ik_runtime {
public:
    explicit ik_runtime(const options & opt) : batch_size(std::max(1, opt.batch)) {
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        model = llama_model_load_from_file(opt.model.c_str(), mparams);
        if (!model) throw std::runtime_error("failed to load model: " + opt.model);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = (uint32_t) opt.ctx;
        cparams.n_batch = (uint32_t) opt.batch;
        cparams.n_ubatch = (uint32_t) opt.batch;
        cparams.n_threads = (uint32_t) opt.threads;
        cparams.n_threads_batch = (uint32_t) opt.threads;
        cparams.offload_kqv = false;

        ctx = llama_init_from_model(model, cparams);
        if (!ctx) throw std::runtime_error("failed to create context");
    }

    ~ik_runtime() {
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
    }

    completion_stats complete(const request & req) {
        const int64_t started = llama_time_us();
        if (req.reset) clear_cache();

        std::vector<llama_token> prompt_tokens = tokenize(model, req.prompt);
        if (prompt_tokens.empty()) throw std::runtime_error("prompt tokenization produced no tokens");
        if ((int32_t) prompt_tokens.size() + req.predict + 1 > (int32_t) llama_n_ctx(ctx)) {
            clear_cache();
            throw std::runtime_error("prompt plus prediction exceeds context");
        }

        size_t common = common_prefix(prompt_tokens, cached_tokens);
        if (common == prompt_tokens.size()) {
            common -= 1; // recompute final prompt logits for generation.
        }
        trim_cache(common);
        const size_t reused = cached_tokens.size();

        std::vector<llama_token> suffix(prompt_tokens.begin() + (ptrdiff_t) cached_tokens.size(), prompt_tokens.end());
        const int32_t decoded = (int32_t) suffix.size();
        int32_t pos = (int32_t) cached_tokens.size();
        if (!suffix.empty()) {
            pos = decode_tokens(ctx, suffix, pos, batch_size);
            if (pos < 0) throw std::runtime_error("prompt decode failed");
            cached_tokens.insert(cached_tokens.end(), suffix.begin(), suffix.end());
        }

        completion_stats stats;
        stats.prompt_tokens = (int32_t) prompt_tokens.size();
        stats.reused_tokens = (int32_t) reused;
        stats.decoded_prompt_tokens = decoded;

        for (int32_t i = 0; i < req.predict; ++i) {
            llama_token next = sample_greedy(ctx, model);
            if (llama_token_is_eog(model, next)) break;
            stats.text += token_piece(model, next);
            std::vector<llama_token> one = { next };
            pos = decode_tokens(ctx, one, (int32_t) cached_tokens.size(), 1);
            if (pos < 0) throw std::runtime_error("decode failed");
            cached_tokens.push_back(next);
            stats.generated_tokens += 1;
        }

        if (!req.save_session.empty()) {
            llama_state_save_file(ctx, req.save_session.c_str(), cached_tokens.data(), cached_tokens.size());
        }

        stats.kv_tokens = llama_get_kv_cache_token_count(ctx);
        stats.elapsed_ms = (llama_time_us() - started) / 1000;
        return stats;
    }

private:
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    int32_t batch_size = 128;
    std::vector<llama_token> cached_tokens;

    static size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) i += 1;
        return i;
    }

    void clear_cache() {
        llama_kv_cache_clear(ctx);
        cached_tokens.clear();
    }

    void trim_cache(size_t keep) {
        keep = std::min(keep, cached_tokens.size());
        if (keep == cached_tokens.size()) return;
        if (!llama_kv_cache_seq_rm(ctx, 0, (llama_pos) keep, -1)) {
            clear_cache();
            return;
        }
        cached_tokens.resize(keep);
    }
};

static void write_ok(const request & req, const completion_stats & stats) {
    std::cout
        << "{"
        << "\"id\":\"" << json_escape(req.id) << "\","
        << "\"ok\":true,"
        << "\"text\":\"" << json_escape(stats.text) << "\","
        << "\"promptTokens\":" << stats.prompt_tokens << ","
        << "\"generatedTokens\":" << stats.generated_tokens << ","
        << "\"reusedTokens\":" << stats.reused_tokens << ","
        << "\"decodedPromptTokens\":" << stats.decoded_prompt_tokens << ","
        << "\"kvTokens\":" << stats.kv_tokens << ","
        << "\"elapsedMs\":" << stats.elapsed_ms
        << "}\n";
    std::cout.flush();
}

static void write_error(const std::string & id, const std::exception & error) {
    std::cout
        << "{"
        << "\"id\":\"" << json_escape(id) << "\","
        << "\"ok\":false,"
        << "\"error\":\"" << json_escape(error.what()) << "\""
        << "}\n";
    std::cout.flush();
}

int main(int argc, char ** argv) {
    options opt;
    if (!parse_args(argc, argv, opt)) {
        usage(argv[0]);
        return 2;
    }

    try {
        ik_runtime runtime(opt);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::string id;
            try {
                request req = parse_request(line);
                id = req.id;
                if (req.shutdown) {
                    std::cout << "{\"id\":\"" << json_escape(req.id) << "\",\"ok\":true,\"shutdown\":true}\n";
                    std::cout.flush();
                    break;
                }
                write_ok(req, runtime.complete(req));
            } catch (const std::exception & error) {
                write_error(id, error);
            }
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    return 0;
}
