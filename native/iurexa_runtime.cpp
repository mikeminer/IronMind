#include "httplib.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

struct options {
    std::string model;
    std::string host = "127.0.0.1";
    std::string model_id = "iurexa";
    fs::path log_dir;
    int32_t ctx = 4096;
    int32_t batch = 128;
    int32_t threads = 0;
    int port = 4141;
    bool cpu_only = true;
    bool allow_public_bind = false;
};

struct chat_message {
    std::string role;
    std::string content;
};

struct completion_result {
    std::string text;
    std::string finish_reason = "stop";
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
};

class runtime_error_with_code : public std::runtime_error {
public:
    runtime_error_with_code(std::string code, std::string message, int status)
        : std::runtime_error(message), code_(std::move(code)), status_(status) {}

    const std::string & code() const { return code_; }
    int status() const { return status_; }

private:
    std::string code_;
    int status_;
};

static std::string now_iso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

static int64_t unix_time_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

class file_logger {
public:
    explicit file_logger(fs::path dir) : dir_(std::move(dir)) {
        fs::create_directories(dir_);
        path_ = dir_ / "iurexa-runtime.log";
    }

    void log(const std::string & level, const std::string & message) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ofstream out(path_, std::ios::app | std::ios::binary);
        out << now_iso() << " [" << level << "] " << message << "\n";
    }

    const fs::path & path() const { return path_; }

private:
    fs::path dir_;
    fs::path path_;
    std::mutex mutex_;
};

static file_logger * g_logger = nullptr;

static void llama_log_callback(ggml_log_level level, const char * text, void *) {
    if (!g_logger || !text) return;
    const char * name = "llama";
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: name = "llama:error"; break;
        case GGML_LOG_LEVEL_WARN:  name = "llama:warn"; break;
        case GGML_LOG_LEVEL_INFO:  name = "llama:info"; break;
        case GGML_LOG_LEVEL_DEBUG: name = "llama:debug"; break;
        default: break;
    }
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) msg.pop_back();
    if (!msg.empty()) g_logger->log(name, msg);
}

static fs::path default_log_dir() {
    if (const char * appdata = std::getenv("APPDATA")) {
        return fs::path(appdata) / "Iurexa" / "logs";
    }
    return fs::temp_directory_path() / "Iurexa" / "logs";
}

static bool is_local_host(const std::string & host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

static int32_t auto_threads() {
    const unsigned int detected = std::thread::hardware_concurrency();
    if (detected == 0) return 4;
    return (int32_t) std::max(1u, detected);
}

static int32_t parse_i32(const char * value, const char * name) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        throw std::runtime_error(std::string("invalid value for ") + name + ": " + value);
    }
    return (int32_t) parsed;
}

static int parse_port(const char * value) {
    const int32_t parsed = parse_i32(value, "--port");
    if (parsed <= 0 || parsed > 65535) {
        throw std::runtime_error(std::string("invalid value for --port: ") + value);
    }
    return (int) parsed;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model model.gguf [--host 127.0.0.1] [--port 4141] [--ctx 4096] [--threads auto|N] [--batch 128] [--cpu-only]\n"
        "       optional: [--model-id iurexa] [--log-dir PATH] [--allow-public-bind]\n",
        argv0);
}

static bool parse_args(int argc, char ** argv, options & opt) {
    opt.log_dir = default_log_dir();
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };

        if (arg == "--model" || arg == "-m") opt.model = next();
        else if (arg == "--host") opt.host = next();
        else if (arg == "--port") opt.port = parse_port(next());
        else if (arg == "--ctx" || arg == "--ctx-size" || arg == "-c") opt.ctx = parse_i32(next(), arg.c_str());
        else if (arg == "--batch" || arg == "--batch-size" || arg == "-b") opt.batch = parse_i32(next(), arg.c_str());
        else if (arg == "--threads" || arg == "-t") {
            const std::string value = next();
            opt.threads = value == "auto" ? auto_threads() : parse_i32(value.c_str(), arg.c_str());
        } else if (arg == "--cpu-only") opt.cpu_only = true;
        else if (arg == "--model-id") opt.model_id = next();
        else if (arg == "--log-dir") opt.log_dir = next();
        else if (arg == "--allow-public-bind") opt.allow_public_bind = true;
        else if (arg == "--help" || arg == "-h") return false;
        else if (opt.model.empty()) opt.model = arg;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (opt.threads <= 0) opt.threads = auto_threads();
    if (opt.model.empty()) return false;
    if (!opt.allow_public_bind && !is_local_host(opt.host)) {
        throw std::runtime_error("Iurexa Runtime binds locally by default; use --host 127.0.0.1 or pass --allow-public-bind explicitly.");
    }
    return true;
}

static uint64_t ram_used_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
        return (uint64_t) (pmc.WorkingSetSize / (1024ull * 1024ull));
    }
#endif
    return 0;
}

static std::string base_name(const std::string & path) {
    try {
        return fs::path(path).filename().string();
    } catch (...) {
        return path;
    }
}

static std::string strip_think_blocks(std::string text) {
    const std::string open = "<think>";
    const std::string close = "</think>";
    for (;;) {
        const size_t start = text.find(open);
        if (start == std::string::npos) break;
        const size_t end = text.find(close, start + open.size());
        if (end == std::string::npos) {
            text.erase(start);
            break;
        }
        text.erase(start, end + close.size() - start);
    }
    while (!text.empty() && (text.front() == '\n' || text.front() == '\r' || text.front() == ' ' || text.front() == '\t')) {
        text.erase(text.begin());
    }
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '\t')) {
        text.pop_back();
    }
    return text;
}

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return value;
}

static size_t find_ci(const std::string & haystack, const std::string & needle) {
    return lower_ascii(haystack).find(lower_ascii(needle));
}

class thinking_stream_filter {
public:
    std::string push(const std::string & piece) {
        pending_ += piece;
        std::string out;
        for (;;) {
            if (inside_think_) {
                const size_t close = find_ci(pending_, "</think>");
                if (close == std::string::npos) {
                    pending_.clear();
                    return out;
                }
                pending_.erase(0, close + 8);
                inside_think_ = false;
                continue;
            }

            const size_t open = find_ci(pending_, "<think>");
            if (open == std::string::npos) {
                if (pending_.size() > max_tag_len_) {
                    const size_t emit = pending_.size() - max_tag_len_;
                    out += pending_.substr(0, emit);
                    pending_.erase(0, emit);
                }
                return out;
            }

            out += pending_.substr(0, open);
            pending_.erase(0, open + 7);
            inside_think_ = true;
        }
    }

    std::string finish() {
        if (inside_think_) {
            pending_.clear();
            inside_think_ = false;
            return "";
        }
        std::string out = pending_;
        pending_.clear();
        return strip_think_blocks(out);
    }

private:
    static constexpr size_t max_tag_len_ = 8;
    std::string pending_;
    bool inside_think_ = false;
};

static std::vector<llama_token> tokenize(const llama_model * model, const std::string & text) {
    int32_t n = llama_tokenize(model, text.c_str(), (int32_t) text.size(), nullptr, 0, false, true);
    if (n < 0) n = -n;
    std::vector<llama_token> tokens((size_t) std::max(1, n));
    const int32_t got = llama_tokenize(model, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), false, true);
    if (got < 0) throw std::runtime_error("tokenization failed");
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

class iurexa_engine {
public:
    explicit iurexa_engine(options opt, file_logger & logger)
        : opt_(std::move(opt)), logger_(logger), started_(std::chrono::steady_clock::now()) {}

    ~iurexa_engine() {
        if (loader_.joinable()) loader_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        if (ctx_) llama_free(ctx_);
        if (model_) llama_free_model(model_);
        if (backend_ready_) llama_backend_free();
    }

    void start_loading() {
        loader_ = std::thread([this]() { this->load_model(); });
    }

    json health() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - started_).count();
        json body = {
            {"ok", model_loaded_},
            {"modelLoaded", model_loaded_},
            {"model", base_name(opt_.model)},
            {"backend", "ik_embedded"},
            {"cpuOnly", true},
            {"ctx", opt_.ctx},
            {"threads", opt_.threads},
            {"batch", opt_.batch},
            {"uptimeSeconds", uptime},
            {"ramUsedMb", ram_used_mb()}
        };
        if (loading_) body["loading"] = true;
        if (!load_error_.empty()) body["error"] = load_error_;
        return body;
    }

    bool model_loaded() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return model_loaded_;
    }

    completion_result complete_chat(
        const std::vector<chat_message> & messages,
        int32_t max_tokens,
        const std::function<bool(const std::string &)> & on_token = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!model_loaded_ || !model_ || !ctx_) {
            throw runtime_error_with_code("model_not_loaded", load_error_.empty() ? "Model is not loaded yet." : load_error_, 503);
        }
        const std::string prompt = render_prompt_locked(messages);
        return complete_prompt_locked(prompt, max_tokens, on_token);
    }

private:
    options opt_;
    file_logger & logger_;
    std::chrono::steady_clock::time_point started_;
    mutable std::mutex mutex_;
    std::thread loader_;
    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    bool backend_ready_ = false;
    bool model_loaded_ = false;
    bool loading_ = false;
    std::string load_error_;
    std::vector<llama_token> cached_tokens_;

    void load_model() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            loading_ = true;
            load_error_.clear();
        }

        logger_.log("info", "loading model: " + opt_.model);
        llama_model * model = nullptr;
        llama_context * ctx = nullptr;
        bool backend_ready = false;

        try {
            if (!fs::exists(opt_.model)) {
                throw std::runtime_error("Model file not found: " + opt_.model);
            }
            if (!fs::is_regular_file(opt_.model)) {
                throw std::runtime_error("Model path is not a file: " + opt_.model);
            }

            llama_log_set(llama_log_callback, nullptr);
            llama_backend_init();
            backend_ready = true;

            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 0;
            mparams.use_mmap = false;
            mparams.use_mlock = false;

            model = llama_model_load_from_file(opt_.model.c_str(), mparams);
            if (!model) throw std::runtime_error("Invalid or unsupported GGUF model: " + opt_.model);

            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = (uint32_t) opt_.ctx;
            cparams.n_batch = (uint32_t) opt_.batch;
            cparams.n_ubatch = (uint32_t) opt_.batch;
            cparams.n_threads = (uint32_t) opt_.threads;
            cparams.n_threads_batch = (uint32_t) opt_.threads;
            cparams.offload_kqv = false;

            ctx = llama_init_from_model(model, cparams);
            if (!ctx) throw std::runtime_error("Failed to create llama context for model: " + opt_.model);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                model_ = model;
                ctx_ = ctx;
                backend_ready_ = backend_ready;
                model_loaded_ = true;
                loading_ = false;
            }
            logger_.log("info", "model loaded: " + opt_.model);
        } catch (const std::exception & error) {
            if (ctx) llama_free(ctx);
            if (model) llama_free_model(model);
            if (backend_ready) llama_backend_free();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                backend_ready_ = false;
                model_loaded_ = false;
                loading_ = false;
                load_error_ = error.what();
            }
            logger_.log("error", error.what());
        }
    }

    std::string render_prompt_locked(const std::vector<chat_message> & messages) const {
        std::vector<std::string> roles;
        std::vector<std::string> contents;
        roles.reserve(messages.size());
        contents.reserve(messages.size());
        std::vector<llama_chat_message> llama_messages;
        llama_messages.reserve(messages.size());
        for (const auto & msg : messages) {
            roles.push_back(msg.role);
            contents.push_back(msg.content);
        }
        for (size_t i = 0; i < messages.size(); ++i) {
            llama_messages.push_back({ roles[i].c_str(), contents[i].c_str() });
        }

        const char * tmpl = llama_model_chat_template(model_, nullptr);
        int32_t n = llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), true, nullptr, 0);
        if (n > 0) {
            std::vector<char> buffer((size_t) n + 1u);
            const int32_t got = llama_chat_apply_template(tmpl, llama_messages.data(), llama_messages.size(), true, buffer.data(), n + 1);
            if (got > 0) return std::string(buffer.data(), (size_t) got);
        }

        std::string prompt;
        for (const auto & msg : messages) {
            prompt += "<|im_start|>";
            prompt += msg.role;
            prompt += "\n";
            prompt += msg.content;
            prompt += "<|im_end|>\n";
        }
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }

    completion_result complete_prompt_locked(
        const std::string & prompt,
        int32_t max_tokens,
        const std::function<bool(const std::string &)> & on_token) {
        max_tokens = std::max<int32_t>(1, max_tokens);
        std::vector<llama_token> prompt_tokens = tokenize(model_, prompt);
        if (prompt_tokens.empty()) {
            throw runtime_error_with_code("invalid_prompt", "Prompt tokenization produced no tokens.", 400);
        }
        if ((int32_t) prompt_tokens.size() + max_tokens + 1 > (int32_t) llama_n_ctx(ctx_)) {
            clear_cache_locked();
            throw runtime_error_with_code("context_overflow", "Prompt plus requested completion exceeds runtime context.", 400);
        }

        size_t common = common_prefix(prompt_tokens, cached_tokens_);
        if (common == prompt_tokens.size() && common > 0) common -= 1;
        trim_cache_locked(common);

        std::vector<llama_token> suffix(prompt_tokens.begin() + (ptrdiff_t) cached_tokens_.size(), prompt_tokens.end());
        int32_t pos = (int32_t) cached_tokens_.size();
        if (!suffix.empty()) {
            pos = decode_tokens(ctx_, suffix, pos, opt_.batch);
            if (pos < 0) throw runtime_error_with_code("decode_failed", "Prompt decode failed.", 500);
            cached_tokens_.insert(cached_tokens_.end(), suffix.begin(), suffix.end());
        }

        completion_result result;
        result.prompt_tokens = (int32_t) prompt_tokens.size();

        for (int32_t i = 0; i < max_tokens; ++i) {
            llama_token next = sample_greedy(ctx_, model_);
            if (llama_token_is_eog(model_, next)) break;

            std::string piece = token_piece(model_, next);
            result.text += piece;
            if (on_token && !piece.empty()) {
                if (!on_token(piece)) {
                    result.finish_reason = "stop";
                    break;
                }
            }

            std::vector<llama_token> one = { next };
            pos = decode_tokens(ctx_, one, (int32_t) cached_tokens_.size(), 1);
            if (pos < 0) throw runtime_error_with_code("decode_failed", "Token decode failed.", 500);
            cached_tokens_.push_back(next);
            result.completion_tokens += 1;
        }

        result.text = strip_think_blocks(result.text);
        return result;
    }

    static size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && a[i] == b[i]) ++i;
        return i;
    }

    void clear_cache_locked() {
        if (ctx_) llama_kv_cache_clear(ctx_);
        cached_tokens_.clear();
    }

    void trim_cache_locked(size_t keep) {
        keep = std::min(keep, cached_tokens_.size());
        if (keep == cached_tokens_.size()) return;
        if (!llama_kv_cache_seq_rm(ctx_, 0, (llama_pos) keep, -1)) {
            clear_cache_locked();
            return;
        }
        cached_tokens_.resize(keep);
    }
};

static std::string json_content(const json & value) {
    return value.dump();
}

static void set_json(httplib::Response & res, const json & body, int status = 200) {
    res.status = status;
    res.set_header("Cache-Control", "no-store");
    res.set_content(json_content(body), "application/json; charset=utf-8");
}

static json error_body(const std::string & code, const std::string & message) {
    return {
        {"error", {
            {"message", message},
            {"type", "iurexa_runtime_error"},
            {"code", code}
        }}
    };
}

static void set_error(httplib::Response & res, const std::string & code, const std::string & message, int status) {
    set_json(res, error_body(code, message), status);
}

static std::string content_to_string(const json & content) {
    if (content.is_string()) return content.get<std::string>();
    if (content.is_array()) {
        std::string out;
        for (const auto & part : content) {
            if (part.is_object() && part.value("type", "") == "text" && part.contains("text")) {
                if (!out.empty()) out += "\n";
                out += part["text"].get<std::string>();
            }
        }
        if (!out.empty()) return out;
    }
    return content.dump();
}

static std::vector<chat_message> parse_messages(const json & body) {
    if (!body.contains("messages") || !body["messages"].is_array()) {
        throw runtime_error_with_code("invalid_request", "messages must be an array.", 400);
    }
    std::vector<chat_message> messages;
    for (const auto & item : body["messages"]) {
        if (!item.is_object()) continue;
        const std::string role = item.value("role", "");
        if (role != "system" && role != "user" && role != "assistant") {
            throw runtime_error_with_code("invalid_request", "message role must be system, user, or assistant.", 400);
        }
        if (!item.contains("content")) {
            throw runtime_error_with_code("invalid_request", "message content is required.", 400);
        }
        messages.push_back({ role, content_to_string(item["content"]) });
    }
    if (messages.empty()) {
        throw runtime_error_with_code("invalid_request", "messages must contain at least one supported message.", 400);
    }
    return messages;
}

static bool wants_thinking(const json & body) {
    if (body.contains("think") && body["think"].is_boolean() && body["think"].get<bool>()) return true;
    return body.contains("reasoning") || body.contains("reasoning_effort");
}

static void add_no_think_directive(std::vector<chat_message> & messages, bool allow_thinking) {
    if (allow_thinking) return;
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role != "user") continue;
        const std::string lowered = lower_ascii(it->content);
        if (lowered.find("/think") == std::string::npos && lowered.find("/no_think") == std::string::npos) {
            if (!it->content.empty() && it->content.back() != ' ' && it->content.back() != '\n') it->content += " ";
            it->content += "/no_think";
        }
        return;
    }
}

static std::string make_id() {
    static std::atomic<uint64_t> counter{0};
    std::ostringstream out;
    out << "chatcmpl-local-" << unix_time_seconds() << "-" << counter.fetch_add(1);
    return out.str();
}

static json completion_response(const std::string & id, const std::string & model_id, const completion_result & result) {
    return {
        {"id", id},
        {"object", "chat.completion"},
        {"created", unix_time_seconds()},
        {"model", model_id},
        {"choices", json::array({
            {
                {"index", 0},
                {"message", {
                    {"role", "assistant"},
                    {"content", result.text}
                }},
                {"finish_reason", result.finish_reason}
            }
        })},
        {"usage", {
            {"prompt_tokens", result.prompt_tokens},
            {"completion_tokens", result.completion_tokens},
            {"total_tokens", result.prompt_tokens + result.completion_tokens}
        }}
    };
}

static std::string sse_line(const json & payload) {
    return "data: " + payload.dump() + "\n\n";
}

static json stream_delta(const std::string & id, const std::string & model_id, const json & delta, const json & finish_reason) {
    return {
        {"id", id},
        {"object", "chat.completion.chunk"},
        {"created", unix_time_seconds()},
        {"model", model_id},
        {"choices", json::array({
            {
                {"index", 0},
                {"delta", delta},
                {"finish_reason", finish_reason}
            }
        })}
    };
}

static httplib::Server * g_server = nullptr;

static void signal_stop(int) {
    if (g_server) g_server->stop();
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

    file_logger logger(opt.log_dir);
    g_logger = &logger;
    logger.log("info", "Iurexa Runtime starting");
    logger.log("info", "log file: " + logger.path().string());

    iurexa_engine engine(opt, logger);
    engine.start_loading();

    httplib::Server server;
    g_server = &server;
    std::signal(SIGINT, signal_stop);
    std::signal(SIGTERM, signal_stop);

    server.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Headers", "content-type, authorization");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.status = 204;
    });

    server.Get("/health", [&engine](const httplib::Request &, httplib::Response & res) {
        set_json(res, engine.health());
    });

    server.Get("/v1/models", [&opt](const httplib::Request &, httplib::Response & res) {
        set_json(res, {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id", opt.model_id},
                    {"object", "model"},
                    {"owned_by", "magistra"}
                }
            })}
        });
    });

    server.Post("/v1/chat/completions", [&engine, &opt, &logger](const httplib::Request & req, httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (!engine.model_loaded()) {
            const json health = engine.health();
            const std::string message = health.value("error", "Model is still loading.");
            set_error(res, "model_not_loaded", message, 503);
            return;
        }

        json body;
        std::vector<chat_message> messages;
        int32_t max_tokens = 512;
        bool stream = false;
        try {
            body = json::parse(req.body);
            messages = parse_messages(body);
            add_no_think_directive(messages, wants_thinking(body));
            max_tokens = body.value("max_tokens", 512);
            if (max_tokens <= 0) max_tokens = 512;
            stream = body.value("stream", false);
        } catch (const runtime_error_with_code & error) {
            set_error(res, error.code(), error.what(), error.status());
            return;
        } catch (const std::exception & error) {
            set_error(res, "invalid_json", error.what(), 400);
            return;
        }

        const std::string id = make_id();
        if (!stream) {
            try {
                completion_result result = engine.complete_chat(messages, max_tokens);
                set_json(res, completion_response(id, opt.model_id, result));
            } catch (const runtime_error_with_code & error) {
                logger.log("error", error.what());
                set_error(res, error.code(), error.what(), error.status());
            } catch (const std::exception & error) {
                logger.log("error", error.what());
                set_error(res, "generation_failed", error.what(), 500);
            }
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        auto state = std::make_shared<std::tuple<std::vector<chat_message>, int32_t, std::string>>(messages, max_tokens, id);
        res.set_chunked_content_provider("text/event-stream", [&engine, &opt, &logger, state](size_t, httplib::DataSink & sink) {
            auto & messages_ref = std::get<0>(*state);
            const int32_t max_tokens_ref = std::get<1>(*state);
            const std::string id_ref = std::get<2>(*state);
            thinking_stream_filter filter;
            auto write = [&sink](const std::string & data) {
                return sink.write(data.data(), data.size());
            };

            bool writable = true;
            try {
                write(sse_line(stream_delta(id_ref, opt.model_id, {{"role", "assistant"}}, nullptr)));
                completion_result result = engine.complete_chat(messages_ref, max_tokens_ref, [&](const std::string & piece) {
                    const std::string visible = filter.push(piece);
                    if (visible.empty()) return true;
                    const std::string chunk = sse_line(stream_delta(id_ref, opt.model_id, {{"content", visible}}, nullptr));
                    if (sink.is_writable && !sink.is_writable()) return false;
                    writable = write(chunk);
                    return writable;
                });
                if (writable) {
                    const std::string tail = filter.finish();
                    if (!tail.empty()) {
                        write(sse_line(stream_delta(id_ref, opt.model_id, {{"content", tail}}, nullptr)));
                    }
                    write(sse_line(stream_delta(id_ref, opt.model_id, json::object(), result.finish_reason)));
                    write("data: [DONE]\n\n");
                }
            } catch (const runtime_error_with_code & error) {
                logger.log("error", error.what());
                write("data: " + error_body(error.code(), error.what()).dump() + "\n\n");
                write("data: [DONE]\n\n");
            } catch (const std::exception & error) {
                logger.log("error", error.what());
                write("data: " + error_body("generation_failed", error.what()).dump() + "\n\n");
                write("data: [DONE]\n\n");
            }
            sink.done();
            return false;
        });
    });

    server.Post("/shutdown", [&logger, &server](const httplib::Request &, httplib::Response & res) {
        logger.log("info", "shutdown requested");
        set_json(res, {{"ok", true}, {"shutdown", true}});
        std::thread([&server]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            server.stop();
        }).detach();
    });

    server.set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            set_error(res, "not_found", "Endpoint not found.", 404);
        }
    });

    logger.log("info", "listening on " + opt.host + ":" + std::to_string(opt.port));
    const bool ok = server.listen(opt.host, opt.port);
    if (!ok) {
        logger.log("error", "failed to listen on " + opt.host + ":" + std::to_string(opt.port));
        return 1;
    }

    logger.log("info", "Iurexa Runtime stopped");
    g_server = nullptr;
    return 0;
}
