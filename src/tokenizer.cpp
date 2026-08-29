#include "tokenizer.h"

#ifdef SEMILLM_HAVE_LLAMA
#include "llama.h"

bool tokenizer_load(const char* path, Tokenizer* t, std::string* err) {
    static bool inited = false;
    if (!inited) { llama_backend_init(); inited = true; }
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;      // load ONLY the vocab — no weights, no GPU alloc
    mp.n_gpu_layers = 0;
    t->model = llama_model_load_from_file(path, mp);
    if (!t->model) { if (err) *err = "llama_model_load_from_file (vocab_only) failed"; return false; }
    t->vocab = llama_model_get_vocab(t->model);
    t->ok = t->vocab != nullptr;
    if (!t->ok && err) *err = "no vocab in model";
    return t->ok;
}

std::vector<int> tokenizer_encode(const Tokenizer& t, const std::string& text, bool add_special) {
    if (!t.ok) return {};
    int cap = (int)text.size() + 8;
    std::vector<int> out(cap);
    int n = llama_tokenize(t.vocab, text.c_str(), (int)text.size(), out.data(), cap, add_special, true);
    if (n < 0) {   // buffer too small: -n is the required size
        out.resize(-n);
        n = llama_tokenize(t.vocab, text.c_str(), (int)text.size(), out.data(), (int)out.size(), add_special, true);
    }
    out.resize(n > 0 ? n : 0);
    return out;
}

std::string tokenizer_piece(const Tokenizer& t, int id) {
    if (!t.ok) return "";
    char buf[512];
    int n = llama_token_to_piece(t.vocab, id, buf, (int)sizeof(buf), 0, false);
    return n > 0 ? std::string(buf, n) : "";
}

void tokenizer_free(Tokenizer* t) {
    if (t->model) { llama_model_free(t->model); t->model = nullptr; t->vocab = nullptr; t->ok = false; }
}

#else  // built without llama.cpp

bool tokenizer_load(const char*, Tokenizer*, std::string* err) {
    if (err) *err = "tokenizer needs llama.cpp: reconfigure with -DSEMILLM_LLAMA_DIR=<built llama.cpp>";
    return false;
}
std::vector<int> tokenizer_encode(const Tokenizer&, const std::string&, bool) { return {}; }
std::string tokenizer_piece(const Tokenizer&, int) { return ""; }
void tokenizer_free(Tokenizer*) {}

#endif
