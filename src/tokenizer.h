// Tokenizer: thin wrapper over llama.cpp's vocab (libllama), loaded VOCAB-ONLY from the
// GGUF (no weights) so it stays off FloatyLLM's streaming path. Gives exact, maintained
// encode/decode for every tokenizer llama.cpp supports (SPM, BPE, ...). Built only when
// configured with -DSEMILLM_LLAMA_DIR=<path to a built llama.cpp>; otherwise the calls
// return an error so the rest of the project still builds (e.g. CI without llama.cpp).
#pragma once
#include <string>
#include <vector>
#include <utility>

struct llama_model;
struct llama_vocab;

struct Tokenizer {
    llama_model* model = nullptr;        // vocab-only model handle (owns the vocab)
    const llama_vocab* vocab = nullptr;
    bool ok = false;
};

// Load the vocab from a GGUF (first shard for split models). vocab_only => no weights.
bool tokenizer_load(const char* gguf_path, Tokenizer* t, std::string* err);

// Encode text -> token ids. add_special adds BOS/EOS per the model's config (matches
// `llama-tokenize` defaults).
std::vector<int> tokenizer_encode(const Tokenizer& t, const std::string& text, bool add_special = true);

// One token id -> its text piece (special tokens rendered as empty by default).
std::string tokenizer_piece(const Tokenizer& t, int id);

int  tokenizer_bos(const Tokenizer& t);
int  tokenizer_eos(const Tokenizer& t);
bool tokenizer_is_eog(const Tokenizer& t, int id);   // end-of-generation (EOS or variants)

// Render conversation messages with the model's built-in chat template. Each message is
// (role, content) with role in {"system","user","assistant"}. add_assistant appends the
// assistant header so the model continues as the assistant. Empty string on failure.
std::string tokenizer_apply_chat(const Tokenizer& t,
                                 const std::vector<std::pair<std::string, std::string>>& msgs,
                                 bool add_assistant);

void tokenizer_free(Tokenizer* t);
