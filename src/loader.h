// Minimal GGUF v3 reader. Parses header, metadata key/values, and the tensor
// directory; exposes typed metadata getters and tensor data pointers. Host-only
// (no CUDA); used to feed real Q8_0 weights + config into the runtime. PLAN sec 8.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

// GGUF metadata value types (subset we store; arrays are parsed but not retained).
enum GgufValueType {
    GGUF_U8 = 0, GGUF_I8 = 1, GGUF_U16 = 2, GGUF_I16 = 3, GGUF_U32 = 4,
    GGUF_I32 = 5, GGUF_F32 = 6, GGUF_BOOL = 7, GGUF_STRING = 8, GGUF_ARRAY = 9,
    GGUF_U64 = 10, GGUF_I64 = 11, GGUF_F64 = 12
};

// ggml tensor types we care about (values match the ggml enum).
enum GgmlType { GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q5_0 = 6,
                GGML_Q8_0 = 8, GGML_Q6_K = 14 };

struct MetaValue {
    int type = 0;        // GgufValueType
    int64_t inum = 0;    // for integer/bool types
    double fnum = 0.0;   // for float types
    std::string str;     // for string type
};

struct TensorInfo {
    std::string name;
    uint32_t ggml_type = 0;
    std::vector<uint64_t> dims;
    uint64_t offset = 0;       // relative to the data section start
};

struct GgufFile {
    uint32_t version = 0;
    std::vector<uint8_t> bytes;                          // whole file, owned
    std::unordered_map<std::string, MetaValue> meta;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, int> tensor_index;   // name -> index in tensors
    size_t data_offset = 0;                              // start of tensor data in bytes
};

// Load + parse a GGUF file. Returns false and sets *err on failure.
bool gguf_load(const char* path, GgufFile* out, std::string* err);

// Typed metadata getters. Return false if the key is missing or the wrong type.
bool gguf_get_u32(const GgufFile& g, const std::string& key, uint32_t* v);
bool gguf_get_f32(const GgufFile& g, const std::string& key, float* v);
bool gguf_get_str(const GgufFile& g, const std::string& key, std::string* v);

// Find a tensor by name (null if absent).
const TensorInfo* gguf_find_tensor(const GgufFile& g, const std::string& name);

// Pointer to a tensor's raw data within the file bytes.
const uint8_t* gguf_tensor_data(const GgufFile& g, const TensorInfo& t);

// Number of elements and byte size of a tensor (byte size for supported types).
uint64_t gguf_tensor_elements(const TensorInfo& t);
uint64_t gguf_tensor_bytes(const TensorInfo& t);
