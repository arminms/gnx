---
title: 2-Bit Packed Sequences
description: Memory-efficient nucleotide storage with gnx::packed_generic_sequence_2bit
---

# 2-Bit Packed Sequences

`gnx::packed_generic_sequence_2bit` (aliased as `gnx::psq2`) stores DNA sequences
using only 2 bits per nucleotide (A=`00`, C=`01`, G=`10`, T=`11`), packing four
bases into each byte.  Compared with one byte per character in `gnx::sq`, this
gives a **4× memory reduction** — critical for whole-genome analysis and
GPU-resident data.

## Bit Layout

Within each byte, bases are stored **MSB-first**:

```
byte[i] = [base 4i | base 4i+1 | base 4i+2 | base 4i+3]
            bits 7-6     5-4         3-2         1-0
```

So the four-base sequence `ACGT` encodes to the single byte `0b00_01_10_11 = 0x1B`.

## Headers and Setup

```cpp
#define FMT_HEADER_ONLY
#include <gnx/psq.hpp>   // brings in gnx::psq2 and gnx::packed_generic_sequence_2bit
```

## Basic Usage

### Creating Sequences

```cpp
// From a string literal
gnx::psq2 s{"ACGTACGT"};

// Using the _psq2 UDL
auto t = "ACGTACGT"_psq2;

// Fill constructor: 8 cytosines
gnx::psq2 c8(8, 'C');          // "CCCCCCCC"

// Count constructor (fills with 'A')
gnx::psq2 a5(5);               // "AAAAA"

// From a pair of iterators (any char-convertible range)
std::string raw = "TTGCAA";
gnx::psq2 u(raw.begin(), raw.end());

// From initializer list
gnx::psq2 v{'A', 'C', 'G', 'T'};
```

### Interoperability with `gnx::sq`

`gnx::psq2` and `gnx::sq` (and any `gnx::generic_sequence`) can be freely converted;
tagged metadata is copied in both directions.

```cpp
// generic_sequence → psq2
gnx::sq src("ACGTACGT"_sq);
src["_id"] = std::string("chr1");
gnx::psq2 packed(src);          // sequence + tagged data copied

// psq2 → generic_sequence
auto back = packed.to_sq();     // Default target: gnx::sq
assert(back == "ACGTACGT");
assert(std::any_cast<std::string>(back["_id"]) == "chr1");

// Cross-comparison works directly
assert(packed == src);
assert(src    == packed);
```

## Element Access

Bases are accessed through **proxy objects** that transparently pack/unpack
the 2-bit representation. The proxy is implicitly convertible to `char`.

```cpp
gnx::psq2 s("ACGT");

// Read
char b0 = s[0];          // 'A'
char b1 = char(s[1]);    // 'C' — explicit cast also works
char b2 = s.get_base(2); // 'G' — named accessor

// Write
s[3] = 'A';              // T → A
assert(s == "ACGA");

// Bounds-checked access
char b = char(s.at(0));
s.at(10) = 'A';          // throws std::out_of_range
```

## Iterators

All iterator categories expected of a random-access container are provided.

```cpp
gnx::psq2 s("ACGTACGT");

// Range-based for loop
for (char c : s)
    fmt::print("{}", c);       // prints ACGTACGT

// Mutable iteration
for (auto it = s.begin(); it != s.end(); ++it)
    *it = 'A';                 // fills with 'A'

// Reverse iteration
std::string rev;
for (auto it = s.crbegin(); it != s.crend(); ++it)
    rev += char(*it);

// Random-access arithmetic
auto it = s.begin();
char third = char(*(it + 2));  // 'G'
auto dist  = s.end() - s.begin(); // == 8
```

## Capacity

```cpp
gnx::psq2 s("ACGTACGTA"); // 9 bases

s.size();       // 9        — number of bases
s.byte_size();  // 3        — packed bytes used (⌈9/4⌉)
s.empty();      // false
s.size_in_memory(); // bytes occupied by container + tagged data (rough estimate)
```

## Tagged Metadata

Tagged metadata works identically to `gnx::sq`:

```cpp
gnx::psq2 s("ACGT");

s["_id"]   = std::string("read-42");
s["score"] = 3.14;

assert(s.has("_id"));
assert(!s.has("missing"));

auto id = std::any_cast<std::string>(s["_id"]);

// Const access throws for missing tags
const gnx::psq2& cs = s;
cs["missing"];  // throws std::out_of_range
```

## Comparison

```cpp
gnx::psq2 a("ACGT"), b("ACGT"), c("TTTT");
gnx::sq   sq_a("ACGT");

a == b;              // true
a != c;              // true
a == "ACGT";         // true  (string_view overload)
"ACGT" == a;         // true  (symmetric)
a == sq_a;           // true  (cross-type)
sq_a == a;           // true  (symmetric)
```

## Serialisation

The stream operators match the `gnx::sq` binary format (size + raw packed bytes
+ optional tagged sections), enabling mixed workflows:

```cpp
gnx::psq2 s("ACGTACGT");
s["_id"] = std::string("seq1");

// Serialize
std::stringstream ss;
ss << s;

// Deserialize
gnx::psq2 loaded;
ss >> loaded;

assert(loaded == "ACGTACGT");
assert(std::any_cast<std::string>(loaded["_id"]) == "seq1");
```

## Memory Savings

For a 3-billion-base human genome, storing bases as `char` costs ≈3 GB; 2-bit
packing reduces that to ≈750 MB, fitting comfortably in most GPU device memories.

```cpp
const std::size_t genome_len = 3'000'000'000UZ;

// gnx::sq: 1 byte/base
auto sq_mem  = genome_len;                   // 3 000 000 000 bytes ≈ 2.8 GiB

// gnx::psq2: 2 bits/base = 1 byte per 4 bases
auto psq_mem = gnx::psq2::num_bytes(genome_len);  //  750 000 000 bytes ≈ 715 MiB

fmt::print("sq  memory : {} MiB\n", sq_mem  / (1024*1024));
fmt::print("psq memory : {} MiB\n", psq_mem / (1024*1024));
```

## Template Parameters

```cpp
template
<   typename ByteContainer = std::vector<uint8_t>
,   typename Map = std::unordered_map<std::string, std::any>
>
class packed_generic_sequence_2bit;
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ByteContainer` | `std::vector<uint8_t>` | Stores the packed bytes; `value_type` must be `uint8_t` |
| `Map` | `std::unordered_map<std::string, std::any>` | Metadata map; lazily allocated via `unique_ptr` |

## Static Helper Functions

| Function | Description |
|----------|-------------|
| `encode(char c)` | Maps `A/a→0`, `C/c→1`, `G/g→2`, `T/t→3`; unknown chars map to `0` |
| `decode(uint8_t bits)` | Maps `0→'A'`, `1→'C'`, `2→'G'`, `3→'T'` |
| `num_bytes(size_type n)` | Returns `⌈n/4⌉` — bytes needed for `n` bases |

## Limitations

- Only the four canonical nucleotides `A`, `C`, `G`, `T` (and their lowercase
  equivalents) are preserved.  IUPAC ambiguity codes (e.g. `N`, `R`) are
  silently mapped to `A`.
- There is no GPU (CUDA/ROCm) specialisation yet; use `gnx::sq` with
  `thrust::device_vector` for GPU kernels.
