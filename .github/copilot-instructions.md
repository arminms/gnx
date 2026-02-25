# Copilot Instructions for gnx

Purpose: Make AI agents immediately productive in this Modern C++ header-only library for biological sequences.

## Big Picture
- **Library type:** Header-only CMake `INTERFACE` target `gnx::gnx` exposing headers under `include/gnx/**`. No compiled sources.
- **Language:** C++20 (utilize Concepts, Ranges, and `std::span` heavily).
- **Parallelism:** Support heterogeneous computing. Code should be compatible with:
  - **NVIDIA CUDA** (via `nvcc`)
  - **AMD ROCm** (via `hipcc`)
  - **OpenMP** (for CPU parallelism)
- **SIMD:** Prioritize explicit SIMD optimizations (AVX2, AVX-512) or portable wrappers where applicable.
- **Core model:** `gnx::generic_sequence<Container>` (aliased as `gnx::sq` for `std::vector<char>`) represents a sequence plus a map of tagged metadata (`std::unordered_map<std::string, std::any>`).
- **I/O format:** Extensible visitors in `gnx/visitor.hpp` serialize/deserialize tagged metadata via two registries: `td_print_visitor` (type-index keyed, using `fmt::memory_buffer`) and `td_scan_visitor` (string keyed). Add types by registering handlers.
- **FASTA/Q gz input:** `gnx/io/fastaqz.hpp` provides `gnx::in::faqz` which loads a single record by `id` from `.fa.gz/.fq.gz` or stdin using `zlib` + `kseq.h` (embedded in `include/gnx/io/kseq.h`).
- **Execution policies:** `gnx/execution.hpp` provides execution policies (`seq`, `par`, `unseq`, `par_unseq`, `cuda`, `rocm`, `oneapi`) for controlling parallelization strategy.
- **Memory utilities:** `gnx/memory.hpp` provides allocators and utilities for heterogeneous memory (host, device, pinned, unified).
- **External deps:** `ZLIB::ZLIB`, `fmt::fmt-header-only` (v11.0.2+), `g3p::g3p`, and `ranx::openmp` (all fetched if not found). `Catch2` and `Google Benchmark` fetched for tests/benchmarks.

## Coding Style & Conventions
- **Naming:** Follow standard C++ library conventions (snake_case for functions/variables, PascalCase for Template Concepts).
- **Output:** Use `fmt::print()` for all console output and `fmt::format()` for string formatting. The library uses fmt (header-only) for modern, type-safe formatting.
- **Memory Management:**
  - Prioritize **Zero-Copy** operations. Use `std::string_view` or `std::span` over `std::string` or `std::vector` when parsing sequences (FASTA/FASTQ).
  - Avoid implicit allocations in hot loops.
- **Safety:** Use `[[nodiscard]]` for pure functions.
- **Error Handling:** Prefer compile-time errors (`static_assert`, `concepts`) over runtime exceptions. Use `fmt::format()` for exception messages.
- **Execution Policies:**
  - Algorithms should provide both policy-free and policy-accepting overloads.
  - Use `requires gnx::is_execution_policy_v<std::decay_t<ExecPolicy>>` to constrain policy-accepting overloads.
  - Implement compile-time dispatch with `if constexpr` to select CPU (seq/par/unseq/par_unseq) vs GPU (cuda/rocm/oneapi) backends.
  - GPU implementations go in `detail::algorithm_device()` with `#if defined(__CUDACC__) || defined(__HIPCC__)` guards.

## GPU & HPC Constraints
- **Device Compatibility:** When writing kernels, ensure code is portable between CUDA and HIP (e.g., use `__host__ __device__` qualifiers).
- **Memory Coalescing:** Ensure data structures are Struct-of-Arrays (SoA) friendly for GPU access.
- **Warp Divergence:** Avoid branchy code in kernels; use branchless programming techniques where possible.

## Documentation
- Use Doxygen-style comments `///` for all public APIs.
- Put documentation and examples in `doc/` in [MyST Markdown](https://mystmd.org/) format.
- Use `doc/quickstart.md` as a template for a comprehensive quickstart guide with code snippets and explanations and add usage for newly defined classes and algorithms to this file.

## Build & Install
- **Build System:** CMake (3.25+).
- **CMake target:** Declared in [CMakeLists.txt](CMakeLists.txt). Uses `INTERFACE` include directories and links `ZLIB::ZLIB` + `g3p::g3p`.
- **Out-of-source build required:** CMake errors for in-source builds.
- **Build (library consumers):** The library has no sources to compile; configure to generate and install headers + package config.
- **Install headers:** [include/CMakeLists.txt](include/CMakeLists.txt) installs `include/gnx/**` to `${CMAKE_INSTALL_INCLUDEDIR}`.
- **Package config:** Generated files `${project}-config.cmake`, `${project}-targets.cmake` under `${CMAKE_INSTALL_LIBDIR}/cmake/gnx` enable `find_package(gnx CONFIG)` in consumer projects.

Example local build + install:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix $HOME/.local
```

## Tests
- **Switch:** `GNX_ENABLE_TESTS` (default ON). Disabled when used as a subproject (`NOT_SUBPROJECT` logic).
- **Framework:** `Catch2` v3 (fetched if missing). See [test/CMakeLists.txt](test/CMakeLists.txt) and [test/unit_tests.cpp](test/unit_tests.cpp).
- **Run:**
```bash
cmake -S . -B build -DGNX_ENABLE_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Benchmarks
- **Switch:** `GNX_ENABLE_BENCHMARKS` (default ON). Disabled when used as a subproject (`NOT_SUBPROJECT` logic).
- **Framework:** `Google Benchmark` v1.7 (fetched if missing). See [perf/CMakeLists.txt](perf/CMakeLists.txt) and [perf/benchmarks.cpp](perf/benchmarks.cpp).
- **Run:**
```bash
cmake -S . -B build -DGNX_ENABLE_BENCHMARKS=ON
cmake --build build
build/perf/benchmarks --benchmark_counters_tabular=true
```

## Key Types & Patterns
- **`gnx::generic_sequence<Container>`:**
  - Sequence storage via `Container` (e.g., `std::vector<char>`); metadata via `_td` tagged data.
  - Subsequence via `operator()(pos, count)`; random access via `operator[]`.
  - I/O: `print(std::ostream&)` and `scan(std::istream&)` encode size, raw sequence, then tagged items `#<tag>|<type>...`.
  - `load(filename, id, Format)` calls a format functor, e.g., `gnx::in::faqz{}`.
- **`gnx::sq_view<Range>`:**
  - Non-owning view over sequence ranges; compatible with C++20 ranges.
  - Zero-copy subsequences and transformations.
- **`gnx::sequence_bank<InterfaceType>`:**
  - Collection wrapper (aliased as `gnx::sqb`) for managing multiple sequences.
  - Abstracts underlying interface (file, stream, database) as a range.
- **Execution Policies (`gnx/execution.hpp`):**
  - CPU: `gnx::execution::seq`, `par`, `unseq`, `par_unseq`
  - GPU: `gnx::execution::cuda`, `rocm`, `oneapi`
  - Use `is_execution_policy_v<T>` trait to constrain algorithm templates.
- **Memory Management (`gnx/memory.hpp`):**
  - Custom allocators for GPU memory: `host_pinned_allocator`, `device_allocator`, `universal_allocator`.
  - Thrust integration for unified memory management across CUDA/ROCm.
- **Lookup Tables (`gnx/lut/*.hpp`):**
  - Substitution matrices: `blosum.hpp`, `pam.hpp`
  - Quality scores: `phred33.hpp`, `phred64.hpp`
  - Sequence validation: `valid.hpp`
- **Visitors (`gnx/visitor.hpp`):**
  - Printing: `td_print_visitor` maps `std::type_index` → printer using `fmt::memory_buffer`. Register with `register_td_print_visitor<T>(lambda)`.
  - Scanning: `td_scan_visitor` maps type-name strings → scanner. Register with `register_td_scan_visitor(type_name, lambda)`.
  - Built-in handlers for `void, bool, int, unsigned, float, double, string, std::vector<int>`.
  - Use `quote_with_delimiter(buf, str, delim)` helper for custom type formatting.
- **FASTAQZ Loader (`gnx/io/fastaqz.hpp`):**
  - `faqz_gen<Container>::operator()` opens gz file/stdin, iterates `kseq_read`, matches `name == id`, fills `_id`, optional `_qs` (quality), `_desc` (comment), and assigns to sequence.
  - Returns `true` if record was found (`s.has("_id")`).
- **Algorithms (`gnx/algorithms/*.hpp`):**
  - `compare.hpp`: Sequence comparison and similarity metrics.
  - `local_align.hpp`: Smith-Waterman local alignment with scoring matrices.
  - `random.hpp`: Random sequence generation using `ranx` RNG library.
  - `valid.hpp`: Sequence validation against alphabet constraints.
  - All algorithms support execution policies for multi-backend parallelism.
  - Pattern: Provide policy-free overload (auto-detects host/device) + policy-accepting overload.

## Conventions & Gotchas
- **Header-only:** Do not add `.cpp` sources; prefer templates or inline implementations in headers.
- **Metadata tags:** Core tags used: `_id`, `_qs` (quality), `_desc` (comment). Maintain these names for interoperability.
- **Visitor symmetry:** When adding a printer for a new type, also add a scanner entry using the exact type-name string produced by the printer.
- **String literal helper:** Global `operator""_sq` creates `gnx::sq` from string literals.
- **Dependencies:**
  - `fmt` v11.0.2+: Modern C++ formatting library (header-only mode).
  - `g3p` v1.2.0: Modern C++ interface library for Gnuplot with Jupyter support.
  - `ranx`: Modern parallel random number generation library (OpenMP component required).
  - Avoid changing dependency versions unless necessary.
- **Thrust vectors:** When GPU support is enabled, `gnx::sq` can use `thrust::device_vector`, `thrust::universal_vector`, or `thrust::host_vector` as container types for heterogeneous computing.

## Usage Examples (from this project)
- Create a sequence and tag:
```cpp
#include <gnx/sq.hpp>
using gnx::sq;
sq s = "ACGT"_sq; s[0] = 'N'; s["_id"] = std::string("read-1");
```
- Load one record from gz FASTA/FASTQ:
```cpp
#include <gnx/sq.hpp>
#include <gnx/io/fastaqz.hpp>
using gnx::sq; using gnx::in::faqz;
sq s; bool ok = s.load("reads.fq.gz", "read-42", faqz{});
```
- Use execution policies with algorithms:
```cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/compare.hpp>
using gnx::sq; using gnx::execution::par;
sq s1 = "ACGTACGT"_sq, s2 = "ACGGACGT"_sq;
auto result = gnx::compare(par, s1, s2);
```
- Extend visitors for a custom type:
```cpp
#include <gnx/visitor.hpp>
struct MyTag { int a; };
register_td_print_visitor<MyTag>([](fmt::memory_buffer& buf, const MyTag& t){ 
    quote_with_delimiter(buf, "MyTag"); 
    fmt::format_to(std::back_inserter(buf), "{}", t.a); 
});
register_td_scan_visitor("MyTag", [](std::istream& is, std::any& a){ int x; is >> x; a = MyTag{x}; });
```
- Output sequences using fmt:
```cpp
#include <fmt/core.h>
#include <gnx/sq.hpp>
using gnx::sq;
sq s = "ACGTACGT"_sq;
fmt::print("Sequence: {}\n", s);  // Uses custom fmt::formatter
```
- **Real examples:** Check `example/` directory for complete working examples:
  - `local_align_example.cpp`: Smith-Waterman alignment with various scoring schemes
  - `local_align_matrix_example.cpp`: Using BLOSUM/PAM matrices for alignment
  - `sq_view_ranges_example.cpp`: C++20 ranges integration with sequence views
  - `valid_example.cpp`: Sequence validation with execution policies

## Integration Points
- **Consumer projects:** Use `find_package(gnx CONFIG)` and link `gnx::gnx`. Ensure `ZLIB` and `g3p` are available or allow FetchContent to fetch them.
- **CI:** See workflow badge referencing `cmake-multi-platform.yml` in the repository GitHub Actions.

## Agent Tips
- Prefer minimal, focused changes inside headers; keep public API stable.
- When adding tests, put them in `test/unit_tests.cpp` and rely on Catch2 v3, ensuring coverage for both Host and Device execution paths..
- When adding benchmarks, use Google Benchmark in `perf/benchmarks.cpp`, ensuring coverage for both Host and Device execution paths.
- Keep `CMakeLists.txt` consistent: `INTERFACE` target, exported config, and install rules.

Feedback: If any workflows or conventions feel unclear or incomplete (e.g., expected tag types, g3p usage patterns), tell us which sections need more detail and propose additions.
