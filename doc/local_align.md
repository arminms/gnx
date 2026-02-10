# Smith-Waterman Local Alignment Algorithm

## Overview

The `gnx::local_align()` function implements the Smith-Waterman algorithm for local sequence alignment. This algorithm finds the optimal local alignment between two sequences, identifying regions of similarity even when the sequences have significant differences elsewhere.

## Header

```cpp
#include <gnx/algorithms/local_align.hpp>
```

## Features

- **Classic Smith-Waterman Algorithm**: Guarantees finding the optimal local alignment
- **Flexible Scoring**: Customizable match, mismatch, and gap penalties
- **Substitution Matrix Support**: BLOSUM (45, 62, 80) and PAM (30, 120, 250) matrices for protein alignment
- **Comprehensive Results**: Returns alignment score, positions, traceback path, and aligned sequences with gaps
- **Generic Interface**: Works with any container that provides iterators (std::string, std::vector, gnx::sq, etc.)
- **Case-Insensitive**: Automatically handles both uppercase and lowercase sequences

## API

### Function Signatures

#### Iterator-based Interface (Nucleotide Sequences)

```cpp
template<typename Iterator1, typename Iterator2>
alignment_result local_align
(   Iterator1 seq1_first
,   Iterator1 seq1_last
,   Iterator2 seq2_first
,   Iterator2 seq2_last
,   int match = 2
,   int mismatch = -1
,   int gap_penalty = -1
);
```

#### Iterator-based Interface (Protein Sequences with Substitution Matrix)

```cpp
template<typename Iterator1, typename Iterator2, typename SubMatrix>
alignment_result local_align
(   Iterator1 seq1_first
,   Iterator1 seq1_last
,   Iterator2 seq2_first
,   Iterator2 seq2_last
,   const SubMatrix& subst_matrix
,   int gap_penalty = -8
);
```

#### Range-based Interface (Convenience)

```cpp
template<std::ranges::input_range Range1, std::ranges::input_range Range2>
alignment_result local_align
(   const Range1& seq1
,   const Range2& seq2
);
```

#### Range-based Interface with Substitution Matrix

```cpp
template<std::ranges::input_range Range1, std::ranges::input_range Range2, typename SubMatrix>
alignment_result local_align
(   const Range1& seq1
,   const Range2& seq2
,   const SubMatrix& subst_matrix
,   int gap_penalty = -8
);
```

### Parameters

- **seq1_first, seq1_last**: Iterators defining the first sequence range
- **seq2_first, seq2_last**: Iterators defining the second sequence range
- **seq1, seq2**: Sequence containers (when using range-based interface)
- **match**: Score awarded for matching characters (default: 2, nucleotides only)
- **mismatch**: Penalty for mismatching characters (default: -1, nucleotides only)
- **gap_penalty**: Penalty for inserting a gap (default: -1 for nucleotides, -8 for proteins)
- **subst_matrix**: Substitution matrix for protein alignment (e.g., `gnx::lut::blosum62`)

### Available Substitution Matrices

#### BLOSUM (BLOcks SUbstitution Matrix)
- **`gnx::lut::blosum45`**: For distantly related sequences (<45% identity)
- **`gnx::lut::blosum62`**: Most commonly used, balanced for general use (~62% identity)
- **`gnx::lut::blosum80`**: For closely related sequences (>80% identity)

#### PAM (Point Accepted Mutation)
- **`gnx::lut::pam30`**: For very closely related sequences (30 mutations per 100 residues)
- **`gnx::lut::pam120`**: For moderately related sequences (120 mutations per 100 residues)
- **`gnx::lut::pam250`**: For distantly related sequences (250 mutations per 100 residues)

### Return Value

Returns an `alignment_result` structure containing:

```cpp
struct alignment_result
{   int score;                                      // Alignment score
    std::size_t max_i;                              // Row index of maximum score
    std::size_t max_j;                              // Column index of maximum score
    std::vector<alignment_direction> traceback;      // Traceback path
    std::string aligned_seq1;                       // First aligned sequence (with gaps)
    std::string aligned_seq2;                       // Second aligned sequence (with gaps)
};
```

- **score**: The optimal local alignment score
- **max_i, max_j**: Matrix positions where the maximum score was found
- **traceback**: Vector of alignment directions (diagonal, up, left, none)
- **aligned_seq1, aligned_seq2**: The aligned sequences with '-' characters representing gaps

## Usage Examples

### Basic Usage (Nucleotide Sequences)

```cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/local_align.hpp>

using gnx::sq;
using gnx::local_align;

// Align two sequences
sq seq1{"ACGTACGT"};
sq seq2{"TACGT"};
auto result = local_align(seq1, seq2);

std::cout << "Score: " << result.score << "\n";
std::cout << "Aligned 1: " << result.aligned_seq1 << "\n";
std::cout << "Aligned 2: " << result.aligned_seq2 << "\n";
```

### Protein Alignment with BLOSUM62

```cpp
#include <gnx/sq.hpp>
#include <gnx/algorithms/local_align.hpp>

using gnx::sq;
using gnx::local_align;
using gnx::lut::blosum62;

// Align protein sequences
sq protein1{"MVHLTPEEKSAV"};
sq protein2{"MVHLTPEEKSAV"};
auto result = local_align(protein1, protein2, blosum62);

std::cout << "Score: " << result.score << "\n";
```

### Comparing Human and Mouse Proteins

```cpp
sq human_globin{"VLSPADKTNVKAAW"};
sq mouse_globin{"VLSAADKTNVKAAW"};  // P->A substitution

auto result = local_align(human_globin, mouse_globin, gnx::lut::blosum62);
// Will find good alignment despite the difference
```

### Custom Scoring Parameters

```cpp
std::string seq1 = "ACGTACGT";
std::string seq2 = "ACGTACGT";

// High match score, strong mismatch penalty, moderate gap penalty
auto result = local_align(seq1.begin(), seq1.end(), 
                          seq2.begin(), seq2.end(), 
                          5,    // match score
                          -3,   // mismatch penalty
                          -2);  // gap penalty
```

### Finding Subsequences

```cpp
// Find where a short sequence aligns within a longer one
sq long_seq{"AAAAAACGTACGTTTTTTT"};
sq short_seq{"ACGTACGT"};
auto result = local_align(long_seq, short_seq);

// Will find the embedded ACGTACGT with perfect score
```

### Detecting SNPs and Indels

```cpp
// Sequences with single nucleotide polymorphism
sq seq1{"ATCGATCGATCG"};
sq seq2{"ATCGCTCGATCG"};  // C instead of A at position 5
auto result = local_align(seq1, seq2);

// Sequences with insertion/deletion
sq seq3{"ATCGATCGATCG"};
sq seq4{"ATCGTCGATCG"};   // Missing 'A'
auto result2 = local_align(seq3, seq4);
```

## Algorithm Details

### Smith-Waterman Algorithm

The Smith-Waterman algorithm uses dynamic programming to find the optimal local alignment:

1. **Initialization**: Creates a scoring matrix with all cells initialized to 0
2. **Filling**: For each cell (i,j):
   - Calculate scores from diagonal (match/mismatch), up (gap in seq2), and left (gap in seq1)
   - Set cell value to max(0, diagonal_score, up_score, left_score)
   - Track the maximum score position
3. **Traceback**: Starting from the maximum score, follow the traceback path until reaching a cell with score 0

### Substitution Matrices

Substitution matrices provide position-independent scoring for amino acid alignments based on observed substitution rates in evolutionarily related proteins.

#### BLOSUM Matrices

BLOSUM (BLOcks SUbstitution Matrix) matrices are derived from conserved regions in protein families:

- **BLOSUM62**: Most widely used; based on sequences with ≤62% identity
  - Balanced for general-purpose protein alignment
  - Suitable for sequences with moderate evolutionary distance
  - Positive scores for similar amino acids, negative for dissimilar
  
- **BLOSUM80**: For closely related sequences (>80% identity)
  - Higher scores for identical residues
  - More stringent penalties for substitutions
  - Best for comparing orthologs or recent paralogs

- **BLOSUM45**: For distantly related sequences (<45% identity)
  - More tolerant of substitutions
  - Useful for detecting distant homology
  - Lower scores overall but better discrimination at large evolutionary distances

#### PAM Matrices

PAM (Point Accepted Mutation) matrices model evolutionary distance:

- **PAM30**: 30 accepted mutations per 100 residues
  - For very closely related sequences
  - High scores for identities, strong penalties for substitutions
  
- **PAM120**: 120 accepted mutations per 100 residues
  - Moderate evolutionary distance
  - Balanced approach similar to BLOSUM62
  
- **PAM250**: 250 accepted mutations per 100 residues
  - For distantly related sequences
  - More tolerant of substitutions
  - Useful for distant homology detection

#### Choosing a Matrix

| Sequence Relationship | Recommended Matrix | Typical % Identity |
|----------------------|-------------------|-------------------|
| Identical/Nearly identical | BLOSUM80, PAM30 | >80% |
| Closely related | BLOSUM62, PAM120 | 40-80% |
| Moderately related | BLOSUM62, PAM120 | 25-40% |
| Distantly related | BLOSUM45, PAM250 | <25% |

### Amino Acid Index Mapping

The implementation supports all 20 standard amino acids plus:
- **B**: Aspartic acid (D) or Asparagine (N)
- **Z**: Glutamic acid (E) or Glutamine (Q)
- **X**: Any amino acid (unknown)
- **\***: Stop codon

### Time and Space Complexity

1. **Initialization**: Creates a scoring matrix with all cells initialized to 0
2. **Filling**: For each cell (i,j):
   - Calculate scores from diagonal (match/mismatch), up (gap in seq2), and left (gap in seq1)
   - Set cell value to max(0, diagonal_score, up_score, left_score)
   - Track the maximum score position
3. **Traceback**: Starting from the maximum score, follow the traceback path until reaching a cell with score 0

### Time and Space Complexity

- **Time Complexity**: O(m × n) where m and n are sequence lengths
- **Space Complexity**: O(m × n) for storing the scoring and traceback matrices

### Default Scoring Scheme

- **Match**: +2
- **Mismatch**: -1
- **Gap**: -1

This scoring scheme works well for nucleotide sequences. For protein sequences or different similarity requirements, adjust the parameters accordingly.

## Performance Considerations

- The algorithm allocates matrices proportional to the product of sequence lengths
- For very long sequences (>10,000 bases), consider using banded alignment or other optimizations
- The implementation is currently CPU-only; GPU acceleration is not yet implemented

## Comparison with Other Alignment Methods

| Algorithm | Type | Use Case | Guarantees Optimal |
|-----------|------|----------|-------------------|
| Smith-Waterman | Local | Finding similar regions | Yes |
| Needleman-Wunsch | Global | End-to-end alignment | Yes |
| BLAST | Local | Database searching | No (heuristic) |
| BWA | Local | Read mapping | No (heuristic) |

## Testing

The implementation includes comprehensive unit tests covering:
- Perfect matches
- Mismatches and SNPs
- Gap handling (insertions/deletions)
- Custom scoring parameters
- Edge cases (empty sequences, single characters)
- Case insensitivity
- Integration with gnx::sq containers

Run tests with:
```bash
cd build
ctest -R local_align --output-on-failure
```

## References

1. Smith, T. F., & Waterman, M. S. (1981). "Identification of common molecular subsequences". Journal of Molecular Biology, 147(1), 195-197.
2. Durbin, R., Eddy, S. R., Krogh, A., & Mitchison, G. (1998). "Biological Sequence Analysis". Cambridge University Press.

## Future Enhancements

Potential improvements for future versions:

- [ ] GPU acceleration using CUDA/HIP
- [ ] Banded alignment for long sequences
- [ ] Affine gap penalties
- [ ] Semi-global alignment variants
- [ ] Parallel execution policy support
- [ ] SIMD optimization for scoring
