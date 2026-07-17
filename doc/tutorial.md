---
title: GNX Tutorial
description: Using GNX in a nutshell.
kernelspec:
  name: xcpp20-openmp
  display_name: C++20-OpenMP
---

# GNX Tutorial

---

## The header file

The `gnx` library is *header-only*. That means we can start using it by just including the necessary header files. That often comes down to *the base* header that contains the core components and also switching to `gnx` namespace:

```{code-cell} cpp
#include <gnx/base>

using namespace gnx;

// optional but makes the output neater
std::locale::global(std::locale("en_US.UTF-8"));
```

:::::{seealso} `cling` include paths
:class: dropdown

If `gnx` is not installed in a standard folder for headers, you can add it to the [cling include paths](xref:cling#chapters/grammar) using one of the following methods:

::::{tab-set}
:::{tab-item} #pragma

```cpp
#pragma cling add_include_path("gnx/include/path")
```
:::

:::{tab-item} .(command)
In a `cling` REPL, but not Jupyter Notebook you can use `.I`:
```cpp
.I "gnx/include/path"
```

:::

::::

:::::

## The `gnx::sq` class
Making a biological sequence in `gnx` is easy:

```{code-cell} cpp
gnx::sq s{"ACGT"};
s
```

:::{hint} Displaying rich content
:class: dropdown
To display an object in the front-end you can omit the last semicolon of a code cell. By doing so, the last expression will be displayed.
:::

Or even easier using *string literals* for its [complementary](wiki:Complementarity_(molecular_biology)) sequence:

```{code-cell} cpp
auto c = "TGCA"_sq;
(s == c)
```

Or from another sequence:

```{code-cell} cpp
auto cc(c);
~cc;    // in-place complementing
(s == cc)
```

It's designed to have a small footprint:

```{code-cell} cpp
s.size()
```

```{code-cell} cpp
sizeof(std::vector<char>)
```

```{code-cell} cpp
s.size_in_memory()
```

## Adding tagged (meta)data

You can add unlimited (meta)data with arbitrary types to a sequence by giving them a name (i.e. tag):

```{code-cell} cpp
s["_id"] = std::string("my-first-sequence");
s["ref_number"] = 123;
s
```

:::{hint} Reserved tags
:class: dropdown
Tags starting with underscore character `_` are reserved for `gnx` internal use. For example, `_id` and `_desc` are used for giving a sequence an id and description. So, when you set them, they will show up as you display the sequence. But, that's not the case for other tagged data and you cannot see them in the output.
:::

And check for their existence:

```{code-cell} cpp
s.has("ref_number")
```

Or changing them:

```{code-cell} cpp
s["ref_number"] = 321;
```

And getting them by providing their tag and type:

```{code-cell} cpp
std::get<int>(s["ref_number"])
```

As you add more tagged data, the size of the sequence increases accordingly:

```{code-cell} cpp
s.size_in_memory()
```

## Random sequences
You can easily generate random <wiki:DNA>/<wiki:RNA>/<wiki:Protein> sequences with the specified length using `gnx::random`:

```{code-cell} cpp
auto dna_seq = random::dna(500);
dna_seq
```
+++
```{code-cell} cpp
auto rna_seq = random::rna(200);
rna_seq
```
+++
```{code-cell} cpp
auto protein_seq = random::protein(600);
protein_seq
```

:::{important} Try it NOW!
:class: dropdown
Rerun the above cells to see if you get different sequence each time.
:::

You can use an arbitrary `unsigned long` seed to get the same sequence each time:
```{code-cell} cpp
auto reproducible_seq = random::dna(100, 666ul);
reproducible_seq
```
As you may already realized, the generated sequences are [uniformly distributed](wiki:Continuous_uniform_distribution). We can use `gnx::count()` *algorithm* or `gnx::gc_content()` *utility* to check that:

```{code-cell} cpp
count(dna_seq)
```
+++
```{code-cell} cpp
gc_content(dna_seq)
```
Or even better, the `gnx::summary()` utility:
```{code-cell} cpp
summary(dna_seq);
```

If you provide a `double` value as the second argument for <wiki:DNA>/<wiki:RNA> instead of an integer, you can change that behavior to get a <wiki:DNA>/<wiki:RNA> sequence with that <wiki:GC-content>:

```{code-cell} cpp
auto gc33_dna = random::dna(500, 33.0);
summary(gc33_dna);
```

## Iterating over bp/aa

Let's try to replace all <wiki:Arginine> (`R`) <wiki:amino_acid>s in our protein sequence with a <wiki:stop_codon> (`*`):

```{code-cell} cpp
for (auto& r : protein_seq)
    if (r == 'R' || r == 'r')
        r = '*';
protein_seq
```
And count them:

```{code-cell} cpp
size_t count{0};
for (auto aa : protein_seq)
    if (aa == '*')
        ++count;
count
```
And turn them back:

```{code-cell} cpp
for (size_t i = 0; i < std::size(protein_seq); ++i)
    if ('*' == protein_seq[i])
        protein_seq[i] = 'R';
protein_seq
```

## Saving sequences
To save a sequence in a file you can simply use its `save()` member function:

```{code-cell} cpp
gc33_dna.save
// supported extensions: .fasta,.fa,.fas,.fna,.faa,.ffn,.frn,.fq,.fastq,.fasta.gz,.fa.gz,.fas.gz,.fna.gz,faa.gz,.ffn.gz,.frn.gz,.fq.gz,fastq.gz
(   "dna_gc_content_33.fa"
// ,   80 // line width
// ,    1 // number of parallel threads
// ,   -1 // compress level (-1 for auto)
);
```

To add a bunch of sequences to a file, you can instantiate a `gnx::file` object and call its `write()` member function for each:

```{code-cell} cpp
gnx::file my_contigs
(   "contigs.fa"
// ,   false // faidx
// ,   80    // line width
// ,    1    // number of parallel threads
// ,   -1    // compress level (-1 for auto)
);

std::vector<size_t> lengths{ 120, 400, 1001, 524 };
for (auto len : lengths)
{   auto contig = random::dna(len, 33.0);
    my_contigs.write(contig);
}
my_contigs.close();
```

:::{important} Try it NOW!
:class: dropdown
As a practice, change the above cell to generate each sequence with a different <wiki:GC-content> defined in another `std::vector<double>`. *Hint*: for a range-based `for` loop you can use [`std::views::iota(e, f)`](https://en.cppreference.com/cpp/ranges/iota_view).
:::

## Non-owning sequence views

`gnx::sq_view` is a lightweight, non-owning view over a `gnx::sq` (or any compatible container), similar to [`std::string_view`](https://en.cppreference.com/cpp/string/basic_string_view). It avoids copying while letting you slice and compare.

```{code-cell} cpp
reproducible_seq
```
+++
```{code-cell} cpp
gnx::sq_view v{reproducible_seq};   // view over reproducible_seq, no copy
v
```
+++
```{code-cell} cpp
// slicing without allocation
auto mid = v.subseq(25, 50);
mid
```
+++
```{code-cell} cpp
// adjust view window
mid.remove_prefix(5);
mid.remove_suffix(10);
mid
```
You can also use the *function operator* (i.e. parenthesis `()`) on the original sequence (i.e. `reproducible_seq`) for the same effect, which is more convenient:
+++
```{code-cell} cpp
auto w = reproducible_seq(25, 50);
w.remove_prefix(5);
w.remove_suffix(10);
w
```
Changing the original sequence proves that they are views indeed:
```{code-cell} cpp
reproducible_seq[50] = 'N';
v
```
+++
```{code-cell} cpp
w
```
Using *function operator* for getting a view, if you drop the length it goes to the end:
```{code-cell} cpp
auto last_ten = reproducible_seq(90);
last_ten
```

## Working with sequence files

### Whole bacterial genomes

With `gnx`, it's easy to work with compressed/uncompressed [FASTA](wiki:FASTA_format) files. First, let's download a sample compressed <wiki:genome> from the [NCBI](wiki:National_Center_for_Biotechnology_Information) ftp site:

```{code-cell} cpp
auto bacterial_genome = wget("genome://GCF_000204255.1_ASM20425v1");
```

Now, let's get a gist of what's inside using `gnx::describe()` function:

```{code-cell} cpp
describe(bacterial_genome());
```

Turned out it's a bacterial whole genome of a *<wiki:Chlamydia>* species, including a **7.6** kb *<wiki:plasmid>*. To work with the smaller *<wiki:plasmid>*, we can use the exact same method we used to instantiate a `gnx::sq` object but this time we provide the filename instead of a sequence, and a *zero-offset* index of 1, as the second argument to load the second sequence in the file:

```{code-cell} cpp
sq plasmid(bacterial_genome(), 1);
```

:::{tip} Loading the plasmid directly
:class: dropdown
If all you need is just the plasmid sequence and don't have anything to do with the whole genome, rather than downloading the file first and then loading the plasmid, you can load the plasmid directly:

**with index:**
```{code-block} cpp
sq plasmid("genome://GCF_000204255.1_ASM20425v1", 1);
```
**with id:**
```{code-block} cpp
sq plasmid("genome://GCF_000204255.1_ASM20425v1", "NC_017288.1");
```
:::

Once loaded, we can call either `gnx::describe()` or `gnx::summary()` to get more information about it:

```{code-cell} cpp
describe(plasmid);
```
+++
```{code-cell} cpp
summary(plasmid);
```

:::{important} Try it NOW!
:class: dropdown
You can also use the plasmid id (`NC_017288.1`) instead of the index for the same effect:
```{code-block} cpp
sq plasmid(bacterial_genome(), "NC_017288.1");
```
:::

:::{tip} Using the string literal
:class: dropdown
It is also possible to use the `string literal` method to load from a sequence file, but it always loads the first sequence in the file, which is the *<wiki:Chlamydia>*'s chromosome (`NC_017287.1` with index `0`) in this case:
```{code-block} cpp
auto chromosome = "genome://GCF_000204255.1_ASM20425v1"_sq;
```
:::

Let's display the sequence, but this time using `gnx::print` *utility* function to have more control over the output format as it's a much longer sequence:

```{code-cell} cpp
// print
// (   plasmid
// // ,   80  // line width
// // ,   1   // start index
// // ,   10  // separator
// );
```

#### [Dot plot](wiki:Dot_plot_(bioinformatics)) analysis

A self-[dot plot](wiki:Dot_plot_(bioinformatics)) compares a sequence against itself. It is useful for identifying <wiki:structural_motif>s, [direct](wiki:direct_repeat) or <wiki:inverted_repeats>, duplications and *low-complexity regions* (LCRs).

```{code-cell} cpp
dotplot(plasmid);
```
+++
```{code-cell} cpp
dotplot(plasmid(0, 500), 1000);
```
We can also [dot plot](wiki:Dot_plot_(bioinformatics)) a <wiki:DNA>/<wiki:RNA> sequence against its [reverse-complement](wiki:Complementarity_(molecular_biology)) (self-complementarity plot) to visualize and identify <wiki:palindromes> (<wiki:inverted_repeats>):

```{code-cell} cpp
auto plasmid_rc(plasmid);
std::ranges::reverse(~plasmid_rc); // in-place reverse-complementing
dotplot(plasmid, plasmid_rc);
```

#### Finding [ORF](wiki:Open_reading_frame)s

```{code-cell} cpp
auto rf1 = translate(plasmid);
print
(   rf1
// ,   color_scheme:: // na | na_inverted | na_warn | aa_warn | aa_clustal | aa_clustal_inverted | orf_identify | mono
// ,   80             // line width
// ,   1              // start index
// ,   10             // separator
);
```
+++
```{code-cell} cpp
auto rf3 = translate(plasmid_rc);
```

### Large (mammalian) genomes with index file (*faidx*)

You can also work with uncompressed/[block compressed](wiki:BGZF) mammalian genomic [FASTA](wiki:FASTA_format) files with accompanying index (`.fai`, `.gzi`) files using `gnx::virtual_vector` class:


```{code-cell} cpp
auto downloaded_human_genome = wget("genome://GCF_000001405.40_GRCh38.p14");
```

```{code-cell} cpp
describe(downloaded_human_genome());
```
+++
```{code-cell} cpp
gnx::virtual_vector<sq> human_genome{downloaded_human_genome()};
```
+++
```{code-cell} cpp
auto NC_012920_1 = human_genome[704];
```
+++
```{code-cell} cpp
summary(NC_012920_1);
```

#### 2-Bit Packed sequences

`gnx::psq2` stores DNA sequences using only 2 bits per nucleotide (A=`00`, C=`01`, G=`10`, T=`11`), packing four bases into each byte. Compared with one byte per character in `gnx::sq`, this gives a **4× memory reduction** — critical for whole-genome analysis and GPU-resident data. Within each byte, bases are stored **MSB-first**:

```
byte[i] = [base 4i | base 4i+1 | base 4i+2 | base 4i+3]
            bits 7-6     5-4         3-2         1-0
```

So the four-base sequence `ACGT` encodes to the single byte `0b00_01_10_11 = 0x1B`

```{code-cell} cpp
gnx::virtual_vector<psq2> packed_human_genome{downloaded_human_genome()};
auto packed_NC_012920_1 = packed_human_genome[704];
packed_NC_012920_1.size_in_memory()
```
+++
```{code-cell} cpp
NC_012920_1.size_in_memory()
```

### WGS (<wiki:Whole_genome_sequencing>) reads 

`gnx` also supports compressed/uncompressed [FASTQ](wiki:FASTQ_format) files coming from [high throughput sequencing](wiki:DNA_sequencing#High-throughput_sequencing_(HTS)_methods) pipelines. Like the [FASTA](wiki:FASTA_format) example, first let's download a small [FASTQ](wiki:FASTQ_format) file containing reads:

```{code-cell} cpp
auto downloaded_reads = wget("sra://SRR10190173_1");
```
Now let investigate what's in it:

```{code-cell} cpp
describe(downloaded_reads())
```
+++
```{code-cell} cpp
summary(downloaded_reads())
```
+++

```{code-cell} cpp
sq read(downloaded_reads(), 12724);
read
```

```{code-cell} cpp
summary(read);
```

## High performance algorithms

### Parallel execution policies

```{code-cell} cpp
describe(download());
```
+++
```{code-cell} cpp
auto chromosome{downloaded()};
```
+++
```{code-cell} cpp
summary(chromosome);
```
+++

**Serial version**
```{code-cell} cpp
%timeit -n 5 auto test = translate(gnx::execution::seq, chromosome);
```
**SIMD version**
```{code-cell} cpp
%timeit -n 5 auto test = translate(gnx::execution::unseq, chromosome);
```

**Parallel version**
```{code-cell} cpp
%timeit -n 5 auto test = translate(gnx::execution::par, chromosome);
```
**Hybrid SIMD/parallel**
```{code-cell} cpp
%timeit -n 5 auto test = translate(gnx::execution::par_unseq, chromosome);
```

### Running on GPU

Running on both <wiki:CUDA> and <wiki:ROCm> devices are supported. Just change `sq`, `psq2`, `_sq` and `_psq2` to corresponding device versions, `dsq`, `dpsq2`, `_dsq` and `_dpsq2`, and you're all set:

```{code-block} cpp
auto device_chromosome = "GCF_000204255.1_ASM20425v1_genomic.fna.gz"_dsq;
```
**CUDA**
```{code-block} cpp
auto device_test = translate(thrust::cuda::par, chromosome);
```
**ROCm**
```{code-block} cpp
auto device_test = translate(thrust::hip::par, chromosome);
```

## Local alignment (<wiki:Smith-Waterman_algorithm>)

```{code-cell} cpp
auto HIV_1 = "genome://GCF_000864765.1_ViralProj15476"_sq;
```
+++
```{code-cell} cpp
describe(HIV_1);
```
+++
```{code-cell} cpp
summary(HIV_1);
```
+++
```{code-cell} cpp
auto HIV_2 = "genome://GCF_000856385.1_ViralProj14991"_sq;
```
+++
```{code-cell} cpp
describe(HIV_2);
```
+++
```{code-cell} cpp

summary(HIV_2);
```
+++
```{code-cell} cpp
dotplot(HIV_1, HIV_2);
```
+++
```{code-cell} cpp
auto alignment = local_align_n(HIV_1, HIV_2);
```
+++
```{code-cell} cpp
print(alignment, 150);
```
+++
```{code-cell} cpp
dotplot(HIV_1(500, 500), HIV_2(1095, 500), 1000);
```
