# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2025 Armin Sobhani
#

## fetching sample genome (Chlamydia psittaci 6BC)
#
FetchContent_Declare(
  sample_genome
  URL      https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/204/255/GCF_000204255.1_ASM20425v1/GCF_000204255.1_ASM20425v1_genomic.fna.gz
  URL_HASH MD5=599418042b774c4e200406d290f4a863
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(sample_genome)

## fetching sample reads (Experiment: SRX6910578)
#
FetchContent_Declare(
  sample_reads
  URL       ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR101/073/SRR10190173/SRR10190173_1.fastq.gz
  URL_HASH MD5=f93602650632e6d0a32201457b4a2385
  DOWNLOAD_NO_EXTRACT TRUE
)
FetchContent_MakeAvailable(sample_reads)

## add path for unit tests
#
add_definitions(-DSAMPLE_GENOME="${sample_genome_SOURCE_DIR}/GCF_000204255.1_ASM20425v1_genomic.fna.gz")
add_definitions(-DSAMPLE_READS="${sample_reads_SOURCE_DIR}/SRR10190173_1.fastq.gz")

## function to add unit tests
#
function(add_unit_tests TEST_NAME TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  target_link_libraries(${TEST_TARGET} PRIVATE
    ${PROJECT_NAME}::${PROJECT_NAME}
    Catch2::Catch2WithMain
    ranx::openmp
  )
if(CUDAToolkit_FOUND)
  set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)
  target_compile_options(${TEST_TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
  )
endif()
if(hip_FOUND)
  set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE HIP)
  target_compile_options(${TEST_TARGET} PRIVATE
    -Wno-unused-result -Wno-pass-failed
    $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx942>
  )
endif()
endfunction()

## defining unit tests
#
add_unit_tests("forward_stream" test_backend_forward_stream.cpp)
add_unit_tests("core" unit_tests.cpp)
add_unit_tests("compare_algorithm" test_algorithm_compare.cpp)
add_unit_tests("complement_algorithm" test_algorithm_complement.cpp)
add_unit_tests("count_algorithm" test_algorithm_count.cpp)
add_unit_tests("io_fastaqz" test_io_fastaqz.cpp)
add_unit_tests("local_align_algorithm" test_algorithm_local_align.cpp)
add_unit_tests("random_algorithm" test_algorithm_random.cpp)
add_unit_tests("translate_algorithm" test_algorithm_translate.cpp)
add_unit_tests("valid_algorithm" test_algorithm_valid.cpp)

## finally adding unit tests
#
catch_discover_tests(test_backend_forward_stream TEST_PREFIX "[Backend] ")
catch_discover_tests(unit_tests TEST_PREFIX "[Core] ")
catch_discover_tests(test_algorithm_compare TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_complement TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_count TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_local_align TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_random TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_translate TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_algorithm_valid TEST_PREFIX "[Algorithm] ")
catch_discover_tests(test_io_fastaqz TEST_PREFIX "[I/O] ")
