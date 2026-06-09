// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <gnx/sq.hpp>
#include <gnx/utility/gc_content.hpp>
#include <gnx/algorithms/random.hpp>

const uint64_t seed_pi{3141592654};

// =============================================================================
// gc_content() and at_content() utilities tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::gc_content()"
,   "[utility][gc_content][gc_ratio][at_content][at_ratio]"
,   std::vector<char>
)
{   typedef TestType T;
    const auto N{10'000};
    gnx::generic_sequence<T> s(N);
    gnx::rand(s.begin(), N, "ACGT", {35, 15, 15, 35}, seed_pi);

    SECTION( "gc-content and gc-ratio" )
    {   REQUIRE_THAT(gnx::gc_content(s), Catch::Matchers::WithinAbs(30.16, 0.001));
        REQUIRE_THAT(gnx::gc_ratio(s), Catch::Matchers::WithinAbs(0.3016, 0.001));
    }

    SECTION( "at-content and at-ratio" )
    {   REQUIRE_THAT(gnx::at_content(s), Catch::Matchers::WithinAbs(69.84, 0.001));
        REQUIRE_THAT(gnx::at_ratio(s), Catch::Matchers::WithinAbs(0.6984, 0.001));
    }
}
