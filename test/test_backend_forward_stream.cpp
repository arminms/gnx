// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Armin Sobhani
//
#include <catch2/catch_all.hpp>

#include <gnx/sq.hpp>
#include <gnx/sqb.hpp>
#include <gnx/algorithms/valid.hpp>
#include <gnx/backend/forward_stream.hpp>

// =============================================================================
// forward stream sequence bank tests
// =============================================================================

TEMPLATE_TEST_CASE
(   "gnx::sequence_bank<forward_stream>"
,   "[backend][forward_stream]"
,   std::vector<char>
)
{   typedef TestType T;

    SECTION( "forward_stream" )
    {   gnx::sequence_bank sb{gnx::forward_stream<gnx::generic_sequence<T>>{SAMPLE_GENOME}};
        for (const auto& s : sb)
        {   CHECK(gnx::valid(s.sequence()));
            CHECK(s.quality().empty());  // No quality scores in this test
        }
    }
}
