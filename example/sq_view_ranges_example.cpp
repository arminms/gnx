// Example demonstrating generic_sequence_view composability with C++20 ranges
#include <fmt/core.h>
#include <gnx/sq.hpp>
#include <gnx/views.hpp>
#include <ranges>
#include <cctype>

int main() {
    using gnx::sq;
    using gnx::sq_view;

    // Create a sequence
    sq s = "ACGTACGTNN"_sq;
    s["_id"] = std::string("seq1");

    fmt::print("Original sequence: {}\n", std::string(s.begin(), s.end()));

    // Create a view
    sq_view view{s};

    // Example 1: Transform and filter using range adaptors
    auto valid_bases = view 
        | std::views::filter([](char c) { return c != 'N'; })
        | std::views::transform([](char c) { return std::tolower(c); });

    fmt::print("Valid bases (lowercase): ");
    for (char c : valid_bases) {
        fmt::print("{}", c);
    }
    fmt::print("\n");

    // Example 2: Reverse and take first 5
    auto reversed = view 
        | std::views::reverse 
        | std::views::take(5);

    fmt::print("Last 5 bases (reversed): ");
    for (char c : reversed) {
        fmt::print("{}", c);
    }
    fmt::print("\n");

    // Example 3: Complex pipeline with multiple transformations
    auto result = view
        | std::views::filter([](char c) { return c == 'A' || c == 'T'; })
        | std::views::transform([](char c) { return c == 'A' ? 'T' : 'A'; })
        | std::views::take(4);

    fmt::print("AT bases complemented (first 4): ");
    for (char c : result) {
        fmt::print("{}", c);
    }
    fmt::print("\n");

    // Example 4: Using view_interface methods
    fmt::print("View is {}\n", view.empty() ? "empty" : "not empty");
    fmt::print("View size: {}\n", view.size());
    fmt::print("First base: {}\n", view.front());
    fmt::print("Last base: {}\n", view.back());

    return 0;
}
