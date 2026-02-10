// Example demonstrating sq_view_gen composability with C++20 ranges
#include <gnx/sq.hpp>
#include <gnx/sq_view.hpp>
#include <iostream>
#include <ranges>
#include <cctype>

int main() {
    using gnx::sq;
    using gnx::sq_view;

    // Create a sequence
    sq s = "ACGTACGTNN"_sq;
    s["_id"] = std::string("seq1");

    std::cout << "Original sequence: " << std::string(s.begin(), s.end()) << "\n";

    // Create a view
    sq_view view{s};

    // Example 1: Transform and filter using range adaptors
    auto valid_bases = view 
        | std::views::filter([](char c) { return c != 'N'; })
        | std::views::transform([](char c) { return std::tolower(c); });

    std::cout << "Valid bases (lowercase): ";
    for (char c : valid_bases) {
        std::cout << c;
    }
    std::cout << "\n";

    // Example 2: Reverse and take first 5
    auto reversed = view 
        | std::views::reverse 
        | std::views::take(5);

    std::cout << "Last 5 bases (reversed): ";
    for (char c : reversed) {
        std::cout << c;
    }
    std::cout << "\n";

    // Example 3: Complex pipeline with multiple transformations
    auto result = view
        | std::views::filter([](char c) { return c == 'A' || c == 'T'; })
        | std::views::transform([](char c) { return c == 'A' ? 'T' : 'A'; })
        | std::views::take(4);

    std::cout << "AT bases complemented (first 4): ";
    for (char c : result) {
        std::cout << c;
    }
    std::cout << "\n";

    // Example 4: Using view_interface methods
    std::cout << "View is " << (view.empty() ? "empty" : "not empty") << "\n";
    std::cout << "View size: " << view.size() << "\n";
    std::cout << "First base: " << view.front() << "\n";
    std::cout << "Last base: " << view.back() << "\n";

    return 0;
}
