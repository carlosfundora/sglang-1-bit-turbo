#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace ngram {

struct Result {
  std::vector<int32_t> token;
  std::vector<uint8_t> mask;

  void truncate(size_t n);
};

struct Node {
  // Ordered (std::map) so fillResult's BFS flatten is deterministic across runs/platforms — the
  // draft-token order and mask row indices no longer depend on hash-table iteration order. (The
  // tree topology and which tokens are included are set during expansion in trie.cpp and are
  // unaffected; this only fixes the sibling emission order.) Synthesized from the deterministic
  // ordering in the rs_ngram_draft Rust port.
  std::map<int32_t, int32_t> next;
};

Result fillResult(int last_token, int draft_token_num, std::vector<Node>& tree, int root);

}  // namespace ngram
