#ifndef __FREQUENT_PATTERN_MINING_H__
#define __FREQUENT_PATTERN_MINING_H__

#include <memory>

#include "../data/documents.h"
#include "../utils/utils.h"

class FrequentPatternMining {
  mutex separateMutex[SUFFIX_MASK + 1];

public:
  FrequentPatternMining(Documents *corpus) : corpus_(corpus) {}

  Documents *corpus_;
  static ULL MAGIC;
  static const int UNKNOWN_LABEL = -1000000000;

  struct Pattern {
    vector<TOKEN_ID_TYPE> tokens;
    int label;
    double probability, quality;
    ULL hashValue;
    TOKEN_ID_TYPE currentFreq;

    void dump(FILE *out) {
      Binary::write(out, currentFreq);
      Binary::write(out, quality);
      Binary::write(out, tokens.size());
      for (auto &token : tokens) {
        Binary::write(out, token);
      }
    }

    void load(FILE *in) {
      Binary::read(in, currentFreq);
      Binary::read(in, quality);
      size_t tokenSize;
      Binary::read(in, tokenSize);
      tokens.clear();
      for (size_t i = 0; i < tokenSize; ++i) {
        TOTAL_TOKENS_TYPE token;
        Binary::read(in, token);
        append(token);
      }
    }

    Pattern(const TOKEN_ID_TYPE &token) {
      tokens.clear();
      hashValue = 0;
      currentFreq = 0;
      label = UNKNOWN_LABEL;
      append(token);
      quality = 1;
    }

    Pattern(const Pattern &other) {
      this->tokens = other.tokens;
      this->hashValue = other.hashValue;
      this->probability = other.probability;
      this->label = other.label;
      this->quality = other.quality;
      this->currentFreq = other.currentFreq;
    }

    Pattern() {
      tokens.clear();
      hashValue = 0;
      currentFreq = 0;
      label = UNKNOWN_LABEL;
      quality = 1;
    }

    inline void shrink_to_fit() { tokens.shrink_to_fit(); }

    inline Pattern substr(int l, int r) const {
      Pattern ret;
      for (int i = l; i < r; ++i) {
        ret.append(tokens[i]);
      }
      return ret;
    }

    inline int size() const { return tokens.size(); }

    inline void append(const TOKEN_ID_TYPE &token) {
      tokens.push_back(token);
      hashValue = hashValue * MAGIC + token + 1;
    }

    inline bool operator==(const Pattern &other) const {
      return hashValue == other.hashValue && tokens == other.tokens;
    }

    inline void show() const {
      for (int i = 0; i < tokens.size(); ++i) {
        cerr << tokens[i] << " ";
      }
      cerr << endl;
    }
  };

  // === global variables ===
  TOTAL_TOKENS_TYPE *unigrams; // 0 .. Documents::maxTokenID
  vector<Pattern> patterns, truthPatterns;
  vector<vector<TOTAL_TOKENS_TYPE>> id2ends;
  unordered_map<ULL, PATTERN_ID_TYPE> pattern2id;

  // ===

  bool isPrime(ULL x) {
    for (ULL y = 2; y * y <= x; ++y) {
      if (x % y == 0) {
        return false;
      }
    }
    return true;
  }

  inline void initialize() {
    MAGIC = corpus_->maxTokenID + 1;
    while (!isPrime(MAGIC)) {
      ++MAGIC;
    }
    cerr << "selected MAGIC = " << MAGIC << endl;
  }

  void addPatternWithoutLocks(const Pattern &pattern,
                              const TOTAL_TOKENS_TYPE &ed,
                              bool addPosition = true);

  void addPattern(const Pattern &pattern, const TOTAL_TOKENS_TYPE &ed,
                  bool addPosition = true);

  vector<bool> noExpansion, noInitial;

  bool pruneByPOSTag(TOTAL_TOKENS_TYPE st, TOTAL_TOKENS_TYPE ed);

  void mine(const Configure &config);
};

#endif
