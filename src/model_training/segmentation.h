#ifndef __SEGMENTATION_H__
#define __SEGMENTATION_H__

#include <cassert>

#include "../frequent_pattern_mining/frequent_pattern_mining.h"
#include "../utils/utils.h"

class Segmentation {
  mutex POSTagMutex[SUFFIX_MASK + 1];
  mutex separateMutex[SUFFIX_MASK + 1];

  //#define TRUTH (patterns.size())
public:
  struct TrieNode {
    unordered_map<TOTAL_TOKENS_TYPE, size_t> children;

    PATTERN_ID_TYPE id;

    TrieNode() {
      id = -1;
      children.clear();
    }
  };
  vector<TrieNode> trie;

  void constructTrie(bool duringTraingStage = true);

private:
  static const double INF;
  vector<vector<TOTAL_TOKENS_TYPE>> total;

public:
  static bool ENABLE_POS_TAGGING;
  static double penalty;
  vector<vector<double>> connect, disconnect;

  void initializePosTags(int n);

  void getDisconnect();

  void normalizePosTags();

  void logPosTags();

private:
  // generated
  int maxLen;
  double *prob;

  void normalize();

  void initialize();

  FrequentPatternMining *pattern_mining_;
  Documents *corpus_;

public:
  double getProb(int id) const { return exp(prob[id]); }

  ~Segmentation() { delete[] prob; }

  Segmentation(bool ENABLE_POS_TAGGING, FrequentPatternMining *pattern_mining,
               Documents *corpus);

  Segmentation(double penalty);

  double viterbi(const vector<TOKEN_ID_TYPE> &tokens, vector<double> &f,
                 vector<int> &pre);

  double viterbi_proba(const vector<TOKEN_ID_TYPE> &tokens, vector<double> &f,
                       vector<int> &pre);

  double viterbi(const vector<TOKEN_ID_TYPE> &tokens,
                 const vector<POS_ID_TYPE> &tags, vector<double> &f,
                 vector<int> &pre);

  double viterbi_proba_randomPOS(const vector<TOKEN_ID_TYPE> &tokens,
                                 vector<double> &f, vector<int> &pre);

  void rectifyFrequency(
      vector<pair<TOTAL_TOKENS_TYPE, TOTAL_TOKENS_TYPE>> &sentences,
      int min_sup);

  double rectifyFrequencyPOS(
      vector<pair<TOTAL_TOKENS_TYPE, TOTAL_TOKENS_TYPE>> &sentences,
      int MIN_SUP);

  double adjustPOSTagTransition(
      vector<pair<TOTAL_TOKENS_TYPE, TOTAL_TOKENS_TYPE>> &sentences,
      int MIN_SUP);

  bool qualify(int id, int length, double multi_thres, double uni_thres);

  double viterbi_for_testing(const vector<TOKEN_ID_TYPE> &tokens,
                             const vector<POS_ID_TYPE> &tags, vector<double> &f,
                             vector<int> &pre, double multi_thres,
                             double uni_thres);

  double viterbi_for_testing(const vector<TOKEN_ID_TYPE> &tokens,
                             vector<double> &f, vector<int> &pre,
                             double multi_thres, double uni_thres);
};

#endif
