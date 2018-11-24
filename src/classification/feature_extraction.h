#ifndef __FEATURE_EXTRACTION_H__
#define __FEATURE_EXTRACTION_H__

#include "../data/documents.h"
#include "../frequent_pattern_mining/frequent_pattern_mining.h"
#include "../model_training/segmentation.h"
#include "../utils/utils.h"

class Features {
public:
  Features(Documents *corpus, FrequentPatternMining *pattern_mining)
      : corpus_(corpus), pattern_mining_(pattern_mining) {}
  Documents *corpus_;
  FrequentPatternMining *pattern_mining_;

  TOTAL_TOKENS_TYPE
  getFrequency(const FrequentPatternMining::Pattern &pattern);

  void extractCompleteness(const FrequentPatternMining::Pattern &pattern,
                           vector<double> &feature);
  // ready for parallel
  void extractStopwords(const FrequentPatternMining::Pattern &pattern,
                        vector<double> &feature);

  // ready for parallel
  void extractPunctuation(PATTERN_ID_TYPE id, vector<double> &feature);

  // ready for parallel
  void extractStatistical(PATTERN_ID_TYPE id, vector<double> &feature);
  PATTERN_ID_TYPE recognize(vector<FrequentPatternMining::Pattern> &truth);

  vector<vector<double>> extract(vector<string> &featureNames);

  // ready for parallel
  void extractPunctuationUnigram(PATTERN_ID_TYPE id, vector<double> &feature);

  void extractCompletenessUnigram(FrequentPatternMining::Pattern &pattern,
                                  vector<double> &feature);
  vector<vector<double>> extractUnigram(vector<string> &featureNames);
};

#endif
