#ifndef __DUMP_H__
#define __DUMP_H__

#include "../data/documents.h"
#include "../frequent_pattern_mining/frequent_pattern_mining.h"
#include "../model_training/segmentation.h"
#include "../utils/utils.h"

class Dump {
public:
  Dump(Documents *corpus, FrequentPatternMining *pattern_mining)
      : corpus_(corpus), pattern_mining_(pattern_mining) {}
  Documents *corpus_;
  FrequentPatternMining *pattern_mining_;

  void loadSegmentationModel(const string &filename, Segmentation *segmenter);

  void dumpSegmentationModel(const string &filename, Segmentation *segmenter,
                             int min_sup);

  void dumpPOSTransition(const string &filename, Segmentation *segmenter);

  void dumpFeatures(const string &filename,
                    const vector<vector<double>> &features,
                    const vector<FrequentPatternMining::Pattern> &truth);

  void dumpLabels(const string &filename,
                  const vector<FrequentPatternMining::Pattern> &truth);

  template <class T>
  void dumpRankingList(const string &filename,
                       vector<pair<T, PATTERN_ID_TYPE>> &order) {
    auto &patterns = pattern_mining_->patterns;

    FILE *out = tryOpen(filename, "w");
    sort(order.rbegin(), order.rend());
    for (size_t iter = 0; iter < order.size(); ++iter) {
      PATTERN_ID_TYPE i = order[iter].second;
      fprintf(out, "%.10f\t", patterns[i].quality);
      for (int j = 0; j < patterns[i].tokens.size(); ++j) {
        fprintf(out, "%d%c", patterns[i].tokens[j],
                j + 1 == patterns[i].tokens.size() ? '\n' : ' ');
      }
    }
    fclose(out);
  }

  void dumpResults(const string &prefix, int min_sup);

  void dumpSalientResults(const string& filename, int min_sup);
  void dumpMultiWordResults(const string& filename, int min_sup);
  void dumpUnigramResults(const  string& filename, int min_sup);
};

#endif
