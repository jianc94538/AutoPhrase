#include "dump.h"
#include "../classification/feature_extraction.h"
#include "../classification/label_generation.h"
#include "../classification/predict_quality.h"
//#include "../utils/commandline_flags.h"
#include "../utils/parameters.h"

void Dump::loadSegmentationModel(const string &filename,
                                 Segmentation *segmenter) {
  auto &patterns = pattern_mining_->patterns;
  auto &truthPatterns = pattern_mining_->truthPatterns;

  FILE *in = tryOpen(filename, "rb");
  bool flag;
  Binary::read(in, flag);
  myAssert(/*ENABLE_POS_TAGGING*/ true == flag,
           "Model and configuration mismatch! whether ENABLE_POS_TAGGING?");
  Binary::read(in, Segmentation::penalty);

  if (flag) {
    cerr << "POS guided model loaded." << endl;
  } else {
    cerr << "Length penalty model loaded." << endl;
    cerr << "\tpenalty = " << Segmentation::penalty << endl;
  }

  // quality phrases & unigrams
  size_t cnt = 0;
  Binary::read(in, cnt);
  patterns.resize(cnt);
  for (size_t i = 0; i < cnt; ++i) {
    patterns[i].load(in);
  }
  cerr << "# of loaded patterns = " << cnt << endl;

  Binary::read(in, cnt);
  truthPatterns.resize(cnt);
  for (size_t i = 0; i < cnt; ++i) {
    truthPatterns[i].load(in);
  }
  cerr << "# of loaded truth patterns = " << cnt << endl;

  if (flag) {
    // POS Tag mapping
    Binary::read(in, cnt);
    corpus_->posTag.resize(cnt);
    for (int i = 0; i < corpus_->posTag.size(); ++i) {
      Binary::read(in, corpus_->posTag[i]);
      corpus_->posTag2id[corpus_->posTag[i]] = i;
    }
    // cerr << "pos tags loaded" << endl;

    // POS Tag Transition
    Binary::read(in, cnt);
    segmenter->connect.resize(cnt);
    for (int i = 0; i < segmenter->connect.size(); ++i) {
      segmenter->connect[i].resize(cnt);
      for (int j = 0; j < segmenter->connect[i].size(); ++j) {
        Binary::read(in, segmenter->connect[i][j]);
      }
    }
    cerr << "POS transition matrix loaded" << endl;
  }

  fclose(in);
}

void Dump::dumpSegmentationModel(const string &filename,
                                 Segmentation *segmenter, int min_sup) {
  auto &patterns = pattern_mining_->patterns;
  auto &truthPatterns = pattern_mining_->truthPatterns;
  auto &unigrams = pattern_mining_->unigrams;

  FILE *out = tryOpen(filename, "wb");
  Binary::write(out, /*ENABLE_POS_TAGGING*/ true);
  Binary::write(out, segmenter->penalty);

  // quality phrases & unigrams
  size_t cnt = 0;
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 ||
        patterns[i].size() == 1 && patterns[i].currentFreq > 0 &&
            unigrams[patterns[i].tokens[0]] >= min_sup) {
      ++cnt;
    }
  }
  Binary::write(out, cnt);
  if (INTERMEDIATE) {
    cerr << "# of phrases dumped = " << cnt << endl;
  }
  for (size_t i = 0; i < patterns.size(); ++i) {
    if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 ||
        patterns[i].size() == 1 && patterns[i].currentFreq > 0 &&
            unigrams[patterns[i].tokens[0]] >= min_sup) {
      patterns[i].dump(out);
    }
  }

  // truth
  if (INTERMEDIATE) {
    cerr << "# of truth dumped = " << truthPatterns.size() << endl;
  }
  Binary::write(out, truthPatterns.size());
  for (size_t i = 0; i < truthPatterns.size(); ++i) {
    truthPatterns[i].dump(out);
  }

  // POS Tag mapping
  Binary::write(out, corpus_->posTag.size());
  for (int i = 0; i < corpus_->posTag.size(); ++i) {
    Binary::write(out, corpus_->posTag[i]);
  }

  // POS Tag Transition
  Binary::write(out, segmenter->connect.size());
  for (int i = 0; i < segmenter->connect.size(); ++i) {
    for (int j = 0; j < segmenter->connect[i].size(); ++j) {
      Binary::write(out, segmenter->connect[i][j]);
    }
  }

  fclose(out);
}

void Dump::dumpPOSTransition(const string &filename, Segmentation *segmenter) {
  FILE *out = tryOpen(filename, "w");
  for (int i = 0; i < corpus_->posTag.size(); ++i) {
    fprintf(out, "\t%s", corpus_->posTag[i].c_str());
  }
  fprintf(out, "\n");
  for (int i = 0; i < corpus_->posTag.size(); ++i) {
    fprintf(out, "%s", corpus_->posTag[i].c_str());
    for (int j = 0; j < corpus_->posTag.size(); ++j) {
      fprintf(out, "\t%.10f", segmenter->connect[i][j]);
    }
    fprintf(out, "\n");
  }
  fclose(out);
}

void Dump::dumpFeatures(const string &filename,
                        const vector<vector<double>> &features,
                        const vector<FrequentPatternMining::Pattern> &truth) {
  FILE *out = tryOpen(filename, "w");
  for (const auto &pattern : truth) {
    PATTERN_ID_TYPE i = pattern_mining_->pattern2id[pattern.hashValue];
    if (features[i].size() > 0) {
      for (int j = 0; j < features[i].size(); ++j) {
        fprintf(out, "%.10f%c", features[i][j],
                j + 1 == features[i].size() ? '\n' : '\t');
      }
    }
  }
  fclose(out);
}

void Dump::dumpLabels(const string &filename,
                      const vector<FrequentPatternMining::Pattern> &truth) {
  FILE *out = tryOpen(filename, "w");
  for (FrequentPatternMining::Pattern pattern : truth) {
    for (int j = 0; j < pattern.tokens.size(); ++j) {
      fprintf(out, "%d%c", pattern.tokens[j],
              j + 1 == pattern.tokens.size() ? '\n' : ' ');
    }
  }
  fclose(out);
}

void Dump::dumpResults(const string &prefix, int min_sup) {
  dumpSalientResults(prefix + "_salient.txt", min_sup);
  dumpMultiWordResults(prefix + "_multi-words.txt", min_sup);
  dumpUnigramResults(prefix + "_unigrams.txt", min_sup);
}

void Dump::dumpSalientResults(const string &filename, int min_sup) {
  auto &patterns = pattern_mining_->patterns;
  auto &unigrams = pattern_mining_->unigrams;

  vector<pair<double, PATTERN_ID_TYPE>> order;
  for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++i) {
    if (patterns[i].size() > 1 && patterns[i].currentFreq > 0 ||
        patterns[i].size() == 1 && patterns[i].currentFreq > 0 &&
            unigrams[patterns[i].tokens[0]] >= min_sup) {
      order.push_back(make_pair(patterns[i].quality, i));
    }
  }
  dumpRankingList(filename, order);
}

void Dump::dumpMultiWordResults(const string &filename, int min_sup) {
  auto &patterns = pattern_mining_->patterns;

  vector<pair<double, PATTERN_ID_TYPE>> order;
  for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++i) {
    if (patterns[i].size() > 1 && patterns[i].currentFreq > 0) {
      order.push_back(make_pair(patterns[i].quality, i));
    }
  }
  dumpRankingList(filename, order);
}

void Dump::dumpUnigramResults(const string &filename, int min_sup) {
  auto &patterns = pattern_mining_->patterns;
  auto &unigrams = pattern_mining_->unigrams;

  vector<pair<double, PATTERN_ID_TYPE>> order;
  for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++i) {
    if (patterns[i].size() == 1 && patterns[i].currentFreq > 0 &&
        unigrams[patterns[i].tokens[0]] >= min_sup) {
      order.push_back(make_pair(patterns[i].quality, i));
    }
  }
  dumpRankingList(filename, order);
}
