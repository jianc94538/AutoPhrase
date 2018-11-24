#ifndef __LABEL_GENERATION_H__
#define __LABEL_GENERATION_H__

#include "../classification/predict_quality.h"
#include "../clustering/clustering.h"
#include "../data/documents.h"
#include "../frequent_pattern_mining/frequent_pattern_mining.h"
#include "../utils/utils.h"

/*
using Documents::totalWordTokens;
using Documents::wordTokens;
using Documents::stopwords;

using FrequentPatternMining::Pattern;
using FrequentPatternMining::patterns;
using FrequentPatternMining::pattern2id;
using FrequentPatternMining::id2ends;
using FrequentPatternMining::unigrams;
*/

class Label {
public:
  Label(Documents *corpus, FrequentPatternMining *pattern_mining)
      : corpus_(corpus), pattern_mining_(pattern_mining) {}
  Documents *corpus_;
  FrequentPatternMining *pattern_mining_;

  inline vector<FrequentPatternMining::Pattern> loadLabels(string filename) {
    auto &pattern2id = pattern_mining_->pattern2id;

    vector<FrequentPatternMining::Pattern> ret;
    FILE *in = tryOpen(filename, "r");
    while (getLine(in)) {
      stringstream sin(line);
      bool valid = true;
      FrequentPatternMining::Pattern p;
      sin >> p.label;
      for (TOKEN_ID_TYPE s; sin >> s;) {
        p.append(s);
      }
      if (p.size() > 0 && pattern2id.count(p.hashValue)) {
        ret.push_back(p);
      }
    }
    cerr << "# of loaded labels = " << ret.size() << endl;
    return ret;
  }

  inline vector<FrequentPatternMining::Pattern>
  loadTruthPatterns(string filename) {
    vector<FrequentPatternMining::Pattern> ret;
    FILE *in = tryOpen(filename, "r");
    while (getLine(in)) {
      stringstream sin(line);
      bool valid = true;
      FrequentPatternMining::Pattern p;
      for (string s; sin >> s;) {
        bool possibleInt = false;
        for (int i = 0; i < s.size(); ++i) {
          possibleInt |= isdigit(s[i]);
        }
        if (possibleInt) {
          TOKEN_ID_TYPE x;
          fromString(s, x);
          if (x < 0) {
            valid = false;
            break;
          }
          p.append(x);
        }
      }
      if (valid) {
        ret.push_back(p);
      }
    }
    fclose(in);
    return ret;
  }

  inline unordered_set<ULL> loadPatterns(string filename, int MAX_POSITIVE) {
    auto &pattern2id = pattern_mining_->pattern2id;

    FILE *in = tryOpen(filename, "r");
    vector<ULL> positivesUnigrams, positiveMultiwords;
    while (getLine(in)) {
      stringstream sin(line);
      bool valid = true;
      FrequentPatternMining::Pattern p;
      for (string s; sin >> s;) {
        bool possibleInt = false;
        for (int i = 0; i < s.size(); ++i) {
          possibleInt |= isdigit(s[i]);
        }
        if (possibleInt) {
          TOKEN_ID_TYPE x;
          fromString(s, x);
          if (x < 0) {
            valid = false;
            break;
          }
          p.append(x);
        }
      }
      if (valid && pattern2id.count(p.hashValue)) {
        if (p.size() > 1) {
          positiveMultiwords.push_back(p.hashValue);
        } else if (p.size() == 1) {
          positivesUnigrams.push_back(p.hashValue);
        }
      }
    }
    fclose(in);

    if (MAX_POSITIVE != -1) {
      sort(positiveMultiwords.begin(), positiveMultiwords.end());
      positiveMultiwords.erase(
          unique(positiveMultiwords.begin(), positiveMultiwords.end()),
          positiveMultiwords.end());
      if (MAX_POSITIVE < positiveMultiwords.size()) {
        srand(time(0) ^ 13548689);
        random_shuffle(positiveMultiwords.begin(), positiveMultiwords.end());
        positiveMultiwords.resize(MAX_POSITIVE);
      }
      sort(positivesUnigrams.begin(), positivesUnigrams.end());
      positivesUnigrams.erase(
          unique(positivesUnigrams.begin(), positivesUnigrams.end()),
          positivesUnigrams.end());
      if (MAX_POSITIVE < positivesUnigrams.size()) {
        srand(time(0) ^ 13548689);
        random_shuffle(positivesUnigrams.begin(), positivesUnigrams.end());
        positivesUnigrams.resize(MAX_POSITIVE);
      }
    }
    unordered_set<ULL> ret;
    for (ULL value : positiveMultiwords) {
      ret.insert(value);
    }
    for (ULL value : positivesUnigrams) {
      ret.insert(value);
    }
    return ret;
  }

  inline vector<FrequentPatternMining::Pattern>
  generateAll(string LABEL_METHOD, string LABEL_FILE, string ALL_FILE,
              string QUALITY_FILE, int max_positive) {
    auto &patterns = pattern_mining_->patterns;

    vector<FrequentPatternMining::Pattern> ret;

    if (LABEL_METHOD.find("E") != -1) { // experts
      vector<FrequentPatternMining::Pattern> truth;
      cerr << "Loading existing labels..." << endl;
      truth = Label::loadLabels(LABEL_FILE);
      bool needPos = LABEL_METHOD.find("EP") != -1;
      bool needNeg = LABEL_METHOD.find("EN") != -1;
      for (PATTERN_ID_TYPE i = 0; i < truth.size(); ++i) {
        if (truth[i].label == 1) {
          if (needPos) {
            ret.push_back(truth[i]);
          }
        } else if (truth[i].label == 0) {
          if (needNeg) {
            ret.push_back(truth[i]);
          }
        }
      }
    }

    if (LABEL_METHOD.find("D") != -1) { // distant training
      bool needPos = LABEL_METHOD.find("DP") != -1;
      bool needNeg = LABEL_METHOD.find("DN") != -1;

      unordered_set<ULL> include = loadPatterns(QUALITY_FILE, max_positive);
      unordered_set<ULL> exclude = loadPatterns(ALL_FILE, max_positive);

      if (max_positive != -1) {
        exclude.clear();
      }

      for (ULL value : include) { // make sure exclude is a super set of include
        exclude.insert(value);
      }
      for (int i = 0; i < ret.size();
           ++i) { // make sure every human label is excluded
        exclude.insert(ret[i].hashValue);
      }

      for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++i) {
        if (patterns[i].size() < 1) {
          continue;
        }
        if (patterns[i].size() == 1 &&
            corpus_->stopwords.count(patterns[i].tokens[0])) {
          ret.push_back(patterns[i]);
          ret.back().label = 0;
        } else if (include.count(patterns[i].hashValue)) {
          if (needPos) {
            ret.push_back(patterns[i]);
            ret.back().label = 1;
          }
        } else if (!exclude.count(patterns[i].hashValue)) {
          if (needNeg) {
            ret.push_back(patterns[i]);
            ret.back().label = 0;
          }
        }
      }
    }

    int cntPositives = 0, cntNegatives = 0;
    for (PATTERN_ID_TYPE i = 0; i < ret.size(); ++i) {
      if (ret[i].label == 1) {
        ++cntPositives;
      } else if (ret[i].label == 0) {
        ++cntNegatives;
      } else {
        assert(false); // It should not happen!
      }
    }

    fprintf(stderr, "\tThe size of the positive pool = %d\n", cntPositives);
    fprintf(stderr, "\tThe size of the negative pool = %d\n", cntNegatives);

    return ret;
  }

  void removeWrongLabels() {
    for (FrequentPatternMining::Pattern &pattern : pattern_mining_->patterns) {
      if (pattern.currentFreq == 0 && pattern.label == 1) {
        pattern.label = pattern_mining_->UNKNOWN_LABEL;
      }
    }
  }
};

#endif
