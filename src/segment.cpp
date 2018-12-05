#include "classification/feature_extraction.h"
#include "classification/label_generation.h"
#include "classification/predict_quality.h"
#include "data/documents.h"
#include "data/dump.h"
#include "frequent_pattern_mining/frequent_pattern_mining.h"
#include "model_training/segmentation.h"
#include "utils/commandline_flags.h"
#include "utils/parameters.h"
#include "utils/utils.h"
#include <memory>

/*using FrequentPatternMining::Pattern;
using FrequentPatternMining::patterns;
*/

vector<double> f;
vector<int> pre;

int highlights = 0, sentences = 0;

void process(const vector<TOTAL_TOKENS_TYPE> &tokens,
             const vector<POS_ID_TYPE> &tags, Segmentation &segmenter,
             const FrequentPatternMining *pattern_mining, FILE *out,
             const Configure &config) {
  const auto &patterns = pattern_mining->patterns;
  ++sentences;
  // if (ENABLE_POS_TAGGING) {
  segmenter.viterbi_for_testing(tokens, tags, f, pre,
                                config.SEGMENT_MULTI_WORD_QUALITY_THRESHOLD,
                                config.SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
  /*} else {
  segmenter.viterbi_for_testing(tokens, f, pre,
                                SEGMENT_MULTI_WORD_QUALITY_THRESHOLD,
                                SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
                                }*/

  int i = (int)tokens.size();
  assert(f[i] > -1e80);
  vector<string> ret;
  const auto& trie = segmenter.trie;
  while (i > 0) {
    int j = pre[i];
    size_t u = 0;
    bool quality = true;
    for (int k = j; k < i; ++k) {
      if (!trie[u].children.count(tokens[k])) {
        quality = false;
        break;
      }
      u = trie[u].children.at(tokens[k]);
    }
    quality &=
        trie[u].id == patterns.size() &&
            ( // These phrases are in the wiki_quality.txt, their quality scores
              // are treated as 1.
                i - j > 1 && 1 >= config.SEGMENT_MULTI_WORD_QUALITY_THRESHOLD ||
                i - j == 1 &&
                    1 >= config.SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD) ||
        trie[u].id < patterns.size() && trie[u].id >= 0 &&
            (patterns[trie[u].id].size() > 1 &&
                 patterns[trie[u].id].quality >=
                     config.SEGMENT_MULTI_WORD_QUALITY_THRESHOLD ||
             patterns[trie[u].id].size() == 1 &&
                 patterns[trie[u].id].quality >=
                     config.SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
    if (quality) {
      ret.push_back("</phrase>");
      ++highlights;
    }
    for (int k = i - 1; k >= j; --k) {
      ostringstream sout;
      sout << tokens[k];
      ret.push_back(sout.str());
    }
    if (quality) {
      ret.push_back("<phrase>");
    }

    i = j;
  }

  reverse(ret.begin(), ret.end());
  for (int i = 0; i < ret.size(); ++i) {
    fprintf(out, "%s%c", ret[i].c_str(), i + 1 == ret.size() ? '\n' : ' ');
  }
}

inline bool byQuality(const FrequentPatternMining::Pattern &a,
                      const FrequentPatternMining::Pattern &b) {
  return a.quality > b.quality + EPS ||
         fabs(a.quality - b.quality) < EPS && a.currentFreq > b.currentFreq;
}

int main(int argc, char *argv[]) {
  Configure config;
  parseCommandFlags(argc, argv, &config);
  std::unique_ptr<Documents> corpus(new Documents);
  std::unique_ptr<FrequentPatternMining> pattern_mining(
      new FrequentPatternMining(corpus.get()));
  Dump dumper(corpus.get(), pattern_mining.get());

  sscanf(argv[1], "%d", &config.NTHREADS);
  omp_set_num_threads(config.NTHREADS);

  sort(pattern_mining->patterns.begin(), pattern_mining->patterns.end(),
       byQuality);

  Segmentation *segmenter;
  segmenter = new Segmentation(config.ENABLE_POS_TAGGING, pattern_mining.get(),
                               corpus.get());
  dumper.loadSegmentationModel(config.SEGMENTATION_MODEL, segmenter);

  segmenter->getDisconnect();
  segmenter->logPosTags();
  // update the current frequent enough patterns
  segmenter->constructTrie(false);

  char currentTag[100];

  FILE *in = tryOpen(config.TEXT_TO_SEG_FILE, "r");
  FILE *posIn = NULL;
  if (config.ENABLE_POS_TAGGING) {
    posIn = tryOpen(config.TEXT_TO_SEG_POS_TAGS_FILE, "r");
  }

  FILE *out = tryOpen("tmp/tokenized_segmented_sentences.txt", "w");

  std::unique_ptr<char> line(new char[MAX_LENGTH + 1]);
  while (getLine(in, line.get())) {
    stringstream sin(line.get());
    vector<TOTAL_TOKENS_TYPE> tokens;
    vector<POS_ID_TYPE> tags;

    string lastPunc = "";
    for (string temp; sin >> temp;) {
      // get pos tag
      POS_ID_TYPE posTagId = -1;
      if (config.ENABLE_POS_TAGGING) {
        myAssert(fscanf(posIn, "%s", currentTag) == 1,
                 "POS file doesn't have enough POS tags");
        if (!corpus->posTag2id.count(currentTag)) {
          posTagId = -1; // unknown tag
        } else {
          posTagId = corpus->posTag2id[currentTag];
        }
      }

      // get token
      bool flag = true;
      TOKEN_ID_TYPE token = 0;
      for (size_t i = 0; i < temp.size() && flag; ++i) {
        flag &= isdigit(temp[i]) || i == 0 && temp.size() > 1 && temp[0] == '-';
      }
      stringstream sin(temp);
      sin >> token;

      if (!flag) {
        string punc = temp;
        if (corpus->separatePunc.count(punc)) {
          process(tokens, tags, *segmenter, pattern_mining.get(), out, config);
          tokens.clear();
          tags.clear();
        }
      } else {
        tokens.push_back(token);
        if (config.ENABLE_POS_TAGGING) {
          tags.push_back(posTagId);
        }
      }
    }
    if (tokens.size() > 0) {
      process(tokens, tags, *segmenter, pattern_mining.get(), out, config);
    }
  }
  fclose(in);
  if (config.ENABLE_POS_TAGGING) {
    fclose(posIn);
  }
  fclose(out);

  cerr << "Phrasal segmentation finished." << endl;
  cerr << "   # of total highlighted quality phrases = " << highlights << endl;
  cerr << "   # of total processed sentences = " << sentences << endl;
  cerr << "   avg highlights per sentence = " << (double)highlights / sentences
       << endl;

  return 0;
}
