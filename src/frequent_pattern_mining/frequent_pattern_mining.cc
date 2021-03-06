#include "frequent_pattern_mining.h"

ULL FrequentPatternMining::MAGIC = 0xabcdef;

void FrequentPatternMining::addPatternWithoutLocks(const Pattern &pattern,
                                                   const TOTAL_TOKENS_TYPE &ed,
                                                   bool addPosition) {
  assert(pattern2id.count(pattern.hashValue));
  PATTERN_ID_TYPE id = pattern2id[pattern.hashValue];
  assert(id < id2ends.size());
  if (patterns[id].currentFreq == 0) {
    patterns[id] = pattern;
  }
  if (addPosition) {
    assert(patterns[id].currentFreq < id2ends[id].size());
    id2ends[id][patterns[id].currentFreq] = ed;
  }
  ++patterns[id].currentFreq;
}

void FrequentPatternMining::addPattern(const Pattern &pattern,
                                       const TOTAL_TOKENS_TYPE &ed,
                                       bool addPosition) {
  assert(pattern2id.count(pattern.hashValue));
  PATTERN_ID_TYPE id = pattern2id[pattern.hashValue];
  assert(id < id2ends.size());
  separateMutex[id & SUFFIX_MASK].lock();
  if (patterns[id].currentFreq == 0) {
    patterns[id] = pattern;
  }
  if (addPosition) {
    assert(patterns[id].currentFreq < id2ends[id].size());
    id2ends[id][patterns[id].currentFreq] = ed;
  }
  ++patterns[id].currentFreq;
  separateMutex[id & SUFFIX_MASK].unlock();
}

bool FrequentPatternMining::pruneByPOSTag(TOTAL_TOKENS_TYPE st,
                                          TOTAL_TOKENS_TYPE ed) {
  // if (ENABLE_POS_PRUNE) {
  POS_ID_TYPE lastPos = corpus_->posTags[ed];
  if (st == ed && noInitial[lastPos] && noExpansion[lastPos]) {
    return true;
  }
  if (st != ed && noExpansion[lastPos]) {
    return true;
  }
  // }
  return false;
}

void FrequentPatternMining::mine(const Configure &config) {
  int MIN_SUP = config.MIN_SUP;
  int LENGTH_THRESHOLD = config.MAX_LEN;

  noExpansion = vector<bool>(corpus_->posTag2id.size(), false);
  noInitial = vector<bool>(corpus_->posTag2id.size(), false);
  // if (ENABLE_POS_PRUNE) {
  FILE *in = tryOpen(config.NO_EXPANSION_POS_FILENAME, "r");
  int type = -1;
  std::unique_ptr<char> line(new char[MAX_LENGTH + 1]);
  while (getLine(in, line.get())) {
    if (strlen(line.get()) == 0) {
      continue;
    }
    stringstream sin(line.get());
    string tag;
    sin >> tag;
    if (tag == "===unigram===") {
      type = 0;
    } else if (tag == "===expansion===") {
      type = 1;
    } else {
      if (corpus_->posTag2id.count(tag)) {
        if (type == 0) {
          noInitial[corpus_->posTag2id[tag]] = true;
        } else if (type == 1) {
          noExpansion[corpus_->posTag2id[tag]] = true;
        }
      }
    }
  }
  fclose(in);

  int cntUnigrams = 0, cntExpansions = 0;
  for (int i = 0; i < noInitial.size(); ++i) {
    cntUnigrams += noInitial[i];
    cntExpansions += noExpansion[i];
  }

  if (INTERMEDIATE) {
    cerr << "# of forbidden initial pos tags = " << cntUnigrams << endl;
    cerr << "# of forbidden expanded pos tags = " << cntExpansions << endl;
  }
  //}

  id2ends.clear();
  patterns.clear();
  pattern2id.clear();

  unigrams = new TOTAL_TOKENS_TYPE[corpus_->maxTokenID + 1];

#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
  for (TOTAL_TOKENS_TYPE i = 0; i <= corpus_->maxTokenID; ++i) {
    unigrams[i] = 0;
  }
#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
  for (TOTAL_TOKENS_TYPE i = 0; i < corpus_->totalWordTokens; ++i) {
    const TOTAL_TOKENS_TYPE &token = corpus_->wordTokens[i];
    if (!pruneByPOSTag(i, i)) {
      separateMutex[token & SUFFIX_MASK].lock();
      ++unigrams[token];
      separateMutex[token & SUFFIX_MASK].unlock();
    }
  }

  // all unigrams should be added as patterns
  // allocate memory
  for (TOTAL_TOKENS_TYPE i = 0; i <= corpus_->maxTokenID; ++i) {
    pattern2id[i + 1] = patterns.size();
    patterns.push_back(Pattern());
  }
  id2ends.resize(patterns.size());
#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
  for (TOTAL_TOKENS_TYPE i = 0; i <= corpus_->maxTokenID; ++i) {
    id2ends[i].resize(unigrams[i] >= MIN_SUP ? unigrams[i] : 0);
  }

  long long totalOcc = 0;
# pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE) reduction(+:totalOcc)
  for (TOTAL_TOKENS_TYPE i = 0; i < corpus_->totalWordTokens; ++i) {
    if (!pruneByPOSTag(i, i)) {
      const TOTAL_TOKENS_TYPE &token = corpus_->wordTokens[i];
      addPattern(Pattern(token), i, unigrams[token] >= MIN_SUP);
      totalOcc += unigrams[token] >= MIN_SUP;
    }
  }
  if (INTERMEDIATE) {
    cerr << "unigrams inserted" << endl;
  }

  PATTERN_ID_TYPE last = 0;
  for (int len = 1; len <= LENGTH_THRESHOLD && last < patterns.size(); ++len) {
    if (INTERMEDIATE) {
      cerr << "# of frequent patterns of length-" << len << " = "
           << patterns.size() - last + 1 << endl;
    }
    PATTERN_ID_TYPE backup = patterns.size();

    unordered_map<ULL, TOTAL_TOKENS_TYPE> threadFreq[config.NTHREADS];
#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
    for (PATTERN_ID_TYPE id = last; id < backup; ++id) {
      assert(patterns[id].size() == 0 || patterns[id].size() == len);
      id2ends[id].shrink_to_fit();
      if (len < LENGTH_THRESHOLD) {
        for (const TOTAL_TOKENS_TYPE &ed : id2ends[id]) {
          TOTAL_TOKENS_TYPE st = ed - len + 1;
          assert(corpus_->wordTokens[st] == patterns[id].tokens[0]);

          if (!corpus_->isEndOfSentence(ed)) {
            if (!pruneByPOSTag(st, ed + 1) &&
                unigrams[corpus_->wordTokens[ed + 1]] >= config.MIN_SUP) {
              ULL newHashValue = patterns[id].hashValue * MAGIC +
                                 corpus_->wordTokens[ed + 1] + 1;

              int tid = omp_get_thread_num();
              ++threadFreq[tid][newHashValue];
            }
          }
        }
      }
    }

    // merge and allocate memory
    vector<pair<ULL, TOTAL_TOKENS_TYPE>> newPatterns;
    for (int tid = 0; tid < config.NTHREADS; ++tid) {
      for (const auto &iter : threadFreq[tid]) {
        const TOTAL_TOKENS_TYPE &freq = iter.second;
        if (freq >= config.MIN_SUP) {
          const ULL &hashValue = iter.first;
          pattern2id[hashValue] = patterns.size();
          patterns.push_back(Pattern());
          newPatterns.push_back(make_pair(hashValue, freq));
        }
      }
      threadFreq[tid].clear();
    }

    id2ends.resize(patterns.size());

#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
    for (size_t i = 0; i < newPatterns.size(); ++i) {
      const ULL &hashValue = newPatterns[i].first;
      const TOTAL_TOKENS_TYPE &freq = newPatterns[i].second;
      id2ends[pattern2id[hashValue]].resize(freq);
    }
    newPatterns.clear();

# pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE) reduction(+:totalOcc)
    for (PATTERN_ID_TYPE id = last; id < backup; ++id) {
      if (len < LENGTH_THRESHOLD) {
        vector<TOTAL_TOKENS_TYPE> positions = id2ends[id];
        for (const TOTAL_TOKENS_TYPE &ed : positions) {
          TOTAL_TOKENS_TYPE st = ed - len + 1;
          assert(corpus_->wordTokens[st] == patterns[id].tokens[0]);

          if (!corpus_->isEndOfSentence(ed)) {
            if (!pruneByPOSTag(st, ed + 1) &&
                unigrams[corpus_->wordTokens[ed + 1]] >= config.MIN_SUP) {
              ULL newHashValue = patterns[id].hashValue * MAGIC +
                                 corpus_->wordTokens[ed + 1] + 1;
              if (pattern2id.count(newHashValue)) {
                Pattern newPattern(patterns[id]);
                newPattern.append(corpus_->wordTokens[ed + 1]);
                assert(newPattern.size() == len + 1);
                newPattern.currentFreq = 0;

                addPatternWithoutLocks(newPattern, ed + 1);
                ++totalOcc;
              }
            }
          }
        }
        /*if (len == 1) {
          id2ends[id].clear();
          id2ends[id].shrink_to_fit();
          }*/
      }
    }
    last = backup;
  }
  id2ends.shrink_to_fit();

  cerr << "# of frequent phrases = " << patterns.size() << endl;
  if (INTERMEDIATE) {
    cerr << "total occurrence = " << totalOcc << endl;
  }

  for (PATTERN_ID_TYPE i = 0; i < patterns.size(); ++i) {
    assert(patterns[i].currentFreq == id2ends[i].size() ||
           id2ends[i].size() == 0);
    assert(patterns[i].size() == 0 || patterns[i].size() == 1 ||
           id2ends[i].size() >= config.MIN_SUP);
  }

#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
  for (TOTAL_TOKENS_TYPE i = 0; i < corpus_->totalWordTokens; ++i) {
    if (pruneByPOSTag(i, i)) {
      const TOTAL_TOKENS_TYPE &token = corpus_->wordTokens[i];
      addPattern(Pattern(token), i, false);
    }
  }

  // update real unigrams for later usages
#pragma omp parallel for schedule(dynamic, PATTERN_CHUNK_SIZE)
  for (TOTAL_TOKENS_TYPE i = 0; i < corpus_->totalWordTokens; ++i) {
    const TOTAL_TOKENS_TYPE &token = corpus_->wordTokens[i];
    if (pruneByPOSTag(i, i)) {
      separateMutex[token & SUFFIX_MASK].lock();
      ++unigrams[token];
      separateMutex[token & SUFFIX_MASK].unlock();
    }
  }
}
