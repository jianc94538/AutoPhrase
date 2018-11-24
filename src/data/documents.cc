#include "documents.h"

void Documents::loadStopwords(const string &filename) {
  FILE *in = tryOpen(filename, "r");
  std::unique_ptr<char> line(new char[MAX_LENGTH + 1]);
  while (getLine(in, line.get())) {
    vector<string> tokens = splitBy(line.get(), ' ');
    if (tokens.size() == 1) {
      TOKEN_ID_TYPE id;
      fromString(tokens[0], id);
      if (id >= 0) {
        stopwords.insert(id);
      }
    }
  }
  if (INTERMEDIATE) {
    cerr << "# of loaded stop words = " << stopwords.size() << endl;
  }
  fclose(in);
}

void Documents::loadAllTrainingFiles(const string &docFile,
                                     const string &posFile,
                                     const string &capitalFile,
                                     bool enable_pos_tagging) {
  // get total number of tokens and the maximum number of tokens
  FILE *in = tryOpen(docFile, "r");
  TOTAL_TOKENS_TYPE totalTokens = 0;
  std::unique_ptr<char> line_ptr(new char[MAX_LENGTH + 1]);
  char *line = line_ptr.get();
  for (; fscanf(in, "%s", line) == 1; ++totalTokens) {
    bool flag = true;
    TOKEN_ID_TYPE id = 0;
    for (TOTAL_TOKENS_TYPE i = 0; line[i] && flag; ++i) {
      flag &= isdigit(line[i]);
      id = id * 10 + line[i] - '0';
    }
    if (flag) {
      maxTokenID = max(maxTokenID, id);
      ++totalWordTokens;
    }
  }
  cerr << "# of total tokens = " << totalTokens << endl;
  if (INTERMEDIATE) {
    cerr << "# of total word tokens = " << totalWordTokens << endl;
  }
  cerr << "max word token id = " << maxTokenID << endl;
  fclose(in);

  idf = new float[maxTokenID + 1];
  isDigital = new bool[maxTokenID + 1];
  for (TOKEN_ID_TYPE i = 0; i <= maxTokenID; ++i) {
    isDigital[i] = false;
  }
  wordTokens = new TOKEN_ID_TYPE[totalWordTokens];
  if (enable_pos_tagging) {
    posTag.clear();
    posTag2id.clear();
    posTags = new POS_ID_TYPE[totalWordTokens];
  }
  wordTokenInfo = new WordTokenInfo[totalWordTokens];

  char currentTag[100];

  in = tryOpen(docFile, "r");
  FILE *posIn = NULL;
  if (enable_pos_tagging) {
    posIn = tryOpen(posFile, "r");
  }
  FILE *capitalIn = tryOpen(capitalFile, "r");

  INDEX_TYPE docs = 0;
  TOTAL_TOKENS_TYPE ptr = 0;
  while (getLine(in, line)) {
    ++docs;
    TOTAL_TOKENS_TYPE docStart = ptr;

    stringstream sin(line);

    myAssert(getLine(capitalIn, line),
             "Captial info file doesn't have enough lines");
    TOTAL_TOKENS_TYPE capitalPtr = 0;

    string lastPunc = "";
    for (string temp; sin >> temp;) {
      // get pos tag
      POS_ID_TYPE posTagId = -1;
      if (enable_pos_tagging) {
        myAssert(fscanf(posIn, "%s", currentTag) == 1,
                 "POS file doesn't have enough POS tags");
        if (!posTag2id.count(currentTag)) {
          posTagId = posTag2id.size();
          posTag.push_back(currentTag);
          posTag2id[currentTag] = posTagId;
        } else {
          posTagId = posTag2id[currentTag];
        }
      }

      // get token
      bool flag = true;
      TOKEN_ID_TYPE token = 0;
      for (size_t i = 0; i < temp.size() && flag; ++i) {
        flag &= isdigit(temp[i]);
        token = token * 10 + temp[i] - '0';
      }

      // get capital info
      int capitalInfo = line[capitalPtr++];

      if (!flag) {
        string punc = temp;
        if (ptr > 0) {
          if (punc == "-") {
            wordTokenInfo[ptr - 1].turnOn(DASH_AFTER);
          }
          if (punc == "\"") {
            wordTokenInfo[ptr - 1].turnOn(QUOTE_AFTER);
          }
          if (punc == ")" && ptr > 0) {
            wordTokenInfo[ptr - 1].turnOn(PARENTHESIS_AFTER);
          }
          if (separatePunc.count(punc)) {
            wordTokenInfo[ptr - 1].turnOn(SEPARATOR_AFTER);
          }
        }
        lastPunc = punc;
      } else {
        wordTokens[ptr] = token;
        if (enable_pos_tagging) {
          posTags[ptr] = posTagId;
        }

        if (lastPunc == "\"") {
          wordTokenInfo[ptr].turnOn(QUOTE_BEFORE);
        } else if (lastPunc == "(") {
          wordTokenInfo[ptr].turnOn(PARENTHESIS_BEFORE);
        }

        if (capitalInfo & 1) {
          wordTokenInfo[ptr].turnOn(FIRST_CAPITAL);
        }
        if (capitalInfo >> 1 & 1) {
          wordTokenInfo[ptr].turnOn(ALL_CAPITAL);
        }
        if (capitalInfo >> 2 & 1) {
          isDigital[token] = true;
        }
        ++ptr;
      }
    }

    // The end of line is also a separator.
    wordTokenInfo[ptr - 1].turnOn(SEPARATOR_AFTER);

    set<TOKEN_ID_TYPE> docSet(wordTokens + docStart, wordTokens + ptr);
    FOR(token, docSet) { ++idf[*token]; }
  }
  fclose(in);

  for (TOKEN_ID_TYPE i = 0; i <= maxTokenID; ++i) {
    idf[i] = log(docs / idf[i] + EPS);
  }

  cerr << "# of documents = " << docs << endl;
  cerr << "# of distinct POS tags = " << posTag2id.size() << endl;
}

void Documents::splitIntoSentences() {
  sentences.clear();
  TOTAL_TOKENS_TYPE st = 0;
  for (TOTAL_TOKENS_TYPE i = 0; i < totalWordTokens; ++i) {
    if (isEndOfSentence(i)) {
      sentences.push_back(make_pair(st, i));
      st = i + 1;
    }
  }
  sentences.shrink_to_fit();
  if (INTERMEDIATE) {
    cerr << "The number of sentences = " << sentences.size() << endl;
  }
}
