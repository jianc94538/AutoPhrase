#ifndef __DOCUMENTS_H__
#define __DOCUMENTS_H__

#include <memory>

#include "../utils/parameters.h"
#include "../utils/utils.h"

class Documents {
public:
  const int FIRST_CAPITAL = 0;
  const int ALL_CAPITAL = 1;
  const int DASH_AFTER = 2;
  const int QUOTE_BEFORE = 3;
  const int QUOTE_AFTER = 4;
  const int PARENTHESIS_BEFORE = 5;
  const int PARENTHESIS_AFTER = 6;
  const int SEPARATOR_AFTER = 7;

  struct WordTokenInfo {
    unsigned char mask;
    // 7 types:
    //    0-th bit: First Char Capital?
    //    1-st bit: All Chars Capital?
    //    2-nd bit: any - after this char?
    //    3-rd bit: any " before this char?
    //    4-th bit: any " after this char?
    //    5-th bit: any ( before this char?
    //    6-th bit: any ) after this char?
    //    7-th bit: any separator punc after this char?
    WordTokenInfo() { mask = 0; }

    inline void turnOn(int bit) { mask |= 1 << bit; }

    inline bool get(int bit) const { return mask >> bit & 1; }
  };
  // === global variables ===
  TOTAL_TOKENS_TYPE totalWordTokens = 0;

  TOKEN_ID_TYPE maxTokenID = 0;

  float *idf;                // 0 .. maxTokenID
  TOKEN_ID_TYPE *wordTokens; // 0 .. totalWordTokens - 1
  POS_ID_TYPE *posTags;      // 0 .. totalWordTokens - 1

  // 0 .. ((totalWordTokens * 7 + 31) / 32) - 1
  WordTokenInfo *wordTokenInfo;
  bool *isDigital; // 0..maxTokenID

  vector<pair<TOTAL_TOKENS_TYPE, TOTAL_TOKENS_TYPE>> sentences;

  map<string, POS_ID_TYPE> posTag2id;
  vector<string> posTag;

  set<TOKEN_ID_TYPE> stopwords;

  const set<string> separatePunc = {",", ".", "\"", ";", "!", ":", "(", ")", "\""};
  // ===
  inline bool hasDashAfter(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens && wordTokenInfo[i].get(DASH_AFTER);
  }

  inline bool hasQuoteBefore(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens && wordTokenInfo[i].get(QUOTE_BEFORE);
  }

  inline bool hasQuoteAfter(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens && wordTokenInfo[i].get(QUOTE_AFTER);
  }

  inline bool hasParentThesisBefore(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens &&
           wordTokenInfo[i].get(PARENTHESIS_BEFORE);
  }

  inline bool hasParentThesisAfter(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens &&
           wordTokenInfo[i].get(PARENTHESIS_AFTER);
  }

  inline bool isFirstCapital(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens && wordTokenInfo[i].get(FIRST_CAPITAL);
  }

  inline bool isAllCapital(TOTAL_TOKENS_TYPE i) {
    return 0 <= i && i < totalWordTokens && wordTokenInfo[i].get(ALL_CAPITAL);
  }

  inline bool isEndOfSentence(TOTAL_TOKENS_TYPE i) {
    return i < 0 || i + 1 >= totalWordTokens ||
           wordTokenInfo[i].get(SEPARATOR_AFTER);
  }

  void loadStopwords(const string &filename);

  void loadAllTrainingFiles(const string &docFile, const string &posFile,
                            const string &capitalFile, bool enable_pos_tagging);

  void splitIntoSentences();
};

#endif
