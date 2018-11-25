#ifndef __PREDICT_QUALITY_H__
#define __PREDICT_QUALITY_H__

#include "../frequent_pattern_mining/frequent_pattern_mining.h"

void predictQuality(vector<FrequentPatternMining::Pattern> &patterns,
                    vector<vector<double>> &features,
                    vector<string> &featureNames);

void predictQualityUnigram(vector<FrequentPatternMining::Pattern> &patterns,
                           vector<vector<double>> &features,
                           vector<string> &featureNames);

#endif
