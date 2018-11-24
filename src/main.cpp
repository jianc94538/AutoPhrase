#include <memory>

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

int main(int argc, char *argv[]) {
  Configure config;
  parseCommandFlags(argc, argv, &config);

  sscanf(argv[1], "%d", &config.NTHREADS);
  omp_set_num_threads(config.NTHREADS);

  cerr << "Loading data..." << endl;
  // load stopwords, documents, and capital information
  std::unique_ptr<Documents> corpus(new Documents);
  corpus->loadStopwords(config.STOPWORDS_FILE);
  corpus->loadAllTrainingFiles(config.TRAIN_FILE, config.POS_TAGS_FILE,
                               config.TRAIN_CAPITAL_FILE, config.ENABLE_POS_TAGGING);
  corpus->splitIntoSentences();

  cerr << "Mining frequent phrases..." << endl;
  std::unique_ptr<FrequentPatternMining> pattern_mining(
      new FrequentPatternMining(corpus.get()));
  pattern_mining->initialize();
  pattern_mining->mine(config);
  // check the patterns
  Dump dumper(corpus.get(), pattern_mining.get());
  if (INTERMEDIATE) {
    vector<pair<TOTAL_TOKENS_TYPE, PATTERN_ID_TYPE>> order;
    for (PATTERN_ID_TYPE i = 0; i < pattern_mining->patterns.size(); ++i) {
      order.push_back(make_pair(pattern_mining->patterns[i].currentFreq, i));
    }
    dumper.dumpRankingList("tmp/frequent_patterns.txt", order);
  }

  // feature extraction
  cerr << "Extracting features..." << endl;
  std::unique_ptr<Features> featurize(
      new Features(corpus.get(), pattern_mining.get()));
  vector<string> featureNames;
  vector<vector<double>> features = featurize->extract(featureNames);

  vector<string> featureNamesUnigram;
  vector<vector<double>> featuresUnigram =
      featurize->extractUnigram(featureNamesUnigram);

  cerr << "Constructing label pools..." << endl;
  std::unique_ptr<Label> label(new Label(corpus.get(), pattern_mining.get()));
  vector<FrequentPatternMining::Pattern> truth =
      label->generateAll(config.LABEL_METHOD, config.LABEL_FILE,
                         config.ALL_FILE, config.QUALITY_FILE, config.MAX_POSITIVE);

  pattern_mining->truthPatterns = label->loadTruthPatterns(config.QUALITY_FILE);
  cerr << "# truth patterns = " << pattern_mining->truthPatterns.size() << endl;
  for (FrequentPatternMining::Pattern p : truth) {
    if (p.label == 1) {
      pattern_mining->truthPatterns.push_back(p);
    }
  }

  TOTAL_TOKENS_TYPE recognized = featurize->recognize(truth);

  /*
  if (config.ENABLE_POS_TAGGING) {
    Segmentation::initializePosTags(corpus.posTag2id.size());
  }
  */

  std::unique_ptr<Segmentation> segmentation(new Segmentation(
      config.ENABLE_POS_TAGGING, pattern_mining.get(), corpus.get()));

  // SegPhrase, +, ++, +++, ...
  for (int iteration = 0; iteration < config.ITERATIONS; ++iteration) {
    if (INTERMEDIATE) {
      fprintf(stderr, "Feature Matrix = %d X %d\n", features.size(),
              features.back().size());
    }
    cerr << "Estimating Phrase Quality..." << endl;
    predictQuality(pattern_mining->patterns, features, featureNames);
    predictQualityUnigram(pattern_mining->patterns, featuresUnigram,
                          featureNamesUnigram);

    /*
    if (iteration == 0) {
        Dump::dumpResults("tmp/distant_training_only");
        break;
    }
    */
    /*
    constructTrie(); // update the current frequent enough patterns
    */
    // check the quality
    if (INTERMEDIATE) {
      char filename[256];
      sprintf(filename, "tmp/iter_%d_quality", iteration);
      dumper.dumpResults(filename, config.MIN_SUP);
    }

    cerr << "Segmenting..." << endl;
    if (INTERMEDIATE) {
      cerr << "[POS Tags Mode]" << endl;
    }
    segmentation->constructTrie();
    double last = 1e100;
    for (int inner = 0; inner < 10; ++inner) {
      double energy = segmentation->adjustPOSTagTransition(corpus->sentences,
                                                           config.MIN_SUP);
      if (fabs(energy - last) / fabs(last) < config.EPS) {
        break;
      }
      last = energy;
    }

    if (INTERMEDIATE) {
      char filename[256];
      sprintf(filename, "tmp/iter_%d_pos_tags.txt", iteration);
      dumper.dumpPOSTransition(filename, segmentation.get());
    }

    segmentation->rectifyFrequencyPOS(corpus->sentences, config.MIN_SUP);

    if (iteration + 1 < config.ITERATIONS) {
      // rectify the features
      cerr << "Rectifying features..." << endl;
      label->removeWrongLabels();

      /*
      // use number of sentences + rectified frequency to approximate the new
      idf double docs = Documents::sentences.size() + EPS; double diff = 0; int
      cnt = 0; for (int i = 0; i < patterns.size(); ++ i) { if
      (patterns[i].size() == 1) { const TOKEN_ID_TYPE& token =
      patterns[i].tokens[0]; TOTAL_TOKENS_TYPE freq = patterns[i].currentFreq;
              double newIdf = log(docs / (freq + EPS) + EPS);
              diff += abs(newIdf - Documents::idf[token]);
              ++ cnt;
              Documents::idf[token] = newIdf;
          }
      }
      */

      features = featurize->extract(featureNames);
      featuresUnigram = featurize->extractUnigram(featureNamesUnigram);
    }

    // check the quality
    if (INTERMEDIATE) {
      char filename[256];
      sprintf(filename, "tmp/iter_%d_frequent_quality", iteration);
      dumper.dumpResults(filename, config.MIN_SUP);
    }
  }

  cerr << "Dumping results..." << endl;
  dumper.dumpResults("tmp/final_quality", config.MIN_SUP);
  dumper.dumpSegmentationModel("tmp/segmentation.model", segmentation.get(), config.MIN_SUP);

  cerr << "Done." << endl;

  return 0;
}
