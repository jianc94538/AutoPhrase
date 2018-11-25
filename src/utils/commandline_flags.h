#ifndef __COMMANDLINE_FLAGS_H__
#define __COMMANDLINE_FLAGS_H__

#include "../utils/parameters.h"
#include "../utils/utils.h"

void parseCommandFlags(int argc, char *argv[], Configure *config) {
  config->SEGMENTATION_MODEL = "";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--iter")) {
      fromString(argv[++i], config->ITERATIONS);
    } else if (!strcmp(argv[i], "--min_sup")) {
      fromString(argv[++i], config->MIN_SUP);
    } else if (!strcmp(argv[i], "--max_len")) {
      fromString(argv[++i], config->MAX_LEN);
    } else if (!strcmp(argv[i], "--discard")) {
      fromString(argv[++i], config->DISCARD);
    } else if (!strcmp(argv[i], "--thread")) {
      fromString(argv[++i], config->NTHREADS);
    } else if (!strcmp(argv[i], "--label")) {
      config->LABEL_FILE = argv[++i];
    } else if (!strcmp(argv[i], "--verbose")) {
      INTERMEDIATE = true;
    } else if (!strcmp(argv[i], "--pos_tag")) {
      config->ENABLE_POS_TAGGING = true;
    } else if (!strcmp(argv[i], "--pos_prune")) {
      config->ENABLE_POS_PRUNE = true;
      config->NO_EXPANSION_POS_FILENAME = argv[++i];
    } else if (!strcmp(argv[i], "--max_positives")) {
      fromString(argv[++i], config->MAX_POSITIVE);
    } else if (!strcmp(argv[i], "--label_method")) {
      config->LABEL_METHOD = argv[++i];
      if (config->LABEL_METHOD != "DPDN" && config->LABEL_METHOD != "EPEN" &&
          config->LABEL_METHOD != "EPDN" && config->LABEL_METHOD != "DPEN") {
        fprintf(stderr, "[Warning] Unknown Label Method: %s\n", argv[i]);
        config->LABEL_METHOD = "DPDN";
      }
    } else if (!strcmp(argv[i], "--negative_ratio")) {
      fromString(argv[++i], config->NEGATIVE_RATIO);
    } else if (!strcmp(argv[i], "--model")) {
      config->SEGMENTATION_MODEL = argv[++i];
    } else if (!strcmp(argv[i], "--highlight-multi")) {
      fromString(argv[++i], config->SEGMENT_MULTI_WORD_QUALITY_THRESHOLD);
    } else if (!strcmp(argv[i], "--highlight-single")) {
      fromString(argv[++i], config->SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
    } else if (!strcmp(argv[i], "--output_salient")){
      config->output_salient_file = argv[++i];
    } else if (!strcmp(argv[i], "--output_multiword")){
      config->output_multiword_file = argv[++i];
    } else if (!strcmp(argv[i], "--output_unigram")){
      config->output_unigram_file = argv[++i];
    } else {
      fprintf(stderr, "[Warning] Unknown Parameter: %s\n", argv[i]);
    }
  }
  if (config->SEGMENTATION_MODEL == "") {
    fprintf(stderr, "=== Current Settings ===\n");
    fprintf(stderr, "Iterations = %d\n", config->ITERATIONS);
    fprintf(stderr, "Minimum Support Threshold = %d\n", config->MIN_SUP);
    fprintf(stderr, "Maximum Length Threshold = %d\n", config->MAX_LEN);
    if (config->ENABLE_POS_TAGGING) {
      fprintf(stderr, "POS-Tagging Mode Enabled\n");
    } else {
      fprintf(stderr, "POS-Tagging Mode Disabled\n");
      fprintf(stderr, "Discard Ratio = %.6f\n", config->DISCARD);
    }
    fprintf(stderr, "Number of threads = %d\n", config->NTHREADS);

    fprintf(stderr, "Labeling Method = %s\n", config->LABEL_METHOD.c_str());
    if (config->LABEL_METHOD.find("E") != -1) {
      fprintf(stderr, "\tLoad labels from %s\n", config->LABEL_FILE.c_str());
    }
    if (config->LABEL_METHOD.find("D") != -1) {
      fprintf(stderr, "\tAuto labels from knowledge bases\n");
      fprintf(stderr, "\tMax Positive Samples = %d\n", config->MAX_POSITIVE);
    }
  } else {
    fprintf(stderr, "=== Current Settings ===\n");
    fprintf(stderr, "Segmentation Model Path = %s\n",
            config->SEGMENTATION_MODEL.c_str());
    fprintf(stderr, "After the phrasal segmentation, only following phrases "
                    "will be highlighted with <phrase> and </phrase>\n");
    fprintf(stderr, "\tQ(multi-word phrases) >= %.6f\n",
            config->SEGMENT_MULTI_WORD_QUALITY_THRESHOLD);
    fprintf(stderr, "\tQ(single-word phrases) >= %.6f\n",
            config->SEGMENT_SINGLE_WORD_QUALITY_THRESHOLD);
  }

  fprintf(stderr, "=======\n");
}

#endif
