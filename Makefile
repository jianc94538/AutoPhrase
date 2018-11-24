CXX = g++
CFLAGS = -std=c++11 -Wall -O3 -msse2  -fopenmp  -I..

BIN = 	./bin/segphrase_train ./bin/segphrase_segment
OBJ =  	./src/frequent_pattern_mining/frequent_pattern_mining.o ./src/classification/feature_extraction.o \
	./src/model_training/segmentation.o ./src/data/documents.o
.PHONY: clean all

all: ./bin $(BIN)

./bin/segphrase_train: 	./src/main.cpp ./src/utils/*.h ./src/frequent_pattern_mining/*.h \
			./src/data/*.h ./src/classification/*.h \
			./src/model_training/*.h \
                        ./src/clustering/*.h $(OBJ)
./bin/segphrase_segment: ./src/segment.cpp ./src/utils/*.h ./src/frequent_pattern_mining/*.h ./src/data/*.h ./src/classification/*.h ./src/model_training/*.h ./src/clustering/*.h

./src/frequent_pattern_mining/frequent_pattern_mining.o: ./src/frequent_pattern_mining/frequent_pattern_mining.cc

./src/model_training/segmentation.o: ./src/model_training/segmentation.cc

./src/classification/feature_extraction.o: ./src/classification/feature_extraction.cc

./src/data/documents.o: ./src/data/documents.cc

./bin:
	mkdir -p bin

LDFLAGS= -pthread -lm -Wno-unused-result -Wno-sign-compare -Wno-unused-variable -Wno-parentheses -Wno-format
$(BIN) : $(OBJ)
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)
$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

clean :
	rm -rf bin; rm $(OBJ)
