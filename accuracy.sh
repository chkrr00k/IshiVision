#!/bin/bash

declare -a TYPES=("knn" "sift" "svm")
declare -a TRAIN_SETS=("100" "200" "1000")
TEST_SET=50
FILE_NAME="data_set"
LOG_FILE="logs.txt"
echo "Started logging on" $LOG_FILE
echo "Test set size:" $TEST_SET
echo -ne "\n"
for K in "${TYPES[@]}"; do
FIRST=true
echo "Testing" $K
    for TS in "${TRAIN_SETS[@]}"; do
        echo "Train set size:" $TS
        if [ "$FIRST" = true ]; then
            python3 test.py -k $K -t -s $TS -d $FILE_NAME -a $TEST_SET
            FIRST=false
        else
            python3 test.py -k $K -l $FILE_NAME -a $TEST_SET
        fi
    done
echo -ne "\n"
done > $LOG_FILE
