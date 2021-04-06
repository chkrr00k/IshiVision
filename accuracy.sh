#!/bin/bash

declare -a TYPES=("sift" "knn" "svm" "gnb" "sksvm" "area")
declare -a TRAIN_SETS=("10" "20" "100")
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
        echo "Train set size:" "$(($TS * 10))"
        if [ "$K" = "sift" ] || [ "$K" = "area" ]; then
            python3 test.py -k $K -t -s $TS -a $TEST_SET
        elif [ "$FIRST" = true ]; then
            python3 test.py -k $K -t -s $TS -d $FILE_NAME -a $TEST_SET
            FIRST=false
        else
            python3 test.py -k $K -l $FILE_NAME -a $TEST_SET
        fi
    done
echo -ne "\n"
done > $LOG_FILE
