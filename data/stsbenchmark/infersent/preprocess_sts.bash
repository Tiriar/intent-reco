#!/usr/bin/env bash

preprocess_exec="sed -f tokenizer.sed"

cp ../sts-dev.csv ../sts-test.csv ../sts-train.csv .

for split in train dev test
do
    fname=sts-${split}.csv
    cut -f1,2,3,4,5 ${fname} > tmp1
    cut -f6 ${fname} | ${preprocess_exec} > tmp2
    cut -f7 ${fname} | ${preprocess_exec} > tmp3
    paste tmp1 tmp2 tmp3 > ${fname}
    rm tmp1 tmp2 tmp3
done
