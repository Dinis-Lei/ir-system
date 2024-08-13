#!/bin/bash

# top-k: 10, 50, 100
# bm25, tfidf (lnc.ltc)
# tiny small medium large

for mod in 1 0.5
do
    for k in 10 50 100
    do
        for mode in "bm25" "tfidf"
        do
            for size in "tiny" "small" "medium"
            do
                python3 main.py searcher --top_k $k ./output/$size ./questions/questions_with_gs/questions_$size.jsonl ./search_out/${size}_results_${mode}_${k}_${mod}.txt ranking.${mode} --ranking.${mode}.B 2 --ranking.${mode}.mod $mod
            done
        done
    done
done