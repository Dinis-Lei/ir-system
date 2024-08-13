#!/bin/bash

# --tk.stemmer potterNLTK snowballNLTK

#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart lnc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_tiny.jsonl.gz ./output/tiny
#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart lnc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_small.jsonl.gz ./output/small
#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart lnc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_medium.jsonl.gz ./output/medium
python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart lnc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_large.jsonl.gz ./output/large

#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart ltc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_tiny.jsonl.gz ./output/tiny
#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart ltc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_small.jsonl.gz ./output/small
#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart ltc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_medium.jsonl.gz ./output/medium
#python3 main.py indexer --indexer.class SPIMIIndexer --tk.stopwords stop_words_english.json --tk.minL 3 --tk.cache 20000 --indexer.n_processes 8 --indexer.tfidf.smart ltc.ltc --indexer.memory_threshold 25 ./collections/pubmed_2022_large.jsonl.gz ./output/large
