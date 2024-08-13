#!/bin/bash

# bash command to solve the assignment 1
# Not finished this will change
python main.py indexer collections/pubmed_tiny.jsonl pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer potterNLTK
python main.py indexer collections/pubmed_small.jsonl pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer potterNLTK
python main.py indexer collections/pubmed_medium pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer potterNLTK
python main.py indexer collections/pubmed_large pubmedSPIMIindex --tk.minL 2 --tk.stopwords stopw.txt --tk.stemmer potterNLTK