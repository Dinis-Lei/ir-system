from utils import dynamically_init_class
from collections import Counter
from math import log10, sqrt, ceil
from index import InvertedIndex
from time import perf_counter
import os
from LFUCache import LFUCache
from heapq import nlargest

def get_min_window(positions):
    len1 = max([pos for position_list in positions for pos in position_list]) + 1
    len2 = len(positions)

    sequence = [len2] * len1
    for token_key, token_pos in enumerate(positions):
        for idx in token_pos:
            sequence[idx] = token_key

    hash_pat = ([1] * len2) + [0]
    hash_str = [0] * (len2+1)

    start = 0
    start_idx = -1
    min_len = float('inf')

    count = 0
    for j, token_key in enumerate(sequence):
        hash_str[token_key] += 1

        if hash_str[token_key] <= hash_pat[token_key]: count += 1

        if count == len2:
            while True:
                if hash_str[sequence[start]] > hash_pat[sequence[start]] or sequence[start] == len2:
                    if hash_str[sequence[start]] > hash_pat[sequence[start]]:
                        hash_str[sequence[start]] -= 1
                    start += 1
                else: break
                
            len_window = j - start + 1
            if min_len > len_window:
                min_len = len_window
                start_idx = start

    window = [start_idx, start_idx+min_len-1] # both inclusive

    return window, min_len

def dynamically_init_searcher(**kwargs):
    """Dynamically initializes a Tokenizer object from this
    module.

    Parameters
    ----------
    kwargs : Dict[str, object]
        python dictionary that holds the variables and their values
        that are used as arguments during the class initialization.
        Note that the variable `class` must be here and that it will
        not be passed as an initialization argument since it is removed
        from this dict.
    
    Returns
        ----------
        object
            python instance
    """
    return dynamically_init_class(__name__, **kwargs)

class BaseSearcher:

    def __init__(self) -> None:
        self.cache = LFUCache(100_000)

    def search(self, index, query_tokens, top_k):
        pass

    def batch_search(self, index, reader, tokenizer, output_file, top_k=20):
        print("searching...")

        with open(f"{output_file}","w") as f:
            ctr = 1
            for question, documents in reader.read_questions():
                #print(f"{ctr/300 : %}", '\r')
                ctr += 1

                # aplies the tokenization to get the query_tokens
                query_tokens = tokenizer.tokenize(question)
                query_tokens = [token[0] for token in query_tokens]

                results, latency = self.search(index, query_tokens, top_k)
                page = results.get_page(0)
                results_list_pmid = [result[0] for result in page]

                if results_list_pmid:
                    evaluation = self.evaluate_ranking(results_list_pmid, documents)
                else: evaluation = None, None, None, None

                f.write(f"{evaluation}|{latency}\n")

        results = self.final_results(output_file)

        with open(f"{output_file}_averages.txt", "w") as f:
            f.write(f"{results}")

    def load_postings(self, index : InvertedIndex, query_tokens):
        term_postings = dict()
        query_tf = dict()
        tokens_idf = list()
        
        for token in query_tokens:
            if token not in term_postings:
                query_tf[token] = 1
                postings = self.cache.get(token)
                if postings == -1:
                    if token in index.pointers:
                        tokens_idf.append(index.pointers[token][0])
                        ptr_bytes = index.pointers[token][1]
                        with open(f"{index.path}/index.index","rb") as f:
                            f.seek(ptr_bytes)
                            try:
                                postings = eval(f.readline().split(b":")[1])
                            except:
                                print(token)
                                print(index.pointers[token])
                                print(f.readline().split(b":"))
                                quit()
                            term_postings[token] = postings
                            self.cache.set(key=token, value=postings)
                else:
                    term_postings[token] = postings
            else:
                query_tf[token] += 1

        return term_postings, query_tf, tokens_idf

    def get_top_k(self, scores : dict, k):
        tops = set()
        minimun = (None, 1)
        for doc_id, score_info in scores.items():
            if len(tops) < k:
                tops.add((doc_id, score_info[0]))
                if len(tops) == k: minimun = min(tops, key=lambda x: x[1])
            else:
                if score_info[0] > minimun[1]:
                    tops.remove(minimun)
                    minimun = min(tops, key=lambda x: x[1])
                    tops.add((doc_id, score_info[0]))
        return tops




    def final_results(self, output_file):
        with open(f"{output_file}", "r") as results:
            precisions = []
            recalls = []
            f_measures = []
            av_precisions = []
            latencies = []
            query_count = 0
            for line in results:
                measures, latency = line.split("|")
                measures = eval(measures)
                latency = eval(latency)
                precisions.append(measures[0])
                recalls.append(measures[1])
                f_measures.append(measures[2])
                av_precisions.append(measures[3])
                latencies.append(latency)
                query_count += 1

        precision = sum(precisions)/len(precisions)
        recall = sum(recalls)/len(recalls)
        f_measure = sum(f_measures)/len(f_measures)
        av_precision = sum(av_precisions)/len(av_precisions)
        latency = sum(latencies)/len(latencies)

        query_throughput = 1/latency

        return precision, recall, f_measure, av_precision, latency, query_throughput
            
    def search_it(self, index, reader, tokenizer, top_k=10):
        while True:
            ipt = input("Input query (q to exit): ")
            if ipt == 'q':
                return
            query_tokens = tokenizer.tokenize(ipt)
            results = self.search(index, query_tokens, top_k)

            # write results to disk
            # TODO

    def evaluate_ranking(self, results : list, documents : list):
        documents = [int(document) for document in documents]
        s_results = set(results)  
        s_documents = set(documents)
        tp = len(s_results.intersection(s_documents))
        fp = len(s_results) - tp
        fn = len(s_documents) - tp
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        if not recall and not precision: f_measure = 0
        else: f_measure = 2*recall*precision/(recall+precision)
        hits = 0
        count = 1
        precisions = []
        for pmid in results:
            if pmid in s_documents: 
                hits += 1
                precisions += [hits/count]
            count += 1
        if not precisions: average_precision = 0
        else: average_precision = sum(precisions)/len(precisions)
        return precision, recall, f_measure, average_precision

class TFIDFRanking(BaseSearcher):

    def __init__(self, smart, B, mod, **kwargs) -> None:
        super().__init__(**kwargs)
        self.smart = smart
        self.B = B
        self.mod = mod
        print("init TFIDFRanking|", f"{smart=} {B=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def search(self, index: InvertedIndex, query_tokens, top_k):
        tic = perf_counter()

        # index must be compatible with tfidf
        term_postings, query_tf, tokens_idf = self.load_postings(index=index, query_tokens=query_tokens)

        #query_tf = Counter(query_tokens)
        #tokens_idf = {token: index.pointers[token][0] for token in query_tf if token in index.pointers}
        if not tokens_idf: avg_idf = 0
        else: avg_idf = sum(tokens_idf)/len(tokens_idf)
        high_idf = {token for token in query_tf if token in index.pointers and index.pointers[token][0] > avg_idf}
        
        doc, query = tuple(self.smart.split("."))
        if query == "ltc":
            query_norm = sqrt(sum([((1+log10(query_tf[term])) * index.pointers[term][1])**2 for term in query_tf if term in index.pointers]))
        else:
            print("Schema not supported!")
            quit()

        scores = dict()

        for token, q_tf in query_tf.items():
            if token in index.pointers:
                for doc_id, d_tf, positions in term_postings[token]:
                    if doc == "lnc":
                        doc_norm = index.mappings[str(doc_id)][2]
                        tf = 1 + log10(d_tf)
                        idf = 1
                    elif doc == "ltc":
                        doc_norm = index.mappings[str(doc_id)][2]
                        tf = 1 + log10(d_tf)
                        idf = index.pointers[token][0]
                    else:
                        print("Schema not supported!")
                        quit()
                    doc_w = tf*idf/doc_norm

                    if query == "ltc":
                        tf = 1+log10(q_tf)
                    else:
                        print("Schema not supported!")
                        quit()
                    query_w = tf*idf/query_norm

                    local_score = query_w*doc_w

                    if doc_id in scores:
                        scores[doc_id][0] += local_score
                        if token in high_idf:
                            scores[doc_id][2] += 1
                            scores[doc_id][1].append(positions)
                    else:
                        # scores -> doc_id: [weight, [positions lists], terms_hit]
                        if token in high_idf:
                            window = positions
                            scores[doc_id] = [local_score, [window], 1]
                        else:
                            scores[doc_id] = [local_score, [], 0]

        if self.B:
            window_size = len(query_tf)
            for doc_id, values in scores.items():
                positions = values[1]
                hits = values[2]
                if positions:
                    diff = get_min_window(positions)[1] - window_size

                if hits < len(high_idf) or not positions:
                    #print("no boost")
                    boost = 1
                elif diff <= window_size or hits == 1:
                    boost = self.B
                    #print("max_boost", boost)
                else:
                    boost = ((self.B-1)/((self.mod*diff)+1)) + 1
                    #print("mid boost", boost)

                scores[doc_id][0] = scores[doc_id][0] * boost

        #scores_final = list(scores.items())
        #scores_final.sort(key=lambda item: item[1][0], reverse=True)

        tops = self.get_top_k(scores, top_k)
        scores_final = list(tops)
        scores_final.sort(key=lambda x: x[1], reverse=True)

        scores_final = [(index.mappings[str(doc_id)][0], score) for doc_id, score in scores_final]
        
        results = Paginator(scores_final, top_k)

        tok = perf_counter()

        return results, (tok-tic)

class BM25Ranking(BaseSearcher):

    def __init__(self, k1, b, B, mod, **kwargs) -> None:
        super().__init__(**kwargs)
        self.k1 = k1
        self.b = b
        self.B = B
        self.mod = mod
        print("init BM25Ranking|", f"{k1=}", f"{b=}, {B=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def search(self, index : InvertedIndex, query_tokens, top_k):
        tic = perf_counter()

        # index must be compatible with bm25
        term_postings, query_tf, tokens_idf = self.load_postings(index=index, query_tokens=query_tokens)
                

        #query_tf = Counter(query_tokens)
        #tokens_idf = {token: index.pointers[token][0] for token in query_tf if token in index.pointers}
        if not tokens_idf: avg_idf = 0
        else: avg_idf = sum(tokens_idf)/len(tokens_idf)
        high_idf = {token for token in query_tf if token in index.pointers and index.pointers[token][0] > avg_idf}

        scores = dict()
        avdl = index.metadata["avdl"]

        for token, _ in query_tf.items():
            if token in index.pointers:
                for doc_id, tf, positions in term_postings[token]:
                    dl = index.mappings[str(doc_id)][1]
                    idf = index.pointers[token][0]
                    B = (1-self.b)+self.b*(dl/avdl)
                    tf_ = tf/B

                    local_score = idf*(((self.k1+1)*tf_)/(self.k1+tf_))

                    if doc_id in scores:
                        scores[doc_id][0] += local_score
                        if token in high_idf:
                            scores[doc_id][2] += 1
                            scores[doc_id][1].append(positions)
                    else:
                        # scores -> doc_id: [weight, [positions lists], terms_hit]
                        if token in high_idf:
                            window = positions
                            scores[doc_id] = [local_score, [window], 1]
                        else:
                            scores[doc_id] = [local_score, [], 0]

        if self.B:
            window_size = len(query_tf)
            for doc_id, values in scores.items():
                positions = values[1]
                hits = values[2]
                if positions:
                    diff = get_min_window(positions)[1] - window_size

                if hits < len(high_idf) or not positions:
                    #print("no boost")
                    continue
                elif diff <= window_size or hits == 1:
                    boost = self.B
                    #print("max_boost", boost)
                else:
                    boost = ((self.B-1)/((self.mod*diff)+1)) + 1
                    #print("mid boost", boost)

                scores[doc_id][0] = scores[doc_id][0] * boost

        #scores_final = list(scores.items())
        #scores_final.sort(key=lambda item: item[1][0], reverse=True)

        #tops = nlargest(n=top_k, iterable=scores.items(), key=lambda item: item[1][0])
        tops = self.get_top_k(scores, top_k)
        scores_final = list(tops)
        scores_final.sort(key=lambda x: x[1], reverse=True)

        scores_final = [(index.mappings[str(doc_id)][0], score) for doc_id, score in scores_final]

        results = Paginator(scores_final, top_k)

        tok = perf_counter()

        return results, tok-tic

class Paginator():

    def __init__(self, obj : list, page_size = 10) -> None:
        self.page_size = page_size
        self.obj = obj
        self.page_number = ceil(len(self.obj)/page_size)

    def get_page(self, page : int):
        if len(self.obj) == 0:
            return self.obj

        if page > self.page_number - 1 or page < 0:
            return None

        if page == self.page_number - 1:
            remainder = len(self.obj) % self.page_size
            if remainder == 0:
                return self.obj[-self.page_size:]
            else:
                return self.obj[-remainder:]
        else:
            return self.obj[self.page_size*page:self.page_size*(page+1)]


