"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: Afonso Campos, Dinis Lei

Indexer module

Holds the code/logic addressing the Indexer class
and the index managment.

"""

from utils import dynamically_init_class
from os import listdir
from math import log10
import math
import time
import os
import psutil
import ast
import json
from collections import Counter
import multiprocessing
from multiprocessing import Queue, Array, Value, Process
from reader import PubMedReader
from tokenizers import Tokenizer

def dynamically_init_indexer(**kwargs):
    """Dynamically initializes a Indexer object from this
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

class BaseIndex:
    """
    Top-level Index class
    
    This loosly defines a class over the concept of 
    an index.
    """

    def get_tokenizer_kwargs(self):
        """
        Index should store the arguments used to initialize the index as aditional metadata
        """
        return {}

    def add_term(self, term, doc_id, *args, **kwargs):
        raise NotImplementedError()
    
    def print_statistics(self):
        raise NotImplementedError()
    
    @classmethod
    def load_from_disk(cls, path_to_folder:str):
        """
        Loads the index from disk, note that this
        the process may be complex, especially if your index
        cannot be fully loaded. Think of ways to coordinate
        this job and have a top-level abstraction that can
        represent the entire index even without being fully load
        in memory.
        
        Tip: The most important thing is to always know where your
        data is on disk and how to easily access it. Recall that the
        disk access are the slowest operation in a computation device, 
        so they should be minimized.
        
        Parameters
        ----------
        path_to_folder: str
            the folder where the index or indexes are stored.
            
        """
        return cls()


class Indexer:
    """
    Top-level Indexer class
    
    This loosly defines a class over the concept of 
    an index.

    """
    
    def __init__(self, 
                 index_instance: BaseIndex,
                 **kwargs):
        super().__init__()
        self._index = index_instance
    
    def get_index(self):
        return self._index
    
    def build_index(self, reader, tokenizer, index_output_folder):
        """
        Holds the logic for the indexing algorithm.
        
        This method should be implemented by more specific sub-classes
        
        Parameters
        ----------
        reader : Reader
            a reader object that knows how to read the collection
        tokenizer: Tokenizer
            a tokenizer object that knows how to convert text into
            tokens
        index_output_folder: str
            the folder where the resulting index or indexes should
            be stored, with some additional information.
            
        """
        raise NotImplementedError()
    

class SPIMIIndexer(Indexer):
    """
    The SPIMIIndexer represents an indexer that
    holds the logic to build an index according to the
    spimi algorithm.

    """
    def __init__(self, 
                 posting_threshold,
                 memory_threshold,
                 n_processes,
                 schema,
                 n_tokenizers,
                 n_indexers,
                 **kwargs):
        # lets suppose that the SPIMIIindex uses the inverted index, so
        # it initializes this type of index
        super().__init__(InvertedIndex(), **kwargs)
        self._index : InvertedIndex
        self.memory_threshold = memory_threshold if memory_threshold else 60
        self.posting_threshold = posting_threshold if posting_threshold else math.inf
        self.n_processes = n_processes
        self.n_tokenizers = n_tokenizers
        self.n_indexers = n_indexers
        self.total_memory = psutil.virtual_memory()[0] # bytes
        self.process = psutil.Process(os.getpid())
        self.term_num = 0
        self.doc_num = 0
        self.curr_id = 0
        self.schema = kwargs["tfidf"]["smart"]
        self.index_output_folder = None
        print("init SPIMIIndexer|", f"{posting_threshold=}, {memory_threshold=}, {n_processes=}, {schema=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

        self.reader = None
        self.tokenizer = None
        self.bytes_read = 0

    def set_up_dirs(self, index_output_folder: str):
        """ Creates temporary folders and permanent if they don't exist"""
        isExist = os.path.exists(index_output_folder)
        if not isExist:
            os.makedirs(index_output_folder)
        else:
            files = listdir(index_output_folder)
            for f in files:
                os.remove(f"{index_output_folder}/{f}")

        isExist = os.path.exists("blocks")
        if not isExist:
            os.makedirs("blocks")
        else:
            files = listdir("blocks")
            for f in files:
                os.remove(f"blocks/{f}")

    def dump_doc_mappings(self):
        """ Stores document mapping """
        new_dict = {tmp[0]: [pmid] + tmp[1:] for pmid, tmp in self.doc_mappings.items()}
        self.doc_mappings.clear()
        with open(f"{self.index_output_folder}/doc_mappings.json","w") as f:
            json.dump(new_dict, f, indent=4)

    def reader_worker(self, documents_queue: Queue):
        """ 
            Producer
            Reads documents and stores them in a documents queue 
        """
        print("INIT READER")
        doc_id = 0
        pmid_check = set()
        for document, pmid, _ in self.reader.read_document():
            # Evaluate repeated documents
            if pmid in pmid_check: 
                continue           
            pmid_check.add(pmid)
            
            # Insert document in queue
            msg = [
                doc_id,
                pmid,
                document,
            ]
            documents_queue.put(msg)
            doc_id += 1

        for _ in range(self.n_tokenizers):
            documents_queue.put(None)
        with self.doc_count.get_lock():
            self.doc_count.value = doc_id
        print("END READER")


    def tokenizer_worker(self, documents_queue: Queue, token_stream_queue: Queue, get_idf=False, process_id = 0):
        """ 
            Consumer/Producer
            Tokenizes items of the documents queue and stores them in token stream queue 
        """
        print(f"INIT TK {process_id}")
        sum_token = 0
        while True:
            # Get and Unpack message
            msg = documents_queue.get()
            if not msg:
                #documents_queue.put(None)
                break 
            doc_id, pmid, document = msg
            
            # Generate token stream
            token_stream = self.tokenizer.tokenize(document)
            token_positions = dict()
            for token, position in token_stream:
                if token in token_positions:
                    token_positions[token] += [position]
                else:
                    token_positions[token] = [position]

            if get_idf == True and self.schema == "ltc.ltc":
                doc_norm = math.sqrt(sum([((1+log10(len(tf)) * log10(self.doc_count.value/self._index.idfs[term]))**2)  for term, tf in token_positions.items()]))
                self.doc_mappings[pmid] += [doc_norm]
                continue


            sum_token += len(token_stream)
            #token_stream_final = Counter(token_stream_final)

            msg = {
                "doc_id": doc_id,
                #"tk_stream": token_stream_final,
                "tk_positions": token_positions,
                "end": None
            }

            token_stream_queue.put(msg)

            tmp = [doc_id, len(token_stream)]
            if self.schema == "lnc.ltc":
                doc_norm = math.sqrt(sum([(1+log10(len(tf)))**2 for _, tf in token_positions.items()])) # weight (whats inside the sum to the power of 2) will depend on schema
                tmp += [doc_norm]
            self.doc_mappings[pmid] = tmp

        with self.sum_token.get_lock():
            self.sum_token.value += sum_token

        token_stream_queue.put({"end": process_id})
        print(f"END TK {process_id}")   

    def SPIMI_worker(self, token_stream_queue: Queue, is_processing: list, n_tk, target_mem = 30, process_id = 0):
        """ 
            Consumer
            Builds a temporary index of items in the token stream queue and stores it in a temporary file
        """
        process = psutil.Process(os.getpid())
        count = 0
        endFlag = False
        print(f"INIT SPIMI-{process_id}")
        current_use = (process.memory_info().rss/self.total_memory)*100
        tk_processes = [True for _ in range(n_tk)]
        #print(f'SPIMI MEM: {target_mem}, {current_use}')
        while True:
            msg = token_stream_queue.get()

            # End Process when every end message is found
            if msg['end'] is not None:
                print(f"PROCESS-{process_id}, END TK-{msg['end']}")
                tk_processes[msg['end']] = False
                token_stream_queue.put(msg)
                if not any(tk_processes):
                    endFlag = True
                    break
                continue

            doc_id = msg['doc_id']
            #token_stream = msg['tk_stream']
            token_positions : dict = msg['tk_positions']

            for term, positions in token_positions.items():
                self._index.add_term(term=term, doc_id=doc_id, freq=len(positions), positions=positions)

            # Memory Check
            if count == 1000:
                current_use = (process.memory_info().rss/self.total_memory)*100
                #print(f'SPIMI MEM: {target_mem}, {current_use}')
                count = 0
            if current_use >= target_mem:
                break
            count += 1

        self._index.write_to_disk(f"blocks/block{doc_id}.index")
        self._index.reset_dictionary()

        is_processing[process_id] = not endFlag
        
        if endFlag:
            print(f"ENDED DOCUMENTS {process_id}")
        else:
            print(f"MEM END SPIMI-{process_id}")



    def mySPIMI(self, bytes_read_start, bytes_read_offset, processing, target_mem = 30, process_id = 0):
        """ SPIMI Worker. Generates an Index Block"""
        cur_byte = bytes_read_start[process_id] + bytes_read_offset[process_id]
        print(f"..INIT SPIMI Process-{process_id}: {cur_byte}")

        process = psutil.Process(os.getpid())

        current_use = (process.memory_info().rss/self.total_memory)*100
        print(f"{current_use = } | {target_mem = }")
        
        endFlag = 1
        sum_token = 0
        # Read its share of documents
        for document, pmid, bytes_r in self.reader.read_document(cur_byte):
            bytes_read_offset[process_id] += bytes_r
            
            if pmid in self.doc_mappings: continue

            # Get a unique doc_id
            with self.doc_id.get_lock():
                doc_id = self.doc_id.value
                self.doc_id.value += 1

            # Generate token stream 
            token_stream = self.tokenizer.tokenize(document)

            token_positions = dict()
            for tuple in token_stream:
                token = tuple[0]
                if token in token_positions:
                    token_positions[token] += [tuple[1]]
                else:
                    token_positions[token] = [tuple[1]]

            token_stream_final = [posting[0] for posting in token_stream]

            sum_token += len(token_stream)

            token_stream_final = Counter(token_stream_final)

            tmp = [doc_id, len(token_stream)]
            if self.schema == "lnc.ltc":
                doc_norm = math.sqrt(sum([(1+log10(tf))**2 for _, tf in token_stream_final.items()])) # weight (whats inside the sum to the power of 2) will depend on schema
                tmp += [doc_norm]
            self.doc_mappings[pmid] = tmp

            # add terms to temporary index
            for term, freq in token_stream_final.items():
                self._index.add_term(term=term, doc_id=doc_id, freq=freq, positions=token_positions[term])

            current_use = (process.memory_info().rss/self.total_memory)*100 
            # Terminate if share of documents is read       
            if bytes_read_offset[process_id] + bytes_read_start[process_id] >= bytes_read_start[process_id + 1]:
                endFlag = 1
                break 
            # Terminate if memory limit is surparssed
            if current_use >= target_mem:
                print(f"Process-{process_id} Mem Limit")
                endFlag = 0
                break

        # Dump block
        self._index.write_to_disk(f"blocks/block{bytes_read_offset[process_id] + bytes_read_start[process_id]}.index")
        self._index.reset_dictionary()
        processing[process_id] = not endFlag

        with self.sum_token.get_lock():
            self.sum_token.value += sum_token

        print(f"..END SPIMI Process-{process_id}: {bytes_read_offset[process_id] + bytes_read_start[process_id]}")

        if endFlag:
            print(f"ENDED DOCUMENTS {process_id}")


    def getIDF(self, bytes_read_start, bytes_read_offset, processing, target_mem = 30, process_id = 0):
        """ Calculate the idf of each document (only supported by ltc.ltc)"""

        cur_byte = bytes_read_start[process_id] + bytes_read_offset[process_id]
        print(f"..INIT SPIMI Process-{process_id}: {cur_byte}")

        process = psutil.Process(os.getpid())

        current_use = (process.memory_info().rss/self.total_memory)*100
        print(f"{current_use = } | {target_mem = }")
        
        endFlag = 1
        ctr = 0
        # Read its share of documents
        for document, pmid, bytes_r in self.reader.read_document(cur_byte):
            bytes_read_offset[process_id] += bytes_r
            token_stream = self.tokenizer.tokenize(document)

            token_stream_final = Counter(token_stream)
            #print(self._index.idfs)
            print(self.doc_count.value)
            doc_norm = math.sqrt(sum([((1+log10(tf) * log10(self.doc_count.value/self._index.idfs[term[0]]))**2)  for term, tf in token_stream_final.items()]))
            self.doc_mappings[pmid] += [doc_norm]


            token_positions = dict()
            #token_stream_final = []
            for token, position in token_stream:
                if token in token_positions:
                    token_positions[token] += [position]
                else:
                    token_positions[token] = [position]
                #token_stream_final.append(token)

    
            #print(f"{token_positions}")
            #print(f"{[self._index.idfs[term] for term in token_positions]}")
            doc_norm2 = math.sqrt(sum([((1+log10(len(tf)) * log10(self.doc_count.value/self._index.idfs[term]))**2)  for term, tf in token_positions.items()]))
            

            print(doc_norm, doc_norm2)

            print("TK", token_stream)
            print("\n")

            print(token_stream_final)
            print(token_positions)

            print("\n\n")

            current_use = (process.memory_info().rss/self.total_memory)*100        

            # Terminate if share of documents is read 
            if bytes_read_offset[process_id] + bytes_read_start[process_id] >= bytes_read_start[process_id + 1]:
                endFlag = 1
                break 
            # Terminate if memory limit is surparssed
            if current_use >= target_mem:
                print(f"Process-{process_id} Mem Limit")
                endFlag = 0
                break
            ctr += 1
            if ctr == 10:
                break

        processing[process_id] = not endFlag

        print(f"..END SPIMI Process-{process_id}: {bytes_read_offset[process_id] + bytes_read_start[process_id]}")

        if endFlag:
            print(f"ENDED DOCUMENTS {process_id}")
    
    def launch_procecess(self, target):
        """ Laucn mutiple workers depending on target function """
        processing = multiprocessing.Array("i", tuple(1 for _ in range(self.n_processes)), lock=False)
        bytes_read_offset = multiprocessing.Array("L", tuple(0 for _ in range(self.n_processes)), lock=False)
        while any(processing):
            current_use = (self.process.memory_info().rss/self.total_memory)*100
            print(f"B4: {current_use}")
            # print(bytes_read)
            processes = []
            for i in range(self.n_processes):
                if processing[i]:
                    args = (
                        [0, math.inf], 
                        bytes_read_offset, 
                        processing, 
                        (self.memory_threshold-current_use)/self.n_processes, 
                        i
                    )
                    processes.append(multiprocessing.Process(target=target, args=args))
                    processes[-1].start()

            for i in range(len(processes)):
                processes[i].join()
                processes[i].close()

            # print(processing)
            current_use = (self.process.memory_info().rss/self.total_memory)*100
            print(f"Af: {current_use}")
    
    def merge(self, index_output_folder):
        """ Merge mutilpe temporary blocks """
        print("Start Merge...")
        docs = set()
        current_use = (self.process.memory_info().rss/self.total_memory)*100
        print(f"{current_use = }")
        buffers = []
        blocks = []
        blocks_file = listdir("blocks")
        self.segments = len(blocks_file)
        dict_size = int(((self.memory_threshold/100)*self.total_memory)/3) if self.memory_threshold != math.inf else math.inf
        byte_size = int((((self.memory_threshold/100)*self.total_memory)-dict_size)/self.segments) if self.memory_threshold != math.inf else -1
        idx = 0
        bypass = 10000
        print("First Read...")
        for i in range(len(blocks_file)):
            blocks.append([])
            fname = blocks_file[i]
            str = f"blocks/{fname}"
            buffers.append([str,0])
            buffer = open(str,"rb")
            for line in buffer.readlines(byte_size):
                buffers[i][1] += len(line)
                blocks[i].append(line.strip().split(":".encode('utf-8')))
            buffer.close()
            idx += 1
        current_use = (self.process.memory_info().rss/self.total_memory)*100
        print(f"{current_use = }")
        
        # Merge temporary blocks
        self._index.reset_dictionary()
        self.n_terms = 0
        terms_added = 0
        while blocks:
            # Get the 1st term alphabetically
            term = min([terms[0][0] for terms in blocks])
            terms_added += 1

            # Get indexes where term appears
            indexes = [idx for idx in range(len(blocks)) if blocks[idx][0][0] == term]

            # Create posting list and remove term from blocks
            postings = list()
            for idx in indexes:
                block_postings = ast.literal_eval(blocks[idx][0][1].decode('utf-8'))
                for posting in block_postings:
                    postings.append(posting)
                    docs.add(posting[0])
                blocks[idx].pop(0)


            postings.sort()
            self._index.dictionary[term] = postings

            # Check for empty blocks and fill/remove them
            empty = [idx for idx in indexes if len(blocks[idx]) == 0]
            for idx in empty:
                if byte_size != math.inf:
                    buffer = open(buffers[idx][0],"rb")
                    buffer.seek(buffers[idx][1])
                    lines = buffer.readlines(byte_size)
                    buffer.close()
                    if len(lines) <= 0: 
                        continue
                    else:
                        for line in lines:
                            buffers[idx][1] += len(line)
                            blocks[idx].append(line.strip().split(":".encode('utf-8')))

            buffers = [buffers[i] for i in range(len(blocks)) if blocks[i]]
            blocks = [block for block in blocks if block]

            # If dictionary size limit is reached dump information to file
            current_use = (self.process.memory_info().rss/self.total_memory)*100
            if len(self._index.dictionary) >= bypass:
                if current_use >= self.memory_threshold or len(self._index.dictionary) >= self.posting_threshold:
                    # Dump to disk
                    print(f"   Dump...")
                    print(f"{len(self._index.dictionary)} {bypass}")
                    current_use = (self.process.memory_info().rss/self.total_memory)*100
                    print(f"   {current_use = }")
                    print(f"   {terms_added = }")
                    terms = sorted(self._index.dictionary)
                    self.term_num += len(terms)
                    bypass = len(terms) if len(terms) > bypass else bypass
                    # string = f"{terms[0].decode('utf-8')}-{terms[-1].decode('utf-8')}"
                    self._index.write_index(index_output_folder, self.doc_count.value)
                    self._index.reset_dictionary()
                    current_use = (self.process.memory_info().rss/self.total_memory)*100
                    print(f"   After Dump...")
                    print(f"   {current_use = }")
                    terms_added = 0
                

        # Dump the remaining thats left in the dictionary
        terms = sorted(self._index.dictionary)
        self.term_num += len(terms)
        # string = f"{terms[0].decode('utf-8')}-{terms[-1].decode('utf-8')}"
        self._index.write_index(index_output_folder,self.doc_count.value)
        self._index.reset_dictionary()
            
        # Delete blocks
        # for fname in listdir("blocks"):
        #     os.remove(f"blocks/{fname}")

        print(f"{len(docs) =}")

    def build_index(self, reader: PubMedReader, tokenizer: Tokenizer, index_output_folder):
        it_tic = time.perf_counter()
        print(f"{self.schema =}")

        if self.schema not in ["ltc.ltc", "lnc.ltc"]:
            print("Schema not supported")
            quit()

        # Set Up
        self.set_up_dirs(index_output_folder)

        self.index_output_folder = index_output_folder
        self.reader = reader
        self.tokenizer = tokenizer

        # Init shared memory variables
        #fsize = reader.get_file_size()
        #mids = reader.get_middle(fsize, self.n_processes)
        #self.bytes_read_start = Array("L", tuple(mids), lock=False)
        self.doc_count = Value('L', 0, lock=True)
        self.sum_token = Value('L', 0, lock=True)
        manager = multiprocessing.Manager()
        self.doc_mappings = manager.dict()
        documents_queue = Queue(100)
        token_stream_queue = Queue(200)
        is_processing = multiprocessing.Array("i", tuple(1 for _ in range(self.n_indexers)), lock=False)

        # Lauch Processes
        rt_tic = time.perf_counter()

        args = (documents_queue, )
        reader_process = Process(target=self.reader_worker, args=args)
        reader_process.start()

        args = [documents_queue, token_stream_queue, False,]
        tk_processes : list[Process] = []
        for i in range(self.n_tokenizers):
            tk_process = Process(target=self.tokenizer_worker, args=args + [i])
            tk_process.start()
            tk_processes.append(tk_process)

        current_use = (self.process.memory_info().rss/self.total_memory)*100
        args = [token_stream_queue, is_processing, self.n_tokenizers, (self.memory_threshold-current_use)/self.n_indexers,]
        #idx_processes : list[Process] = [None for _ in range(n_indexers)]
        while any(is_processing):
            for p in is_processing:
                print(p,end= ' ')
            print()
            idx_processes: list[Process] = []
            for i in range(self.n_indexers):
                if is_processing[i]:
                    idx_process = Process(target=self.SPIMI_worker, args=args + [i])
                    idx_process.start()
                    idx_processes.append(idx_process)
            for p in idx_processes:
                p.join()
                p.close()

        reader_process.join()
        reader_process.close()
        for p in tk_processes:
            p.join()
            p.close()




        #rt_tic = time.perf_counter()
        #self.launch_procecess(target=self.mySPIMI)
        rt_toc = time.perf_counter()
        self.rt_time = rt_toc - rt_tic

        # print(f"TIME {self.rt_time}")
        # print(f"{self.doc_id.value = }")
        # print(f"{len(self.doc_mappings) =}")
        # print(self.doc_id.value)
        # print(self.sum_token.value)
        
        avdl = self.sum_token.value/self.doc_count.value
        # print("Average doc len:", avdl)
        metadata = tokenizer.get_args()
        metadata["doc_num"] = self.doc_count.value
        metadata["avdl"] = avdl
        metadata["schema_indexer"] = self.schema

        if self.schema == "lnc.ltc":
            self.dump_doc_mappings()
        with open(f"{index_output_folder}/metadata.txt", "w") as file:
            json.dump(metadata, file, indent=4)
            
        m_tic = time.perf_counter()
        self.merge(self.index_output_folder)
        m_toc = time.perf_counter()
        self.merging_time = m_toc - m_tic

        if self.schema == "ltc.ltc":
            # doc mappings with correct norm (tf*idf)
            rt_tic = time.perf_counter()
            args = (documents_queue, )
            reader_process = Process(target=self.reader_worker, args=args)
            reader_process.start()
        
            args = [documents_queue, token_stream_queue, True, ]
            tk_processes : list[Process] = []
            for i in range(self.n_tokenizers):
                tk_process = Process(target=self.tokenizer_worker, args=args + [i])
                tk_process.start()
                tk_processes.append(tk_process)
            
            reader_process.join()
            reader_process.close()
            for p in tk_processes:
                p.join()
                p.close()

            rt_toc = time.perf_counter()
            self.rt_time += rt_toc - rt_tic


            self.dump_doc_mappings()

        it_toc = time.perf_counter()
        self.indexing_time = it_toc - it_tic

        self.print_statistics(index_output_folder)
    
    def print_statistics(self, output):
        print("--- Indexing Statistics ---")
        print(f"Total indexing time: {self.indexing_time}")
        print(f"Merging time: {self.merging_time}")
        print(f"Read and Tokenize time: {self.rt_time}")
        print(f"Number of temporary index segments written to disk: {self.segments}")
        size = 0
        for path, _, files in os.walk(output):
            for f in files:
                fp = os.path.join(path, f)
                size += os.path.getsize(fp)
        print(f"Total index size on disk: {size}")
        print(f"Vocabulary size: {self.term_num}")
        print("---------------------------")

class BaseIndexer(Indexer):
    
    def __init__(self, 
                 posting_threshold, 
                 memory_threshold, 
                 **kwargs):
        # lets suppose that the SPIMIIindex uses the inverted index, so
        # it initializes this type of index
        super().__init__(InvertedIndex(), **kwargs)
        print("init Base|", f"{posting_threshold=}, {memory_threshold=}")

    def build_index(self, reader, tokenizer, index_output_folder):
        pass

class InvertedIndex(BaseIndex):
    
    # make an efficient implementation of an inverted index

    def __init__(self) -> None:
        super().__init__()
        self.dictionary = {}
        self.idfs = dict()
        self.byte_ptr = 0
        self.n_calls = 0

        # Search
        self.pointers = dict()
        self.metadata = dict()
        self.mappings = dict()
        self.path = ''

    def get_tokenizer_kwargs(self):
        """
        Index should store the arguments used to initialize the index as aditional metadata
        """

        return {"class": self.metadata["class"], "minL": int(self.metadata["minL"]), "stopwords_path": self.metadata["stopwords_path"], "stemmer": self.metadata["stemmer"]} if self.metadata else None

    @classmethod
    def load_from_disk(cls, path_to_folder:str):
        index = InvertedIndex()
        index.load_mappings(path_to_folder)
        index.load_metadata(path_to_folder)
        index.load_pointers(path_to_folder)
        index.path = path_to_folder
        return index

    def load_metadata(self, path_to_folder:str):
        with open(f"{path_to_folder}/metadata.txt","r") as f:
            self.metadata = json.load(f)

    def load_mappings(self, path_to_folder:str):
        with open(f"{path_to_folder}/doc_mappings.json","r") as f:
            self.mappings = json.load(f)

    def load_pointers(self, path_to_folder:str):
        with open(f"{path_to_folder}/pointers.csv", "rb") as file:
            for line in file:
                line = line.decode().strip().split(",")
                self.pointers[line[0]] = [float(line[1])] + [int(line[2])]
    
    def print_statistics(self):
        print("Print some stats about this index.. This should be implemented by the base classes")
        print(f"Number of temporary index segments written to disk: {self.n_calls}")
    
    def add_term(self, term, doc_id, freq, positions, *args, **kwargs):
        if term not in self.dictionary:
            self.dictionary[term] = [(int(doc_id),freq,positions)]
            #self.doc_freq[term] = 1
        else:
            self.dictionary[term].append((int(doc_id),freq,positions))
            #self.doc_freq[term] += 1 

    def write_to_disk(self, path_to_file:str, flg = 1):
        """Write the inverted index to a file"""
        if self.dictionary:
            self.n_calls += flg
            file = open(path_to_file, "wb")
            sorted_terms = list(self.dictionary.keys())
            sorted_terms.sort()
            print(f"WRITE TO {path_to_file} {len(sorted_terms)} terms")
            file.write(''.join(f"{term}:{self.dictionary[term]}\n" for term in sorted_terms).encode())
            file.close()

    def write_index(self, path_to_folder:str, doc_num:int):
        if self.dictionary:
            index = open(path_to_folder+"/index.index", "ab")
            pointers = open(path_to_folder+"/pointers.csv", "ab")
            sorted_terms = list(self.dictionary.keys())
            sorted_terms.sort()
            for term in sorted_terms:
                try:
                    idf = log10(doc_num/len(self.dictionary[term]))
                    self.idfs[term.decode()] = idf
                except:
                    print(f"{doc_num = }")
                    print(f"{len(self.dictionary[term]) = }")
                    print(f"{self.dictionary[term] = }")
                    quit()
                pointers.write(f"{term.decode()},{idf},{int(self.byte_ptr)}\n".encode())
                string = f"{term.decode()}:{self.dictionary[term]}\n".encode()
                index.write(string)
                self.byte_ptr += len(string)
            index.close()
            pointers.close()

    def reset_dictionary(self):
        self.dictionary.clear()
    
    
