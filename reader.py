"""
Base template created by: Tiago Almeida & Sérgio Matos
Authors: Afonso Campos, Dinis Lei

Reader module

Holds the code/logic addressing the Reader class
and how to read text from a specific data format.

"""

import math
import struct
from utils import dynamically_init_class

import json
import gzip
import os
import io


def dynamically_init_reader(**kwargs):
    """Dynamically initializes a Reader object from this
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


class Reader:
    """
    Top-level Reader class
    
    This loosly defines a class over the concept of 
    a reader.
    
    Since there are multiple ways for implementing
    this class, we did not defined any specific method 
    in this started code.

    """
    def __init__(self, 
                 path_to_collection:str, 
                 **kwargs):
        super().__init__()
        self.path_to_collection = path_to_collection

    def read_document():
        pass
        
    
class PubMedReader(Reader):
    
    def __init__(self, 
                 path_to_collection:str,
                 **kwargs):
        super().__init__(path_to_collection, **kwargs)
        print("init PubMedReader|", f"{self.path_to_collection=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")
        self.bytes_read = 0
        self.curr_id = 0

    def get_file_size(self):
        stats = os.stat(self.path_to_collection)
        with gzip.open(self.path_to_collection, 'rb') as f:
            if stats.st_size > 4294967296:
                file_size = f.seek(0, io.SEEK_END)
                return file_size
            else:
                f.seek(-4, 2)
                return struct.unpack('I', f.read(4))[0]

    def get_middle(self, size, n_process):
        mids = [0]
        with gzip.open(self.path_to_collection, 'rb') as f:
            inc = size//n_process
            for i in range(1,n_process+1):
                f.seek(inc*i)
                offset = 0
                while True:
                    char = f.read(1)
                   
                    if char == b'\n' or char == b'':
                        mids.append(inc*i + offset+1)
                        break
                    offset += 1
        return mids
                
    def read_document(self, bytes_read=0):
        with gzip.open(self.path_to_collection, 'rb') as file:
            file.seek(bytes_read)
            for line in file:
                tmp = json.loads(line)
                bytes_read += len(line)
                yield tmp['title'] + " " + tmp['abstract'], int(tmp['pmid']), len(line)




class QuestionsReader(Reader):
    def __init__(self, 
                 path_to_questions:str,
                 **kwargs):
        super().__init__(path_to_questions, **kwargs)
        # I do not want to refactor Reader and here path_to_collection does not make any sense.
        # So consider using self.path_to_questions instead (but both variables point to the same thing, it just to not break old code)
        self.path_to_questions = self.path_to_collection
        print("init QuestionsReader|", f"{self.path_to_questions=}")
        if kwargs:
            print(f"{self.__class__.__name__} also caught the following additional arguments {kwargs}")

    def read_questions(self):
        with open(f"{self.path_to_questions}","r") as f:
            for line in f:
                tmp = json.loads(line)
                yield tmp["query_text"], tmp["documents_pmid"]
