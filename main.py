"""
Created by: Tiago Almeida & Sérgio Matos
Last update: 27-09-2022

Main python CLI interface for the IR class 
assigments.

The code was tested on python version 3.9.
Older versions (until 3.6) of the python 
interpreter may run this code [No verification
was performed, yet].

"""

import argparse
from core import engine_logic, add_more_options_to_indexer

class Params:
    """
    A container class used to store group of parameters.
    
    Note: Students can ignore this class, this is only used to
    help with grouping of similar arguments at the CLI.

    Attributes
    ----------
    params_keys : List[str]
        a list that holds the all of the name of the parameters
        added to this class

    """
    def __init__(self):
        self.params_keys = []
        
    def __str__(self):
        attr_str = ", ".join([f"{param}={getattr(self,param)}" for param in self.params_keys])
        return f"Params({attr_str})"
    
    def __repr__(self):
        return self.__str__()
    
    def add_parameter(self, parameter_name, parameter_value):
        """
        Adds at runtime a parameter with its respective 
        value to this object.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter/variable (identifier)
        parameter_value: object
            Value of the variable
        """
        if len(parameter_name)>1:
            if parameter_name[0] in self.params_keys:
                getattr(self, parameter_name[0]).add_parameter(parameter_name[1:], parameter_value)
            else:
                new_parms = Params()
                new_parms.add_parameter(parameter_name[1:], parameter_value)
                setattr(self, parameter_name[0], new_parms)
                self.params_keys.append(parameter_name[0])

        else:
            setattr(self, parameter_name[0], parameter_value)
            self.params_keys.append(parameter_name[0])
        
    def get_kwargs(self) -> dict:
        """
        Gets all of the parameters stored inside as
        python keyword arguments.
        
        Returns
        ----------
        dict
            python dictionary with variable names as keys
            and their respective value as values.
        """
        kwargs = {}
        for var_name in self.params_keys:
            value = getattr(self, var_name)
            if isinstance(value,Params):
                value = value.get_kwargs()

            kwargs[var_name] = value

        # is a nested dict with only one key? if so, tries to simplify if possible
        if len(kwargs)==1:# and any(isinstance(i,dict) for i in kwargs.values()
            key = list(kwargs.keys())[0]
            if isinstance(kwargs[key], dict):
                return kwargs[key] # just ignores the first dict

        return kwargs

    def get_kwargs_without_defaults(self):
        cli_recorder = CLIRecorder()
        kwargs = self.get_kwargs()
        # remove the arguments that were not specified on the terminal
        return { k:v for k,v in kwargs.items() if k in cli_recorder}


def shared_reader(parser, default_value=None):
    # this functions is for code reusability 
    parser.add_argument('--reader.class', 
                        type=str, 
                        default=default_value,
                        help=f'Type of reader to be used to process the input data. (default={default_value}).')

class Singleton(type):
    """
    Python cookbook
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class CLIRecorder(metaclass=Singleton):
    def __init__(self):
        self.args = set()

    def add_arg(self, arg):
        self.args.add(arg.split(".")[-1])

    def __contains__(self, value):
        return value in self.args

class RecordArgument(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
        self.cli_args = CLIRecorder()

    def __call__(self, parser, namespace, values, option_string=None):
        self.cli_args.add_arg(self.dest)
        setattr(namespace, self.dest, values)

def shared_tokenizer(parser, default_value=None):
    parser.add_argument('--tk.class', 
                                    #dest="class",
                                    type=str, 
                                    action=RecordArgument,
                                    default="PubMedTokenizer",
                                    help='Type of tokenizer to be used to process the loaded document. (default=PubMedTokenizer).')
    
    parser.add_argument('--tk.minL',
                                    #dest="minL",
                                    type=int, 
                                    action=RecordArgument,
                                    default=None,
                                    help='Minimum token length. The absence means that will not be used (default=None).')
    
    parser.add_argument('--tk.stopwords_path',
                                    #dest="stopwords_path",
                                    type=str,
                                    action=RecordArgument,
                                    default=None,
                                    help='Path to the file that holds the stopwords. The absence means that will not be used (default=None).')
    
    parser.add_argument('--tk.stemmer',
                                    #dest="stemmer",
                                    type=str, 
                                    action=RecordArgument,
                                    default=None,
                                    help='Type of stemmer to be used. The absence means that will not be used (default=None).')

    parser.add_argument('--tk.cache',
                                    #dest="cache",
                                    type=int, 
                                    action=RecordArgument,
                                    default=None,
                                    help='Enables cache of tokens and sets its value')

def grouping_args(args):
    """
    Auxiliar function to group the arguments group
    the optional arguments that belong to a specific group.
    
    A group is identified with the dot (".") according to the
    following format --<group name>.<variable name> <variable value>.
    
    This method will gather all of the variables that belong to a 
    specific group. Each group is represented as an instance of the
    Params class.
    
    For instance:
        indexer.posting_threshold and indexer.memory_threshold, will be 
        assigned to the same group "indexer", which can be then accessed
        through args.indexer
        
    Parameters
        ----------
        args : argparse.Namespace
            current namespace from argparse
            
    Returns
        ----------
        argparse.Namespace
            modified namespace after performing the grouping
    """
    
    
    namespace_dict = vars(args)
    keys = set(namespace_dict.keys())
    for x in keys:
        if "." in x:
            group_name, *param_name = x.split(".")
            if group_name not in namespace_dict:
                namespace_dict[group_name] = Params()
            
            namespace_dict[group_name].add_parameter(param_name, namespace_dict[x])
            
            del namespace_dict[x]
    
    return args

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="CLI interface for the IR engine")
    
    # operation modes
    # - indexer
    # - searcher
    mode_subparsers = parser.add_subparsers(dest='mode', 
                                            required=True)
    
    ############################
    ## Indexer CLI interface  ##
    ############################
    indexer_parser = mode_subparsers.add_parser('indexer', 
                                                help='Indexer help')
    
    indexer_parser.add_argument('path_to_collection', 
                                type=str, 
                                help='Name of the folder or file that holds the document collection to be indexed.')
    
    indexer_parser.add_argument('index_output_folder', 
                                type=str, 
                                help='Name of the folder where all the index related files will be stored.')
    
    indexer_settings_parser = indexer_parser.add_argument_group('Indexer settings', 'This settings are related to how the inverted index is built and stored.')
    indexer_settings_parser.add_argument('--indexer.class', 
                                    type=str, 
                                    default="SPIMIIndexer",
                                    help='(default=SPIMIIndexer).')
    
    indexer_settings_parser.add_argument('--indexer.posting_threshold', 
                                    type=int, 
                                    default=None,
                                    help='Maximum number of postings that each index should hold.')
    
    indexer_settings_parser.add_argument('--indexer.memory_threshold', 
                                    type=int, 
                                    default=None,
                                    help='Maximum limit of RAM that the program (index) should consume.')

    indexer_settings_parser.add_argument('--indexer.n_processes', 
                                    type=int, 
                                    default=1,
                                    help='Number of parallel processes the program should use')

    indexer_settings_parser.add_argument('--indexer.n_tokenizers', 
                                    type=int, 
                                    default=8,
                                    help='Number of parallel tokenizers processes the program should launch')

    indexer_settings_parser.add_argument('--indexer.n_indexers', 
                                    type=int, 
                                    default=1,
                                    help='Number of parallel indexers processes the program should launch')

    indexer_settings_parser.add_argument('--indexer.schema', 
                                    type=str, 
                                    default="lnc.ltc",
                                    help='Indexing schema.')
        
    indexer_doc_parser = indexer_parser.add_argument_group('Indexer document processing settings', 'This settings are related to how the documents should be loaded and processed to tokens.')

    indexer_settings_parser.add_argument('--indexer.bm25.cache_in_disk', 
                                    action="store_true",
                                    help='The index will cache all intermediate values in order to speed up the BM25 computations.')

    indexer_settings_parser.add_argument('--indexer.bm25.k1', 
                                    type=float, default=1.2,
                                    help='The k1 value to be used if --indexer.bm25.cache_in_disk is enabled.')

    indexer_settings_parser.add_argument('--indexer.bm25.b', 
                                    type=float, default=0.7,
                                    help='The b value to be used if --indexer.bm25.cache_in_disk is enabled.')
    
    indexer_settings_parser.add_argument('--indexer.tfidf.cache_in_disk', 
                                    action="store_true",
                                    help='The index will cache all intermediate values in order to speed up the TFIDF computations.')

    indexer_settings_parser.add_argument('--indexer.tfidf.smart',
                                    type=str,
                                    default="lnc.ltc",
                                    help='The smart notation to be used if --indexer.tfidf.cache_in_disk is enabled.')
    
        
    indexer_doc_parser = indexer_parser.add_argument_group('Indexer document processing settings', 'This settings are related to how the documents should be loaded and processed to tokens.')
    
    # corpus reader
    shared_reader(indexer_doc_parser, "PubMedReader")
    
    # tokenizer
    shared_tokenizer(indexer_doc_parser)
    
    add_more_options_to_indexer(indexer_parser, indexer_settings_parser, indexer_doc_parser)
    
    ############################
    ## Searcher CLI interface ##
    ############################
    searcher_parser = mode_subparsers.add_parser('searcher', help='Searcher help')

    searcher_parser.add_argument('index_folder', 
                                type=str, 
                                help='Folder where all the index related files will be loaded.')

    searcher_parser.add_argument('path_to_questions', 
                                type=str, 
                                help='Path to the file that contains the question to be processed, one per line.')

    searcher_parser.add_argument('output_file', 
                                type=str, 
                                help='File where the found documents will be returned for each question.')

    searcher_parser.add_argument('--top_k', 
                                type=int,
                                default=1000,
                                help='Number maximum of documents that should be returned per question.')

    # Searcher also specifies a reader
    # question reader
    shared_reader(searcher_parser, "QuestionsReader")
    
    # Searcher also specifies a tokenizer
    # tokenizer
    shared_tokenizer(searcher_parser)

    # mutual exclusive searching modes
    searcher_modes_parser = searcher_parser.add_subparsers(dest='ranking_mode', required=True)
    
    bm25_mode_parser = searcher_modes_parser.add_parser('ranking.bm25', help='Uses the BM25 as the searching method')
    bm25_mode_parser.add_argument("--ranking.bm25.class", type=str, default="BM25Ranking")
    bm25_mode_parser.add_argument("--ranking.bm25.k1", type=float, default=1.2)
    bm25_mode_parser.add_argument("--ranking.bm25.b", type=float, default=0.6)
    bm25_mode_parser.add_argument("--ranking.bm25.B", type=float, default=None)
    bm25_mode_parser.add_argument("--ranking.bm25.mod", type=float, default=1.0)

    tfidf_mode_parser = searcher_modes_parser.add_parser('ranking.tfidf', help='Uses the TFIDF as the searching method')
    tfidf_mode_parser.add_argument("--ranking.tfidf.class", type=str, default="TFIDFRanking")
    tfidf_mode_parser.add_argument("--ranking.tfidf.smart", type=str, default="lnc.ltc")
    tfidf_mode_parser.add_argument("--ranking.tfidf.B", type=float, default=None)
    tfidf_mode_parser.add_argument("--ranking.tfidf.mod", type=float, default=1.0)
    # CLI parsing
    #args = parser.parse_args()
    args = grouping_args(parser.parse_args())

    engine_logic(args)
