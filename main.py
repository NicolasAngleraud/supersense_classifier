import pickle
import random
from collections import defaultdict
import argparse
import torch
import Wiktionary as wi
import SupersenseClassifier as clf
from sklearn.model_selection import train_test_split


def keep_n_elements(lst, n):
    random.shuffle(lst)
    if len(lst) >= n:
        return lst[:n]
    else:
        return lst


def split_data(data_list):
    train_data, test_data = train_test_split(data_list, test_size=0.2)
    dev_data, test_data = train_test_split(test_data, test_size=0.5)
    return train_data, dev_data, test_data


def get_data_sets(seeds_file="seeds_checked_V3.txt"):
    id2def_supersense = {}
    id2defwithlemma_supersense = {}
    supersense2ids = defaultdict(list)
    train = []
    dev = []
    test = []
    seeds_file = open(seeds_file, 'r', encoding="utf-8")
    # build dictionnary structures
    headers = seeds_file.readline().strip("\n").strip().split("\t")
    id = 0
    while 1:
        id += 1
        line = seeds_file.readline()
        if not line:
            break
        sline = line.strip("\n").strip().split("\t")
        for header, value in zip(headers, sline):
            if header == "supersense":
                supersense = value
            if header == "lemma":
                lemma = value
            if header == "definition":
                value = value.replace("DEF:", "")
                definition = value
                definitionwithlemma = f"{lemma} : {value}"
        if supersense in SUPERSENSES:
            id2def_supersense[id] = (definition, supersense)
            id2defwithlemma_supersense[id] = (definitionwithlemma, supersense)
            supersense2ids[supersense].append(id)

    # build id lists for classification sets
    for supersense in supersense2ids:
        train_ids, dev_ids, test_ids = split_data(supersense2ids[supersense])
        train += train_ids
        dev += dev_ids
        test += test_ids

    # serialize the different structures created
    with open("id2def_supersense.pkl", "wb") as file:
        pickle.dump(id2def_supersense, file)
    with open("id2defwithlemma_supersense.pkl", "wb") as file:
        pickle.dump(id2defwithlemma_supersense, file)
    with open("supersense2ids.pkl", "wb") as file:
        pickle.dump(supersense2ids, file)
    with open("train.pkl", "wb") as file:
        pickle.dump(train, file)
    with open("dev.pkl", "wb") as file:
        pickle.dump(dev, file)
    with open("test.pkl", "wb") as file:
        pickle.dump(test, file)
    seeds_file.close()


def get_fixed_nb_examples_data_sets(n=100, seeds_file="seeds_checked_V3.txt"):
    id2def_supersense = {}
    id2defwithlemma_supersense = {}
    supersense2ids = defaultdict(list)
    train = []
    dev = []
    test = []
    seeds_file = open(seeds_file, 'r', encoding="utf-8")
    # build dictionnary structures
    headers = seeds_file.readline().strip("\n").strip().split("\t")
    id = 0
    while 1:
        id += 1
        line = seeds_file.readline()
        if not line:
            break
        sline = line.strip("\n").strip().split("\t")
        for header, value in zip(headers, sline):
            if header == "supersense":
                supersense = value
            if header == "lemma":
                lemma = value
            if header == "definition":
                value = value.replace("DEF:", "")
                definition = value
                definitionwithlemma = f"{lemma} : {value}"
        if supersense in SUPERSENSES:
            id2def_supersense[id] = (definition, supersense)
            id2defwithlemma_supersense[id] = (definitionwithlemma, supersense)
            supersense2ids[supersense].append(id)

    # build id lists for classification sets
    for supersense in supersense2ids:

        train_ids, dev_ids, test_ids = split_data(keep_n_elements(supersense2ids[supersense], n=n))
        train += train_ids
        dev += dev_ids
        test += test_ids

    # serialize the different structures created
    with open(f"{n}_id2def_supersense.pkl", "wb") as file:
        pickle.dump(id2def_supersense, file)
    with open(f"{n}_id2defwithlemma_supersense.pkl", "wb") as file:
        pickle.dump(id2defwithlemma_supersense, file)
    with open(f"{n}_supersense2ids.pkl", "wb") as file:
        pickle.dump(supersense2ids, file)
    with open(f"{n}_train.pkl", "wb") as file:
        pickle.dump(train, file)
    with open(f"{n}_dev.pkl", "wb") as file:
        pickle.dump(dev, file)
    with open(f"{n}_test.pkl", "wb") as file:
        pickle.dump(test, file)
    seeds_file.close()


def get_inference_data_set(inference_data_file):
    pass


# supersenses acknowleged
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

# relations to be acknowledged while reading wiktionary ttl files
allowed_relations = ["rdf:type",
                     "lexinfo:partOfSpeech",
                     "ontolex:canonicalForm",
                     "ontolex:sense",
                     "dbnary:describes",
                     "dbnary:synonym",
                     "lexinfo:gender",
                     "skos:definition",
                     "skos:example",
                     "rdfs:label",
                     "dbnary:partOfSpeech"]

# rdf types to be acknowledged while reading wiktionary ttl files
allowed_rdf_types = ["ontolex:LexicalSense",
                     "ontolex:Form",
                     "ontolex:LexicalEntry",
                     "dbnary:Page",
                     "ontolex:Word",
                     "ontolex:MultiWordExpression"]

# grammatical categories to be acknowledged while reading wiktionary ttl files
allowed_categories = ["lexinfo:noun", '"-nom-"']

# labels in definitions leading to ignore a lexical sense while reading wiktionary ttl files
labels_to_ignore = ["vieilli", "archaïque", "désuet", "archaïque, orthographe d’avant 1835"]

# language token to be acknowledged while reading wiktionary ttl files
lang = "fra"


class Parameters:
    def __init__(self, nb_epochs=100, batch_size=50, hidden_layer_size=300, patience=5, lr=0.00025, frozen=True, max_seq_length=50):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.patience = patience
        self.lr = lr
        self.frozen = frozen
        self.max_seq_length = max_seq_length
        self.keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "max_seq_length"]


# parameter combinations to find the values to make the best classifier
PARAMETER_COMBINATIONS = []
# best parameters to serialize
BEST_PARAMETERS = Parameters()
# basic parameters to test the classifier
TEST_PARAMETERS = Parameters()


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_mode", choices=['parse', 'classify', 'get_data_sets'], help="Sets the main purpose of the program, that is to say parsing a file linked to seeds or use a classifier for supersense classification.")
    parser.add_argument('-parsing_file', default="fr_dbnary_ontolex.ttl", help='wiktionary file (dbnary dump, in turtle format) or serialized Wiktionary instance (pickle file).')
    parser.add_argument("-statistics_file", default="seeds_checked_V3.txt", help="")
    parser.add_argument("-checked_seeds_file", default="seeds_checked_V3.txt", help="")
    parser.add_argument("-train_file", default="100_train.pkl", help="")
    parser.add_argument("-dev_file", default="100_dev.pkl", help="")
    parser.add_argument("-test_file", default="100_test.pkl", help="")
    parser.add_argument("-id2def_sup_file", default="100_id2def_supersense.pkl", help="")
    parser.add_argument("-id2deflem_sup_file", default="100_id2defwithlemma_supersense.pkl", help="")
    parser.add_argument("-corpus_file", default="sequoia.deep_and_surf.parseme.frsemcor", help="")
    parser.add_argument('-parsing_mode', choices=['read', 'filter', 'check_seeds', 'read_and_dump', 'statistics', 'get_seeds_from_corpus'], help="Sets the mode for the parsing: read, filter, check_seeds or read_and_dump.")
    parser.add_argument("-inference_data_file", default=None, help="File containing the data for inference.")
    parser.add_argument("-wiktionary_dump", default="wiki_dump.pkl", help="Serialized Wiktionary instance containig all the annoted data for the classifier to be trained and evaluated.")
    parser.add_argument("-definition_begins_with_lemma", action="store_true", help="")
    parser.add_argument("-classifier_mode", choices=['test','combinations','dump', 'evaluation', 'inference'], default='test', help="")
    parser.add_argument("-dataset_mode", choices=['full','n'], default='n', help="")
    parser.add_argument("-nb_max_examples_per_supersense", default='100', help="")
    parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser_args()

    if args.main_mode == "parse":
        parsing_file = args.parsing_file

        # creation of an instance of the Parser class meant to parse wiktionary related files
        wiki_parser = wi.Parser(categories=allowed_categories,
                                relations=allowed_relations,
                                lang=lang,
                                labels_to_ignore=labels_to_ignore,
                                trace=args.trace,
                                wiki_dump=args.wiktionary_dump,
                                rdf_types=allowed_rdf_types,
                                file=parsing_file,
                                statistics_file=args.statistics_file,
                                checked_seeds_file=args.checked_seeds_file,
                                corpus_file=args.corpus_file,
                                seeds="seeds_ClassSemWikt_06_04-23.xlsx",
                                )
        wiki_parser.set_parser_mode(args.parsing_mode)
        wiki_parser.parse_file()

    if args.main_mode == "classify":
        train_examples, dev_examples, test_examples = clf.encoded_examples_split(args.definition_begins_with_lemma,
                                                                                 train=args.train_file,
                                                                                 dev=args.dev_file,
                                                                                 test=args.test_file,
                                                                                 id2def_sup=args.id2def_sup_file,
                                                                                 id2deflem_sup=args.id2deflem_sup_file)
        if args.classifier_mode == 'test':
            test_params = TEST_PARAMETERS
            classifier = clf.SupersenseTagger(test_params)
            clf.training(test_params, train_examples, dev_examples, classifier)
            clf.evaluation(test_examples, classifier)
        elif args.classifier_mode == 'combinations':
            parameter_combinations = PARAMETER_COMBINATIONS
            for params in parameter_combinations:
                classifier = clf.SupersenseTagger(params)
                clf.training(params, train_examples, dev_examples, classifier)
                clf.evaluation(test_examples, classifier)
        elif args.classifier_mode == 'dump':
            params = BEST_PARAMETERS
            classifier = clf.SupersenseTagger(params)
            clf.training(params, train_examples, dev_examples, classifier)
            torch.save(classifier, 'mlp_model.pt')
            torch.save(classifier.state_dict(), 'mlp_model_state_dict.pt')
        elif args.classifier_mode == 'evaluation':
            params = BEST_PARAMETERS
            # Create an instance of your MLP model
            classifier = clf.SupersenseTagger(params)
            # Load the serialized model
            classifier.load_state_dict(torch.load('mlp_model_state_dict.pt'))
            classifier.eval()  # Set the model to evaluation mode
            clf.evaluation(test_examples, classifier)
        elif args.classifier_mode == 'inference':
            params = BEST_PARAMETERS
            inference_data_set = get_inference_data_set(args.inference_data_file)
            # Create an instance of your MLP model
            classifier = clf.SupersenseTagger(params)
            # Load the serialized model
            classifier.load_state_dict(torch.load('mlp_model_state_dict.pt'))
            classifier.eval()  # Set the model to evaluation mode
            clf.inference(inference_data_set, classifier)
    if args.main_mode == "get_data_sets":
        if args.dataset_mode == "full":
            get_data_sets()
        elif args.dataset_mode == "n":
            get_fixed_nb_examples_data_sets(n=int(args.nb_max_examples_per_supersense))
