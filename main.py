import pickle
import random
from collections import defaultdict
import argparse
import torch
import SupersenseClassifier as clf
from sklearn.model_selection import train_test_split


# supersenses acknowleged
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']


class Parameters:
    def __init__(self, nb_epochs=10, batch_size=25, hidden_layer_size=300, patience=5, lr=0.00025, frozen=True, max_seq_length=50):
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
    parser.add_argument("device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
    parser.add_argument("-classifier_mode", choices=['test','combinations','dump', 'evaluation', 'inference'], default='test', help="")
    parser.add_argument("-train_file", default="train.pkl", help="")
    parser.add_argument("-dev_file", default="dev.pkl", help="")
    parser.add_argument("-test_file", default="test.pkl", help="")
    parser.add_argument("-id2def_sup_file", default="id2def_supersense.pkl", help="")
    parser.add_argument("-id2deflem_sup_file", default="id2defwithlemma_supersense.pkl", help="")
    parser.add_argument("-inference_data_file", default=None, help="File containing the data for inference.")
    parser.add_argument("-definition_begins_with_lemma", action="store_true", help="")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser_args()

    # DEVICE setup
    device_id = args.device_id
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + args.device_id)

    # Encoding the examples from the datasets
    train_examples, dev_examples, test_examples = clf.encoded_examples_split(DEVICE,
                                                                             args.definition_begins_with_lemma,
                                                                             train=args.train_file,
                                                                             dev=args.dev_file,
                                                                             test=args.test_file,
                                                                             id2def_sup=args.id2def_sup_file,
                                                                             id2deflem_sup=args.id2deflem_sup_file)
    # Classification program
    if args.classifier_mode == 'test':
        with open("logs.txt", 'r', encoding="utf-8") as file:
            test_params = TEST_PARAMETERS
            classifier = clf.SupersenseTagger(test_params, DEVICE)
            clf.training(test_params, train_examples, dev_examples, classifier, DEVICE, file)
            clf.evaluation(test_examples, classifier, file)
"""
    elif args.classifier_mode == 'combinations':
        parameter_combinations = PARAMETER_COMBINATIONS
        for params in parameter_combinations:
            classifier = clf.SupersenseTagger(params, DEVICE)
            clf.training(params, train_examples, dev_examples, classifier, DEVICE)
            clf.evaluation(train_examples, classifier)
            clf.evaluation(dev_examples, classifier)

    elif args.classifier_mode == 'dump':
        params = BEST_PARAMETERS
        classifier = clf.SupersenseTagger(params, DEVICE)
        clf.training(params, train_examples, dev_examples, classifier, DEVICE)
        torch.save(classifier, 'mlp_model.pt')
        torch.save(classifier.state_dict(), 'mlp_model_state_dict.pt')

    elif args.classifier_mode == 'evaluation':
        params = BEST_PARAMETERS
        classifier = clf.SupersenseTagger(params, DEVICE)
        classifier.load_state_dict(torch.load('mlp_model_state_dict.pt'))
        classifier.eval()
        clf.evaluation(test_examples, classifier)

    elif args.classifier_mode == 'inference':
        params = BEST_PARAMETERS
        inference_data_set = args.inference_data_file
        classifier = clf.SupersenseTagger(params, DEVICE)
        classifier.load_state_dict(torch.load('mlp_model_state_dict.pt'))
        classifier.eval()
        clf.inference(inference_data_set, classifier, DEVICE)
"""
