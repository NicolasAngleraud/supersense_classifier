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
    def __init__(self, nb_epochs=1000, batch_size=25, hidden_layer_size=300, patience=15, lr=0.00025, frozen=True, max_seq_length=50, window_example=10, definition_mode='definition'):
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.patience = patience
        self.lr = lr
        self.frozen = frozen
        self.max_seq_length = max_seq_length
        self.window_example = window_example
        self.definition_mode = definition_mode
        self.keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "max_seq_length", "window_example", "definition_mode"]


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_parser_args()

    # DEVICE setup
    device_id = args.device_id
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:" + args.device_id)

    # Classification program
    with open("logs_4.txt", 'w', encoding="utf-8") as file:
        for split_id in range(1, 2):
            for def_mode in ['definition_with_lemma_and_labels']:
                for lr in [0.00001, 0.000001, 0.0000001]:
                    for patience in [10]:
                        # Encoding the examples from the datasets
                        train_examples, dev_examples, test_examples = clf.encoded_examples_split(DEVICE,
                                                                                                 def_mode=def_mode,
                                                                                                 train=f"{split_id}_train.pkl",
                                                                                                 dev=f"{split_id}_dev.pkl",
                                                                                                 test=f"{split_id}_test.pkl",
                                                                                                 id2data=f"{split_id}_id2data.pkl",
                                                                                                 )

                        params = Parameters(lr=lr, definition_mode=def_mode, patience=patience)
                        file.write(f"split_id:{split_id};")
                        classifier = clf.SupersenseTagger(params, DEVICE)
                        clf.training(params, train_examples, dev_examples, classifier, DEVICE, file)
                        clf.evaluation(dev_examples, classifier, DEVICE, file)
                        file.write("\n")
