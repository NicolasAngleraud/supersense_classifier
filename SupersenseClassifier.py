import pickle
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sacremoses
from random import shuffle
import numpy as np
from transformers import AutoModel, AutoTokenizer
from matplotlib import pyplot as plt


# Definition of the supersenses and index structure
SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES)}

NB_CLASSES = len(supersense2i)
MODEL_NAME = "flaubert/flaubert_base_cased"
HIDDEN_LAYER_SIZE = 128
PATIENCE = 3
LR = 0.005
PADDING_TOKEN_ID = 2
NB_EPOCHS = 10
BATCH_SIZE = 2


def encoded_examples_split(DEVICE, def_with_lemma, train="100_train.pkl", dev="100_dev.pkl", test="100_test.pkl",
                           id2def_sup="100_id2def_supersense.pkl", id2deflem_sup="100_id2defwithlemma_supersense.pkl"):
    # loads the structures necessary to build examples sets
    with open(train, "rb") as file:
        train_ids = pickle.load(file)
    with open(dev, "rb") as file:
        dev_ids = pickle.load(file)
    with open(test, "rb") as file:
        test_ids = pickle.load(file)
    with open(id2def_sup, "rb") as file:
        id2def_supersense = pickle.load(file)
    with open(id2deflem_sup, "rb") as file:
        id2defwithlemma_supersense = pickle.load(file)

    # build examples sets
    if not def_with_lemma:
        train_examples = [id2def_supersense[id] for id in train_ids]
        dev_examples = [id2def_supersense[id] for id in dev_ids]
        test_examples = [id2def_supersense[id] for id in test_ids]
    else:
        train_examples = [id2defwithlemma_supersense[id] for id in train_ids]
        dev_examples = [id2defwithlemma_supersense[id] for id in dev_ids]
        test_examples = [id2defwithlemma_supersense[id] for id in test_ids]

    # encodes the examples of each set
    encoded_train_examples = []
    encoded_dev_examples = []
    encoded_test_examples = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME).to(DEVICE)

    for example in train_examples:
        definition = example[0]
        supersense = example[1]
        definition_encoding = tokenizer.encode(text=definition, add_special_tokens=True)
        supersense_encoding = supersense2i[supersense]
        encoded_train_examples.append((definition_encoding, supersense_encoding))
    for example in dev_examples:
        definition = example[0]
        supersense = example[1]
        definition_encoding = tokenizer.encode(text=definition, add_special_tokens=True)
        supersense_encoding = supersense2i[supersense]
        encoded_dev_examples.append((definition_encoding, supersense_encoding))
    for example in test_examples:
        definition = example[0]
        supersense = example[1]
        definition_encoding = tokenizer.encode(text=definition, add_special_tokens=True)
        supersense_encoding = supersense2i[supersense]
        encoded_test_examples.append((definition_encoding, supersense_encoding))

    return encoded_train_examples, encoded_dev_examples, encoded_test_examples


def pad_batch(encodings_batch, padding_token_id=2, max_seq_length=100):
    padding_size = max(len(sublist) for sublist in encodings_batch)
    padding_size = min(padding_size, max_seq_length)
    for sentence_encoding in encodings_batch:
        if len(sentence_encoding) < padding_size:
            while len(sentence_encoding) < padding_size:
                sentence_encoding.append(padding_token_id)
        else:
            while len(sentence_encoding) > padding_size:
                sentence_encoding.pop(-2)
    return torch.tensor(encodings_batch, dtype=torch.long)


class SupersenseTagger(nn.Module):

    def __init__(self, params, DEVICE, bert_model_name=MODEL_NAME):
        super(SupersenseTagger, self).__init__()
        # definition of the bert model attribute
        self.bert_model = AutoModel.from_pretrained(bert_model_name, output_attentions=True).to(DEVICE)
        # freezes the parameters of the bert embeddings if specified
        if params.frozen:
            for param in self.bert_model.parameters():
                param.requires_grad = False
        # size of the embedding layer (here it will be the size of a bert embedding from the model given)
        #@@ plm_emb_size
        self.embedding_layer_size = self.bert_model.config.hidden_size
        # size if the hidden layer
        self.hidden_layer_size = params.hidden_layer_size
        # size of the vocabulary over which the forward method of the MLP will give probabilities
        self.output_size = NB_CLASSES
        # definition of the parameters for the linear operation between the embedding layer and the hidden layer
        self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)
        # definition of the parameters for the linear operation between the hidden layer and the output layer
        self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)

    def forward(self, padded_encodings): #@@ prendre en entrée un tenseur de taille batch_size, max_seq_length_for_this_batch

        # the tokenization encoding is given to the FlauBERT model which outputs the contextual embeddings for every definition
        bert_output = self.bert_model(padded_encodings, return_dict=True) # SHAPE [len(definitions), max_length, embedding_size]

        batch_contextual_embeddings = bert_output.last_hidden_state[:,0,:] # from [batch_size , max_seq_length, plm_emb_size] to [batch_size, plm_emb_size]

        # linear combination from embedding layer to hidden layer
        out = self.linear_1(batch_contextual_embeddings) # SHAPE [len(definitions), hidden_layer_size]

        # relu activation function at the hidden layer
        out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

        # linear combination from hidden layer to output layer
        out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

        # log-softmax operation to get the log-probabilities for each supersense for each definition
        return F.log_softmax(out, dim=1)

    def predict(self, definitions_batch_encodings):
        with torch.no_grad():
            log_probs = self.forward(definitions_batch_encodings)
            predicted_indices = torch.argmax(log_probs, dim=1).tolist()
        return [SUPERSENSES[i] for i in predicted_indices]

    def evaluate(self, examples_batch_encodings, DEVICE):
        with torch.no_grad():
            X, Y = zip(*examples_batch_encodings)
            X = pad_batch(X, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
            Y_gold = torch.tensor(Y).to(DEVICE)
            Y_pred = torch.argmax(self.forward(X), dim=1)

        # Find the indices where predictions and gold classes differ
        error_indices = torch.nonzero(Y_pred != Y_gold).squeeze()

        # Get the predicted and gold classes for the errors
        errors = [(SUPERSENSES[Y_pred[i].item()], SUPERSENSES[Y_gold[i].item()]) for i in error_indices]

        return errors, torch.sum((Y_pred == Y_gold).int()).item()


def training(parameters, train_examples, dev_examples, classifier, DEVICE, file):
    # print("TRAINING")
    file.write("TRAINING\n")
    # get every parameter in a local variable
    for param in parameters.keys:
        locals()[param] = getattr(parameters, param)
        # print(param, locals()[param])
        file.write(f"{param}: {locals()[param]}\n")

    # instance of NSupersenseTagger
    my_supersense_tagger = classifier

    # to store the training losses at each epoch
    train_losses = []
    train_accuracies = []
    # to store the dev losses after each epoch
    dev_losses = []
    dev_accuracies = []
    loss_function = nn.NLLLoss()

    # the optimizer is the instance that will actually update the declared parameters
    optimizer = optim.Adam(my_supersense_tagger.parameters(), lr=locals()["lr"])

    for epoch in range(locals()["nb_epochs"]):
        # print("EPOCH: ", epoch)
        epoch_loss = 0
        dev_epoch_loss = 0

        train_epoch_accuracy = 0
        dev_epoch_accuracy = 0

        # shuffle data
        shuffle(train_examples)
        i = 0
        j = 0
        # iteration over every batch from the training examples
        while i < len(train_examples):
            train_batch = train_examples[i: i + locals()["batch_size"]]

            i += locals()["batch_size"]

            X_train, Y_train = zip(*train_batch)

            padded_encodings = pad_batch(X_train, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
            Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)

            my_supersense_tagger.zero_grad()
            log_probs = my_supersense_tagger(padded_encodings)

            predicted_indices = torch.argmax(log_probs, dim=1)
            train_epoch_accuracy += torch.sum((predicted_indices == Y_train).int()).item()

            loss = loss_function(log_probs, Y_train)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # end of the epoch for train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_epoch_accuracy / len(train_examples))

        while j < len(dev_examples):
            dev_batch = dev_examples[j: j + locals()["batch_size"]]
            j += locals()["batch_size"]
            X_dev, Y_dev = zip(*dev_batch)
            dev_padded_encodings = pad_batch(X_dev, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
            Y_dev = torch.tensor(Y_dev, dtype=torch.long).to(DEVICE)
            dev_log_probs = my_supersense_tagger(dev_padded_encodings)

            predicted_indices = torch.argmax(dev_log_probs, dim=1)
            dev_epoch_accuracy += torch.sum((predicted_indices == Y_dev).int()).item()

            dev_loss = loss_function(dev_log_probs, Y_dev)
            dev_epoch_loss += dev_loss.item()

        dev_losses.append(dev_epoch_loss)
        dev_accuracies.append(dev_epoch_accuracy / len(dev_examples))

        # Early stopping
        if epoch > locals()["patience"]:
            if all(dev_losses[i] > dev_losses[i - 1] for i in range(-1, -locals()["patience"], -1)):
                # print(f"EARLY STOPPING: EPOCH = {epoch}")
                file.write(f"EARLY STOPPING: EPOCH = {epoch}\n")
                break

    # print(f"Train losses = {[round(train_loss, 2) for train_loss in train_losses]}")
    file.write(f"Train losses = {[round(train_loss, 2) for train_loss in train_losses]}\n")
    # print(f"Dev losses = {[round(dev_loss, 2) for dev_loss in dev_losses]}")
    file.write(f"Dev losses = {[round(dev_loss, 2) for dev_loss in dev_losses]}\n")
    # print(f"Train accuracies = {[round(train_accuracy, 2) for train_accuracy in train_accuracies]}")
    file.write(f"Train accuracies = {[round(train_accuracy, 2) for train_accuracy in train_accuracies]}\n")
    # print(f"Dev accuracies = {[round(dev_accuracy, 2) for dev_accuracy in dev_accuracies]}")
    file.write(f"Dev accuracies = {[round(dev_accuracy, 2) for dev_accuracy in dev_accuracies]}\n")

    abs = np.arange(len(train_losses))
    plt.plot(abs, train_losses, label='train loss')
    plt.plot(abs, dev_losses, label='dev loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.show()
    plt.savefig(f"train_losses.png")
    plt.close()

    abs = np.arange(len(train_accuracies))
    plt.plot(abs, train_accuracies, label='train accuracy')
    plt.plot(abs, dev_accuracies, label='dev accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(f"train_accuracies.png")
    plt.close()


def evaluation(examples, classifier, file):
    batch_size = 100
    i = 0
    nb_good_preds = 0
    errors_list = []
    while i < len(examples):
        evaluation_batch = examples[i: i + batch_size]
        i += batch_size
        partial_errors_list, partial_nb_good_preds = classifier.evaluate(evaluation_batch)
        errors_list += partial_errors_list
        nb_good_preds += partial_nb_good_preds

    # print(f"ACCURACY test set = {nb_good_preds/len(examples)}")
    file.write(f"ACCURACY test set = {nb_good_preds/len(examples)}\n")

    counter = Counter(errors_list)
    most_common_errors = counter.most_common(10)
    # print(f"Erreurs les plus courantes: {most_common_errors}")
    file.write(f"Erreurs les plus courantes: {most_common_errors}\n")


def inference(inference_data_set, classifier, DEVICE):
    pass


class Baseline:
    pass


class MostFrequentSequoia(Baseline):
    def __init__(self):
        pass


class MostFrequentWiktionary(Baseline):
    def __init__(self):
        pass


class MostFrequentTrainingData(Baseline):
    def __init__(self):
        pass
