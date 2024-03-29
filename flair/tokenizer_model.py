import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Union, List
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from flair.datasets import DataLoader
from flair.datasets import SentenceDataset

import flair
flair.device = 'cuda:1'
from flair.data import DataPoint
from flair.training_utils import Result

import torch
import torch.nn as nn
import torch.nn.functional as F
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)


class LabeledString(DataPoint):

    def __init__(self, string: str):
        super().__init__()
        self.string = string

    def __str__(self) -> str:
        # add Sentence labels to output if they exist
        sentence_labels = f"  − Labels: {self.annotation_layers}" if self.annotation_layers != {} else ""

        return f'String: "{self.string}" {sentence_labels}'

    def __repr__(self) -> str:
        # add Sentence labels to output if they exist
        sentence_labels = f"  − Labels: {self.annotation_layers}" if self.annotation_layers != {} else ""

        return f'String: "{self.string}" {sentence_labels}'


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

    # Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class FlairTokenizer(flair.nn.Model):

    def __init__(self,
                 letter_to_ix={},  # character dictionary
                 embedding_dim=4096,
                 hidden_dim=256,
                 num_layers=1,
                 use_CSE=False,
                 tag_to_ix={'B': 0, 'I': 1, 'E': 2, 'S': 3, 'X': 4},
                 learning_rate=0.1,
                 use_CRF=False,
                 dropout=0. #dropout: float = 0.
                 ):

        super(FlairTokenizer, self).__init__()
        self.letter_to_ix = letter_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_CSE = use_CSE
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {y: x for x, y in self.tag_to_ix.items()}
        self.learning_rate = learning_rate
        self.use_CRF = use_CRF
        self.dropout = dropout

        if not self.use_CSE:
            self.character_embeddings = nn.Embedding(len(letter_to_ix), embedding_dim)
            self.embeddings = self.character_embeddings

        elif self.use_CSE:
            self.flair_embedding = FlairEmbeddings
            self.lm_f: LanguageModel = self.flair_embedding('multi-forward').lm
            self.lm_b: LanguageModel = self.flair_embedding('multi-backward').lm
            self.embeddings = self.flair_embedding

        if self.use_CRF:
            self.START_TAG = '<START>'
            self.STOP_TAG = '<STOP>'
            self.tag_to_ix = {"B": 0, "I": 1, "E": 2, 'S': 3, 'X': 4, self.START_TAG: 5, self.STOP_TAG: 6}
            self.ix_to_tag = {y: x for x, y in self.tag_to_ix.items()}
            self.tagset_size = len(self.tag_to_ix)
            # Matrix of transition parameters.  Entry i,j is the score of
            # transitioning *to* i *from* j.
            self.transitions = nn.Parameter(
                torch.randn(len(self.tag_to_ix), len(self.tag_to_ix), device=flair.device))
            # These two statements enforce the constraint that we never transfer
            # to the start tag and we never transfer from the stop tag
            self.transitions.data[self.tag_to_ix[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag_to_ix[self.STOP_TAG]] = -10000

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=False, bidirectional=True,dropout=self.dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, len(self.tag_to_ix))
        self.loss_function = nn.NLLLoss()

        self.to(flair.device)

    def _forward_alg(self, feats):  # feats = the output: lstm_features
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000., device=flair.device)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    @abstractmethod
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1, device=flair.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long, device=flair.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000., device=flair.device)
        init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    @abstractmethod
    # def neg_log_likelihood(self, sentence, tags):
    def neg_log_likelihood(self, feats, tags):  # loss = self.neg_log_likelihood(lstm_feats,targets)
        # feats = self.forward_loss(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, data_points):
        # Get the emission scores from the BiLSTM
        loss, packed_sent, packed_tags, lstm_feats = self.forward_loss(data_points, foreval=True)
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        out_list = [self.ix_to_tag[int(o)] for o in tag_seq]
        tag_seq_str = ''
        for o in out_list:
            tag_seq_str += o
        return tag_seq_str

    def prepare_cse(self, sentence, batch_size=1):
        if (batch_size == 1) & (not isinstance(sentence, list)):
            embeds_f = self.lm_f.get_representation([sentence], '\n', '\n')[1:-1, :, :]
            embeds_b = self.lm_b.get_representation([sentence], '\n', '\n')[1:-1, :, :]
        if (batch_size == 1) & (isinstance(sentence, list)):
            embeds_f = self.lm_f.get_representation(sentence, '\n', '\n')[1:-1, :, :]
            embeds_b = self.lm_b.get_representation(sentence, '\n', '\n')[1:-1, :, :]
        else:
            embeds_f = self.lm_f.get_representation(list(sentence), '\n', '\n')[1:-1, :, :]
            embeds_b = self.lm_b.get_representation(list(sentence), '\n', '\n')[1:-1, :, :]
        return torch.cat((embeds_f, embeds_b), dim=2)

    @staticmethod
    def prepare_batch(data_points_str, to_ix):
        tensor_list = []
        for seq in data_points_str:
            idxs = [to_ix[w] for w in seq]
            tensor = torch.tensor(idxs, dtype=torch.long, device=flair.device)
            tensor_list.append(tensor)
        sequence = pad_sequence(tensor_list, batch_first=False)
        # print(sequence.size())
        # batch_tensor = sequence.squeeze()
        # print(batch_tensor.size())
        # if len(batch_tensor.shape) == 1:
        #     return batch_tensor.view(-1, 1)  # add batch_size = 1 as dimension
        # else:
        return sequence

    @staticmethod
    def find_token(sentence_str):
        token = []
        word = ''
        for i, tag in enumerate(sentence_str[1]):
            if tag == 'S':
                token.append(sentence_str[0][i])
                continue
            if tag == 'X':
                continue
            if (tag == 'B') | (tag == 'I'):
                word += sentence_str[0][i]
                continue
            if tag == 'E':
                word += sentence_str[0][i]
                token.append(word)
                word = ''
        return token

    def prediction_str(self, input):
        ix_to_tag = {y: x for x, y in self.tag_to_ix.items()}
        output = [torch.argmax(i) for i in input]
        out_list = [ix_to_tag[int(o)] for o in output]
        out_str = ''
        for o in out_list: out_str += o
        return out_str

    @abstractmethod
    def forward_loss(
            self,
            data_points: Union[List[DataPoint], DataPoint],
            foreval=False,
    ) -> torch.tensor:

        """Performs a forward pass and returns a loss tensor for backpropagation. Implement this to enable training."""

        if isinstance(data_points,
                      LabeledString):  # make sure data_points is a list, doesn't matter how many elements inside
            data_points = [data_points]  # FIXME: why is this not working?

        input_sent, input_tags = [], []
        for sent in data_points:
            input_sent.append((sent.string))
            input_tags.append(sent.get_labels('tokenization')[0].value)
        batch_size = len(data_points)

        batch_input_tags = self.prepare_batch(input_tags, self.tag_to_ix).to(flair.device)
        if self.use_CSE:
            embeds = self.prepare_cse(input_sent, batch_size=batch_size).to(flair.device)
        else:
            batch_input_sent = self.prepare_batch(input_sent, self.letter_to_ix)
            embeds = self.character_embeddings(batch_input_sent)

        out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(out.view(embeds.shape[0], embeds.shape[1], -1))
        tag_scores = F.log_softmax(tag_space, dim=2)  # dim = (len(data_points),batch,len(tag))

        length_list = []
        for sentence in data_points:
            length_list.append(len(sentence.string))

        packed_sent, packed_tags = '', ''
        for string in input_sent: packed_sent += string
        for tag in input_tags: packed_tags += tag

        packed_tag_space = torch.tensor([], dtype=torch.long, device=flair.device)
        packed_tag_scores = torch.tensor([], dtype=torch.long, device=flair.device)
        packed_batch_input_tags = torch.tensor([], dtype=torch.long, device=flair.device)

        for i in np.arange(batch_size):
            packed_tag_scores = torch.cat((packed_tag_scores, tag_scores[:length_list[i], i, :]))
            packed_tag_space = torch.cat((packed_tag_space, tag_space[:length_list[i], i, :]))
            packed_batch_input_tags = torch.cat((packed_batch_input_tags, batch_input_tags[:length_list[i], i]))

        if not self.use_CRF:
            tag_predict = self.prediction_str(packed_tag_scores)
            loss = self.loss_function(packed_tag_scores, packed_batch_input_tags)
            if foreval:
                return loss, packed_sent, packed_tags, tag_predict
            else:
                return loss

        else:  # extract lstm_features for CRF layer
            lstm_feats = packed_tag_space
            loss = self.neg_log_likelihood(lstm_feats, packed_batch_input_tags)

            if foreval:
                return loss, packed_sent, packed_tags, lstm_feats
            else:
                return loss

    @abstractmethod
    def evaluate(
            self,
            sentences: Union[List[DataPoint], Dataset],
            out_path: Path = None,
            embedding_storage_mode: str = "none",
            mini_batch_size: int = 32,
            num_workers: int = 8,
    ) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        if not isinstance(sentences, Dataset):
            sentences = SentenceDataset(sentences)

        data_loader = DataLoader(sentences, batch_size=mini_batch_size, num_workers=num_workers)
        eval_loss = 0
        with torch.no_grad():
            error_sentence = []
            R_score, P_score, F1_score = [], [], []

            for batch in data_loader:  # assume input of evaluate fct is the whole dataset, and each element is a batch
                if self.use_CRF:
                    loss, packed_sent, packed_tags, lstm_feats = self.forward_loss(batch, foreval=True)
                    packed_tag_predict = self.forward(batch)
                else:
                    loss, packed_sent, packed_tags, packed_tag_predict = self.forward_loss(batch, foreval=True)

                reference = self.find_token((packed_sent, packed_tags))
                candidate = self.find_token((packed_sent, packed_tag_predict))

                inter = [c for c in candidate if c in reference]

                if len(candidate) != 0:
                    R = len(inter) / len(reference)
                    P = len(inter) / len(candidate)
                else:
                    R, P = 0, 0  # when len(candidate) = 0, which means the model fail to extract any token from the sentence
                    error_sentence.append((packed_sent, packed_tags, packed_tag_predict))
                if (len(candidate) != 0) & ((R + P) != 0):  # if R = P = 0, meaning len(inter) = 0, R+P = 0
                    F1 = 2 * R * P / (R + P)
                else:
                    F1 = 0
                    if (packed_sent, packed_tags, packed_tag_predict) not in error_sentence:
                        error_sentence.append((packed_sent, packed_tags, packed_tag_predict))

                R_score.append(R)
                P_score.append(P)
                F1_score.append(F1)

                eval_loss += loss

            detailed_result = (
                "\nResults:"
                f"\n- F1-score : {np.mean(F1_score)}"
                f"\n- Precision-score : {np.mean(P_score)}"
                f"\n- Recall-score : {np.mean(R_score)}"
            )

            # line for log file
            log_header = "F1_score"
            log_line = f"\t{np.mean(F1_score)}"

            result = Result(
                main_score=np.mean(F1_score),
                log_line=log_line,
                log_header=log_header,
                detailed_results=detailed_result,
            )

            return (result, eval_loss)

    def _get_state_dict(self):
        model_state = {
            "state_dict": self.state_dict(),
            'letter_to_ix': self.letter_to_ix,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'use_CSE': self.use_CSE,
            'tag_to_ix': self.tag_to_ix,
            'learning_rate': self.learning_rate,
            'use_CRF': self.use_CRF
        }
        return model_state

    @staticmethod
    def _init_model_with_state_dict(state):
        model = FlairTokenizer(
            letter_to_ix=state['letter_to_ix'],
            embedding_dim=state['embedding_dim'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            use_CSE=state['use_CSE'],
            tag_to_ix=state['tag_to_ix'],
            learning_rate=state['learning_rate'],
            use_CRF=state['use_CRF']
        )
        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name
