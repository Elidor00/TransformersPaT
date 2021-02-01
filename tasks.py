import logging
import os
from typing import List, TextIO, Union

from conllu import parse_incr

from utils_ner import InputExample, Split, TokenClassificationTask

logger = logging.getLogger(__name__)


class NER(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                # handle metadata instead of using parse_incr(f)
                # if line.startswith("#"):
                #     break
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[self.label_idx].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                writer.write(line)
                if not preds_list[example_id]:
                    example_id += 1
            elif preds_list[example_id]:
                output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                writer.write(output_line)
            else:
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    def set_labels(self, data_dir: str, mode: Union[Split, str]):
        pass

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Chunk(NER):
    def __init__(self):
        # in CONLL2003 dataset chunk column is second-to-last
        super().__init__(label_idx=-2)

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return [
                "O",
                "B-ADVP",
                "B-INTJ",
                "B-LST",
                "B-PRT",
                "B-NP",
                "B-SBAR",
                "B-VP",
                "B-ADJP",
                "B-CONJP",
                "B-PP",
                "I-ADVP",
                "I-INTJ",
                "I-LST",
                "I-PRT",
                "I-NP",
                "I-SBAR",
                "I-VP",
                "I-ADJP",
                "I-CONJP",
                "I-PP",
            ]


class POS(TokenClassificationTask):
    def __init__(self, labels=None):
        if labels is None:
            labels = set()
        self.labels = labels

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                words = []
                labels = []
                for token in sentence:
                    words.append(token["form"])
                    labels.append(token["upos"])
                assert len(words) == len(labels)
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        error = 0
        # print("input: ", test_input_reader)  # test.txt
        # print("list: ", preds_list)  # lista di liste che contiene tutti i PoS tag per ogni frase in test.txt
        for sentence in parse_incr(test_input_reader):
            s_p = preds_list[example_id]  # [['VERB', 'DET', 'PROPN', 'PROPN', 'PUNCT'],
            if len(sentence) == len(s_p):
                out = ""
                for token in sentence:
                    out += f'{token["form"]} ({token["upos"]}|{s_p.pop(0)}) '
                out += "\n"
                writer.write(out)
            else:
                error += 1
            example_id += 1
        assert error == 0

    def set_labels(self, data_dir: str, mode: Union[Split, str]):
        self.labels.update(
            [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X",
            ]
        )

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            return self.labels


class DEPREL(TokenClassificationTask):

    def __init__(self, labels=None):
        if labels is None:
            labels = set()
        self.labels = labels

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                words = []
                labels = []
                for token in sentence:
                    words.append(token["form"])
                    labels.append(token["deprel"])
                assert len(words) == len(labels)
                if words:
                    # Create all the examples for token classification (train and test) with the correlating features
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        total = 0
        for sentence in parse_incr(test_input_reader):
            s_p = preds_list[example_id]
            out = ""
            for token in sentence:
                total += 1
                out += f'{token["form"]} ({token["deprel"]}|{s_p.pop(0)}) '
            out += "\n"
            writer.write(out)
            example_id += 1
        assert total == sum([len(elem) for elem in preds_list])
        assert example_id == len(preds_list)
        print("Nr.: ", example_id, " of sentences analyzed")  # 482 sentences
        print("Nr.: ", total, " of token analyzed")  # 10417 token

    def set_labels(self, data_dir: str, mode: Union[Split, str]):
        """
        set the set of labels for prediction
        """
        for file in mode:
            print("Extracting labels from: ", file.value)
            file_path = os.path.join(data_dir, f"{file.value}.txt")
            with open(file_path, encoding="utf-8") as f:
                for sentence in parse_incr(f):
                    for token in sentence:
                        self.labels.add(token["deprel"])
        print("label: ", self.labels)
        print("label len: ", len(self.labels))

    def get_labels(self, path: str) -> Union[List[str], dict]:
        """
        get the setted labels to predict
        """
        if path and self.labels == 0:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            return self.labels


class RELPOS(TokenClassificationTask):
    """
    # relative position of token's head
    self.pos = 0 if self.head == 0 else self.head - self.id

    max sentence length = 310
    """

    def __init__(self, labels=None):
        if labels is None:
            labels = set()
        self.labels = labels

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                words = []
                labels = []
                for token in sentence:
                    words.append(token["form"])
                    labels.append('0' if token["head"] == 0 else str(token["head"] - token["id"]))
                assert len(words) == len(labels)
                if words:
                    # Create all the examples for token classification (train and test) with the correlating features
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        total = 0
        left_threshold = -50
        right_threshold = 50
        for sentence in parse_incr(test_input_reader):
            s_p = preds_list[example_id]
            out = ""
            for token in sentence:
                total += 1
                label_val = 0 if token["head"] == 0 else token["head"] - token["id"]
                if left_threshold < label_val < right_threshold:
                    # self.labels.add('0' if token["head"] == 0 else str(token["head"] - token["id"]))
                    out += f'{token["form"]} ({label_val}|{s_p.pop(0)}) '
                else:
                    out += f'{token["form"]} (<unk>|{s_p.pop(0)}) '
            out += "\n"
            writer.write(out)
            example_id += 1
        assert total == sum([len(elem) for elem in preds_list])
        assert example_id == len(preds_list)
        print("Nr.: ", example_id, " of sentences analyzed")  # 482 sentences
        print("Nr.: ", total, " of token analyzed")  # 10417 token

    def set_labels(self, data_dir: str, mode: Union[Split, str]):
        """
        set the set of labels for prediction
        """
        left_threshold = -50
        right_threshold = 50
        self.labels.add("<unk>")
        for file in mode:
            print("Extracting labels from: ", file.value)
            file_path = os.path.join(data_dir, f"{file.value}.txt")
            with open(file_path, encoding="utf-8") as f:
                for sentence in parse_incr(f):
                    for token in sentence:
                        # labels must have str type
                        label_val = 0 if token["head"] == 0 else token["head"] - token["id"]
                        if left_threshold < label_val < right_threshold:
                            # self.labels.add('0' if token["head"] == 0 else str(token["head"] - token["id"]))
                            self.labels.add(str(label_val))
        print("label: ", self.labels)
        print("label len: ", len(self.labels))

    def get_labels(self, path: str) -> List[str]:
        """
        get the setted labels to predict
        """
        if path:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            # delta for relative position from -50 to 50 (english UD - PaT original article)
            # 0 -> 80, 137, 140 ... -1 -> -150, others 17 from -151 to -307
            # tot. labels = 203
            # result = list(range(-310, 145))
            # return list(map(str, result))
            return self.labels
