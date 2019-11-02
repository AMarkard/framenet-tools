from framenet_tools.data_handler.annotation import Annotation
from framenet_tools.config import ConfigManager
from framenet_tools.data_handler.reader import DataReader

from torchtext import data
import torch

from framenet_tools.role_identification.roleidnetwork import RoleIdNetwork
from framenet_tools.utils.static_utils import shuffle_concurrent_lists


def get_dataset(reader: DataReader):
    """
    Loads the dataset and combines the necessary data

    :param reader: The reader that contains the dataset
    :return: xs: A list of sentences appended with its FEE
             ys: A list of frames corresponding to the given sentences
    """

    xs = []
    ys = []

    for annotation_sentences in reader.annotations:
        for annotation in annotation_sentences:
            xs.append(annotation.sentence)
            ys.append(annotation.frame)

    return xs, ys


class RoleIdentifier(object):

    def __init__(self, cM: ConfigManager):
        self.cM = cM
        self.network = None
        self.input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )

    def predict_roles(self, annotation: Annotation):
        """
        Predict roles for all spans contained in the given annotation object

        NOTE: Manipulates the given annotation object!

        :param annotation: The annotation object to predict the roles for
        :return:
        """

        # TODO

    def train(self, m_reader: DataReader, m_reader_dev: DataReader):
        """

        :param m_reader:
        :param m_reader_dev:
        :return:
        """

        embedding_layer = self.gen_embedding_layer(m_reader)

        train_set = self.get_dataset(m_reader)
        dev_set = self.get_dataset(m_reader_dev)

        self.network = RoleIdNetwork(self.cM, embedding_layer)

    def gen_embedding_layer(self, reader: DataReader):
        """
        Helper function to generate a embedding layer from

        :param reader: The reader to generate the embedding layer from
        :return: The embedding layer
        """

        input_field = data.Field(
            dtype=torch.long, use_vocab=True, preprocessing=None
        )  # , fix_length= max_length) #No padding necessary anymore, since avg
        output_field = data.Field(dtype=torch.long)
        data_fields = [("Sentence", input_field), ("Frame", output_field)]

        xs = []
        ys = []

        new_xs, new_ys = get_dataset(reader)
        xs += new_xs
        ys += new_ys

        shuffle_concurrent_lists([xs, ys])

        examples = [data.Example.fromlist([x, y], data_fields) for x, y in zip(xs, ys)]

        dataset = data.Dataset(examples, fields=data_fields)

        input_field.build_vocab(dataset)
        output_field.build_vocab(dataset)

        input_field.vocab.load_vectors("glove.6B.300d")

        embed = torch.nn.Embedding.from_pretrained(input_field.vocab.vectors)

        self.input_field = input_field

        return embed

    def get_dataset(self, reader: DataReader):
        """

        :param reader:
        :return:
        """

        # Stub
