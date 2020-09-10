import tensorflow as tf
import numpy as np

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i


        self.character = dict_character + ['[blank]']
        # self.character = dict_character
    
    def encode(self, text, batch_max_length=25):
        res_text = []
        lengths = []
        for t in text:
            length = len(t)
            s = ''.join(t)
            s = [self.dict[char] for char in s]
            res_text.append(s)
            lengths.append(length)

        return res_text, lengths
    
    def decode(self, text_index, length):
        texts = []
        index = 0
        num_char = len(self.character)
        for t, l in zip(text_index, length):
            char_list = []
            for i in range(l):
                if t[i] != (num_char - 1) and (not (i>0 and t[i-1] == t[i])):
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)
            texts.append(text)
            index += l

        return texts

class AttnLabelConverter():
    def __init__(self, character):
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i
    
    def encode(self, texts, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            texts: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in texts]    # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = np.zeros((len(texts), batch_max_length + 1), dtype=np.uint8)
        for i, t in enumerate(texts):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = text   # batch_text[:, 0] = [GO] token
        
        return batch_text, length
    
    def decode(self, text_index, length):
        texts = []
        batch_size = len(text_index)
        for index in range(batch_size):
            text = []
            for i in text_index[index, :]:
                if i == 1: # self.character[i] == '[s]'
                    break
                print(i, type(i))
                text.append(self.character[i])

            text = ''.join(text)
            texts.append(text)
        
        return texts

def dense_to_sparse(dense_tensor, out_type):
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(0, dense_tensor.dtype)))
    values = tf.gather_nd(dense_tensor, indices)
    shape = tf.shape(dense_tensor, out_type=out_type)
    return tf.SparseTensor(indices, values, shape)

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def convert2one_hot(vectors, dim):
    converted_vectors = []
    for v in vectors:
        tmp_vec = np.zeros((len(v), dim), dtype=int)
        for i, e in enumerate(v):
            tmp_vec[i][int(e)] = 1

        converted_vectors.append(tmp_vec)
          
    return np.stack(converted_vectors)

if __name__ == '__main__':
    ctc = CTCLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
    t, l = ctc.encode(['sang'])
    print(t, l)
    print(ctc.decode(t, l))
