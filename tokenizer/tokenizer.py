# %% cel
from typing_extensions import List
import sentencepiece as spm
import os


class Tokenizer:
    def __init__(self,special_token):
        self.sp = spm.SentencePieceProcessor()
        self.special_token = special_token

    def load(self,path):
        self.sp.load(path)

    def trian(self,path:List,out_path,vocab_size:int,num_sentence:int):
        options = dict(
          # input spec
          input=path,
          input_format="text",
          # output spec
          model_prefix=out_path, # output filename prefix
          # algorithm spec
          # BPE alg
          model_type="bpe",
          vocab_size=vocab_size,
          # normalization
          normalization_rule_name="identity", # ew, turn off normalization
          remove_extra_whitespaces=False,
          input_sentence_size=num_sentence, # max number of training sentences
          max_sentence_length=4192, # max number of bytes per sentence
          seed_sentencepiece_size=1000000,
          shuffle_input_sentence=True,
          # rare word treatment
          character_coverage=0.99995,
          byte_fallback=True,
          # merge rules
          split_digits=True,
          split_by_unicode_script=True,
          split_by_whitespace=True,
          split_by_number=True,
          max_sentencepiece_length=16,
          add_dummy_prefix=True,
          allow_whitespace_only_pieces=True,
          # special tokens
          unk_id=0, # the UNK token MUST exist
          bos_id=1, # the others are optional, set to -1 to turn off
          eos_id=2,
          pad_id=3,
          user_defined_symbols = self.special_token,
          # systems
          num_threads=os.cpu_count(), # use ~all system resources
        )

        spm.SentencePieceTrainer.train(**options)

    def encode(self,text,out_type = "int"):
        if(out_type == "str"):
            return self.sp.encode_as_pieces(text)
        return self.sp.encode(text)
    def get_token_id(self, token):
        if token == "bos":
            return self.sp.bos_id()
        elif token == "eos":
            return self.sp.eos_id()
        elif token == "unk":
            return self.sp.unk_id()
        elif token == "pad":
            return self.sp.pad_id()
        else:
            return self.sp.piece_to_id(token)
    def decode(self,tokens):
        return self.sp.decode(tokens)
    def get_vocab_size(self):
        return self.sp.get_piece_size()


# %% cell1
if __name__ == "__main__":
    special_token = ["<en|gu>", "<gu|en>","<en>", "<gu>"]
    tokenizer = Tokenizer(special_token)
    tokenizer.trian(["mixed_data.txt"],"tokenizer",10000,0)

    # %% cell2
    tokenizer.load("tokenizer.model")

    print(tokenizer.encode("<en|gu> <pad> વપરાશકર્તા દ્વારા આપવામાં આવતી ઇનપુટ સામાન્ય જમણે-થી-ડાબે ફોર્મેટને બદલે ડાબેથી જમણે દાખલ થાય છે"))
    print(tokenizer.encode("<en|gu> <pad> વપરાશકર્તા દ્વારા આપવામાં આવતી ઇનપુટ સામાન્ય જમણે-થી-ડાબે ફોર્મેટને બદલે ડાબેથી જમણે દાખલ થાય છે",out_type="str"))
    print(tokenizer.decode(tokenizer.encode("વપરાશકર્તા દ્વારા આપવામાં આવતી ઇનપુટ સામાન્ય જમણે-થી-ડાબે ફોર્મેટને બદલે ડાબેથી જમણે દાખલ થાય છે")))
    # %% cell3
    print(tokenizer.get_vocab_size())
