import sentencepiece as spm

class BaizhuTokenizer:
    def __init__(self, model_path="weights/tokenizer.model"):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text:str):
        return self.sp.EncodeAsIds(text)
    def decode(self, ids):
        return self.sp.DecodeIds(ids)
    @property
    def bos_id(self):
        return self.sp.bos_id()
    @property
    def eos_id(self):
        return self.sp.eos_id()
