class BaizhuTokenizer:
    def __init__(self):
        # 基础字符映射，预留bos eos
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0
        # 字符字典，动态构建
        self.char2id = {"<pad>":0, "<bos>":1, "<eos>":2}
        self.id2char = {0:"<pad>", 1:"<bos>", 2:"<eos>"}

    def encode(self, text: str):
        ids = []
        for ch in text:
            if ch not in self.char2id:
                new_id = len(self.char2id)
                self.char2id[ch] = new_id
                self.id2char[new_id] = ch
            ids.append(self.char2id[ch])
        return ids

    def decode(self, ids):
        text = ""
        for idx in ids:
            c = self.id2char.get(idx, "�")
            if c in ["<bos>","<eos>","<pad>"]:
                continue
            text += c
        return text
