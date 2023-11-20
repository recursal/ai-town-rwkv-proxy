########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# Source: https://github.com/BlinkDL/ChatRWKV/blob/main/tokenizer/rwkv_tokenizer.py
########################################################################################################

import os, sys, time, random

print('''
#######################################################################################################################
Supports special tokens
#######################################################################################################################

This tokenizer is not used in any RWKV models yet. I plan to use it for the future multilang RWKV models.

Benefits:

* Good support of most languages, from European to CJK to Arabic and Hindi and more.

* Clean vocab. Good for code too. Vocab size = 65525 (use 0 for <|endoftext|>).

* Good at numbers: the numerical tokens are '0'~'9', '10'~'99', ' 0'~' 9', ' 10'~' 99'.

* Very easy tokenization:

** The input text must be in UTF-8.

** Greedy encoding: always pick the longest (in bytes) token (with the highest id) that matches your UTF-8 bytes.

* The tokenization result is surprisingly good, because the vocab respects word boundaries and UTF-8 boundaries.

For 10x faster speed:
mypyc rwkv_tokenizer.py
python3 -c "import rwkv_tokenizer"

#######################################################################################################################
''')

########################################################################################################
# Tokenizer #1 (reference, naive, slow)
########################################################################################################


########################################################################################################
# Tokenizer #2 (trie, faster) https://github.com/TkskKurumi/ChatRWKV-TRIE-Tokenizer
########################################################################################################

class TRIE:
    __slots__ = tuple("ch,to,values,front".split(","))
    to:list
    values:set
    def __init__(self, front=None, ch=None):
        self.ch = ch
        self.to = [None for ch in range(256)]
        self.values = set()
        self.front = front

    def __repr__(self):
        fr = self
        ret = []
        while(fr!=None):
            if(fr.ch!=None):
                ret.append(fr.ch)
            fr = fr.front
        return "<TRIE %s %s>"%(ret[::-1], self.values)
    
    def add(self, key:bytes, idx:int=0, val=None):
        if(idx == len(key)):
            if(val is None):
                val = key
            self.values.add(val)
            return self
        ch = key[idx]
        if(self.to[ch] is None):
            self.to[ch] = TRIE(front=self, ch=ch)
        return self.to[ch].add(key, idx=idx+1, val=val)
    
    def find_longest(self, key:bytes, idx:int=0):
        u:TRIE = self
        ch:int = key[idx]
        
        while(u.to[ch] is not None):
            u = u.to[ch]
            idx += 1
            if(u.values):
                ret = idx, u, u.values
            if(idx==len(key)):
                break
            ch = key[idx]
        return ret

class TRIE_TOKENIZER():
    def __init__(self, file_name):
        self.vocab_size = 65535
        self.idx2special = {}
        self.specialstr2idx = {}
        self.idx2specialstr = {}
        self.idx2token = {}
        sorted = [] # must be already sorted
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            
            
            x = eval(l[l.index(' '):l.rindex(' ')])
            
            if idx > 65529:
                self.idx2specialstr[idx] = x
                self.specialstr2idx[x] = idx
            
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            
            if idx > 65529:
                self.idx2special[idx] = x
                
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k,v in self.idx2token.items():
            self.token2idx[v] = int(k)
            
        self.special2idx = {}
        for k,v in self.idx2special.items():
            self.special2idx[v] = int(k)

        self.root = TRIE()
        for t, i in self.token2idx.items():
            _ = self.root.add(t, val=(t, i))

    def encodeBytes(self, src:bytes):
        idx:int = 0
        tokens = []
        
        while (idx < len(src)):
            _idx:int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert(idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)
        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src):
        tokens = []
        
        i = 0
        buffer_str = ""
        while i < len(src):
            buffer_str += src[i]
            for s in self.specialstr2idx.keys():
                if buffer_str.endswith(s):
                    tokens.extend(self.encodeBytes(buffer_str[:-len(s)].encode('utf-8')))
                    tokens.append(self.specialstr2idx[s])
                    buffer_str = ""
                    break
                    
            i += 1
            
        if buffer_str:
            tokens.extend(self.encodeBytes(buffer_str.encode('utf-8')))
        
        return tokens

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.idx2token

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
        print()

