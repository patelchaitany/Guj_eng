# %% cell1
import numpy as np
import regex as re
from scipy.special import digamma
import collections
from utils import Trie
import token

class  Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0
        self.merges = {}
        self.token = {}
        self.char = {}
    def get_stats(self,corpous):
        stats = {}
        for word in corpous:
            for pair in zip(word[:],word[1:]):
                stats[pair] = stats.get(pair,0) + 1

        return stats

    def get_merge(self,text,pair,idx):
        new_text = []
        i = 0
        while i < len(text):
            if i < len(text)- 1 and text[i] == pair[0] and text[i+1] == pair[1]:
                new_text.append(idx)
                i += 2
            else:
                new_text.append(text[i])
                i += 1

        return new_text

    def build_vocab(self, corpus, num_merges):
        corpus = corpus.replace(" ","_")
        for i in corpus:
            self.char[ord(i)] = self.char.get(ord(i),0) + 1
        fillter = "|".join(
                [
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                    r"""\p{N}{1,3}""",
                    r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                    r"""\s*[\r\n]+""",
                    r"""\s+(?!\S)""",
                    r"""\s+""",
                ]
            )
        pattern = re.compile(fillter)

        new_corpus = re.findall(pattern,corpus)
        for i in range(len(new_corpus)):
            new_corpus[i] = new_corpus[i].encode("utf-8")
        stats = self.get_stats(new_corpus)
        for i in range(num_merges):
            idx = 256 + i
            best = max(stats,key = stats.get)
            id1,id2 = best
            if(best):
                self.vocab[idx] = "".join([self.vocab.get(id1,chr(id1)),self.vocab.get(id2,chr(id2))])
                self.token[idx] = stats[best]
            self.merges[best] = idx
            for i in range(len(new_corpus)):
                new_corpus[i] = self.get_merge(new_corpus[i],best,idx)
            stats = self.get_stats(new_corpus)


class Sentencpice:
    def __init__(self):
        self.vocab = None
        self.vocab_size = None
        self.trie = None
        self.maxlen = None

    def _initalize_(self,vocab,end):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.trie = Trie(end)
        norm = sum(list(vocab.values()))
        logsum = digamma(norm + 1)
        maxlen = 0
        for tok,val in vocab.items():
            logval = digamma(val + 1) - logsum
            self.trie.insert(tok,logval)
            maxlen = max(maxlen,len(tok))

        self.maxlen = maxlen

    def forward(self,text):
        N = len(text)
        d = [-np.inf]*(N+1)
        p = [None]*(N+1)
        d[0] = 0
        for i in range(1,N+1):
            for j in range(max(0,i- self.maxlen),i):

                final_token = text[j:i]
                # final_val = self.trie.val_serch(final_token)
                final_val = self.trie.val_search(final_token)
                if final_val and d[j] + final_val > d[i]:
                    d[i] = d[j] + final_val
                    p[i] = len(final_token)

            if p[i] is None:
                raise ValueError("Token not found")

        return d[-1],p

    def backward(self,text,p):

        idx = len(p)
        tokenization = []
        while idx > 1:
            next_idx = idx-p[idx-1]
            tok = text[next_idx-1:idx-1]
            tokenization.append(tok)
            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def E_step(self,tokens):
        counts = collections.Counter(tokens)
        norm = sum(list(counts.values()))
        logsum = digamma(norm + 1)
        for tok,val in counts.items():
            logval = digamma(val + 1) - logsum
            self.trie.set_value(tok,logval)

    def M_step(self,text):
        loss,p = self.forward(text)
        tokens = self.backward(text,p)
        self.E_step(tokens)
        return tokens,loss

    def EM_step(self,text,tokens):
        self.E_step(tokens)
        tokens,loss = self.M_step(text)
        return tokens,loss

    def EM_round(self,text,tokens,delta=0.01,max_iter = 10):
        tokens ,old_loss = self.M_step(text)
        for i in range(max_iter):
            tokens,loss = self.EM_step(text,tokens)
            if old_loss - loss < delta:
                break
            old_loss = loss

    def prune_tokens(self,tokens,characters,vocab_size,trim_factor=0.2):
        sorted_tokens = tokens.most_common()
        N = len(sorted_tokens)
        n_trim = int(N*trim_factor)
        for i in reversed(range(N)):
            if N<=vocab_size:
                return False
            if n_trim<=0:
                break
            token = sorted_tokens[i][0]

            if token not in characters:
                self.trie.set_value(token,0)
                token.pop(token)
                n_trim -= 1
                N-=1
        if n_trim>0:
            raise ValueError("Not enough tokens to prune")
        return True

    def fit(self,text,tokens,vocab_size,delta = 0.01,max_iter = 5,max_round = 5):
        text = re.sub(' ','_',text)
        if vocab_size>len(tokens):
            raise ValueError("Vocab size is greater than number of tokens")

        self._initalize_(tokens,"<|END|>")
        for i in range(max_round):
            self.EM_round(text,tokens,delta,max_iter)
            # if self.prune_tokens(tokens,text,vocab_size):
                # break

    def genralize_forward_step(self,text,nbest_size = 1):
        N = len(text)
        d = [-np.inf]*(N+1)
        p = [None]*(N+1)
        d[0]=0
        for i in range(1, N+1):
            d_queue = []
            p_queue = []
            for j in range(max(i-self.maxlen, 0), i):
                final_token = text[j:i]
                final_value = self.trie.val_search(final_token)
                if final_value:
                    curr_d = d[j]+final_value
                    curr_p = len(final_token)
                    d[i] = max(d[i], curr_d)
                    d_queue.append(curr_d)
                    p_queue.append(curr_p)
            ids = np.argsort(d_queue)[-nbest_size:]
            p[i] = [p_queue[z] for z in ids]
        return p

    def generalize_backward_step(self,text,p):
        idx = len(p)
        tokenization = []
        while idx > 1:
            back_steps = np.random.choice(p[idx-1])
            next_idx = idx-back_steps
            tok = text[next_idx-1:idx-1]
            tokenization.append(tok)
            idx = next_idx
        tokenization = list(reversed(tokenization))
        return tokenization

    def tokenize(self,text,nbest_size=1):
        text = re.sub(' ','_',text)
        p = self.genralize_forward_step(text,nbest_size)
        return self.generalize_backward_step(text,p)

# %% text
if __name__ == "__main__":
    text  = "Hell.It was quite clear that it would not have happened — world the family would not have been disgraced and the world of Gammer would not have been stunned and horrified — if Chawker Minor had not made the Grand Tour."
    tokenizer = Tokenizer()
    tokenizer.build_vocab(text,100)
    vocab = tokenizer.vocab

    new_tokens = {}
    for i in vocab:
        new_tokens[vocab[i]] = tokenizer.token[i]
    for i in tokenizer.char:
        new_tokens[chr(i)] = tokenizer.char[i]
    print(new_tokens)
    sentencpice = Sentencpice()
    sentencpice.fit(text,new_tokens,3)
    print(sentencpice.tokenize("Hello world"))
