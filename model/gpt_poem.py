import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import trange


class GPTPoem(nn.Module):
    def __init__(self, model_path, rhyme_path="../cache/rhyme.json"):
        super(GPTPoem, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)

        if isinstance(rhyme_path, dict):
            rhyme_dict = rhyme_path
        else:
            with open(rhyme_path, 'r', encoding='utf-8') as f:
                rhyme_dict = json.load(f)

        self.pingze_board_index = rhyme_dict['pingze_board_index']
        self.chr2rhyme = rhyme_dict['chr2rhyme']
        self.len_rhymes = rhyme_dict['len_rhymes']
        self.find_rhymes = rhyme_dict['find_rhymes']

    def forward(self, text, **kwargs):
        if text:
            inputs = self.tokenizer('[CLS]' + text, add_special_tokens=False, return_tensors="pt")
            return self.model(**inputs)
        return self.model(**kwargs)

    def generate(self, prefix='', topic='', len_sentence=5, num_sentence=8, top_k=42, top_p=0.0):
        if prefix.startswith('[CLS]'):
            prefix=prefix[5:]
        prefix = ''.join(prefix.split())

        self.model.eval()

        rhyme_feet_index = get_rhyme_index(len_sentence, num_sentence)
        rhyme_feet = random.choice(range(self.len_rhymes))
        occupied_rhyme_feet_ids = []

        len_poem = (len_sentence + 1) * num_sentence
        for i in trange(len(prefix), len_poem, desc='Generating'):
            if i == 0:
                prefix = self.generate_next_chr(topic, top_k=top_k, top_p=top_p)
            elif (i + 1) % (len_sentence + 1) == 0:
                next_chr = '。' if (i + 1) % (2 * len_sentence + 2) == 0 else '，'
                prefix += next_chr
            else:
                topic_prefix = insert_topic(prefix, topic, len_sentence) if topic else prefix
                if i in rhyme_feet_index:
                    next_chr = self.generate_next_chr(topic_prefix, rhyme_feet, occupied_rhyme_feet_ids,
                                                      top_k=top_k, top_p=top_p)
                else:
                    next_chr = self.generate_next_chr(topic_prefix, top_k=top_k, top_p=top_p)
                prefix += next_chr

        return prefix

    def generate_next_chr(self, prefix, rhyme_feet=None, occupied_rhyme_feet_ids=None, top_k=42, top_p=0.0):
        with torch.no_grad():
            outputs = self.forward(prefix)
        next_token_logits = outputs.logits[0, -1, :]
        next_token_logits[self.tokenizer.convert_tokens_to_ids('[UNK]，')] = -float('Inf')
        if rhyme_feet:
            rhyme_feet_tokens = self.find_rhymes[rhyme_feet]
            rhyme_feet_ids = self.tokenizer.convert_tokens_to_ids(rhyme_feet_tokens)
            next_token_logits[[i for i in range(len(next_token_logits)) if i not in rhyme_feet_ids]] = -float('Inf')
            next_token_logits[occupied_rhyme_feet_ids] = -float('Inf')
        else:
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        while True:
            next_token_ids = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
            next_chr = self.tokenizer.decode(next_token_ids)
            if is_chinese_char(next_chr):
                break
        if rhyme_feet:
            occupied_rhyme_feet_ids += next_token_ids.tolist()
        return next_chr


def get_rhyme_index(len_sentence, num_sentence):
    interval = 2 if num_sentence > 4 else 1
    rhyme_feet_index = [(i + 1) * (len_sentence + 1) * interval - 2 for i in range(num_sentence // interval)]
    return rhyme_feet_index


def insert_topic(prefix, topic, len_sentence=5):
    insert_index_in_sen = [j * 2 for j in range(len_sentence // 2)]
    insert_num_of_sen = len(prefix) // (len_sentence + 1) - 1
    insert_index_in_poem = set([i * (len_sentence + 1) + j
                                for i in range(insert_num_of_sen)
                                for j in insert_index_in_sen])
    topic_prefix = [random.choice(topic) if i in insert_index_in_poem else c for i, c in enumerate(prefix)]
    return ''.join(topic_prefix)


def top_k_top_p_filtering(logits, top_k=42, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def is_word(word):
    return all(['a' <= cp <= 'z' for cp in word.lower()])


def is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if 0x4E00 <= cp <= 0x9FFF or \
            0x3400 <= cp <= 0x4DBF or \
            0x20000 <= cp <= 0x2A6DF or \
            0x2A700 <= cp <= 0x2B73F or \
            0x2B740 <= cp <= 0x2B81F or \
            0x2B820 <= cp <= 0x2CEAF or \
            0xF900 <= cp <= 0xFAFF or \
            0x2F800 <= cp <= 0x2FA1F:
        return True
    return False
