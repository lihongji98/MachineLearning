from collections import defaultdict


def generate_voc(lang):
    vocab_table = defaultdict(lambda: len(vocab_table))
    vocab_table.default_factory = vocab_table.__len__

    with open(f'./voc/voc_{lang}.txt', 'r', encoding='utf-8') as f:
        for line in f:
            token, index_str = line.strip().split()
            vocab_table[token] = int(index_str)

    return vocab_table

