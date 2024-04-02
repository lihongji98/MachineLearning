import collections


def voc_generate(lang):
    bpe_vocab = collections.Counter()
    with open(f'./voc.{lang}', 'r', encoding='utf-8') as f:
        for line in f:
            token, count = line.strip().split()
            bpe_vocab[token] = int(count)

    sorted_bpe_vocab = sorted(bpe_vocab.items(), key=lambda x: x[1], reverse=True)

    vocab_table = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    index = 4

    for token, count in sorted_bpe_vocab:
        vocab_table[token] = index
        index += 1

    with open(f'./voc_{lang}.txt', 'w', encoding='utf-8') as f:
        for token, idx in vocab_table.items():
            f.write(f'{token}\t{idx}\n')


if __name__ == "__main__":
    voc_generate("en")
    voc_generate("no")
