import re
from typing import List

import torch

from transformer.inference import beam_search_decoder
from transformer.models.model import Transformer
from transformer.utils import generate_voc_buffer, glue_tokens_to_sentence

import subprocess


end_symbols: List[str] = []


def preprocess_string():
    cmd = [r"D:\pycharm_projects\MachineLearning\transformer\preprocess.bat"]
    null_device = subprocess.DEVNULL
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=null_device, text=True, check=True)


def delete_infer_buffer():
    cmd = r"del /F /Q D:\pycharm_projects\MachineLearning\transformer\infer.no"
    null_device = subprocess.DEVNULL
    try:
        subprocess.run(cmd, shell=True, stdout=null_device, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


def post_process_string():
    cmd = [r"D:\pycharm_projects\MachineLearning\transformer\post_process.bat"]
    null_device = subprocess.DEVNULL
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=null_device, text=True, check=True)


def preprocess(src_sentence: str):
    global end_symbols
    print(end_symbols)
    src_sentence = re.findall(r'[^.!?]+[.!?]?', src_sentence)
    src_sentence = [s.strip() for s in src_sentence if s.strip()]

    with open("infer.no", "w", encoding="UTF-8") as _file:
        for s in src_sentence:
            end_symbols.append(s[-1])
            if s != "":
                _file.write(s)
                _file.write("\n")

    preprocess_string()

    with open("infer.no", "r", encoding="UTF-8") as _file:
        _lines = [line.rstrip() for line in _file]
    _line_to_translate = [_lines[i].split(" ") for i in range(len(_lines))]

    delete_infer_buffer()

    return _line_to_translate


def post_process(_predicting_sentence):
    global end_symbols
    with open("infer.no", "w", encoding="UTF-8") as file:
        for line in _predicting_sentence:
            file.write(line)
            file.write("\n")
    file.close()

    post_process_string()

    with open("infer.no", "r", encoding="UTF-8") as file:
        lines = [line.rstrip() for line in file]
        # lines = [line + '.' for line in lines if not re.search(r'[.!?]$', line) and re.search(r'\w+$', line)]
        for i in range(len(lines)):
            if not re.search(r'[.!?]$', line) and re.search(r'\w+$', lines[i]):
                lines[i] += end_symbols[i]

    delete_infer_buffer()

    lines = " ".join(lines)

    return lines


if __name__ == "__main__":
    src_example = "Ja, alt i orden. Bare hyggelig. Vi ser fram til deres opphold her hos oss. På gjensyn."
    lines_to_translate = preprocess(src_example)

    voc_trg = generate_voc_buffer("en", 16000)
    reversed_trg_dict = {v: k for k, v in voc_trg.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_num=16000, trg_vocab_num=16000, max_len=128, embedding_dim=512, stack_num=4, ffn_dim=2048, qkv_dim=64, head_dim=8, device=device).to(device)
    checkpoint = torch.load(r"D:\pycharm_projects\MachineLearning\transformer\parameters\best_3.17_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])

    predicting_sentences = []
    for line_to_translate in lines_to_translate:
        model.eval()
        with torch.no_grad():
            predicting_sentence = beam_search_decoder(model, line_to_translate, device=device, beam_width=10, vocab_size=16000).cpu().numpy()
            predicting_sentence = [reversed_trg_dict.get(predicting_sentence[i], "<unk>") for i in range(len(predicting_sentence))][1:-1]
        predicting_sentence = glue_tokens_to_sentence(predicting_sentence)

        predicting_sentences.append(predicting_sentence)

    predicting_sentences = post_process(predicting_sentences)

    print(predicting_sentences)
