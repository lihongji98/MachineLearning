import torch
from torch.utils.data import DataLoader

from utils import generate_voc_buffer, PairDataset


def beam_search_decoder(model, src, start_token=1, end_token=2, beam_width=5, max_length=128, device="cuda"):
    with torch.no_grad():
        beam = [(torch.tensor([start_token]), 0)]

        src_voc = generate_voc_buffer("no")
        src_tokens = [[src_voc.get(src[i], src_voc["<unk>"]) for i in range(len(src))]]
        infer_dataset = PairDataset(src_tokens, [[]])
        infer_data_loader = DataLoader(infer_dataset, batch_size=1)
        source, _ = next(iter(infer_data_loader))
        source = source.to(device)

        for _ in range(max_length):
            new_beam = []
            for sequence, score in beam:
                if sequence.view(-1)[-1] == end_token:
                    new_beam.append((sequence, score))
                    continue
                sequence = sequence.to(device)
                target = sequence.unsqueeze(0).to(device)
                next_token_probs = torch.softmax(model(source, target), dim=-1)[:, -1, :].flatten()
                top_tokens = torch.argsort(next_token_probs)[-beam_width:]
                for token in top_tokens:
                    new_sequence = torch.cat((sequence, token.unsqueeze(0)), dim=0)
                    new_score = score + torch.log(next_token_probs[token])
                    new_beam.append((new_sequence, new_score))

            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_width]

    return beam[0][0]


src_example = "og Gud Herren tok mennesket og satte ham i Edens have til å dyrke og vokte den .".split(" ")
trg_example = "Yahweh God took the man , and put him into the garden of Eden to dress it and to keep it .".split(" ")
