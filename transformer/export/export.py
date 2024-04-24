import torch

from transformer.models.model import Transformer

device = torch.device("cpu")
model = Transformer(src_vocab_num=16000, trg_vocab_num=16000, max_len=128, embedding_dim=512, stack_num=4, ffn_dim=2048,
                    qkv_dim=64, head_dim=8, device=device).to(device)
checkpoint = torch.load(r"D:\pycharm_projects\MachineLearning\transformer\parameters\best_3.17_checkpoint.pth")
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

src_input = torch.randint(1, 16000, size=(1, 128), dtype=torch.long)
trg_input = torch.randint(1, 16000, size=(1, 1), dtype=torch.long)

input_names = ['encoder_input', 'decoder_input']
output_names = ['output']

torch.onnx.export(model,
                  (src_input, trg_input),
                  "No-En-Transformer.onnx",
                  export_params=True,
                  opset_version=16,
                  do_constant_folding=True,
                  input_names=input_names,
                  output_names=output_names,
                  dynamic_axes={'decoder_input': {1: 'sequence_length'}}
                  )
print(" ")
print('Model has been converted to ONNX!')
