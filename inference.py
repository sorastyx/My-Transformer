import torch
from transformer import Transformer
import os
from transformers import AutoTokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=False)
    tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>"})

    encoder_vocab_size, decoder_vocab_size = len(tokenizer), len(tokenizer)#用len(tokenizer)获取实际所有token数量
    pad_idx = tokenizer.pad_token_id
    d_model = 512
    num_layes = 6
    heads = 8
    d_ff = 1024
    dropout = 0.1
    max_seq_len = 40
    batch_size = 1
    epochs = 35

    model = Transformer(encoder_vocab_size, decoder_vocab_size, pad_idx, d_model, num_layes, heads, d_ff, dropout,
                        max_seq_len)
    model.to(device)

    if os.path.exists("./best_model.pt"):
        model.load_state_dict(torch.load("./best_model.pt", map_location=device))
        print("Success：best_model.pt")
    else:
        print("Failure: best_model.pt，random parameters！")
        exit()

    input = "No other mountain in the world is so high as Mt. Everest."

    input = tokenizer(input, padding="max_length", max_length=max_seq_len, truncation=True, return_tensors="pt")[
        "input_ids"]
    input = input.to(device)

    decoder_input = torch.ones(batch_size, max_seq_len, dtype=torch.long).to(device) * pad_idx  # 准备一个全部都是填充符ID的矩阵
    decoder_input[:, 0] = tokenizer.bos_token_id  # 把每个样本第0个位置改为bos开始符

    model.eval()
    eos_id = tokenizer.eos_token_id
    print("正在翻译中，请稍后。")
    with torch.no_grad():
        for i in range(1, decoder_input.shape[1]):
            logits = model(input, decoder_input)
            logits=torch.softmax(logits,dim=-1)
            next_token = torch.argmax(logits[:, i - 1, :], dim=-1)
            decoder_input[:, i] = next_token
            if eos_id is not None and torch.all(next_token == eos_id):#torch.all用于检查张量中的所有元素是否都满足某个条件
                break

        ids = decoder_input[0].tolist()#取第一个样本
        if ids[0] == tokenizer.bos_token_id:
            ids = ids[1:]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]#.index()是列表的内置方法，表示取该元素在列表中的索引
        ids = [x for x in ids if x != pad_idx]

        result = tokenizer.decode(ids, skip_special_tokens=True)
        print("翻译结果：", result)


