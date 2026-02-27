import torch
from torch import nn,optim
from transformer import Transformer
from transformers import AutoTokenizer
from dataset import *
import tqdm,os
from torch.utils.data import DataLoader
if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=AutoTokenizer.from_pretrained("./tokenizer",use_fast=False)
    tokenizer.add_special_tokens({"bos_token":"<s>","eos_token":"</s>"})

    encoder_vocab_size,decoder_vocab_size=len(tokenizer),len(tokenizer)#用len(tokenizer)获取实际所有token数量，包含新增的特殊token
    pad_idx=tokenizer.pad_token_id
    print(pad_idx)
    d_model=512
    num_layes=6
    heads=8
    d_ff=1024
    dropout = 0.1
    max_seq_len = 40
    batch_size = 256
    epochs = 35

    model=Transformer(encoder_vocab_size,decoder_vocab_size,pad_idx,d_model,num_layes,heads,d_ff,dropout,max_seq_len)
    model.to(device)

    train_dataset=EnglishChineseDataset(tokenizer,"./data/cmn.txt",max_seq_len)
    evaluation_dataset=EnglishChineseDataset(tokenizer,"./data/evaluation.txt",max_seq_len)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    evaluation_loader = DataLoader(evaluation_dataset,batch_size=batch_size,shuffle=False)

    optimizer=optim.Adam(model.parameters(),lr=1e-4)
    loss_function=nn.CrossEntropyLoss(ignore_index=pad_idx)

    best_evaluation_loss=float("inf")#把最佳模型权重初始化为正无穷，这样在第一轮验证以后一定会至少先保存一次

    with tqdm.tqdm(total=epochs) as t:
        for epoch in range(epochs):
            model.train()
            """
            train_loader是一个 DataLoader,每次迭代会自动从数据集中取出一个 batch的数据(我的batchsize=4，所以1个bacth有4条数据）。
            在datasets中，Dataset库我写的getitem方法会返回一个三元组，Dataloader会利用该方法将每次返回的三个张量
            沿着batch维度拼接后赋值给我下面写的三元组(encoder_input,decoder_input,decoder_label)
            enumerate()负责在迭代的同时给取出的每一个batch一个编号index
            """
            for index,(encoder_input,decoder_input,decoder_label) in enumerate(train_loader):
                encoder_input,decoder_input,decoder_label=encoder_input.to(device),decoder_input.to(device),decoder_label.to(device)
                outputs=model(encoder_input,decoder_input)
                preds=torch.argmax(outputs,-1)#取当前时间步中对词表中所有词预测分数最高的那个作为当前时间步的预测。注意这里未使用softmax，是未归一化的分数，训练时不用softmax因为softmax单增，分数最高也是概率最高。-1指的是沿最后一个维度
                label_mask=(decoder_label!=pad_idx)#找出哪些位置是填充位将其设为false

                correct=(preds==decoder_label)
                acc=torch.sum(label_mask*correct)/torch.sum(label_mask)
                #上面得到的outputs形状 [batch,seq_len,decoder_vocab_size]
                outputs=outputs.reshape(-1,outputs.shape[-1])#这步相当于做了展平操作:[batch*seq_len,decoder_vocab_size]
                decoder_label=decoder_label.reshape(-1)
                """
                这两步要展平是因为你下面要送入CrossEntropy计算损失，而交叉熵一般要求维度是
                input:[N,C]N是样本个数，C是类别数。这里展平就相当于有batch*seq_len个样本，而类别数就等于词表中词元的个数
                laebl:[N]
                """
                train_loss=loss_function(outputs,decoder_label)

                optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)#梯度裁剪，防止梯度爆炸，对模型所有参数生效，如果梯度<=1不做改动，如果>1会按同一个比例将梯度缩放到1
                optimizer.step()

                if index%50==0:
                    print(f"training:current epoch:{epoch+1} total epoch:{epochs} current iter:{index} total:{len(train_loader)} train loss = {train_loss.item():.4f} acc = {acc.item():.4f}")
                    #index表示进行到第几个batch，len(train_loader)表示总共有几个batch,loss和acc都是张量所以需要通过item()转换为Python标量便于打印

            
            print("Successfully save model !")
            model.eval()#将模型切换至评估验证模式，关闭dropout，batchnorm等训练时候才用的策略
            epoch_evaluation_loss=0.0
            epoch_evaluation_acc=0.0
            evaluation_steps=0
            with torch.no_grad():
                for index,(encoder_input,decoder_input,decoder_label) in enumerate(evaluation_loader):
                  encoder_input,decoder_input,decoder_label=encoder_input.to(device),decoder_input.to(device),decoder_label.to(device)
                  outputs=model(encoder_input,decoder_input)
                  preds=torch.argmax(outputs,-1)
                  label_mask=(decoder_label!=pad_idx)

                  correct=(preds==decoder_label)
                  evaluation_acc=torch.sum(label_mask*correct)/torch.sum(label_mask)
                  outputs=outputs.reshape(-1,outputs.shape[-1])
                  decoder_label=decoder_label.reshape(-1)
                  evaluation_loss=loss_function(outputs,decoder_label)

                  epoch_evaluation_loss+=evaluation_loss.item()
                  epoch_evaluation_acc+=evaluation_acc.item()
                  evaluation_steps+=1

            avg_evaluation_loss=epoch_evaluation_loss/max(evaluation_steps,1)
            avg_evaluation_acc=epoch_evaluation_acc/max(evaluation_steps,1)

            print(f"evaluation: current epoch:{epoch+1} total epoch:{epochs} current iter:{index} total batch:{len(evaluation_loader)} evaluation_loss:{avg_evaluation_loss:.4f} evaluation_acc:{avg_evaluation_acc:.4f}")

            torch.save(model.state_dict(),"best_model.pt")#意思是从model.state_dict()这个里面拿到模型里所有可学习参数（权重、偏置）的字典后保存到"best_model.pt"这个文件下
            
            if avg_evaluation_loss<best_evaluation_loss:
                best_evaluation_loss=avg_evaluation_loss
                torch.save(model.state_dict(),"best_model.pt")
                print(f"Saved best checkpoint: best_model.pt (evaluation_loss={best_evaluation_loss:.4f})")
            t.update(1)#表示把tqdm进度条前进一格


