from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import torch
import random
def split_data (data_path):
    data=open(data_path,'r',encoding="utf-8").readlines()#readlines()这个函数用于逐行读取文件并且返回一个列表，列表中每个元素代表从每行中读取的内容
    random.shuffle(data)
    train_data=data[:int(0.95*len(data))]#因为切片要求的数据类型是int，所以需要强制转换，这两个东西拿到的都是列表，还未写入文件
    evaluation_data=data[int(0.95*len(data)):]
    open("./data/train.txt",'w',encoding="utf-8").writelines(train_data)#创建磁盘文件，将列表中的东西写入磁盘中
    open("./data/evaluation.txt",'w',encoding="utf-8").writelines(evaluation_data)

def count_max_seq_len (data_path):
    data=open(data_path,'r',encoding="utf-8").readlines()
    max_seq_len=0
    for i in data:
        english,chinese=i.strip().split("\t")[:2]#strip()操作是去除多余空格换行符等空白字符，split那个是按照"\t"制表符将字符串切分成一个列表。机器翻译数据集通常以"\t"分隔源语言和目标语言，切片代表取的是英文和中文
        max_seq_len=max(max_seq_len,len(tokenizer(english)["input_ids"]),len(tokenizer(chinese)["input_ids"]))#这段是拿着历史最大长度和当前所取的这一行的中英文最大长度取max
        #["input_ids"]是指的tokenizer将原字符串映射为整数ID后其实会返回一个字典，包含两个键input_ids和attention_mask,其中input_ids才是我们真正想要的数字ID
    print(max_seq_len)

class EnglishChineseDataset(Dataset):
    def __init__(self,tokenizer,data_path,max_seq_len=64):
        self.tokenizer=tokenizer
        self.data=open(data_path,'r',encoding="utf-8").readlines()
        self.max_seq_len=max_seq_len
        self.data_cache={}
        if self.tokenizer.bos_token_id is None:
            raise ValueError("tokenizer.bos_token_id is None, please add bos_token first")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer.eos_token_id is None, please add eos_token first")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):#这个index实际上是Dataloader这个库根据__len__方法返回数据总行数，实际就是data那个列表里面一共有多少个元素，index就表示这些元素的索引
        if index in self.data_cache:
            return self.data_cache[index]
        encoder_input,decoder_input=self.data[index].strip().split("\t")[:2]#shape:[max_seq_len]
        encoder_input=self.tokenizer(encoder_input,padding="max_length",max_length=self.max_seq_len,truncation=True,return_tensors="pt")["input_ids"]#padding="max_length"指的是填充到最大长度保持对齐，truncation是超过最大长度进行截断，return_tensors="pt"指的是返回pytorch张量
        target_ids=self.tokenizer(decoder_input,add_special_tokens=False)["input_ids"]
        decoder_in_ids=[self.tokenizer.bos_token_id]+target_ids#输入需要右移加bos,因为这是列表，所以注意添加的顺序
        decoder_label_ids=target_ids+[self.tokenizer.eos_token_id]#标签需要带上eos

        decoder_in_ids=decoder_in_ids[:self.max_seq_len]
        decoder_label_ids=decoder_label_ids[:self.max_seq_len]

        if len(decoder_in_ids)<self.max_seq_len:
            decoder_in_ids=decoder_in_ids+[self.tokenizer.pad_token_id]*(self.max_seq_len-len(decoder_in_ids))
        if len(decoder_label_ids)<self.max_seq_len:
            decoder_label_ids=decoder_label_ids+[self.tokenizer.pad_token_id]*(self.max_seq_len-len(decoder_label_ids))

        decoder_in=torch.LongTensor(decoder_in_ids)#分词decoder_in和decoder_label的时候没有设置return_tensors.pt，所以形状已经是[max_seq_len]
        decoder_label=torch.LongTensor(decoder_label_ids)
        encoder_input=torch.LongTensor(encoder_input).squeeze(0)
        self.data_cache[index]=(encoder_input,decoder_in,decoder_label)#当你在tokenizer中设置return_tensors="pt"时，他默认你传入的是多个句子，所以返回张量时候的维度会是(batch,max_seq_len)我这里因为是一个句子一个句子地按索引逐条取的，所以不需要batch这个维度
        return encoder_input,decoder_in,decoder_label


if __name__=="__main__":
    tokenizer=AutoTokenizer.from_pretrained("./tokenizer",use_fast=False)
    tokenizer.add_special_tokens({"bos_token":"<s>","eos_token":"</s>"})
    print(tokenizer.bos_token,tokenizer.bos_token_id)
    dataset= EnglishChineseDataset(tokenizer,"./data/train.txt",40)
    print(dataset[0])