import math
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from torch import nn;
#提前说明：register_parameter()用于定义可学习参数（可变量） register_buffer()用于定义不可学习参数（不更新的常量）
#input shape [batch,max_seq_len,d_model]
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len=512):    #这里长度是随便先定义的默认值，等后面操作还可以截断
       super().__init__()

       #position 的shape [max_seq_len,1]
       position=torch.arange(0,max_seq_len).unsqueeze(1)#先unsqueeze方便等会再升一次维度到batch的三维
       
       #[d_model/2]
       item = (1/10000)**(torch.arange(0,d_model,2)/d_model)
       
       #[max_seq_len, d_model/2]
       tmp_pos=position*item #这里就用到广播机制了，只要两张量维度不同且有其中一个维度为1的，若进行逐元素操作运算就会广播对齐维度
       
       pe=torch.zeros(max_seq_len,d_model)#创建位置编码矩阵
       pe[:,0::2]=torch.sin(tmp_pos)#这个切片操作左边取的形状本身就是[max_seq_len, d_model/2]，所以这步未广播
       pe[:,1::2]=torch.cos(tmp_pos)

       pe=pe.unsqueeze(0)#[1,max_seq_len,d_model]，增一个维度为batch预留，后面对于位置编码矩阵肯定是与embedding相加，所以广播机制会发力，就不用管别的了
       self.register_buffer("pe",pe,False)
    def forward (self,x):
        batch,seq_len,_=x.shape
        pe=self.pe
        return x+pe[:,:seq_len,:]#这里的seq_len就是我们自己指定的最大截取长度

def attention(query,key,value,mask=None,dropout=None):#缩放点积注意力，mask包括对填充位的和decoder部分上三角的
    d_k=key.shape[-1]#取key最后一个维度作为缩放因子，即每个向量的特征维度,实际上d_k=d_q=d_model=d_v

    #这里query,key,value shape：[batch,max_seq_len,d_k]
    att_=torch.matmul(query,key.transpose(-2,-1))/d_k**0.5#Q*KT/sqrt(d_k)

    if mask is not None:
        att_=att_.masked_fill(mask,-1e10)
    
    #att_score shape [batch,max_seq_len,max_seq_len]
    att_score=torch.softmax(att_,dim=-1)

    if dropout is not None:
        att_score=dropout(att_score)
    
    #最终返回shape [batch,max_seq_len,d_k]
    return torch.matmul(att_score,value)

class MutiHeadAttention (nn.Module):
    def __init__(self, heads, d_model,dropout=0.1):
        #注意括号里面传的只是形参，用来赋值的，而不是代表对象属性的值
        super().__init__()
        assert d_model%heads==0
        self.q_linear=nn.Linear(d_model,d_model,False)#这里就是调用线性层函数将输入乘对应可学习权重矩阵得到qkv
        self.k_linear=nn.Linear(d_model,d_model,False)#Linear前两个形参都是输入张量的形状的最后一个维度数
        self.v_linear=nn.Linear(d_model,d_model,False)#最后一个bool值表示是否加偏置
        self.dropout=nn.Dropout(dropout)
        self.linear=nn.Linear(d_model,d_model,False)
        self.heads=heads
        self.d_k=d_model//heads
        self.d_model=d_model
    def forward (self,q,k,v,mask):
        batch=q.shape[0]
        #维度变化# [n,seq_len,d_model] -> [n,seq_len,heads,d_k] -> [n,heads,seq_len,d_k]
        #注意：必须先reshape成(batch,seq_len,heads,d_k)再transpose，不能直接reshape成(batch,heads,seq_len,d_k)
        #否则会把不同位置的特征混在一起，导致注意力和掩码全部错位！实际上画图分多头也是先分头(即d_model整除),再调整维度的位置
        q=self.q_linear(q).reshape(batch,-1,self.heads,self.d_k).transpose(1,2)
        k=self.k_linear(k).reshape(batch,-1,self.heads,self.d_k).transpose(1,2)
        v=self.v_linear(v).reshape(batch,-1,self.heads,self.d_k).transpose(1,2)
        out=attention(q,k,v,mask,self.dropout)
        #要把heads维度去掉还原回原来形状[batch,max_seq_len,d_model]
        out=out.transpose(1,2).contiguous().reshape(batch,-1,self.d_model)
        out=self.linear(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model,d_ff,dropout=0.1):#d_ff是隐藏层维度
        super().__init__()
        self.ffn=nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,False),
            nn.Dropout(dropout)
        )
    def forward (self,x):
        return self.ffn(x)
    
class EncoderLayer (nn.Module):#因为encoder它论文中说了可以多个进行堆叠，所以先定义每一层的encoder结构
    def __init__(self, heads, d_model,d_ff,dropout=0.1):
        super().__init__()
        self.self_muti_head_att=MutiHeadAttention(heads,d_model,dropout)
        self.ffn=FFN(d_model,d_ff,dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(2)])#等价于[layernorm,layernorm]因为encoder有两个归一化层
        self.dropout=nn.Dropout(dropout)
    def forward (self,x,mask=None):
        multi_head_att_out=self.self_muti_head_att(x,x,x,mask)
        multi_head_att_out=self.norms[0](x+multi_head_att_out)
        ffn_out=self.ffn(multi_head_att_out)
        ffn_out=self.norms[1](ffn_out+multi_head_att_out)
        return ffn_out
    
class Encoder (nn.Module):
    def __init__(self, vocab_size,pad_idx,d_model,num_layers,heads,d_ff,dropout=0.1,max_seq_len=512):#这里vocab_size是指的embedding表行数，也即将token映射成ID的词表大小
        super().__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model,pad_idx)
        self.position_encode=PositionalEncoding(d_model,max_seq_len)
        self.encoder_layers=nn.ModuleList([EncoderLayer(heads,d_model,d_ff,dropout)for i in range(num_layers)])

    def forward (self,x,src_mask=None):
        embedded_x=self.embedding(x)*math.sqrt(self.d_model)#原论文要求对embedding乘sqrt(d_model)进行缩放，防止位置编码信号淹没词嵌入
        pos_encode_x=self.position_encode(embedded_x)
        for layers in self.encoder_layers:
            pos_encode_x=layers(pos_encode_x,src_mask)
        return pos_encode_x
    
class DecoderLayer (nn.Module):
    def __init__(self, heads,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.masked_att=MutiHeadAttention(heads,d_model,dropout)#因为我们在注意力模块里面已经把掩码考虑进去了，所以这里直接复用多头注意力即可
        self.att=MutiHeadAttention(heads,d_model,dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(3)])
        self.ffn=FFN(d_model,d_ff,dropout)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,encoder_kv,encoder_decoder_mask=None,decoder_mask=None):#encoder_decoder_mask指的是在进行交叉多头注意力的时候
        #使用的应该是纯对padding项的掩码(在q*KT之后加上掩码)；decoder_mask指的是掩码多头注意力时候加上的自回归生成的上三角掩码
        mask_att_out=self.masked_att(x,x,x,decoder_mask)
        mask_att_out=self.norms[0](x+mask_att_out)
        att_out=self.att(mask_att_out,encoder_kv,encoder_kv,encoder_decoder_mask)
        att_out=self.norms[1](mask_att_out+att_out)
        ffn_out=self.ffn(att_out)
        ffn_out=self.norms[2](att_out+ffn_out)
        return ffn_out

class Decoder (nn.Module):
    def __init__(self,vocab_size,pad_idx,d_model,num_layers,heads,d_ff,dropout=0.1,max_seq_len=512 ):
        super().__init__()
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model,pad_idx)
        self.position_encode = PositionalEncoding(d_model,max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(heads,d_model,d_ff,dropout) for i in range(num_layers)])
    
    def forward(self,x,encoder_kv,encoder_decoder_mask=None,decoder_mask=None):
        embedded_x=self.embedding(x)*math.sqrt(self.d_model)#原论文要求对embedding乘sqrt(d_model)进行缩放
        pos_encode_x=self.position_encode(embedded_x)
        for layers in self.decoder_layers:
            pos_encode_x=layers(pos_encode_x,encoder_kv,encoder_decoder_mask,decoder_mask)
        return pos_encode_x
    
class Transformer (nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,pad_idx,d_model,num_layers,heads,d_ff,dropout=0.1,max_seq_len=512):
        super().__init__()
        self.encoder=Encoder(enc_vocab_size,pad_idx,d_model,num_layers,heads,d_ff,dropout,max_seq_len)
        self.decoder=Decoder(dec_vocab_size,pad_idx,d_model,num_layers,heads,d_ff,dropout,max_seq_len)
        self.linear=nn.Linear(d_model,dec_vocab_size)#最后要根据表还原回去，所以输出维度是解码器词表大小
        self.pad_idx=pad_idx


    def generate_mask(self,query,key,is_triu_mask=False):
        '''
        注意创建mask的那个步骤是在将原始token映射成词表中的ID的时候就根据他们的ID创建了,而加入掩码则是
        在Q*KT/sqrt(dk)之后才加上
        所以一开始刚创建掩码矩阵的形状是(batch,seq_len)，相当于有几个batch，每个batch的token排成一行
        '''
        device=query.device
        batch,seq_q=query.shape
        batch,seq_k=key.shape
        #这里一开始没加入unsqueeze形状是(bacth,seq_k)，创建掩码只需对key执行即可，加上两个unsqueeze之后是(batch,1,1,seq_k)
        mask=(key==self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask=mask.expand(batch,1,seq_q,seq_k).to(device)#因为掩码矩阵是在Q*KT之后加上，所以维度需要与那两者匹配，形状就是(batch,heads,seq_q,seq_k)
        if is_triu_mask:#创建自回归生成的上三角矩阵
            #torch.triu一般习惯于创建二维上三角矩阵，所以先创建二维之后再升维度，diagonal表示沿着对角线向上移动1个之后往上的元素是1，其余是0
            decoder_triu_mask=torch.triu(torch.ones(seq_q,seq_k,dtype=torch.bool),diagonal=1)
            decoder_triu_mask=decoder_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch,1,seq_q,seq_k).to(device)
            return mask|decoder_triu_mask#这里是把padding掩码和自回归掩码做或的操作
        return mask
    
    def forward(self,input,output_shifted):
        encoder_mask=self.generate_mask(input,input)
        encoder_out=self.encoder(input,encoder_mask)
        decoder_mask=self.generate_mask(output_shifted,output_shifted,True)
        encoder_decoder_mask=self.generate_mask(output_shifted,input)#注意这里交叉注意力掩码矩阵也是要在刚映射到词表ID的时候进行创建，所以这里的output_shifted,input只是为了获取掩码矩阵的尺寸而已。因为交叉注意力的Q来自解码器，K来自编码器，而交叉注意力QKT之后的seq尺寸与原始输入一样
        decoder_out=self.decoder(output_shifted,encoder_out,encoder_decoder_mask,decoder_mask)
        out=self.linear(decoder_out)
        return out
    

if __name__=="__main__":
    # PositionEncoding(512,100)
    att = Transformer(100,200,0,512,6,8,1024,0.1)
    x = torch.randint(0,100,(4,64))
    y = torch.randint(0,200,(4,64))
    out = att(x,y)
    print(out.shape)






        
        



        



        