import math, torch ,torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PretrainedConfig

# 模型配置类，继承自transformers里面的pretrainedconfig
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"     # 模型的类型是minimind
    # init, 参数分别为：每个token的隐藏向量维度为768，默认有8层transformers block，**kwargs表示可以接收其他参数
    def __init__(self, hidden_size = 768, num_hidden_layers = 8, use_moe = False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size      # 保存隐藏层维度。后面 embedding、attention、MLP 都会用它。
        self.num_hidden_layers = num_hidden_layers      # 保存transformer的层数
        self.use_moe = use_moe      # 是否使用moe架构
        self.dropout = kwargs.get("dropout", 0.0)   # 是否使用dropout，从kwargs里面提取，如果没写就是0
        self.vocab_size = kwargs.get("vocab_size",6400)     # 词表的大小，默认6400
        self.bos_token_id = kwargs.get("bos_token_id",1)        # 句子开始token的id
        self.eos_token_id = kwargs.get("eos_token_id"),2        # 句子token结束的id，生成的时候可以用它判断何时停止
        self.pad_token_id = kwargs.get("pad_token_id",0)        # 保存padding token的id，当batch内句子的长度不一致的时候，用padding token补齐
        self.flash_attn = kwargs.get("flash_attn",True)         # 是否使用pytorch的flash attention这一高效的attention实现
        self.num_attention_heads = kwargs.get("num_attention_heads",8)     # multi-head attention里面的attention head的数量
        self.num_key_value_heads = kwargs.get("num_key_value_heads", self.num_attention_heads)  # 保存 key/value head 数量。这里默认等于 attention head 数量，也就是普通多头注意力。以后如果小于 num_attention_heads，就是 GQA。
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)        # 保存每个attention head的维度
        self.hidden_act = kwargs.get("hidden_act", "silu")      # 保存 MLP 里的激活函数名。"silu" 是 LLaMA/MiniMind 常用激活函数。
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64)*64)  # 保存前馈网络中间层维度

        
