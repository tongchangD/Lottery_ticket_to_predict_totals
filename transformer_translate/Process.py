import pandas as pd
import torchtext
from torchtext import data
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import config

def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data).read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()
    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data).read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    tokenize = lambda x: x.split()

    # TRG = data.Field(lower=True, tokenize=tokenize,init_token='<sos>', eos_token='<eos>')
    TRG = data.Field(lower=True, tokenize=tokenize,init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=tokenize)
    # print("TRG",TRG)
    # print("opt.premodels",opt.premodels)
    if opt.premodels and os.path.exists(opt.load_weights+"/"+opt.premodels_path):  # 是否加载 预训练模型
        try:
            print("loading presaved fields...", opt.load_weights + "/" + opt.premodels_path)
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("11error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
    return (SRC, TRG)

def create_dataset(opt, SRC, TRG):
    print("creating dataset and iterator... ")
    raw_data = {'src': [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    # print("raw_data",raw_data)
    # 此处开始制作 一个 csv 文件
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    # print("mask",len(mask),mask)
    df = df.loc[mask]
    # print("df.loc[mask]",df.loc[mask])
    df.to_csv("translate_transformer_temp.csv", index=False)
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)
    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)
    # os.remove('translate_transformer_temp.csv')
    # 此处 删除 制作的 csv 文件
    if not opt.premodels or os.path.exists(opt.load_weights+"/"+opt.premodels_path) :  # 加载权重
        SRC.build_vocab(train)  # 制作数据词表　
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            if not os.path.exists(opt.load_weights):
                os.mkdir(opt.load_weights)
                print("weights folder already exists, run program with -load_weights weights to load them")
            pickle.dump(SRC, open(config.weights+'/SRC.pkl', 'wb'))
            pickle.dump(TRG, open(config.weights+'/TRG.pkl', 'wb'))
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']
    return train_iter