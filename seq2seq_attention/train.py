import os
import time
import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import config
from data_loader import Language, PairsLoader, get_couplets
from model import AttnDecoderRNN, EncoderRNN
from utils import as_minutes, time_since
from criterion import LanguageModelCriterion

SOS_token = 0
EOS_token = 1

writer = SummaryWriter(config.MODEL_DIR+"/events")
if not os.path.exists(config.MODEL_DIR):
    os.makedirs(config.MODEL_DIR)
if not os.path.exists(config.MODEL_DIR+"/events"):
    os.makedirs(config.MODEL_DIR+"/events")

chinese = Language(vocab_file="./couplet/vocabs")  # 导入 词向量
train_pairs = get_couplets(data_dir="./couplet/train")
val_pairs = get_couplets(data_dir="./couplet/test")

train_dataloader = PairsLoader(chinese, train_pairs, batch_size=config.BATCH_SIZE, max_length=config.MAX_LENGTH, target_length=config.TARGET_LENGTH)
val_dataloader = PairsLoader(chinese, val_pairs, batch_size=config.BATCH_SIZE, max_length=config.MAX_LENGTH, target_length=config.TARGET_LENGTH)

# Initialize models
encoder = EncoderRNN(chinese.n_words, config.HIDDEN_SIZE,
                     config.NUM_LAYER, max_length=config.MAX_LENGTH)
decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
                         chinese.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)
if config.RESTORE:
    encoder_path = os.path.join(config.PRE_MODEL_DIR, "encoder_2000.pth")
    decoder_path = os.path.join(config.PRE_MODEL_DIR, "decoder_2000.pth")
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    print("预训练模型加载完成")
# Move models to GPU
if config.USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.LR)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.LR)
criterion = LanguageModelCriterion()  # nn.NLLLoss(ignore_index=0)

start = time.time() 
plot_losses = []
print_loss_total = 0
plot_loss_total = 0

def forward(input_variable, target_variable, mask_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, batch_size,train=False):
    # encoder_step_schedule = optim.lr_scheduler.StepLR(step_size=20, gamma=0.9, optimizer=encoder_optimizer)  # tcd
    # decoder_step_schedule = optim.lr_scheduler.StepLR(step_size=20, gamma=0.9, optimizer=decoder_optimizer)  # tcd
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0  # Added onto for each word
    # Get size of input and target sentences
    target_length = target_variable.size()[1]
    # Run words through encoder
    encoder_hidden, encoder_cell = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_variable, encoder_hidden, encoder_cell)

    # Prepare input and output variables
    decoder_input = Variable(torch.zeros(batch_size, 1).long())
    decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
    # Use last hidden state from encoder to start decoder
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    decoder_outputs = []
    for di in range(target_length):
        decoder_output, decoder_context, decoder_hidden, decoder_cell, decoder_attention = decoder(
            decoder_input, decoder_context, decoder_hidden, decoder_cell, encoder_outputs)
        # loss += criterion(decoder_output, target_variable[:, di])
        decoder_outputs.append(decoder_output)
        # decoder_input = target_variable[:, di]  # Next target is next input
        decoder_input = torch.from_numpy(np.array([di]*batch_size)).cuda() # target_variable[:, di]  # Next target is next input

    decoder_predict = torch.cat(decoder_outputs, 1).view(batch_size, target_length, -1)
    loss = criterion(decoder_predict, target_variable, mask_variable)
    if train:
        encoder.zero_grad()  # 清除梯度 # tcd
        decoder.zero_grad()  # 清除梯度 # tcd
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.CLIP)  # 裁剪参数可迭代的梯度范数
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.CLIP)  # 裁剪参数可迭代的梯度范数
        encoder_optimizer.step()
        decoder_optimizer.step()
    # encoder_step_schedule.step()  # tcd
    # decoder_step_schedule.step()  # tcd
    return loss.item() # / target_length






for epoch in range(1, config.NUM_ITER + 1):
    # Get training data for this cycle
    input_index, output_index, mask_batch = next(train_dataloader.load())

    input_variable = Variable(torch.LongTensor(input_index))
    output_variable = Variable(torch.LongTensor(output_index))
    mask_variable = Variable(torch.FloatTensor(mask_batch))
    
    if config.USE_CUDA:
        input_variable = input_variable.cuda()
        output_variable = output_variable.cuda()
        mask_variable = mask_variable.cuda()
    # Run the train function
    loss = forward(input_variable, output_variable, mask_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion,
                 batch_size=config.BATCH_SIZE)


    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss
    print_loss_avg=1
    if epoch % config.PRINT_STEP == 0:
        print_loss_avg = print_loss_total / config.PRINT_STEP
        print_loss_total = 0
        writer.add_scalar('train_Loss',   print_loss_avg,epoch)
        print('epoch： %d 耗时%s 损失值在 %.8f' % \
              (epoch, time_since(start, epoch / config.NUM_ITER), print_loss_avg))
    if epoch % config.CHECKPOINT_STEP == 0:# or print_loss_avg <= 0.5: # 构建预训练模型 取消注释
        ########################
        input_variable, output_variable, mask_variable=val_dataloader.load_evl()
        input_variable = Variable(torch.LongTensor(input_index))
        output_variable = Variable(torch.LongTensor(output_index))
        mask_variable = Variable(torch.FloatTensor(mask_batch))

        if config.USE_CUDA:
            input_variable = input_variable.cuda()
            output_variable = output_variable.cuda()
            mask_variable = mask_variable.cuda()

        loss = forward(input_variable, output_variable, mask_variable, encoder, decoder,encoder_optimizer, decoder_optimizer, criterion,batch_size = config.BATCH_SIZE,train=True)

        writer.add_scalar('val_Loss',   print_loss_avg,epoch)
        print('测试_结果_epoch： %d 耗时%s 损失值在 %.8f' % \
              (epoch, time_since(start, epoch / config.NUM_ITER), print_loss_avg))
        ################################
        encoder_path = os.path.join(config.MODEL_DIR, "encoder_%s.pth"%epoch)
        decoder_path = os.path.join(config.MODEL_DIR, "decoder_%s.pth"%epoch)
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        print("%d models has saved%s"%(epoch,decoder_path))
writer.close()
print("done")

