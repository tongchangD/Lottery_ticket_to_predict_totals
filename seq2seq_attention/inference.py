import os

import torch
from torch.autograd import Variable

import config
from data_loader import Language, PairsLoader
from model import AttnDecoderRNN, EncoderRNN


def indexes_from_sentence(language, sentence):
    # indexes = [language.word2index[word] for word in sentence if word != ""]
    # indexes = [language.word2index[word] if word != "" and word in language.word2index.keys() else language.word2index['<>'] for word in sentence.split(' ')]
    # indexes = [language.word2index[word] if word in language.word2index.keys() else language.word2index['<>'] for word in sentence.replace(' ','')]
    # indexes = [language.word2index[word] if word in language.word2index.keys() else language.n_words + 1 for word in sentence.replace(' ','')]
    indexes = [language.word2index[word] if word in language.word2index.keys() else language.n_words + 1 for word in sentence.split(" ")]

    if len(indexes) > config.MAX_LENGTH:
        indexes = indexes[:config.MAX_LENGTH]
    # indexes.append(EOS_token)
    # print (indexes)
    return indexes


def pad_sentence(sentence):
    # pad on the right
    results = [0 for i in range(config.MAX_LENGTH+1)]
    for i in range(len(sentence)):
        results[i] = sentence[i]
    return results


# def load_model_param(language, model_dir):
#     encoder = EncoderRNN(language.n_words, config.HIDDEN_SIZE,
#                          config.NUM_LAYER, max_length=config.MAX_LENGTH + 1)
#     decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
#                              language.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)
#
#     encoder_path = os.path.join(model_dir, "encoder_400.pth")
#     decoder_path = os.path.join(model_dir, "decoder_400.pth")
#     encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
#     decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
#     encoder.eval()
#     decoder.eval()
#     return encoder, decoder


def inference(sentence, language,MODEL_DIR,codersum):
    encoder = EncoderRNN(language.n_words, config.HIDDEN_SIZE,
                         config.NUM_LAYER, max_length=config.MAX_LENGTH+1)
    decoder = AttnDecoderRNN(config.ATT_MODEL, config.HIDDEN_SIZE,
                             language.n_words, config.NUM_LAYER, dropout_p=config.DROPOUT)

    encoder_path = os.path.join(MODEL_DIR, "encoder_"+str(codersum)+".pth")
    decoder_path = os.path.join(MODEL_DIR, "decoder_"+str(codersum)+".pth")
    encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    encoder.eval()
    decoder.eval()
    batch_size = 1

    input_index = indexes_from_sentence(language, sentence)
    input_index = pad_sentence(input_index)  # 填充
    input_variable = torch.LongTensor([input_index])
    encoder_hidden, encoder_cell = encoder.init_hidden(batch_size)
    encoder_outputs, encoder_hidden, encoder_cell = encoder(
        input_variable, encoder_hidden, encoder_cell)

    decoder_input = torch.zeros(batch_size, 1).long()
    decoder_context = torch.zeros(batch_size, decoder.hidden_size)
    decoder_hidden = encoder_hidden
    decoder_cell = encoder_cell
    if config.USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoded_words = []

    # Run through decoder
    for di in range(config.MAX_LENGTH):
        decoder_output, decoder_context, decoder_hidden, decoder_cell, _ = decoder(
            decoder_input, decoder_context, decoder_hidden, decoder_cell, encoder_outputs)

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == 0:
            break
        else:
            decoded_words.append(language.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        if config.USE_CUDA:
            decoder_input = decoder_input.cuda()

    return "".join(decoded_words)


# if __name__ == "__main__":
#     chinese = Language(vocab_file="./couplet/vocabs")
#     sentence = "爱的魔力转圈圈"
#     words = inference(sentence, model_dir=config.MODEL_DIR, language=chinese)
#     print(words)
