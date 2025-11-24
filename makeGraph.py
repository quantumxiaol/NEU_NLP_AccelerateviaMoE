import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import math
import torch
from tool.DataTool import *
from tool.Global import *
import torch.nn.functional as F
from model.model4Graph import Transformer
from utils.beam_search import beamSearch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # 从环境变量获取设备，如果没有设置则使用默认值
    import os
    from dotenv import load_dotenv
    load_dotenv()
    device_str = os.getenv("DEVICE")
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print("running on device: ", device)
    encoder_chars, decoder_chars, max_enc_seq_length, max_dec_seq_length = calculate_data()
    print('encoder_chars:', len(encoder_chars))
    print('decoder_chars:', len(decoder_chars))
    print('max_enc_seq_length:', max_enc_seq_length)
    print('max_dec_seq_length:', max_dec_seq_length)

    enc_vocab2id = {word: i for i, word in enumerate(encoder_chars)}
    enc_id2vocab = {i: word for i, word in enumerate(encoder_chars)}

    dec_vocab2id = {word: i for i, word in enumerate(decoder_chars)}
    dec_id2vocab = {i: word for i, word in enumerate(decoder_chars)}

    print('-----------------')
    print(enc_vocab2id[char_space])
    print(dec_vocab2id[char_space])
    print('-----------------')
    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, 0, 0, 0.1, device=device)
    # 先加载到CPU，然后移动到目标设备（确保跨设备兼容性：CUDA/CPU/MPS）
    m_state_dict = torch.load('./save/de2en_2k.pt', map_location='cpu')
    model.load_state_dict(m_state_dict)
    model.to(device)
    model.eval()
    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
    with torch.no_grad():
        # 已知序列
        test_s = open(test_file_path, 'r', encoding='utf-8').readlines()
        test_size = 500
        graphM = torch.zeros(3, d_ff, d_ff).to(device)
        num = 0
        for line in test_s[:test_size]:
            print("{} / {}".format(num, test_size))
            num += 1
            enc_input = line.split('\t')[0]
            enc_pre_1 = enc_input.replace(" ", "")
            enc_pre_1 = enc_pre_1.replace("<e>", " ")

            target_sentence = line.split("\t")[1]
            target_sentence = target_sentence.replace(" ", "")
            target_sentence = target_sentence.replace("<e>", " ")
            k = 3
            enc_input = char_start + char_space + enc_input + char_space + char_end
            search_sources, search_result,  graphTmp= beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,
                                                       enc_input, k, graphCal=True, eChoice=False, eList=None)
            graphM += graphTmp
            tmp = graphM.cpu()
        np.save('./Expert/wIndex.npy', np.array(graphM.cpu()))






