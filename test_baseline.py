"""
测试基础Transformer模型（基线测试）

该脚本用于测试训练好的基础Transformer模型的翻译性能。
功能包括：
- 加载训练好的模型权重
- 使用beam search进行翻译
- 计算BLEU分数评估翻译质量
- 统计翻译时间

使用方法：
    python test_baseline.py

注意：该脚本与vali.py功能类似，都是用于验证基础模型性能。
"""
# *_*coding:utf-8 *_*
import datetime
from nltk.translate.bleu_score import sentence_bleu
import math
from tool.DataTool import *
import torch.nn.functional as F
from model.Transformer import Transformer
from utils.beam_search import beamSearch

import time
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

    model = Transformer(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    # m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    if torch.cuda.is_available():
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    else:
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location=torch.device('cpu'))
    model.load_state_dict(m_state_dict)
    model.to(device)

    model.eval()

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0

    # 计算用时
    start_time = time.time()
    with torch.no_grad():
        # 已知序列
        test_s = open(test_file_path, 'r', encoding='utf-8').readlines()
        # 取前50条进行测试
        # test_size = 872
        test_size = 50
        for line in test_s[:test_size]:
            enc_input = line.split('\t')[0]
            enc_pre_1 = enc_input.replace(" ", "")
            enc_pre_1 = enc_pre_1.replace("<e>", " ")

            target_sentence = line.split("\t")[1]
            target_sentence = target_sentence.replace(" ", "")
            target_sentence = target_sentence.replace("<e>", " ")

            print("原文：{}".format(enc_pre_1))
            print("参考译文：{}".format(target_sentence))
            k = 3
            enc_input = char_start + char_space + enc_input + char_space + char_end
            search_sources, search_result = beamSearch(model, enc_id2vocab, enc_vocab2id, dec_id2vocab, dec_vocab2id,
                                                       enc_input, k)
            for i in range(k):
                dec_input = search_result[i]
                # 将下标转化成句子
                sent = ''
                for w in dec_input:
                    sent += dec_id2vocab[w] + ' '
                sent = sent.replace(" ", "")
                sent = sent.replace("<e>", " ")
                sent = sent.replace(char_start, "")
                sent = sent.replace(char_end, "")

                bleu_score_1 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(1, 0, 0, 0))
                bleu_score_2 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 1, 0, 0))
                bleu_score_3 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 1, 0))
                bleu_score_4 += sentence_bleu([target_sentence.split(char_space)], sent.split(char_space),
                                              weights=(0, 0, 0, 1))

                print('{:.3f},{}'.format(search_sources[i], sent))
            print(" ")
        bleu_score_1 = bleu_score_1 / test_size / k
        bleu_score_2 = bleu_score_2 / test_size / k
        bleu_score_3 = bleu_score_3 / test_size / k
        bleu_score_4 = bleu_score_4 / test_size / k
        print("bleu：{:.4f},{:.4f},{:.4f},{:.4f}".format(bleu_score_1,
                                                        bleu_score_2,
                                                        bleu_score_3,
                                                        bleu_score_4))

        print("mean bleu：{:.4f}".format((bleu_score_1 + bleu_score_2 + bleu_score_3 + bleu_score_4) / 4))

    finish_time = time.time()
    print("time:{:.4f}".format(finish_time - start_time))
