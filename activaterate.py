# *_*coding:utf-8 *_*
from nltk.translate.bleu_score import sentence_bleu
import math
import torch
from tool.DataTool import *
import torch.nn.functional as F
from model.TransformerWithActivateRate import Transformer as TransformerBase
from utils.beam_search import beamSearch


# 创建一个包装类，使模型兼容beamSearch（只返回2个值）
class TransformerWrapper(TransformerBase):
    def forward(self, enc_input, dec_input, enc_output=None):
        result = super().forward(enc_input, dec_input, enc_output)
        # 返回格式为 (enc_output, result, enc_rates, dec_rates)
        # 但beamSearch只需要前两个值
        return result[0], result[1]  # 只返回 enc_output, result

import warnings
warnings.filterwarnings("ignore")





if __name__ == '__main__':
    device = torch.device("cpu")
    print("cpu模式")

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

    # 使用基础模型来统计激活率
    model_base = TransformerBase(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    
    # 根据实际设备设置 map_location
    if torch.cuda.is_available():
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location="cuda:{}".format(map_gpu_index))
    else:
        m_state_dict = torch.load('./save/de2en_2k_0020.pt', map_location=torch.device('cpu'))
    # 使用 strict=False 因为模型结构稍有不同（移除weight引用属性）
    model_base.load_state_dict(m_state_dict, strict=False)
    model_base.eval()
    
    # 创建包装模型用于beamSearch（兼容旧接口）
    model = TransformerWrapper(len(encoder_chars), len(decoder_chars), d_model, d_ff, num_layers, num_heads, device, 0, 0, 0.1)
    model.load_state_dict(m_state_dict, strict=False)
    model.eval()

    # 用于统计激活率（未激活的神经元比例）
    encoder_activation_rates = [[] for _ in range(num_layers)]  # 每层的激活率列表
    decoder_activation_rates = [[] for _ in range(num_layers)]
    total_samples = 0

    print("\n开始统计FFN层激活率...")
    with torch.no_grad():
        # 先统计激活率
        test_s = open(test_file_path, 'r', encoding='utf-8').readlines()
        test_size_for_rate = min(100, len(test_s))  # 使用前100条数据统计激活率
        
        for line_idx, line in enumerate(test_s[:test_size_for_rate]):
            if line_idx % 10 == 0:
                print(f"  处理进度: {line_idx}/{test_size_for_rate}")
            
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            enc_input = parts[0]
            target_sentence = parts[1]
            
            enc_input_ids = [enc_vocab2id.get(char, 0) for char in enc_input.split()]
            if len(enc_input_ids) == 0:
                continue
                
            # 构造一个简单的解码输入用于前向传播（只需要开始标记即可）
            dec_input_ids = [dec_vocab2id.get(char_start, 0)]
            
            enc_input_tensor = torch.tensor([enc_input_ids], device=device)
            dec_input_tensor = torch.tensor([dec_input_ids], device=device)
            
            try:
                _, _, enc_rates, dec_rates = model_base(enc_input_tensor, dec_input_tensor)
                if enc_rates is not None:
                    for layer_idx, rate in enumerate(enc_rates):
                        encoder_activation_rates[layer_idx].append(rate.item())
                if dec_rates is not None:
                    for layer_idx, rate in enumerate(dec_rates):
                        decoder_activation_rates[layer_idx].append(rate.item())
                total_samples += 1
            except Exception as e:
                continue
    
    # 计算并输出激活率统计
    print("\n" + "="*60)
    print("FFN层激活率统计结果（未激活神经元比例）")
    print("="*60)
    
    print("\n【编码器 Encoder】")
    for layer_idx in range(num_layers):
        if encoder_activation_rates[layer_idx]:
            avg_rate = sum(encoder_activation_rates[layer_idx]) / len(encoder_activation_rates[layer_idx])
            activated_rate = 1 - avg_rate  # 激活率 = 1 - 未激活率
            print(f"  第 {layer_idx+1} 层: 未激活比例 = {avg_rate:.4f} ({avg_rate*100:.2f}%), "
                  f"激活比例 = {activated_rate:.4f} ({activated_rate*100:.2f}%)")
    
    print("\n【解码器 Decoder】")
    for layer_idx in range(num_layers):
        if decoder_activation_rates[layer_idx]:
            avg_rate = sum(decoder_activation_rates[layer_idx]) / len(decoder_activation_rates[layer_idx])
            activated_rate = 1 - avg_rate  # 激活率 = 1 - 未激活率
            print(f"  第 {layer_idx+1} 层: 未激活比例 = {avg_rate:.4f} ({avg_rate*100:.2f}%), "
                  f"激活比例 = {activated_rate:.4f} ({activated_rate*100:.2f}%)")
    
    # 计算总体平均
    if encoder_activation_rates[0]:
        all_enc_rates = [r for layer in encoder_activation_rates for r in layer]
        avg_enc_rate = sum(all_enc_rates) / len(all_enc_rates) if all_enc_rates else 0
        print(f"\n编码器平均未激活比例: {avg_enc_rate:.4f} ({avg_enc_rate*100:.2f}%)")
        print(f"编码器平均激活比例: {1-avg_enc_rate:.4f} ({(1-avg_enc_rate)*100:.2f}%)")
    
    if decoder_activation_rates[0]:
        all_dec_rates = [r for layer in decoder_activation_rates for r in layer]
        avg_dec_rate = sum(all_dec_rates) / len(all_dec_rates) if all_dec_rates else 0
        print(f"解码器平均未激活比例: {avg_dec_rate:.4f} ({avg_dec_rate*100:.2f}%)")
        print(f"解码器平均激活比例: {1-avg_dec_rate:.4f} ({(1-avg_dec_rate)*100:.2f}%)")
    
    print("\n" + "="*60)
    print("开始翻译测试...")
    print("="*60 + "\n")

    bleu_score_1 = 0
    bleu_score_2 = 0
    bleu_score_3 = 0
    bleu_score_4 = 0
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
