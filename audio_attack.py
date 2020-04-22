import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import random

import client
from components import Instance, Dimension

cache_path = "./cache/"
result_path = "./result/"
model_path = "DeepSpeech/model/output_graph.pbmm"
lm_path = "DeepSpeech/model/lm.binary"
trie_path = "DeepSpeech/model/trie"
output_file = "result/result.wav"


def levenshtein(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)
    return edit[len(str1)][len(str2)]


def load_wav(input_wave_file):
    """
    根据文件名导入wav文件
    :param input_wave_file: wav文件名
    :return: wav文件的numpy数组
    """
    fs, audio = wav.read(input_wave_file)
    assert fs == 16000  # 确保音频频率为16kHz
    print("Load wav from", input_wave_file)
    return audio


def save_wav(audio, output_wav_file):
    """
    导出wav文件到指定文件
    :param audio: 经过处理的音频numpy数组
    :param output_wav_file: 导出文件路径
    :return: null
    """
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    print('Save wav to', output_wav_file)


def create_random_sample(delta, audio, ds, target, cache_id):
    """
    生成随机对抗样本
    :param delta: 搜索空间
    :param audio: 原始音频
    :param ds: deepspeech模型
    :param target: 目标文本
    :param cache_id: 随机缓存ID
    :return: 四元组(perturbation, text, dd, path)
    """
    perturbation = []
    for i in range(delta.shape[0]):
        perturbation.append(random.randint(delta[i][0], delta[i][1]))
    perturbation = np.array(perturbation, dtype=np.int16)

    return create_sample(perturbation, audio, ds, target, cache_id)


def create_sample(perturbation, audio, ds, target, cache_id):
    """
    根据某个扰动生成对抗样本
    :param perturbation: 给定扰动
    :param audio: 原始音频
    :param ds: deepspeech模型
    :param target: 目标文本
    :param cache_id: 随机缓存ID
    :return: 四元组(perturbation, text, dd, path)
    """
    sample = []
    for i in range(audio.shape[0]):
        sample.append(audio[i] + perturbation[i])
    sample = np.array(sample, dtype=np.int16)
    sample = np.array(np.clip(np.round(sample), -2 ** 15, 2 ** 15 - 1), dtype=np.int16)

    path = cache_path + "sample_" + str(cache_id) + ".wav"
    save_wav(sample, path)
    text = client.run_deep_speech(ds, lm_path, trie_path, path)
    dd = levenshtein(text, target)
    per_dict = {"perturb": perturbation, "text": text, "dd": dd, "path": path}
    return per_dict


def delete_wav(path):
    """
    根据路径删除一个缓存文件
    :param path:
    :return:
    """
    if os.path.exists(path):
        os.remove(path)


def delete_cache(perturb_set):
    """
    删除缓存
    :param perturb_set: 缓存列表
    :return:
    """
    for perturbation in enumerate(perturb_set):
        path = perturbation["path"]
        delete_wav(path)


class AudioAttack:
    """
    算法类
    """

    def __init__(self, audio, dim, ds, len, pn, ss, mi, ub):
        self.audio = audio          # 原始音频文件的numpy数组
        self.dimension = dim        # 存储音频维度及可接受的扰动范围
        self.region = np.zeros((dim.getSize(), 2))      # 搜索空间
        self.region[:, 0] = self.dimension.getMin()     # 搜索空间各维度的扰动最小值
        self.region[:, 1] = self.dimension.getMax()     # 搜索空间各维度的扰动最大值
        self.ds = ds                # deep speech模型
        self.audio_length = len     # 音频长度(某种意义上即为维度)
        self.positive_num = pn      # 最优集大小
        self.sample_size = ss       # 备选集大小
        self.max_iteration = mi     # 最大扰动次数
        self.uncertain_bits = ub    # 每次更新搜索空间的最大更新维度数

        self.ini_label = None       # 原始音频的识别结果label
        self.tar_label = None       # 攻击的target label

        self.label = []             # 更新最优集和备选集时搜索空间的维度变化集
        self.pop = []               # 备选集
        self.pos_pop = []           # 最优集
        self.next_pop = []          # 存放每次更新最优集和备选集时的新扰动集
        self.optional = []          # 最优扰动

    def clear(self):
        self.pop = []
        self.pos_pop = []
        self.next_pop = []
        self.optional = []
        return

    def random_instance(self, dim, region):
        """
        生成全随机扰动
        :param dim:
        :param region:
        :return:
        """
        # completely random
        inst = Instance(dim)
        ins = np.random.uniform(region[0][0], region[0][1], dim.getSize())
        inst.setFeatures(ins)
        return inst

    def pos_random_instance(self, dim, region, label, pos):
        """
        根据某一扰动和收缩的搜索空间，生成对应位置随机变化的扰动
        :param dim:
        :param region:
        :param label:
        :param pos:
        :return:
        """
        ins = Instance(dim)
        ins.CopyFromInstance(pos)
        for i in range(len(label)):
            temp = random.uniform(region[label[i]][0], region[label[i]][1])
            ins.setFeature(label[i], temp)
        return ins

    def reset_model(self):
        """
        重置搜索空间
        :return:
        """
        self.region[:, 0] = self.dimension.getMin()
        self.region[:, 1] = self.dimension.getMax()
        self.label = []
        return

    def update_pop(self):
        """
        将根据新搜索空间新生成的扰动和原有扰动合并并取优
        :return:
        """
        self.next_pop.sort(key=lambda instance: instance.getFitness())
        self.pos_pop, self.pop = [], []
        for i in range(self.positive_num):
            self.pos_pop.append(self.next_pop[i])
        for i in range(self.sample_size):
            self.pop.append(self.next_pop[self.positive_num + i])
        if self.optional.getFitness() > self.pos_pop[0].getFitness():
            self.optional = self.pos_pop[0].CopyInstance()
        return

    def init(self):
        """
        初始化，生成长度为batch_size的随机扰动并取最优的positive_num + sample_size个扰动，生成pos_pop和pop集
        :return:
        """
        self.clear()
        self.reset_model()

        temp = []
        batch_size = 150
        for i in range(batch_size):
            ins = self.random_instance(self.dimension, self.region)
            temp.append(ins)

        self.run_once(temp)
        temp.sort(key=lambda instance: instance.getFitness())

        i = 0
        while i < self.positive_num:
            self.pos_pop.append(temp[i])
            i += 1
        while i < self.positive_num + self.sample_size:
            self.pop.append(temp[i])
            i += 1
        self.optional = self.pos_pop[0].CoptInstance()
        return

    def run_once(self, popset):
        """
        将popset中的所有扰动加上音频本身跑一次deep speech
        :param popset: 扰动数组，长度不定
        :return: 设定popset的fitness值
        """
        batch_size = len(popset)
        batch_input = np.zeros((batch_size, self.audio_length))
        for i in range(batch_size):
            batch_input[i] = popset[i] + self.audio
        # 预测
        for i in range(batch_size):
            pred_str = client.run_deep_speech(self.ds, batch_input[i])
            # 使用pred_str计算不满足度，回填入popset
            popset[i].setFitness(self.compute_dis(pred_str))
        return

    def compute_dis(self, pred):
        """
        根据识别的str计算不满足度
        :param pred:
        :return:
        """
        return -1

    def shrink(self, ins):
        """
        随机取pos_pop集的某个扰动，根据pop集的情况随机收缩搜索空间
        :param ins:
        :return:
        """
        opt_num = 0
        while opt_num < self.uncertain_bits:
            chosen_dim = random.randint(0, self.dimension.getSize() - 1)
            greater, less, max_, min_ = 0, 0, self.dimension.getMax(), self.dimension.getMin()
            stand = ins.getFeature(chosen_dim)
            for i in range(0, self.sample_size):
                temp = self.pop[i].getFeature(chosen_dim)
                if temp >= stand:
                    less += 1
                    min_ = temp if (temp < min_) else min_
                else:
                    greater += 1
                    max_ = temp if (temp > max_) else max_
            if greater >= less:
                self.region[chosen_dim][0] = random.uniform(max_, stand)
            else:
                self.region[chosen_dim][1] = random.uniform(stand, min_)
            self.label.append(chosen_dim)
            opt_num += 1
        self.label.sort()
        return

    def opt(self, label, target=None):
        self.ini_label = label
        self.tar_label = target

        # 初始化
        self.init()

        # run
        for iter in range(self.max_iteration - 1):
            # 如果fitness满足要求则退出

            next_pop = []
            for sam in range(self.sample_size):
                self.reset_model()
                chosen_pos = random.randint(0, self.positive_num - 1)
                self.shrink(self.pos_pop[chosen_pos])
                ins = self.pos_random_instance(self.dimension, self.region, self.label, self.pos_pop[chosen_pos])
                next_pop.append(ins)
            self.run_once(next_pop)
            self.next_pop = next_pop + self.pos_pop + self.pop

        return
