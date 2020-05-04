import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import random

import client
from components import Instance

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


class AudioAttack:
    """
    算法类
    """

    def __init__(self, audio, ds, threshold, len, ss, mi, mr, ex):
        self.audio = audio  # 原始音频文件的numpy数组
        self.audio_length = len  # 音频长度(某种意义上即为维度)
        self.threshold = threshold  # 扰动范围阈值
        self.ds = ds  # deep speech模型
        self.sample_size = ss  # 备选集大小
        self.max_iteration = mi  # 最大扰动次数
        self.move_rate = mr  # 每次更新搜索空间的更新概率
        self.max_expand = ex  # 增大阈值范围的迭代次数

        self.ini_label = client.run_deep_speech(self.ds, self.audio)  # 原始音频的识别结果label
        self.tar_label = None  # 攻击的target label

        self.pop = []  # 备选集
        self.optional = None  # 最优扰动

    def clear(self):
        self.pop = []
        self.optional = None
        return

    def random_instance(self):
        """
        生成全随机扰动
        :param dim:
        :param region:
        :return:
        """
        # completely random
        inst = Instance()
        features = np.random.randint(0, 1, self.audio_length, dtype='int16')
        for i in range(self.audio_length):
            rate = np.random.rand()
            if rate > self.move_rate:
                features[i] = np.random.randint(-self.threshold, self.threshold)
        inst.setFeatures(features)
        regions = np.zeros((self.audio_length, 1))
        inst.setRegions(regions)
        inst.setLen(self.audio_length)
        return inst

    def update_optional(self):
        """
        将根据新搜索空间新生成的扰动和原有扰动合并并取优
        :return:
        """
        self.pop.sort(key=lambda instance: instance.getFitness())
        self.optional = self.pop[0]
        return

    def init(self):
        """
        初始化，生成长度为batch_size的随机扰动并取最优的positive_num + sample_size个扰动，生成pos_pop和pop集
        :return:
        """
        print("Initializing...")
        self.clear()

        self.pop = []
        for i in range(self.sample_size):
            ins = self.random_instance()
            self.pop.append(ins)

        self.run_once()
        self.pop.sort(key=lambda instance: instance.getFitness())
        print("Finish creating " + str(self.sample_size) + " samples.")

        self.optional = self.pop[0].CopyInstance()
        return

    def run_once(self, rtype=0):
        """
        将popset中的所有扰动加上音频本身跑一次deep speech
        :param popset: 扰动数组，长度不定
        :param rtype:
        :return: 设定popset的fitness值
        """
        batch_size = len(self.pop)
        # 预测
        for i in range(rtype, batch_size):
            audio_interrupted = self.audio + self.pop[i].getFeatures()
            pred_str = client.run_deep_speech(self.ds, audio_interrupted)
            # 使用pred_str计算不满足度，回填入pop
            fitness = self.compute_dis(pred_str)
            print("result: " + pred_str + ", fitness: " + str(fitness))
            self.pop[i].setString(pred_str)
            self.pop[i].setFitness(fitness)
        return

    def compute_dis(self, pred):
        """
        根据识别的str计算不满足度
        :param pred:
        :return:
        """
        return levenshtein(pred, self.tar_label)

    def opt(self, target=None):
        self.tar_label = target

        # 初始化
        self.init()

        # run
        expand = 0
        for iter in range(self.max_iteration - 1):
            print("iter time: " + str(iter))

            if expand == self.max_expand:
                expand = 0
                self.threshold *= 2
                print("threshold expanded to " + str(self.threshold))

            # 如果fitness满足要求则退出
            print("best string: " + self.optional.getString())
            if self.optional.getFitness() == 0:
                print("Find attack in " + str(iter) + " times. File saved in result.txt.")
                # 保存文件
                return

            self.pop = []
            self.pop.append(self.optional)
            for i in range(self.sample_size):
                ins = self.optional.CopyInstance()
                for index in range(ins.getLen()):
                    rate = np.random.rand()
                    if rate < self.move_rate:
                        # 更新该维度的阈值和扰动
                        feature = ins.getFeature(index)
                        region = ins.getRegion(index)
                        if feature > region:
                            region = feature + self.threshold
                            feature = np.random.randint(low=region - self.threshold, high=region + self.threshold,
                                                        dtype='int16')
                            ins.setRegion(index, region)
                            ins.setFeature(index, feature)
                        elif feature < region:
                            region = feature - self.threshold
                            feature = np.random.randint(low=region - self.threshold, high=region + self.threshold,
                                                        dtype='int16')
                            ins.setRegion(index, region)
                            ins.setFeature(index, feature)
                        else:
                            ins.setFeature(index,
                                           np.random.randint(low=region - self.threshold, high=region + self.threshold,
                                                             dtype='int16'))
                self.pop.append(ins)

            self.run_once(rtype=1)
            self.pop.sort(key=lambda instance: instance.getFitness())
            if self.optional.Equal(self.pop[0]):
                expand += 1
            else:
                expand = 0
                self.optional = self.pop[0].CopyInstance()

        return


m_audio_path = "DeepSpeech/audio/4507-16021-0012.wav"
m_model_path = "DeepSpeech/model/output_graph.pbmm"
m_lm_path = "DeepSpeech/model/lm.binary"
m_trie_path = "DeepSpeech/model/trie"
m_ds = client.load_model(m_model_path)
m_ds = client.update_ds(m_ds, m_lm_path, m_trie_path)
m_audio = client.get_audio_array(m_ds, m_audio_path)
m_target = 'open the door'

m_threshold = 1024
m_len = m_audio.shape[0]
m_ss = 4
m_mi = 100
m_mr = 0.1
m_ex = 24
m_audio_attack = AudioAttack(audio=m_audio,
                             ds=m_ds,
                             threshold=m_threshold,
                             len=m_len,
                             ss=m_ss,
                             mi=m_mi,
                             mr=m_mr,
                             ex=m_ex)
m_audio_attack.opt(m_target)
