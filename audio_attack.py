import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter

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

    def __init__(self, audio, ds, threshold, len, pn, ss, mi, mr, ex, ns):
        self.dis_group = [[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                          [2, 1, 4, 4, 4, 4, 4, 2, 4, 4, 2, 3, 4],
                          [2, 4, 1, 3, 3, 4, 3, 4, 4, 4, 4, 4, 4],
                          [2, 4, 3, 1, 2, 4, 3, 4, 4, 4, 4, 4, 4],
                          [2, 4, 3, 2, 1, 4, 3, 4, 3, 4, 4, 3, 4],
                          [2, 4, 4, 4, 4, 1, 2, 4, 3, 2, 3, 3, 4],
                          [2, 4, 3, 3, 3, 2, 1, 3, 4, 4, 4, 2, 4],
                          [2, 2, 4, 4, 4, 4, 3, 1, 2, 4, 4, 4, 3],
                          [2, 4, 4, 4, 3, 3, 4, 2, 1, 4, 4, 2, 4],
                          [2, 4, 4, 4, 4, 2, 4, 4, 4, 1, 4, 3, 4],
                          [2, 2, 4, 4, 4, 3, 4, 4, 4, 4, 1, 4, 4],
                          [2, 3, 4, 4, 3, 3, 2, 4, 2, 3, 4, 1, 4],
                          [2, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 1]]
        self.audio = audio  # 原始音频文件的numpy数组
        self.audio_length = len  # 音频长度(某种意义上即为维度)
        self.base_threshold = threshold
        self.threshold = threshold  # 扰动范围阈值
        self.ds = ds  # deep speech模型
        self.positive_num = pn  # 最优集大小
        self.sample_size = ss  # 备选集大小
        self.max_iteration = mi  # 最大扰动次数
        self.move_rate = mr  # 每次更新搜索空间的更新概率
        self.max_expand = ex  # 增大阈值范围的迭代次数
        self.mutation_rate = mr
        self.noise_stdev = ns

        self.ini_label = client.run_deep_speech(self.ds, self.audio)  # 原始音频的识别结果label
        self.ini_fitness = -1
        self.max_fitness = -1
        self.tar_label = None  # 攻击的target label

        self.string_list = []

        self.pop = []  # 最优集
        self.pos_pop = []  # 备选集
        self.optional = None  # 最优扰动

        self.group = {' ': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 1, 'f': 5, 'g': 3, 'h': 6, 'i': 7, 'j': 8, 'k': 3,
                      'l': 4, 'm': 9, 'n': 9, 'o': 10, 'p': 2, 'q': 3, 'r': 11, 's': 12, 't': 4, 'u': 10, 'v': 5,
                      'w': 10, 'x': 3, 'y': 8, 'z': 12, '\'': 0, '-': 0}

    def clear(self):
        self.pop = []
        self.optional = None
        return

    def dis_char(self, ch1, ch2):
        if ch1 == ch2:
            return 0
        return self.dis_group[self.group[ch1]][self.group[ch2]]

    def dis_string(self, str1, str2):
        m = len(str1)
        n = len(str2)
        if m * n == 0:
            return 2 * (m + n)
        str1 = "0" + str1
        str2 = "0" + str2
        dis = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        dis[1][1] = self.dis_char(str1[1], str2[1])
        for j in range(2, n + 1):
            dis[1][j] = min(dis[1][j - 1] + 2, 2 * (j - 1) + self.dis_char(str1[1], str2[j]))

        for i in range(2, m + 1):
            dis[i][1] = min(dis[i - 1][1] + 2, 2 * (i - 1) + self.dis_char(str1[i], str2[1]))
            for j in range(2, n + 1):
                dis[i][j] = dis[i][j - 1] + 2
                for left in range(1, i):
                    k_left = i - left
                    min_part = 4
                    for k_right in range(k_left + 1, i + 1):
                        min_part = min(min_part, self.dis_char(str1[k_right], str2[j]))
                    dis[i][j] = min(dis[i][j], min_part + dis[k_left][j - 1] + 2 * (i - k_left - 1))
                dis[i][j] = min(dis[i][j], self.dis_char(str1[1], str2[j]) + 2 * (i + j - 1))
        return dis[m][n]

    def random_instance(self):
        """
        生成全随机扰动
        :param dim:
        :param region:
        :return:
        """
        # completely random
        inst = Instance()
        features = np.random.randint(-2 ** 15, 2 ** 15 - 1, self.audio_length) * self.noise_stdev
        features = self.highpass_filter(features)
        mask = np.random.rand(self.audio_length) < self.mutation_rate
        inst.setFeatures(features * mask + self.audio)
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
        初始化
        :return:
        """
        print("audio length: " + str(self.audio_length))
        print("original string: " + self.ini_label)
        print("Initializing...")
        self.clear()

        self.pop = []
        self.pos_pop = []
        for i in range(self.positive_num):
            ins = self.random_instance()
            self.pop.append(ins)

        self.run_once()
        self.pop.sort(key=lambda instance: instance.getFitness())
        self.pos_pop = self.pop[:self.positive_num]
        print("Finish creating " + str(self.positive_num) + " samples.")

        self.optional = self.pos_pop[0].CopyInstance()
        return

    def highpass_filter(self, data, cutoff=7000, fs=16000, order=10):
        b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
        return lfilter(b, a, data)

    def run_once(self):
        """
        将popset中的所有扰动加上音频本身跑一次deep speech
        :param popset: 扰动数组，长度不定
        :param rtype:
        :return: 设定popset的fitness值
        """
        batch_size = len(self.pop)
        # 预测
        for i in range(batch_size):
            audio_interrupted = np.array(np.clip(self.pop[i].getFeatures(), -2 ** 15, 2 ** 15),
                                         dtype='int16')
            pred_str = client.run_deep_speech(self.ds, audio_interrupted)
            # 使用pred_str计算不满足度，回填入pop
            fitness = self.compute_dis(pred_str)
            print("result " + str(i) + "\t: " + pred_str + ", fitness: " + str(fitness) + ", max interrupt: " + str(
                max(abs(audio_interrupted - self.audio))))
            self.pop[i].setString(pred_str)
            self.pop[i].setFitness(fitness)
        return

    def compute_dis(self, pred):
        """
        根据识别的str计算不满足度
        :param pred:
        :return:
        """
        # return levenshtein(pred, self.tar_label)
        return self.dis_string(pred, self.tar_label)

    def roulette(self):
        self.pop.sort(key=lambda instance: instance.getFitness())
        if self.pop[0].getFitness() > (self.max_fitness * 2 / 3):
            print("加速收缩")
            self.pos_pop = self.pop[:self.positive_num]
            return

        roulette_rank = []
        roulette_sum = 0
        length = len(self.pop)
        for i in range(length):
            if self.pop[i].getFitness() == 0:
                self.pos_pop = self.pop[i]
                return
            else:
                rank = int(10000 / self.pop[i].getFitness())
            roulette_sum += rank
            roulette_rank.append(roulette_sum)
        self.pos_pop = []
        i = 0
        while i < self.positive_num:
            rand = np.random.randint(roulette_sum)
            choose_rank = -1
            for j in range(0, length):
                if rand < roulette_rank[j]:
                    choose_rank = j
                    break

            if self.pop[choose_rank] not in self.pos_pop:
                self.pos_pop.append(self.pop[choose_rank])
                i += 1

        self.pos_pop.sort(key=lambda instance: instance.getFitness())
        return

    def crossover(self):
        sample1 = self.pos_pop[np.random.randint(self.positive_num)].getFeatures()
        sample2 = self.pos_pop[np.random.randint(self.positive_num)].getFeatures()
        mask = np.random.rand(self.audio_length) < 0.5
        return sample1 * mask + sample2 * (1 - mask)

    def mutation(self, sample):
        noise = np.random.randint(-2 ** 15, 2 ** 15 - 1, self.audio_length) * self.noise_stdev
        noise = self.highpass_filter(noise)
        mask = np.random.rand(self.audio_length) < self.mutation_rate
        return sample + noise * mask

    def save_result(self, number):
        f = open("result_" + str(number) + ".txt", 'w', encoding='utf-8')
        for i in range(len(self.string_list)):
            f.write("result\t" + str(i) + "\t:\t" + self.string_list[i] + "\n")
        f.close()

    def opt(self, target=None):
        self.tar_label = target
        self.max_fitness = self.dis_string(target, self.ini_label)

        # 初始化
        self.init()

        self.string_list = []

        # run
        time = 0
        while time < self.max_iteration:
            print("iter time: " + str(time))

            for i in range(self.positive_num):
                print("best " + str(i) + ": " + self.pos_pop[i].getString() + ", fitness: " + str(
                    self.pos_pop[i].getFitness()))

            self.string_list.append(self.optional.getString())
            if time % 100 == 0:
                self.save_result(time)

            # 如果fitness满足要求则退出
            if self.optional.getFitness() == 0:
                print("Find attack in " + str(time) + " times. File saved in result.txt.")
                # 保存文件
                self.save_result(time)
                return

            self.pop = []
            for i in range(self.positive_num * self.sample_size):
                sample = self.crossover()
                ins = Instance()
                ins.setLen(self.audio_length)
                ins.setFeatures(self.mutation(sample))
                self.pop.append(ins)

            self.run_once()
            self.pop.extend(self.pos_pop)
            self.roulette()
            self.optional = self.pos_pop[0].CopyInstance()

            time += 1

        self.save_result(time)
        return


m_audio_path = "DeepSpeech/audio/4507-16021-0012.wav"
m_model_path = "DeepSpeech/model/output_graph.pbmm"
m_lm_path = "DeepSpeech/model/lm.binary"
m_trie_path = "DeepSpeech/model/trie"
m_ds = client.load_model(m_model_path)
m_ds = client.update_ds(m_ds, m_lm_path, m_trie_path)
m_audio = client.get_audio_array(m_ds, m_audio_path)
m_target = 'restart'

m_threshold = 256
m_len = m_audio.shape[0]
m_pn = 10
m_ss = 5
m_mi = 3000
m_mr = 0.1
m_ex = 8
m_ns = 0.01
m_audio_attack = AudioAttack(audio=m_audio,
                             ds=m_ds,
                             threshold=m_threshold,
                             len=m_len,
                             pn=m_pn,
                             ss=m_ss,
                             mi=m_mi,
                             mr=m_mr,
                             ex=m_ex,
                             ns=m_ns)
m_audio_attack.opt(m_target)
