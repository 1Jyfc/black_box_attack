import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import client

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


class RunAttack:
    """
    算法类
    """

    def __init__(self, input_wav_file, output_wav_file, target_phrase, s, k, t, epsilon):
        self.s = s  # sample size
        self.k = k  # ranking threshold
        self.t = t  # max iteration time
        self.epsilon = epsilon  # 搜索空间范围
        self.input_audio = load_wav(input_wav_file).astype(np.float32)
        self.u = int(self.input_audio.shape[0] * 0.01)
        self.output_wav_file = output_wav_file
        self.target_phrase = target_phrase

    def run(self):
        """
        算法运行
        :return: 符合要求的扰动
        """
        ''' 1. 初始化搜索空间 '''
        ''' 2. 随机在搜索空间中生成s+k个扰动，作为集合B '''
        ''' 3. 分别将这些扰动加到原音频，计算不满足度 '''
        ''' 4. 选取不满足度最小的扰动x '''
        ''' 5. 开始迭代 '''
        for iter_time in range(self.t):
            ''' 1. 若x已经满足攻击要求，则退出 '''
            ''' 2. 将B中不满足度最低的k个扰动作为B+，剩余作为B- '''
            ''' 3. 开始缩小搜索空间 '''
            for search_time in range(self.s):
                ''' 1. 在B+中任取一个扰动b+ '''
                ''' 2. 找出u个点逐一操作 '''
                for unit_time in range(self.u):
                    ''' 1. 在b+中随意找到一个点p '''
                    ''' 2. 在B-中的各个扰动找到相同p位置的扰动值，计算比b+中p位置扰动值大/小的个数 '''
                    ''' 3. 若比b+中p位置大的扰动多，则缩小p点搜索空间上界，反之缩小下界 '''
                ''' 3. 根据b+和新的搜索空间，重新生成一个扰动，加入B '''
                ''' 4. 重置搜索空间 '''
            ''' 4. 在B中选取不满足度最小的s+k个扰动作为新的B，选取不满足度最小的扰动作为新的x，开始下一轮迭代 '''
        ''' 6. 经过最大迭代轮数尚未攻击成功 '''


if __name__ == '__main__':
    cache_folder = os.path.exists(cache_path)
    if not cache_folder:
        os.makedirs(cache_path)
    result_folder = os.path.exists(result_path)
    if not cache_folder:
        os.makedirs(result_path)

    parser = argparse.ArgumentParser(description='Running Attack.')
    parser.add_argument('--audio', required=True,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--target', required=True,
                        help='Target phrase for running attack')
    args = parser.parse_args()
    attack = RunAttack(args.audio, output_file, args.target, s=3, k=2, t=100, epsilon=100)
    attack.run()
