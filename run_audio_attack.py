import numpy as np
import scipy.io.wavfile as wav
import os
import argparse
import random
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


def delete_cache(perturb_set):
    """
    删除缓存
    :param perturb_set: 缓存列表
    :return:
    """
    for perturbation in enumerate(perturb_set):
        path = perturbation["path"]
        if os.path.exists(path):
            os.remove(path)


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
        self.audio_len = self.input_audio.shape[0]
        self.u = int(self.audio_len * 0.01)
        self.output_wav_file = output_wav_file
        self.target_phrase = target_phrase

    def run(self):
        """
        算法运行
        :return: 符合要求的扰动
        """
        cache_id = 0
        ds = client.load_model(model_path)
        ''' 1. 初始化搜索空间 '''
        delta = np.array([[-self.epsilon, self.epsilon] * self.audio_len])
        ''' 2. 随机在搜索空间中生成s+k个扰动，作为集合B '''
        ''' 3. 分别将这些扰动加到原音频，计算不满足度 '''
        perturb_set = []
        for i in range(self.s + self.k):
            cache_id += 1
            perturb_set.append(create_random_sample(delta, self.input_audio, ds, self.target_phrase, cache_id))
        ''' 4. 选取不满足度最小的扰动x '''
        perturb_set.sort(key=lambda x: x["dd"], reverse=False)
        perturb_x = perturb_set[0]
        ''' 5. 开始迭代 '''
        for iter_time in range(self.t):
            ''' 1. 若x已经满足攻击要求，则退出 '''
            if perturb_x["dd"] == 0:
                result_audio = load_wav(perturb_x["path"])
                save_wav(result_audio, result_path)
                print("Iterating", iter_time, "times, success to attack. Result saved in", result_path)
                delete_cache(perturb_set)
                return
            ''' 2. 将B中不满足度最低的k个扰动作为B+，剩余作为B- '''
            ''' 3. 开始缩小搜索空间 '''
            for search_time in range(self.s):
                ''' 1. 在B+中任取一个扰动b+ '''
                perturb_b_plus = perturb_set[random.randint(0, self.k - 1)]["perturb"]
                ''' 2. 找出u个点逐一操作 '''
                u_list = random.sample(range(0, self.audio_len), self.u).sort()
                for p in u_list:
                    ''' 1. 在b+中随意找到一个点p '''
                    ''' 2. 在B-中的各个扰动找到相同p位置的扰动值，计算比b+中p位置扰动值大/小的个数 '''
                    perturb_unit_p = perturb_b_plus[p]
                    ge = []
                    le = []
                    for i in range(self.k, self.s + self.k - 1):
                        perturb_b_minus = perturb_set[i]["perturb"]
                        p_minus = perturb_b_minus[p]
                        if p_minus > perturb_unit_p:
                            ge.append(p_minus)
                        else:
                            le.append(p_minus)
                    ''' 3. 若比b+中p位置大的扰动多，则缩小p点搜索空间上界，反之缩小下界 '''
                    if len(ge) > len(le):
                        ge.sort(reverse=False)
                        delta[p][1] = random.randint(perturb_unit_p, ge[0])
                    else:
                        le.sort(reverse=True)
                        delta[p][0] = random.randint(le[0], perturb_unit_p)
                ''' 3. 根据b+和新的搜索空间，重新生成一个扰动，加入B '''
                new_perturb = perturb_b_plus[:]
                for p in u_list:
                    new_perturb[p] = random.randint(delta[p][0], delta[p][1])
                cache_id += 1
                perturb_set.append(create_sample(new_perturb, self.input_audio, ds, self.target_phrase, cache_id))
                ''' 4. 重置搜索空间 '''
                delta = np.array([[-self.epsilon, self.epsilon] * self.audio_len])
            ''' 4. 在B中选取不满足度最小的s+k个扰动作为新的B，选取不满足度最小的扰动作为新的x，开始下一轮迭代 '''
            perturb_set.sort(key=lambda x: x["dd"], reverse=False)
            perturb_set = perturb_set[:self.s + self.k]
            perturb_x = perturb_set[0]
        ''' 6. 经过最大迭代轮数尚未攻击成功 '''
        print("Iterating", self.t, "times, we don't get a satisfying attack.")
        result_audio = load_wav(perturb_x["path"])
        print("The most nearly attack sample is saved in " + result_path + ", whose result is " + perturb_x["text"])
        save_wav(result_audio, result_path)
        delete_cache(perturb_set)


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
