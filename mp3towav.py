from pydub import AudioSegment


def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath + ".mp3")
    song.export(filepath + ".wav", format="wav")

trans_mp3_to_wav("../audio/sunrise")