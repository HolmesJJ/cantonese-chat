import json
import wave
import base64
import pyaudio
import keyboard
import requests

from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
from paddlespeech.cli.tts.infer import TTSExecutor

TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
TRANSLATION_URL = "https://aip.baidubce.com/rpc/2.0/mt/v2/speech-translation"
GRANT_TYPE = "client_credentials"
API_KEY = ""
SECRET_KEY = ""

FORMAT1 = pyaudio.paInt16
FORMAT2 = "wav"
CHANNELS = 1
RATE = 16000
CHUNK = 1024

CANTONESE1 = "yue"
MANDARIN = "zh"
CANTONESE2 = "canton"

PADDLE_MODEL = "fastspeech2_canton"
OPENAI_MODEL = "gpt-4"
OPENAI_KEY = ""

USER_AUDIO_PATH = "user.wav"
AI_AUDIO_PATH = "ai.wav"


def get_access_token():
    params = {
        "grant_type": GRANT_TYPE,
        "client_id": API_KEY,
        "client_secret": SECRET_KEY,
    }
    response = requests.post(TOKEN_URL, params=params)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception("获取Access Token失败")


def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT1, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("录音中... 按下 'Enter' 停止录音")
    frames = []
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        if keyboard.is_pressed("enter"):
            break
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf = wave.open(USER_AUDIO_PATH, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT1))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def speech_recognition(access_token):
    with open(USER_AUDIO_PATH, "rb") as f:
        speech_data = f.read()
    speech_base64 = base64.b64encode(speech_data).decode("utf-8")
    headers = {"Content-Type": "application/json"}
    params = {
        "format": FORMAT2,
        "from": CANTONESE1,
        "to": MANDARIN,
        "voice": speech_base64
    }
    response = requests.post(f"{TRANSLATION_URL}?access_token={access_token}",
                             headers=headers,
                             data=json.dumps(params))
    data = response.json()
    if data.get("result"):
        return data["result"]
    else:
        raise Exception(f"识别失败，错误信息：{data['err_msg']}")


def get_response(prompt_content):
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "你是一位说粤语的助手。",
            },
            {
                "role": "user",
                "content": prompt_content,
            }
        ],
        temperature=0.7
    )
    return response.choices[0].message.content


def run():
    tts_executor = TTSExecutor()
    while True:
        print("用户：")
        record_audio()
        question = speech_recognition(get_access_token())
        print(f"    粤语：{question['source']}")
        print(f"    普通话：{question['target']}")
        answer = get_response(question['target'])
        answer = answer.rstrip("。") if answer.endswith("。") else answer
        print("AI：")
        print(f"    粤语：{answer}")
        tts_executor(text=answer, am=PADDLE_MODEL, lang=CANTONESE2, output=AI_AUDIO_PATH)
        audio = AudioSegment.from_file(AI_AUDIO_PATH)
        play(audio)


if __name__ == "__main__":
    run()
