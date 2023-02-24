# Voice GPT-3 robot (Google Cloud STT, Google Cloud TTS, OpenAI API)
# Author: ZorTik
import os
import time

import numpy
import openai

import numpy as np
import sounddevice as sd

from google.cloud import speech, texttospeech

threshold = 0.3
frequency = 16000
duration = 8

openai_max_tokens = 100
openai_temperature = 0.8

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/key.json"
os.environ["OPENAI_API_KEY"] = "your-key-here"


def await_continue():
    input("Press something to start")


def record(dur):
    inp = None

    frames = []

    def callback(indata, outdata, frames, time, status):
        volume_norm = np.linalg.norm(indata) * 10
        frames
        if volume_norm < 0.1:
            pass

    inp = sd.InputStream(
        channels=1,
        samplerate=frequency,
        dtype='int16')

    inp.start()
    data, overflowed = inp.read(int(dur * frequency))

    return np.array(data)


class Recorder:
    rec_now = False
    rec = None

    def __init__(self, continue_by_press=True):
        self.speech_client = speech.SpeechClient()
        self.tts_client = texttospeech.TextToSpeechClient()
        self.continue_by_press = continue_by_press
        pass

    def loop(self):
        await_continue()
        while True:
            if self.rec_now:
                print("Listening...")
                self.rec = record(duration)
                self.handle()
                self.rec_now = False

                if not self.continue_by_press:
                    print("Resting...")
                    time.sleep(5)
                else:
                    await_continue()
            else:
                audio = record(1)
                if self.validateNotEmpty(audio):
                    self.rec_now = True

    def handle(self):
        if not self.validateNotEmpty(self.rec):
            print("Playback is empty!")
            return

        print(f"Bytes: {len(self.rec.tobytes())}")

        res = self.recognize(self.rec)

        if len(res.results) == 0:
            print("No recognition.")
        else:
            transcript = res.results[0].alternatives[0].transcript

            if len(transcript) > 0:
                completion = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=transcript,
                    max_tokens=openai_max_tokens,
                    temperature=openai_temperature
                )

                bot_response = completion.choices[0].text
                synt_input = texttospeech.SynthesisInput(text=bot_response)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="cs-CZ",
                    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
                )
                synt_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16
                )
                synt_res = self.tts_client.synthesize_speech(input=synt_input, voice=voice, audio_config=synt_config)
                synt_res_arr = numpy.frombuffer(synt_res.audio_content, dtype=numpy.int16)

                print("Playing...")

                sd.play(synt_res_arr, frequency, blocking=True)
        pass

    def recognize(self, audio):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="cs-CZ"
        )

        au_bytes = audio.tobytes()
        au_to_send = speech.RecognitionAudio(
            content=au_bytes
        )

        response = self.speech_client.recognize(config=config, audio=au_to_send)

        return response

    @staticmethod
    def validateNotEmpty(audio):
        return np.max(np.abs(audio)) > threshold


if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]

    Recorder().loop()
