# APIを登録しないと動かない
import speech_recognition

r = speech_recognition.Recognizer()
with speech_recognition.AudioFile('wavs/Take01_TrimTrim.wav') as src:
    audio = r.record(src)
# print(r.recognize_google(audio, key='AIzaSyAD0YmU63J9jveJQcJWTlMH9L0dDSSaRM0', language='ja-JP'))
print(r.recognize_google(audio, language='ja-JP'))
