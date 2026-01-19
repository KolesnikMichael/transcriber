import logging
import argparse
import os
import speech_recognition
from pydub import AudioSegment

def convert_audio_to_text(audio_file_path, audio_format):
  wav_audio_file_path = audio_file_path.with_suffix(".wav")        

  audio = AudioSegment.from_file(audio_file_path, format=audio_format)
  audio.export(wav_audio_file_path, format="wav")
  
  recognizer = speech_recognition.Recognizer()
  text = ""

  try:
    with speech_recognition.AudioFile(audio_file_path) as source:
      audio_data = recognizer.record(source)
      text = recognizer.recognize_google(audio_data)
  except speech_recognition.UnknownValueError:
    print("Could not recognize speech in audio")
  except speech_recognition.RequestError as e:
    print("Speech recognition service unavailable")
  finally:
    os.remove(audio_file_path)

  return text


def save_text_to_file(text, audio_file_path):
  text_file_path = os.path.splitext(audio_file_path)[0] + ".txt"
  with open(text_file_path, "w") as text_file:
      text_file.write(text)
      print(f"Text file saved to: {text_file_path}")


def parse_arguments():
  parser = argparse.ArgumentParser(
    prog='transcriber',
    description='converts audio to text')
  parser.add_argument('-a', '--audio', action='store_true', required=True, help='path to audio file')
  parser.add_argument('-f', '--format', action='store_true', required=True, help='format of audio file, e.g. ogg, mp3, wav, flac, au')

  return parser.parse_args()


def main():
  args = parse_arguments()
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.basicConfig(format="%(asctime)s (%(levelname)s): %(name)s - %(message)s", level=logging.INFO)
  logger = logging.getLogger(__name__)

  if args.audio and args.format:
    text = convert_audio_to_text(args.audio, args.format)
    save_text_to_file(text, args.audio)
  else:
    print("Speech recognition service unavailable")
    exit(1)


if __name__ == "__main__":
  main()