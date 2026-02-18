import argparse
import json
import os
from pathlib import Path
import whisperx
from whisperx.diarize import DiarizationPipeline
from mistralai import Mistral
import httpx
# import gc

os.environ['PATH'] += os.pathsep + os.path.abspath('bin')
SETTINGS_PATH = Path(__file__).resolve().with_name("settings.json")

def load_settings():
  if not SETTINGS_PATH.exists():
    raise FileNotFoundError(f"Missing settings file: {SETTINGS_PATH}")
  with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    settings = json.load(f)
  return settings

def convert_audio_to_text(audio_file, hf_token):
  # $ whisperx "2025-07-17 13-40-38.mkv" --compute_type int8 --device cpu --hf_token <token> --diarize --language ru --output_format txt

  model = whisperx.load_model("large-v2", 'cpu', compute_type='int8', download_root='.model', language='ru')

  audio = whisperx.load_audio(audio_file)
  result = model.transcribe(audio, batch_size=4)
  print(result["segments"]) # before alignment

  # delete model if low on GPU resources
  # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

  model_a, metadata = whisperx.load_align_model(language_code=result["language"], device='cpu')
  result = whisperx.align(result["segments"], model_a, metadata, audio, 'cpu', return_char_alignments=False)

  print(result["segments"]) # after alignment

  # delete model if low on GPU resources
  # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

  diarize_model = DiarizationPipeline(token=hf_token, device='cpu')
  diarize_segments = diarize_model(audio)

  result = whisperx.assign_word_speakers(diarize_segments, result)
  print(diarize_segments)
  print(result["segments"]) # segments are now assigned speaker IDs

  lines = ''
  speaker = ''
  for segment in result["segments"]:
    if speaker == segment['speaker']:
      lines += segment['text'] + ' '
    else:
      lines += f"\n[{segment['speaker']}]:{segment['text']}"
      speaker = segment['speaker']

  return lines

def convert_text_to_mom(text_file, mistral_api_key):
  mistral_client = Mistral(api_key=mistral_api_key, client=httpx.Client(verify=False))
  model = "mistral-large-latest"
  system_prompt = """
Как менеджер проектов телекоммуникационной компании, ты должен преобразовать стенограмму совещания в Minutes of Meeting (MOM). 
MOM должен быть составлен в формате Markdown согласно примеру ниже:
<начало примера МОМ>
## Обсужденные вопросы и договоренности:
1. Подтвержены два основных сценария продажи для пилотного проекта - покупка в офисе продаж и покупка комплекта у промоутера
2. Сценарии покупки через веб-сайт и колл-центр будут рассмотрены по результатам пилотного проекта

## Открытые вопросы:
1. При покупке конвергента в офисе продаж является ли оплата обязательной частью процесса продажи или она может быть проведена позже?
2. При покупке конвергента в офисе продаж будет ли сотрудник продаж конфигурировать и сервис ШПД?

## Дальнейшие действия:
1. Команда заказчика (Иван Иванов)  - рассмотреть открытые вопросы для их разрешения на следующих встречах
2. Команда исполнителя (Петр Петров) - обсудить детали создания проекта в Jira для организации монтажных работ
<конец примера МОМ>
При составлении МОМ используй из примера только назввания разделов, содержание удали. Будь лаконичен, выдерживай стиль из примера. Не добавляй примечаний, не добавляй полей для метаинформации о встрече.
  """
  text = open(text_file, "r", encoding="utf-8").read()
  response = mistral_client.chat.complete(
    model=model,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": text}
    ]
  )
  return response.choices[0].message.content

def parse_arguments():
  parser = argparse.ArgumentParser(
    prog='Transcriber',
    description='creates diarized text transcription from audio or video file\nand converts text to MOM',)
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('-a', '--audio', help='path to audio or video file')
  group.add_argument('-t', '--text', help='path to text file')

  return parser.parse_args()

def main():
  args = parse_arguments()
  settings = load_settings()
  hf_token = settings["hf_token"]
  mistral_api_key = settings["mistral_api_key"]

  if args.audio:
    audio_file = args.audio
    text_file = os.path.splitext(audio_file)[0] + ".txt"
    mom_file = os.path.splitext(audio_file)[0] + "-mom.md"
    if os.path.exists(audio_file):
      text = convert_audio_to_text(audio_file, hf_token)
      with open(text_file, "w", encoding="utf-8") as f:
          f.write(text)
      text = convert_text_to_mom(text_file, mistral_api_key)
      with open(mom_file, "w", encoding="utf-8") as f:
        f.write(text)
    else:
      print(f"Cannot find input audio file {audio_file}")
      exit(1)
  elif args.text:
    text_file = args.text
    mom_file = os.path.splitext(text_file)[0] + "-mom.md"
    if os.path.exists(text_file):
      text = convert_text_to_mom(text_file, mistral_api_key)
      with open(mom_file, "w", encoding="utf-8") as f:
        f.write(text)
    else:
      print(f"Cannot find input text file {text_file}")
      exit(1)
      
if __name__ == "__main__":
  main()