import sounddevice as sd
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Tải mô hình và bộ xử lý
model_name = "whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name, language="en")
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Thiết lập các tham số cho thu âm
RATE = 16000
BUFFER_DURATION = 10 # Thời gian của mỗi đoạn buffer thu âm (giây)
BUFFER_SIZE = int(RATE * BUFFER_DURATION)  # Kích thước của buffer


# Biến lưu buffer âm thanh
buffer = np.zeros(BUFFER_SIZE, dtype='float32')
current_pos = 0

# Biến lưu trữ chuỗi mới
new_transcript = ""
listening_for_command = False

# Hàm để chuyển đổi âm thanh thành văn bản
def transcribe_audio(model, processor, waveform):
    inputs = processor(waveform, sampling_rate=RATE, return_tensors="pt").input_features
    predicted_ids = model.generate(inputs)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Hàm callback để thu âm
def audio_callback(indata, frames, time, status):
    global buffer, current_pos, new_transcript, listening_for_command

    if status:
        print(status)

    # Thêm dữ liệu mới vào buffer
    if current_pos + frames < BUFFER_SIZE:
        buffer[current_pos:current_pos + frames] = indata[:, 0]
        current_pos += frames
    else:
        # Khi buffer đầy, xử lý buffer và đặt lại vị trí hiện tại
        buffer[current_pos:] = indata[:BUFFER_SIZE - current_pos, 0]
        waveform = buffer.copy()
        current_pos = frames - (BUFFER_SIZE - current_pos)
        buffer[:current_pos] = indata[BUFFER_SIZE - current_pos:, 0]

        # Xử lý và dịch buffer thành văn bản
        transcription = transcribe_audio(model, processor, waveform)
        print("Kết quả:", transcription)

        if "hey zara" in transcription.lower():
            listening_for_command = True
            print("Bắt đầu ghi lại các từ sau 'Hey Zara'")

        if listening_for_command:
            if "stop zara" in transcription.lower():
                print("Ngừng ghi lại khi nghe thấy 'Stop Zara'")
                listening_for_command = False
            else:
                new_transcript += " " + transcription
                print("new_transcript:", new_transcript)

# Khởi động stream thu âm
with sd.InputStream(callback=audio_callback, channels=1, samplerate=RATE, blocksize=BUFFER_SIZE):
    print("Đang thu âm và dịch trực tiếp... Nhấn Ctrl+C để dừng.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Dừng thu âm.")
        print("Final new_transcript:", new_transcript)
 