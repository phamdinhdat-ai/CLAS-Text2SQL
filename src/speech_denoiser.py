import sounddevice as sd 
import torch 
import  sys 
import argparse





def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device
    
    

def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
    except ValueError:
        message = (f"Invalid {kind} audio interface {device}.\n . It's seem like your audio device is not available.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps




if __name__ == '__main__':
    print("Checking audio device .................")
    
    sample_rate = 16000
    
    device_in = parse_audio_device(0)
    caps = query_devices(device=device_in, kind='input')
    channels_in = min(caps['max_input_channels'], 2)
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=sample_rate,
        channels=channels_in)

    device_out = parse_audio_device('out')
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=sample_rate,
        channels=channels_out)

    stream_in.start()
    stream_out.start()
    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    sr_ms = sample_rate / 1000
    stride_ms = 1 / sr_ms
    print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")
    while True:
        try:
            if current_time > last_log_time + log_delta:
                last_log_time = current_time
                tpf = stre * 1000
                rtf = tpf / stride_ms
                print(f"time per frame: {tpf:.1f}ms, ", end='')
                print(f"RTF: {rtf:.1f}")
                
                
                
        except: 
            pass