# Real-Time-Audio-Frequency-Visualizer-Using-FFT


Description:To develop a real-time system that captures live audio signals and visualizes the corresponding frequency spectrum using the Fast Fourier Transform (FFT), showcasing the application of core DSP principles.



CODE:
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time

# Setup
CHUNK = 1024  # Reduced buffer size for better real-time processing
RATE = 44100
FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_SIZE = CHUNK * 2  # Optimized buffer size

def initialize_stream():
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        if not stream.is_active():
            stream.start_stream()
        return p, stream
    except Exception as e:
        print(f"Error initializing audio stream: {e}")
        if 'p' in locals():
            p.terminate()
        return None, None

try:
    # Initialize PyAudio with retry mechanism
    max_retries = 3
    retry_count = 0
    p, stream = None, None
    
    while retry_count < max_retries and (p is None or stream is None):
        p, stream = initialize_stream()
        if p is None or stream is None:
            retry_count += 1
            print(f"Retrying stream initialization ({retry_count}/{max_retries})...")
            time.sleep(1)
    
    if p is None or stream is None:
        raise Exception("Failed to initialize audio stream after multiple attempts")

    plt.ion()
    fig, ax = plt.subplots()
    x = np.arange(0, CHUNK)
    line, = ax.plot(x, np.random.rand(CHUNK))
    ax.set_ylim(-5000, 5000)

    while True:
        try:
            if stream is None or not stream.is_active():
                print("Reinitializing audio stream...")
                if 'p' in locals() and p is not None:
                    p.terminate()
                p, stream = initialize_stream()
                if p is None or stream is None:
                    print("Failed to reinitialize stream, retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            fft_data = np.abs(np.fft.fft(data))[:CHUNK // 2]
            line.set_ydata(data)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)  # Slightly increased delay for stability
        except OSError as e:
            print(f"OSError: {e}")
            stream = None  # Force reinitialization
            continue
        except Exception as e:
            print(f"Error reading audio stream: {e}")
            continue

except KeyboardInterrupt:
    print("\nStopping the audio stream...")

finally:
    # Cleanup
    if 'stream' in locals():
        if stream.is_active():
            stream.stop_stream()
        stream.close()
    if 'p' in locals():
        p.terminate()
    plt.close()
