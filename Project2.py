import sys
# Add paths to site-packages for required libraries
sys.path.append('/opt/anaconda3/lib/python3.12/site-packages')
import os
import wave
import pyaudio
import threading
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Global variables
is_recording = False

def start_recording():
    global is_recording
    is_recording = True
    print("Recording started...")

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped...")

def record_audio(filename="recording.wav"):
    """
    Record audio from the microphone and save it to a WAV file.
    """
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Mono audio
    rate = 16000  # Sampling rate

    p = pyaudio.PyAudio()
    print("Press Enter to start recording.")
    input()  # Wait for user to press Enter
    start_recording()

    # Open stream for recording
    stream = p.open(format=sample_format, channels=channels, rate=rate,
                    input=True, frames_per_buffer=chunk)

    frames = []

    def record():
        while is_recording:
            data = stream.read(chunk)
            frames.append(data)

    recording_thread = threading.Thread(target=record)
    recording_thread.start()

    # Wait for user to stop recording
    input("Press Enter to stop recording.")
    stop_recording()
    recording_thread.join()

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data to a WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))

    print(f"Audio saved as {filename}")
    return filename

def transcribe_and_differentiate(filename):
    """
    Perform speaker diarization based on silence detection and transcription.
    Keep track of the same speaker number for consecutive segments.
    """
    recognizer = sr.Recognizer()

    # Load the audio file using pydub for segmentation based on silence
    audio = AudioSegment.from_wav(filename)

    # Split the audio into chunks based on silence
    chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

    results = []
    speaker_count = 1
    previous_chunk_end = 0  # Keep track of the end of the last chunk (in milliseconds)
    speaker_segments = []

    for chunk in chunks:
        # Export each chunk to a temporary file
        chunk_filename = f"temp_chunk{speaker_count}.wav"
        chunk.export(chunk_filename, format="wav")

        # Recognize speech in each chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)
            try:
                # Recognize the text
                text = recognizer.recognize_google(audio_data)
                
                # If the current chunk is very close to the last, use the same speaker number
                current_chunk_duration = len(chunk)  # Duration of the chunk in milliseconds
                if previous_chunk_end != 0 and (chunk[0].duration_seconds * 1000 - previous_chunk_end) < 1000:
                    # If the time gap between this chunk and the last one is short, it's the same speaker
                    results.append(f"Speaker {speaker_count}: {text}")
                else:
                    # Otherwise, assign a new speaker
                    speaker_count += 1
                    results.append(f"Speaker {speaker_count}: {text}")

                # Update previous_chunk_end with the current chunk's end time (in milliseconds)
                previous_chunk_end = chunk[0].duration_seconds * 1000 + current_chunk_duration  # Add chunk length
            except sr.UnknownValueError:
                results.append(f"Speaker {speaker_count}: [Unintelligible]")
            except sr.RequestError as e:
                results.append(f"Speaker {speaker_count}: [Error: {e}]")
        
        # Clean up the temporary file
        os.remove(chunk_filename)

    return results

if __name__ == "__main__":
    # Step 1: Record audio
    audio_file = record_audio()

    # Step 2: Process and transcribe
    print("Processing audio for speaker diarization and transcription...")
    transcription = transcribe_and_differentiate(audio_file)

    # Step 3: Display results
    print("\nTranscription:")
    for line in transcription:
        print(line)

    # Cleanup
    os.remove(audio_file)
