
import os
import wave
import datetime
import pandas as pd
import numpy as np
from pyannote.audio import Pipeline
#from pyannote import  audio
#from audio import pipeline
from pydub import AudioSegment
import speech_recognition as sr
import pyaudio
import subprocess
import datetime



token = os.getenv("Token_VALUE")
if token is None:
    raise ValueError("YOUR_AUTH_TOKEN_ENV_VARIABLE environment variable is not set.")

# Get the current date and time in YYYY-MM-DD_HH-MM-SS format
datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set the file name to the current date and time with a .wav extension
filename = f"{datetime_str}.wav"

# Define the arecord command and options
# arecord_cmd = ["arecord", "-D", "hw:CARD=I82801AAICH", "-f", "cd", "-d", "10", filename]
# Check if the 'test.wav' file exists in the current directory
if os.path.exists('test.wav'):
    filename = 'test.wav'  # Use 'test.wav' if it exists
# Run the arecord command using subprocess
#subprocess.run(arecord_cmd)

# Create a directory with the same name as the audio file
dirname = os.path.splitext(filename)[0]
if not os.path.exists(dirname):
    os.mkdir(dirname)

# Save the filename of the most recent recording to a file
with open("most_recent_recording.txt", "w") as f:
    f.write(filename)

with open("directory_name.txt", "w") as f:
    f.write(dirname)
# Perform speaker diarization on the audio file
p#ipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                               #     use_auth_token="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")

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                    use_auth_token=token)

diarization = pipeline(filename)

# Write the speaker diarization output to an RTTM file
rttm_filename = os.path.join(dirname, "diarization.rttm")
with open(rttm_filename, "w") as rttm:
    diarization.write_rttm(rttm)

# Write the speaker diarization output to an RTTM file
rttm_filename = os.path.join(dirname, "diarization.rttm")
print(f"RTTM file path: {rttm_filename}")
with open(rttm_filename, "w") as rttm:
    diarization.write_rttm(rttm)
print(f"RTTM file exists: {os.path.exists(rttm_filename)}")


# Read the RTTM file output into a pandas DataFrame
df = pd.read_csv(rttm_filename, sep=" ", header=None, usecols=[3,4,7], names="tbeg tdur stype".split())
 # Wait for the speaker diarization pipeline to complete

# Format the start and end times as timedelta objects
def td_time_format(td):
    parts = td.components
    return f"{parts.minutes}:{parts.seconds:02}.{parts.milliseconds:03}"
df["tbeg_fmt"] = pd.to_timedelta(df.tbeg, unit="s").apply(td_time_format)
df["tend_fmt"] = pd.to_timedelta(df.tbeg + df.tdur, unit="s").apply(td_time_format)

# Get consecutive speaker runs
speaker_runs = {
    speaker: [
        np.array(grp)[[0,-1]].tolist()
        for grp in np.split(group, np.where(np.diff(group) != 1)[0]+1)]
    for speaker, group in df.groupby("stype").agg("tbeg_fmt").groups.items()
}

# 'Roll up' the timestamps over consecutive runs by inverting the dict
speaker_order = sorted(
    [{speaker: run} for speaker, runs in speaker_runs.items() for run in runs],
    key=lambda d: [*d.values()]
)
rollup_records = [
    {
        "tbeg": df.tbeg[start_idx],
        "tdur": df.tbeg[stop_idx] + df.tdur[stop_idx] - df.tbeg[start_idx],
        "stype": df.stype[start_idx],
        "tbeg_fmt": df.tbeg_fmt[start_idx],
        "tend_fmt": df.tend_fmt[stop_idx],
    }
    for order in speaker_order
    for speaker, (start_idx, stop_idx) in order.items()
]
rollup_df = df.from_records(rollup_records)

# Replace the speaker labels with more readable names
rollup_df["stype"] = rollup_df.stype.replace("SPEAKER_00", "Name0").replace("SPEAKER_01", "Name1").replace("SPEAKER_02", "Name2").replace("SPEAKER_03", "Name3")

# Print the final speaker turns
print(rollup_df[["tbeg_fmt", "tend_fmt","stype"]])

# Load the original audio file
audio = AudioSegment.from_file(filename, format="wav")

# Iterate over each row in the speaker turns DataFrame
for i, row in rollup_df.iterrows():
    # Extract the start and end times of the current speaker turn
    start_time = row["tbeg"] * 1000  # convert to milliseconds
    end_time = (row["tbeg"] + row["tdur"]) * 1000  # convert to milliseconds

    # Extract the audio segment corresponding to the current speaker turn
    segment = audio[start_time:end_time]

    # Save the audio segment to a file in the directory with the same name as the audio file
    segment_filename = os.path.join(dirname, f"speaker_{i}.wav")
    segment.export(segment_filename, format="wav")
    
print(f"Speaker diarization output saved in folder: {dirname}")


# Save the filename of the most recent recording to a file
with open("latest_recording_folder.txt", "w") as f:
    f.write(dirname)  # Save the path of the most recent folder to the "most_recent_recording.txt" file




#Audio recognition


# Read the path of the most recently created folder from a file
with open("latest_recording_folder.txt", "r") as f:
    directory = f.read().strip()

# Specific word to search for in the audio files
specific_word = 'help'

# Initialize the speech recognizer
r = sr.Recognizer()

# Initialize a list to store the filenames
help_files = []

# Loop through the audio files in the directory
for filename in os.listdir(directory):
    # Check if the file is an audio file
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        # Load the audio file with pydub
        audio_file = os.path.join(directory, filename)
        audio = AudioSegment.from_file(audio_file)

        # Convert the audio to PCM WAV format
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_file, format='wav')

        # Read the audio data with speech_recognition
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)

        try:
            # Transcribe the audio to text
            text = r.recognize_google(audio_data)
            # Check if the specific word is present in the transcribed text
            if specific_word in text:
                # Add the filename to the list
                help_files.append(filename)
                # Print the name of the file
                print(filename)
                # Print the transcribed text
                print(f"Transcribed text: {text}")
        except Exception as e:
            # Handle the case where the audio cannot be transcribed
            print(f"Could not transcribe audio file: {filename}")
            #print(f"Error: {e}")

# Check if the help_files list is not empty
if help_files:
    print(f"Help is needed in the following files: {', '.join(help_files)}")
     # Write the list of filenames to a file in the directory with the same name as the audio file
    help_files_filename = os.path.join(directory, "help_files.txt")
    with open(help_files_filename, "w") as f:
        f.write("\n".join(help_files))
    print(f"The list of files that need help is saved in {help_files_filename}")
else:
    print("No help is needed.")
    
    
    
    
    
 #distance code 
    
    
import librosa
import numpy as np


#No, this code has a syntax error in the inner `for` loop of the distance calculation section. The `filename` variable is repeated, which causes a `SyntaxError`.

#Here's the corrected code:


import librosa
import numpy as np
import os
import speech_recognition as sr


# Read the path of the most recently created folder from a file
with open("latest_recording_folder.txt", "r") as f:
    directory = f.read().strip()

# Load the audio file
#audio_file = "output.wav"
#y, sr = librosa.load(audio_file, sr=None, mono=True)

# Specify the location of the microphone

mic_location = np.array([0, 0, 0])

# Calculate the location of the sound source for each audio file that needs help
with open(os.path.join(directory, "help_files.txt"), "r") as f:
    help_files = f.read().splitlines()

for filename in help_files:
    # Load the audio file
    audio_file = os.path.join(directory, filename)
    y, sr = librosa.load(audio_file, sr=None, mono=True)

    # Calculate the location of the sound source
    c = 2*343.2/1000  # Speed of sound in air in meters/second
    delta_t = np.argmax(y) / sr
    distance = delta_t * c
    location = mic_location + np.array([distance, 0, 0])

    # Print the calculated distance and location
    print(f"Distance for {filename}: {distance} meters")
    print(f"Location for {filename}: {location}")

    





