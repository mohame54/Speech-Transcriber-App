# Speech Recognition RESTful API with Whisper Model
### This RESTful API is designed for speech recognition using the Whisper ASR model. It leverages the Flask framework to create an interface for transcribing audio files.

## Features
**ASR Model**: The API employs the Whisper ASR model for accurate and efficient speech recognition.

**RESTful Interface**: Utilizing Flask, the API offers a RESTful interface for seamless integration and communication.

**Multiple Decoding Strategies**: The API supports various decoding strategies, including greedy decoding and beam search decoding(under development).

**Pretrained Models**: ONNX Runtime is utilized for efficient model inference, and the pretrained models can be found in the Pretrained directory.

## Getting Started
### Installation:

**Clone this repository to your local machine Install the required dependencies using** `pip`
```bash
pip install -r requirements.txt
```
## Run the API:

Execute python app.py to start the Flask server and download `(if needed)` the optimized models.
```bash
python app.py
```
API Endpoints:

The main endpoint for transcribing audio is **/transcribe**.
Sending Requests:

Send a POST request to the **/transcribe** endpoint with your audio file and optional generation parameters.
**Example using requests in Python:**

```python
##Copy code
import requests
import base64
url = "http://127.0.0.1:5000/transcribe"

# Replace 'audio_file_path' with the actual path to your audio file
audio_file_path = './test-wavs/recording.wav'

# Read the audio file content as binary data
with open(audio_file_path, 'rb') as audio_file:
    audio_content = audio_file.read()

# Encode the binary data as base64
encoded_audio_content = base64.b64encode(audio_content).decode('utf-8')

# Create the data dictionary with both file content and additional parameters
data = {
    # audio file kwargs
    'audio_file': {
        'filename': 'audio.wav',
        'content': encoded_audio_content,
        'content_type': 'audio/wav'
    },
    # Optional generation kwargs
    'generation_kwargs': {
        'decoding': 'greedy',
        "return_multiple":True
    }
}
# Send the data and wait for the response.
response = requests.post(url, json=data)
```
Additional Information
ONNX Runtime:

The API utilizes ONNX Runtime for efficient and optimized model inference.
Pretrained Models:

Pretrained Whisper models can be found in the `Pretrained directory.`
