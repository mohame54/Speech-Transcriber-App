from flask import Flask, request
from flask_restful import Api, Resource
import io
import os
import base64


# local
from whisper import WhisperConfig, WhisperInference, load_wav
from whisper import GreedyDecoding


app = Flask(__name__)
api = Api(app)
CWR = os.getcwd()

# Whisper default config
config = WhisperConfig(
    encoder_path=os.path.join(CWR, "whisper\\Pretrained\\encoder.int8.onnx"),
    decoder_path=os.path.join(CWR, "whisper\\Pretrained\\decoder.int8.onnx"),
)
# Initialize the inference
inf = WhisperInference(config)


class AudioTranscriber(Resource):
    def __init__(self):
        super().__init__()
        self.recording_thread = None
        
    def post(self):
        # Load and read the audio file
        data = request.json
        audio_file = data.get("audio_file")
        generation_kwargs = data.get("generation_kwargs", {})
        if audio_file is None:
            return {"message":"could't find any audio files!"}, 400 
        audio_file =  base64.b64decode(audio_file['content'])   
        #audio_file.seek(0)
        audio_data, _ = load_wav(io.BytesIO(audio_file))
        # Transcribe the loaded wav
        hyps = inf(audio_data, **generation_kwargs)
        if isinstance(inf.decoding, GreedyDecoding):
            hyps = [hyps]
        texts = inf.decode(hyps)    
        return {'transcriptions': texts}, 200


api.add_resource(AudioTranscriber, '/transcribe')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000", debug=True)
