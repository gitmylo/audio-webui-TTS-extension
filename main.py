import webui.extensionlib.callbacks as cb
import webui.modules.models as mod


def create_coqui():
    from webui.ui.tabs import settings
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    import numpy as np
    import torch
    import gradio
    import gc

    class CoquiTTS(mod.TTSModelLoader):
        no_install = True
        model = 'Coqui TTS'

        current_model: TTS = None
        current_model_name: str = None

        def load_model(self, progress=gradio.Progress()):
            pass

        def unload_model(self):
            self.current_model_name = None
            self.current_model = None
            gc.collect()
            torch.cuda.empty_cache()

        def tts_speakers(self):
            if self.current_model is None:
                return gradio.update(choices=[]), gradio.update(choices=[])
            speakers = list(
                dict.fromkeys([speaker.strip() for speaker in self.current_model.speakers])) if self.current_model.is_multi_speaker else []
            languages = list(dict.fromkeys(self.current_model.languages)) if self.current_model.is_multi_lingual else []
            return gradio.update(choices=speakers), gradio.update(choices=languages)

        def _components(self, **quick_kwargs):
            with gradio.Row(visible=False) as r1:
                selected_tts = gradio.Dropdown(ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False).list_tts_models(), label='TTS model', info='The TTS model to use for text-to-speech',
                                               allow_custom_value=True, **quick_kwargs)
                selected_tts_unload = gradio.Button('💣', variant='primary tool offset--10', **quick_kwargs)

            with gradio.Row(visible=False) as r2:
                speaker_tts = gradio.Dropdown(self.tts_speakers()[0]['choices'], label='TTS speaker',
                                              info='The speaker to use for the TTS model, only for multi speaker models.', **quick_kwargs)
                speaker_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)

            with gradio.Row(visible=False) as r3:
                lang_tts = gradio.Dropdown(self.tts_speakers()[1]['choices'], label='TTS language',
                                           info='The language to use for the TTS model, only for multilingual models.', **quick_kwargs)
                lang_tts_refresh = gradio.Button('🔃', variant='primary tool offset--10', **quick_kwargs)

            speaker_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])
            lang_tts_refresh.click(fn=self.tts_speakers, outputs=[speaker_tts, lang_tts])

            def load_tts(model):
                if self.current_model_name != model:
                    unload_tts()
                    self.current_model_name = model
                    self.current_model = TTS(model, gpu=True if torch.cuda.is_available() and settings.get('tts_use_gpu') else False)
                return gradio.update(value=model), *self.tts_speakers()

            def unload_tts():
                if self.current_model is not None:
                    self.current_model = None
                    self.current_model_name = None
                    gc.collect()
                    torch.cuda.empty_cache()
                return gradio.update(value=''), *self.tts_speakers()

            selected_tts_unload.click(fn=unload_tts, outputs=[selected_tts, speaker_tts, lang_tts])
            selected_tts.select(fn=load_tts, inputs=selected_tts, outputs=[selected_tts, speaker_tts, lang_tts])

            text_input = gradio.TextArea(label='Text to speech text',
                                         info='Text to speech text if no audio file is used as input.', **quick_kwargs)

            return selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input, r1, r2, r3



        def get_response(self, *inputs, progress=gradio.Progress()):
            selected_tts, selected_tts_unload, speaker_tts, speaker_tts_refresh, lang_tts, lang_tts_refresh, text_input = inputs
            if self.current_model_name != selected_tts:
                if self.current_model is not None:
                    self.current_model = None
                    self.current_model_name = None
                    gc.collect()
                    torch.cuda.empty_cache()
                self.current_model_name = selected_tts
                self.current_model = TTS(selected_tts, gpu=True if torch.cuda.is_available() and settings.get('tts_use_gpu') else False)
            audio = np.array(self.current_model.tts(text_input, speaker_tts if self.current_model.is_multi_speaker else None, lang_tts if self.current_model.is_multi_lingual else None))
            audio_tuple = (self.current_model.synthesizer.output_sample_rate, audio)
            return audio_tuple, None
    return CoquiTTS()


cb.register_by_name('webui.tts.list', lambda: create_coqui())
cb.register_by_name('webui.settings', lambda: {
    'tts_use_gpu': {
        'tab': '🐸 Coqui TTS',
        'type': bool,
        'default': False,
        'readname': 'use gpu',
        'description': 'Use the GPU for TTS',
        'el_kwargs': {}  # Example
    },
})
