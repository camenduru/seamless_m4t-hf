import os

import gradio as gr
import numpy as np
import torch
import torchaudio
from seamless_communication.models.inference.translator import Translator

from lang_list import (
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

DESCRIPTION = "# SeamlessM4T"

TASK_NAMES = [
    "S2ST (Speech to Speech translation)",
    "S2TT (Speech to Text translation)",
    "T2ST (Text to Speech translation)",
    "T2TT (Text to Text translation)",
    "ASR (Automatic Speech Recognition)",
]

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds

DEFAULT_TARGET_LANGUAGE = "French"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translator = Translator(
    model_name_or_card="multitask_unity_large",
    vocoder_name_or_card="vocoder_36langs",
    device=device,
    sample_rate=AUDIO_SAMPLE_RATE,
)


def predict(
    task_name: str,
    audio_source: str,
    input_audio_mic: str,
    input_audio_file: str,
    input_text: str,
    source_language: str,
    target_language: str,
) -> tuple[tuple[int, np.ndarray] | None, str]:
    task_name = task_name.split()[0]
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]

    if task_name in ["S2ST", "S2TT", "ASR"]:
        if audio_source == "microphone":
            input_data = input_audio_mic
        else:
            input_data = input_audio_file

        arr, org_sr = torchaudio.load(input_data)
        new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
        max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
        if new_arr.shape[1] > max_length:
            new_arr = new_arr[:, :max_length]
            gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
        torchaudio.save(input_data, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))
    else:
        input_data = input_text
    text_out, wav, sr = translator.predict(
        input=input_data,
        task_str=task_name,
        tgt_lang=target_language_code,
        src_lang=source_language_code,
    )
    if task_name in ["S2ST", "T2ST"]:
        return (sr, wav.cpu().detach().numpy()), text_out
    else:
        return None, text_out


def update_audio_ui(audio_source: str) -> tuple[dict, dict]:
    mic = audio_source == "microphone"
    return (
        gr.update(visible=mic, value=None),  # input_audio_mic
        gr.update(visible=not mic, value=None),  # input_audio_file
    )


def update_input_ui(task_name: str) -> tuple[dict, dict, dict, dict]:
    task_name = task_name.split()[0]
    if task_name == "S2ST":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2ST_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "S2TT":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "T2ST":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True, choices=S2ST_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "T2TT":
        return (
            gr.update(visible=False),  # audio_box
            gr.update(visible=True),  # input_text
            gr.update(visible=True),  # source_language
            gr.update(
                visible=True, choices=T2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    elif task_name == "ASR":
        return (
            gr.update(visible=True),  # audio_box
            gr.update(visible=False),  # input_text
            gr.update(visible=False),  # source_language
            gr.update(
                visible=True, choices=S2TT_TARGET_LANGUAGE_NAMES, value=DEFAULT_TARGET_LANGUAGE
            ),  # target_language
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


def update_output_ui(task_name: str) -> tuple[dict, dict]:
    task_name = task_name.split()[0]
    if task_name in ["S2ST", "T2ST"]:
        return (
            gr.update(visible=True, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    elif task_name in ["S2TT", "T2TT", "ASR"]:
        return (
            gr.update(visible=False, value=None),  # output_audio
            gr.update(value=None),  # output_text
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        task_name = gr.Dropdown(
            label="Task",
            choices=TASK_NAMES,
            value=TASK_NAMES[0],
        )
        with gr.Row():
            source_language = gr.Dropdown(
                label="Source language",
                choices=TEXT_SOURCE_LANGUAGE_NAMES,
                value="English",
                visible=False,
            )
            target_language = gr.Dropdown(
                label="Target language",
                choices=S2ST_TARGET_LANGUAGE_NAMES,
                value=DEFAULT_TARGET_LANGUAGE,
            )
        with gr.Row() as audio_box:
            audio_source = gr.Radio(
                label="Audio source",
                choices=["file", "microphone"],
                value="file",
            )
            input_audio_mic = gr.Audio(
                label="Input speech",
                type="filepath",
                source="microphone",
                visible=False,
            )
            input_audio_file = gr.Audio(
                label="Input speech",
                type="filepath",
                source="upload",
                visible=True,
            )
        input_text = gr.Textbox(label="Input text", visible=False)
        btn = gr.Button("Translate")
        with gr.Column():
            output_audio = gr.Audio(
                label="Translated speech",
                autoplay=False,
                streaming=False,
                type="numpy",
            )
            output_text = gr.Textbox(label="Translated text")

    audio_source.change(
        fn=update_audio_ui,
        inputs=audio_source,
        outputs=[
            input_audio_mic,
            input_audio_file,
        ],
        queue=False,
        api_name=False,
    )
    task_name.change(
        fn=update_input_ui,
        inputs=task_name,
        outputs=[
            audio_box,
            input_text,
            source_language,
            target_language,
        ],
        queue=False,
        api_name=False,
    ).then(
        fn=update_output_ui,
        inputs=task_name,
        outputs=[output_audio, output_text],
        queue=False,
        api_name=False,
    )

    btn.click(
        fn=predict,
        inputs=[
            task_name,
            audio_source,
            input_audio_mic,
            input_audio_file,
            input_text,
            source_language,
            target_language,
        ],
        outputs=[output_audio, output_text],
        api_name="run",
    )
demo.queue(max_size=50).launch()
