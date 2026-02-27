
import gradio as gr
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from google import genai
from google.genai import types


# Maps evaluation speaker names → Gemini prebuilt voice names
SPEAKERS = {
    "Aoede":    "Bright and expressive",
    "Charon":   "Deep and authoritative",
    "Fenrir":   "Confident and clear",
    "Kore":     "Warm and friendly",
    "Leda":     "Calm and measured",
    "Orus":     "Natural conversational tone",
    "Puck":     "Youthful and energetic",
    "Schedar":  "Melodic and smooth",
    "Zephyr":   "Balanced characteristics",
    "Achird":   "Expressive delivery",
}

GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"
GEMINI_SAMPLE_RATE = 24000  # Gemini TTS returns 24 kHz Linear16 PCM

SPEAKER_LIST = list(SPEAKERS.keys())

EXPORT_DIR = Path("./exports")
STATE_FILE = Path("./evaluation_state.json")


class AppState:
    def __init__(self):
        self.samples: List[Dict] = []
        self.current_idx: int = 0
        self.scores: Dict[str, Dict[str, Dict]] = {}
        self.audio_cache: Dict[str, Dict[str, Tuple]] = {}
        self.dataset_path: Optional[str] = None
        self.gemini_client: Optional[genai.Client] = None

        EXPORT_DIR.mkdir(exist_ok=True)

    def set_api_key(self, api_key: str) -> str:
        """Initialize (or re-initialize) the Gemini client with the given API key."""
        try:
            self.gemini_client = genai.Client(api_key=api_key.strip())
            self.audio_cache = {}  # invalidate cache when key changes
            return "✓ Gemini API key accepted"
        except Exception as e:
            self.gemini_client = None
            return f"✗ Failed to set API key: {e}"

    
    def load_dataset(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.samples = []
            self.dataset_path = filepath
            
            if "categories" in data:
                for cat_name, cat_data in data["categories"].items():
                    if isinstance(cat_data, dict):
                        for subcat, items in cat_data.items():
                            if isinstance(items, list):
                                for item in items:
                                    if "text" in item:
                                        self.samples.append({
                                            "id": item.get("id", f"s{len(self.samples)}"),
                                            "text": item["text"],
                                            "translation": item.get("translation", ""),
                                            "category": f"{cat_name}/{subcat}",
                                            "difficulty": item.get("difficulty", "medium")
                                        })
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if "text" in item:
                        self.samples.append({
                            "id": item.get("id", f"s{i}"),
                            "text": item["text"],
                            "translation": item.get("translation", ""),
                            "category": item.get("category", ""),
                            "difficulty": item.get("difficulty", "medium")
                        })
            
            restored = self._load_state()
            if not restored:
                self.current_idx = 0
            
            restore_msg = f" (restored {len(self.scores)} scores)" if restored else ""
            return f"✓ Loaded {len(self.samples)} samples{restore_msg}"
        except Exception as e:
            return f"✗ Error: {e}"
    
    def get_sample(self, idx: int = None) -> Optional[Dict]:
        if idx is None:
            idx = self.current_idx
        if 0 <= idx < len(self.samples):
            return self.samples[idx]
        return None
    
    def generate_audio(self, speaker: str) -> Tuple:
        sample = self.get_sample()
        if not sample:
            return None

        cache_key = f"{sample['id']}_{speaker}"
        if sample['id'] in self.audio_cache and cache_key in self.audio_cache[sample['id']]:
            return self.audio_cache[sample['id']][cache_key]

        if self.gemini_client:
            try:
                voice_name = speaker  # speaker names map 1-to-1 with Gemini voice names
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_TTS_MODEL,
                    contents=sample["text"],
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name,
                                )
                            )
                        ),
                    ),
                )
                # Gemini returns raw Linear16 PCM bytes
                audio_data = response.candidates[0].content.parts[0].inline_data.data
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                result = (GEMINI_SAMPLE_RATE, audio_array)
            except Exception as e:
                print(f"Gemini TTS error: {e}")
                result = self._demo_audio(sample["text"], speaker)
        else:
            result = self._demo_audio(sample["text"], speaker)

        if sample['id'] not in self.audio_cache:
            self.audio_cache[sample['id']] = {}
        self.audio_cache[sample['id']][cache_key] = result

        return result

    def _demo_audio(self, text: str, speaker: str) -> Tuple:
        duration = min(0.08 * len(text), 4.0)
        sr = GEMINI_SAMPLE_RATE
        t = np.linspace(0, duration, int(sr * duration))
        freq = 150 + SPEAKER_LIST.index(speaker) * 15 if speaker in SPEAKER_LIST else 150
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        env = np.ones_like(t)
        attack = int(0.05 * sr)
        release = int(0.1 * sr)
        if len(env) > attack:
            env[:attack] = np.linspace(0, 1, attack)
        if len(env) > release:
            env[-release:] = np.linspace(1, 0, release)
        audio = (audio * env).astype(np.float32)
        return (sr, audio)
    
    def save_score(self, speaker: str, score: int, notes: str):
        sample = self.get_sample()
        if sample:
            if sample['id'] not in self.scores:
                self.scores[sample['id']] = {}
            self.scores[sample['id']][speaker] = {
                "score": score,
                "notes": notes,
                "timestamp": datetime.now().isoformat()
            }
            self._save_state()
    
    def get_score(self, speaker: str) -> Tuple[int, str]:
        sample = self.get_sample()
        if sample and sample['id'] in self.scores and speaker in self.scores[sample['id']]:
            data = self.scores[sample['id']][speaker]
            return data.get("score", 3), data.get("notes", "")
        return 3, ""
    
    def _save_state(self):
        try:
            state_data = {
                "last_saved": datetime.now().isoformat(),
                "current_idx": self.current_idx,
                "scores": self.scores,
                "dataset_path": self.dataset_path
            }
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Auto-save failed: {e}")
    
    def _load_state(self) -> bool:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                self.scores = state_data.get("scores", {})
                self.current_idx = state_data.get("current_idx", 0)
                if self.current_idx >= len(self.samples):
                    self.current_idx = 0
                print(f"✓ Restored {len(self.scores)} scored samples")
                return len(self.scores) > 0
            except Exception as e:
                print(f"Failed to load state: {e}")
        return False
    
    def clear_state(self):
        if STATE_FILE.exists():
            try:
                STATE_FILE.unlink()
            except Exception as e:
                print(f"Failed to delete state: {e}")
        self.scores = {}
        self.current_idx = 0
        self.audio_cache = {}
    
    def export_csv(self) -> str:
        filepath = EXPORT_DIR / f"gemini_tts_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        sample_lookup = {s['id']: s for s in self.samples}
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_id', 'speaker', 'score', 'notes', 'text', 'translation', 'difficulty', 'timestamp'])
            
            for sample_id, speakers in self.scores.items():
                sample = sample_lookup.get(sample_id, {})
                for speaker, data in speakers.items():
                    writer.writerow([
                        sample_id, speaker, data['score'], data.get('notes', ''),
                        sample.get('text', ''), sample.get('translation', ''),
                        sample.get('difficulty', ''), data.get('timestamp', '')
                    ])
        
        return str(filepath)
    
    def export_json(self) -> str:
        filepath = EXPORT_DIR / f"gemini_tts_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_samples": len(self.samples),
            "scored_count": len(self.scores),
            "scores": self.scores,
            "samples": {s['id']: s for s in self.samples}
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def progress_stats(self) -> str:
        total = len(self.samples)
        scored = len(self.scores)
        pct = (scored / total * 100) if total > 0 else 0
        return f"Scored {scored}/{total} samples ({pct:.1f}%)"


state = AppState()


def set_api_key(api_key: str) -> str:
    return state.set_api_key(api_key)


def load_data(file):
    if file is None:
        return "Upload a dataset file", "", "", "", 0, state.progress_stats()
    
    msg = state.load_dataset(file)
    sample = state.get_sample()
    
    if sample:
        return (
            msg,
            sample['text'],
            sample.get('translation', ''),
            f"{sample['id']} | {sample.get('category', '')} | {sample.get('difficulty', '')}",
            state.current_idx,
            state.progress_stats()
        )
    return msg, "", "", "", 0, state.progress_stats()


def navigate(direction: int):
    state.current_idx = max(0, min(len(state.samples) - 1, state.current_idx + direction))
    state._save_state()
    sample = state.get_sample()
    
    if sample:
        return (
            sample['text'],
            sample.get('translation', ''),
            f"{sample['id']} | {sample.get('category', '')} | {sample.get('difficulty', '')}",
            state.current_idx,
            state.progress_stats()
        )
    return "", "", "", state.current_idx, state.progress_stats()


def navigate_prev():
    return navigate(-1)


def navigate_next():
    return navigate(1)


def jump_to(idx: int):
    state.current_idx = max(0, min(len(state.samples) - 1, int(idx)))
    return navigate(0)


def clear_progress():
    state.clear_state()
    return "✓ Progress cleared", state.progress_stats()


def generate_Aoede():
    audio = state.generate_audio("Aoede")
    saved_score, saved_notes = state.get_score("Aoede")
    return audio, saved_score, saved_notes

def generate_Charon():
    audio = state.generate_audio("Charon")
    saved_score, saved_notes = state.get_score("Charon")
    return audio, saved_score, saved_notes

def generate_Fenrir():
    audio = state.generate_audio("Fenrir")
    saved_score, saved_notes = state.get_score("Fenrir")
    return audio, saved_score, saved_notes

def generate_Kore():
    audio = state.generate_audio("Kore")
    saved_score, saved_notes = state.get_score("Kore")
    return audio, saved_score, saved_notes

def generate_Leda():
    audio = state.generate_audio("Leda")
    saved_score, saved_notes = state.get_score("Leda")
    return audio, saved_score, saved_notes

def generate_Orus():
    audio = state.generate_audio("Orus")
    saved_score, saved_notes = state.get_score("Orus")
    return audio, saved_score, saved_notes

def generate_Puck():
    audio = state.generate_audio("Puck")
    saved_score, saved_notes = state.get_score("Puck")
    return audio, saved_score, saved_notes

def generate_Schedar():
    audio = state.generate_audio("Schedar")
    saved_score, saved_notes = state.get_score("Schedar")
    return audio, saved_score, saved_notes

def generate_Zephyr():
    audio = state.generate_audio("Zephyr")
    saved_score, saved_notes = state.get_score("Zephyr")
    return audio, saved_score, saved_notes

def generate_Achird():
    audio = state.generate_audio("Achird")
    saved_score, saved_notes = state.get_score("Achird")
    return audio, saved_score, saved_notes


GENERATE_FUNCTIONS = {
    "Aoede":   generate_Aoede,
    "Charon":  generate_Charon,
    "Fenrir":  generate_Fenrir,
    "Kore":    generate_Kore,
    "Leda":    generate_Leda,
    "Orus":    generate_Orus,
    "Puck":    generate_Puck,
    "Schedar": generate_Schedar,
    "Zephyr":  generate_Zephyr,
    "Achird":  generate_Achird,
}


def save_Aoede(score, notes):
    state.save_score("Aoede", int(score), notes or "")
    return "✓ Saved score for Aoede", state.progress_stats()

def save_Charon(score, notes):
    state.save_score("Charon", int(score), notes or "")
    return "✓ Saved score for Charon", state.progress_stats()

def save_Fenrir(score, notes):
    state.save_score("Fenrir", int(score), notes or "")
    return "✓ Saved score for Fenrir", state.progress_stats()

def save_Kore(score, notes):
    state.save_score("Kore", int(score), notes or "")
    return "✓ Saved score for Kore", state.progress_stats()

def save_Leda(score, notes):
    state.save_score("Leda", int(score), notes or "")
    return "✓ Saved score for Leda", state.progress_stats()

def save_Orus(score, notes):
    state.save_score("Orus", int(score), notes or "")
    return "✓ Saved score for Orus", state.progress_stats()

def save_Puck(score, notes):
    state.save_score("Puck", int(score), notes or "")
    return "✓ Saved score for Puck", state.progress_stats()

def save_Schedar(score, notes):
    state.save_score("Schedar", int(score), notes or "")
    return "✓ Saved score for Schedar", state.progress_stats()

def save_Zephyr(score, notes):
    state.save_score("Zephyr", int(score), notes or "")
    return "✓ Saved score for Zephyr", state.progress_stats()

def save_Achird(score, notes):
    state.save_score("Achird", int(score), notes or "")
    return "✓ Saved score for Achird", state.progress_stats()


SAVE_FUNCTIONS = {
    "Aoede":   save_Aoede,
    "Charon":  save_Charon,
    "Fenrir":  save_Fenrir,
    "Kore":    save_Kore,
    "Leda":    save_Leda,
    "Orus":    save_Orus,
    "Puck":    save_Puck,
    "Schedar": save_Schedar,
    "Zephyr":  save_Zephyr,
    "Achird":  save_Achird,
}


def export_results_csv():
    if len(state.scores) == 0:
        return None
    return state.export_csv()


def export_results_json():
    if len(state.scores) == 0:
        return None
    return state.export_json()


def create_app():
    with gr.Blocks(title="Gemini TTS Eval") as app:

        gr.Markdown("""
        # Gemini TTS Evaluation
        **Set API Key → Load → Generate → Listen → Score → Navigate → Export**

        *Auto-save enabled: Your progress is saved automatically after each score.*
        """)

        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="📁 Upload Test Dataset (JSON)", file_types=[".json"])
                load_status = gr.Textbox(label="Status", interactive=False)
                clear_btn = gr.Button("🗑️ Clear Saved Progress", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("**Gemini API Key**")
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your Gemini API key…",
                    type="password",
                )
                api_key_btn = gr.Button("Set API Key", variant="primary")
                api_key_status = gr.Textbox(label="", interactive=False)

        gr.Markdown("---")

        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", size="sm")
            sample_idx = gr.Number(value=0, label="Sample #", precision=0, scale=1)
            next_btn = gr.Button("Next ➡️", size="sm")
            progress = gr.Textbox(label="Progress", interactive=False, scale=2)

        sample_info = gr.Textbox(label="Sample Info", interactive=False)
        sample_text = gr.Textbox(label="Text", lines=2, interactive=False)
        translation = gr.Textbox(label="Translation", lines=1, interactive=False)

        gr.Markdown("---")

        gr.Markdown("### 🎧 Voice Evaluation (MOS: 1=Poor, 5=Excellent)")

        with gr.Tabs():
            for speaker, desc in SPEAKERS.items():
                with gr.Tab(f"{speaker}"):
                    gr.Markdown(f"**{speaker}** — {desc}")

                    gen_btn = gr.Button("🎵 Generate Audio", variant="primary")
                    audio_out = gr.Audio(label="Audio", interactive=False)

                    with gr.Row():
                        score_slider = gr.Slider(1, 5, 3, step=1, label="Score (1-5)", scale=1)
                        notes_input = gr.Textbox(label="Notes", placeholder="Optional notes...", scale=2)

                    with gr.Row():
                        save_btn = gr.Button("💾 Save Score", variant="secondary")
                        save_status = gr.Textbox(label="", interactive=False, scale=2)

                    gen_btn.click(
                        fn=GENERATE_FUNCTIONS[speaker],
                        inputs=[],
                        outputs=[audio_out, score_slider, notes_input]
                    )

                    save_btn.click(
                        fn=SAVE_FUNCTIONS[speaker],
                        inputs=[score_slider, notes_input],
                        outputs=[save_status, progress]
                    )

        gr.Markdown("---")

        gr.Markdown("### 📤 Export Results")
        with gr.Row():
            export_csv_btn = gr.Button("Export CSV")
            export_json_btn = gr.Button("Export JSON")
            csv_file = gr.File(label="CSV Download")
            json_file = gr.File(label="JSON Download")
        
        file_input.change(
            fn=load_data,
            inputs=[file_input],
            outputs=[load_status, sample_text, translation, sample_info, sample_idx, progress]
        )

        api_key_btn.click(
            fn=set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )

        clear_btn.click(
            fn=clear_progress,
            outputs=[load_status, progress]
        )
        
        prev_btn.click(
            fn=navigate_prev,
            outputs=[sample_text, translation, sample_info, sample_idx, progress]
        )
        
        next_btn.click(
            fn=navigate_next,
            outputs=[sample_text, translation, sample_info, sample_idx, progress]
        )
        
        sample_idx.submit(
            fn=jump_to,
            inputs=[sample_idx],
            outputs=[sample_text, translation, sample_info, sample_idx, progress]
        )
        
        export_csv_btn.click(fn=export_results_csv, outputs=[csv_file])
        export_json_btn.click(fn=export_results_json, outputs=[json_file])
        
        gr.Markdown("""
        ---
        **Tips:**
        - Set your Gemini API key first, then load a dataset
        - Generate audio for each voice tab, listen, and score
        - Scores persist when navigating between samples
        - Progress auto-saves after each evaluation
        - Export your results when finished
        """)
    
    return app


if __name__ == "__main__":
    EXPORT_DIR.mkdir(exist_ok=True)
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        allowed_paths=[str(EXPORT_DIR)]
    )