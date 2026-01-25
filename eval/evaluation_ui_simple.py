
import gradio as gr
import json
import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path


from maliba_ai.tts.inference import BambaraTTSInference
from maliba_ai.config.settings import Speakers


SPEAKERS = {
    "Bourama": "Most stable and accurate",
    "Adama": "Natural conversational tone", 
    "Moussa": "Clear pronunciation",
    "Modibo": "Expressive delivery",
    "Seydou": "Balanced characteristics",
    "Amadou": "Warm and friendly voice",
    "Bakary": "Deep, authoritative tone",
    "Ngolo": "Youthful and energetic",
    "Ibrahima": "Calm and measured",
    "Amara": "Melodic and smooth",
}

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
        self.tts = None
        
        EXPORT_DIR.mkdir(exist_ok=True)
        self.tts = BambaraTTSInference()

    
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
    
    def generate_audio(self, speaker: str, temp: float, top_k: int, top_p: float, max_tokens: int) -> Tuple:
        sample = self.get_sample()
        if not sample:
            return None
        
        cache_key = f"{sample['id']}_{speaker}_{temp}_{top_k}_{top_p}"
        if sample['id'] in self.audio_cache and cache_key in self.audio_cache[sample['id']]:
            return self.audio_cache[sample['id']][cache_key]
        
        if self.tts:
            try:
                speaker_enum = getattr(Speakers, speaker, Speakers.Bourama)
                audio = self.tts.generate_speech(
                    text=sample["text"],
                    speaker_id=speaker_enum,
                    temperature=temp,
                    top_k=top_k,
                    top_p=top_p,
                    max_new_audio_tokens=max_tokens
                )
                result = (16000, audio)
            except Exception as e:
                print(f"TTS error: {e}")
                result = self._demo_audio(sample["text"], speaker)
        else:
            result = self._demo_audio(sample["text"], speaker)
        
        if sample['id'] not in self.audio_cache:
            self.audio_cache[sample['id']] = {}
        self.audio_cache[sample['id']][cache_key] = result
        
        return result
    
    def _demo_audio(self, text: str, speaker: str) -> Tuple:
        duration = min(0.08 * len(text), 4.0)
        sr = 16000
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
        filepath = EXPORT_DIR / f"bambara_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        filepath = EXPORT_DIR / f"bambara_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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


def generate_Bourama(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Bourama", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Bourama")
    return audio, saved_score, saved_notes

def generate_Adama(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Adama", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Adama")
    return audio, saved_score, saved_notes

def generate_Moussa(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Moussa", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Moussa")
    return audio, saved_score, saved_notes

def generate_Modibo(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Modibo", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Modibo")
    return audio, saved_score, saved_notes

def generate_Seydou(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Seydou", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Seydou")
    return audio, saved_score, saved_notes

def generate_Amadou(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Amadou", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Amadou")
    return audio, saved_score, saved_notes

def generate_Bakary(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Bakary", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Bakary")
    return audio, saved_score, saved_notes

def generate_Ngolo(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Ngolo", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Ngolo")
    return audio, saved_score, saved_notes

def generate_Ibrahima(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Ibrahima", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Ibrahima")
    return audio, saved_score, saved_notes

def generate_Amara(temp, top_k, top_p, max_tokens):
    audio = state.generate_audio("Amara", temp, int(top_k), top_p, int(max_tokens))
    saved_score, saved_notes = state.get_score("Amara")
    return audio, saved_score, saved_notes


GENERATE_FUNCTIONS = {
    "Bourama": generate_Bourama,
    "Adama": generate_Adama,
    "Moussa": generate_Moussa,
    "Modibo": generate_Modibo,
    "Seydou": generate_Seydou,
    "Amadou": generate_Amadou,
    "Bakary": generate_Bakary,
    "Ngolo": generate_Ngolo,
    "Ibrahima": generate_Ibrahima,
    "Amara": generate_Amara,
}


def save_Bourama(score, notes):
    state.save_score("Bourama", int(score), notes or "")
    return "✓ Saved score for Bourama", state.progress_stats()

def save_Adama(score, notes):
    state.save_score("Adama", int(score), notes or "")
    return "✓ Saved score for Adama", state.progress_stats()

def save_Moussa(score, notes):
    state.save_score("Moussa", int(score), notes or "")
    return "✓ Saved score for Moussa", state.progress_stats()

def save_Modibo(score, notes):
    state.save_score("Modibo", int(score), notes or "")
    return "✓ Saved score for Modibo", state.progress_stats()

def save_Seydou(score, notes):
    state.save_score("Seydou", int(score), notes or "")
    return "✓ Saved score for Seydou", state.progress_stats()

def save_Amadou(score, notes):
    state.save_score("Amadou", int(score), notes or "")
    return "✓ Saved score for Amadou", state.progress_stats()

def save_Bakary(score, notes):
    state.save_score("Bakary", int(score), notes or "")
    return "✓ Saved score for Bakary", state.progress_stats()

def save_Ngolo(score, notes):
    state.save_score("Ngolo", int(score), notes or "")
    return "✓ Saved score for Ngolo", state.progress_stats()

def save_Ibrahima(score, notes):
    state.save_score("Ibrahima", int(score), notes or "")
    return "✓ Saved score for Ibrahima", state.progress_stats()

def save_Amara(score, notes):
    state.save_score("Amara", int(score), notes or "")
    return "✓ Saved score for Amara", state.progress_stats()


SAVE_FUNCTIONS = {
    "Bourama": save_Bourama,
    "Adama": save_Adama,
    "Moussa": save_Moussa,
    "Modibo": save_Modibo,
    "Seydou": save_Seydou,
    "Amadou": save_Amadou,
    "Bakary": save_Bakary,
    "Ngolo": save_Ngolo,
    "Ibrahima": save_Ibrahima,
    "Amara": save_Amara,
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
    with gr.Blocks(title="Bambara TTS Eval") as app:
        
        gr.Markdown("""
        # MALIBA-AI Bambara TTS Evaluation
        **Load → Generate → Listen → Score → Navigate → Export**
        
        *Auto-save enabled: Your progress is saved automatically after each score.*
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(label="📁 Upload Test Dataset (JSON)", file_types=[".json"])
                load_status = gr.Textbox(label="Status", interactive=False)
                clear_btn = gr.Button("🗑️ Clear Saved Progress", size="sm")
            
            with gr.Column(scale=1):
                gr.Markdown("**Generation Parameters**")
                temp = gr.Slider(0.1, 1.5, 0.8, step=0.1, label="Temperature")
                top_k = gr.Slider(1, 100, 50, step=5, label="Top-K")
                top_p = gr.Slider(0.1, 1.0, 1.0, step=0.05, label="Top-P")
                max_tok = gr.Slider(512, 4096, 2048, step=256, label="Max Tokens")
        
        gr.Markdown("---")
        
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", size="sm")
            sample_idx = gr.Number(value=0, label="Sample #", precision=0, scale=1)
            next_btn = gr.Button("Next ➡️", size="sm")
            progress = gr.Textbox(label="Progress", interactive=False, scale=2)
        
        sample_info = gr.Textbox(label="Sample Info", interactive=False)
        sample_text = gr.Textbox(label="Bambara Text", lines=2, interactive=False)
        translation = gr.Textbox(label="Translation", lines=1, interactive=False)
        
        gr.Markdown("---")
        
        gr.Markdown("### 🎧 Speaker Evaluation (MOS: 1=Poor, 5=Excellent)")
        
        with gr.Tabs():
            for speaker, desc in SPEAKERS.items():
                with gr.Tab(f"{speaker}"):
                    gr.Markdown(f"**{speaker}** - {desc}")
                    
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
                        inputs=[temp, top_k, top_p, max_tok],
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
        - Generate audio for each speaker tab, listen, and score
        - Scores persist when navigating between samples
        - Progress auto-saves after each evaluation
        - Export your results when finished
        
        *MALIBA-AI - AI for Mali's Languages* 🇲🇱
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