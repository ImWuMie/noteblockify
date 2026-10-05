"""Convert audio (or MIDI) to the best-sounding NBS arrangement.

Audio is transcribed to MIDI by muscriptor first; the MIDI then goes
through noteblockify.song.arrange. Existing MIDI files are converted directly.
"""

from pathlib import Path

from muscriptor import TranscriptionModel

from noteblockify.song import arrange

AUDIO = Path("yuai.mp3")
MIDI = AUDIO.with_suffix(".mid")
NBS = AUDIO.with_suffix(".nbs")

if not MIDI.exists():
    model = TranscriptionModel.load_model("medium")
    MIDI.write_bytes(model.transcribe_to_midi(AUDIO))
    print(f"wrote {MIDI} ({MIDI.stat().st_size} bytes)")

song = arrange(MIDI)
keys = [note.key for note in song.notes]
assert keys and min(keys) >= 0 and max(keys) <= 87
song.save(NBS)
print(f"wrote {NBS} ({NBS.stat().st_size} bytes), {len(song.notes)} notes, "
      f"keys {min(keys)}-{max(keys)}, tempo {song.header.tempo} tps, "
      f"{song.header.song_layers} layers")
