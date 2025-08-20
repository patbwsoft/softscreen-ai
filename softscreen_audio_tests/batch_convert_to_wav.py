import os
from pydub import AudioSegment

# Supported formats to convert
FORMATS = [".mp3", ".ogg", ".flac", ".m4a", ".aac"]
CATEGORIES = ['Bark', 'Doorbell', 'Gunshot', 'Explosion', 'Baby Cry', 'Weather', 'Human']

print("🔄 Starting batch conversion...")

for label in CATEGORIES:
    folder = os.path.join(os.getcwd(), label)
    for filename in os.listdir(folder):
        name, ext = os.path.splitext(filename)
        ext = ext.lower()

        if ext in FORMATS:
            in_path = os.path.join(folder, filename)
            out_path = os.path.join(folder, name + ".wav")

            try:
                audio = AudioSegment.from_file(in_path)
                audio.export(out_path, format="wav")
                print(f"✅ Converted {filename} → {name}.wav")

                # Optional: delete the original file
                os.remove(in_path)
            except Exception as e:
                print(f"❌ Failed on {filename}: {e}")

print("✅ All done.")