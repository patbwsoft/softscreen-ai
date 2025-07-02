import os
import yaml
from datetime import datetime

def read_last_reflection():
    journal_folder = os.path.join("outputs", "mello_journal")
    if not os.path.exists(journal_folder):
        return None

    journal_files = sorted(
        [f for f in os.listdir(journal_folder) if f.endswith(".txt")],
        key=lambda x: os.path.getmtime(os.path.join(journal_folder, x)),
        reverse=True
    )

    if not journal_files:
        return None

    last_file_path = os.path.join(journal_folder, journal_files[0])
    with open(last_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = {
        "takeaways": [],
        "top_class": None,
        "avg_conf": None,
        "peak_memory": None
    }

    for line in lines:
        line = line.strip()
        if "Average Confidence:" in line:
            try:
                data["avg_conf"] = float(line.split(":")[1])
            except:
                pass
        if "Peak Memory Count:" in line:
            try:
                data["peak_memory"] = int(line.split(":")[1])
            except:
                pass
        if "Most Frequent Object:" in line:
            data["top_class"] = line.split(":")[1].strip()
        if line.startswith("Consider") or line.startswith("Confidence"):
            data["takeaways"].append(line)

    return data

def adjust_settings(takeaways, data):
    config_path = os.path.join(os.path.dirname(__file__), 'settings.yaml')

    if not os.path.exists(config_path):
        print("⚠️ No settings.yaml found. Cannot adjust settings.")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    modified = False

    for tip in takeaways:
        if "Confidence was low" in tip:
            config["confidence_threshold"] = round(min(config["confidence_threshold"] + 0.05, 0.9), 2)
            print(f"📈 Raised confidence_threshold to {config['confidence_threshold']}")
            modified = True

        if "tracked a lot" in tip:
            config["max_missing_frames"] = min(config["max_missing_frames"] + 2, 15)
            print(f"🧠 Increased max_missing_frames to {config['max_missing_frames']}")
            modified = True

    if modified:
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        print("✅ Mello self-tuning complete.\n")

    # 🧠 New: Suggest improvements
    suggestions = []
    if data.get("top_class") and data["top_class"] not in config.get("blur_labels", []):
        suggestions.append(f"🔍 Consider blurring '{data['top_class']}' — it appeared frequently.")

    if data.get("avg_conf") and data["avg_conf"] > 0.85:
        suggestions.append("⚙️ Confidence was high — could consider lowering detection threshold slightly.")

    if data.get("peak_memory") and data["peak_memory"] < 3:
        suggestions.append("📉 Low memory count — consider reducing min_object_area.")

    if suggestions:
        suggestion_path = os.path.join("outputs", "mello_journal", f"mello_suggestions_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        with open(suggestion_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(suggestions))
        print(f"💡 Mello has suggestions:\n- " + "\n- ".join(suggestions))

def run_pre_session_adjustments():
    print("🤖 Running Mello introspection...")
    data = read_last_reflection()
    if data:
        adjust_settings(data.get("takeaways", []), data)
    else:
        print("🟡 No previous reflections to adjust from.")

def get_model_name():
    model_to_use = "yolov8n.pt"

    data = read_last_reflection()
    if not data:
        return model_to_use

    unexpected_labels = ["cow", "sheep", "zebra", "horse"]
    if data["top_class"] in unexpected_labels:
        print(f"🤔 Top class was '{data['top_class']}', which may indicate misclassification.")
        print("🧬 Upgrading to 'yolov8m.pt' for better recognition.")
        return "yolov8m.pt"

    return model_to_use
