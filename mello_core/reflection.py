import os
from datetime import datetime
from collections import Counter

class MelloReflection:
    def __init__(self, session_data, memory, config):
        self.session_data = session_data
        self.memory = memory
        self.config = config

    def generate_reflection(self):
        try:
            mode = self.config.get("reflection_mode", "pro")
            dev_mode = self.config.get("dev_reflection", True)

            total_tracked = self.session_data.get('total_objects', 0)
            avg_conf = round(self.session_data.get('avg_confidence', 0.0), 2)
            unique_labels = self.session_data.get('unique_labels', [])
            top_labels = Counter(unique_labels).most_common(3)
            label_summary = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in top_labels])
            conf_dips = self.session_data.get('confidence_dips', [])
            frame_count = self.session_data.get('frame_count', 0)
            blur_areas = self.session_data.get('blur_areas', [])
            avg_area = round(sum(blur_areas) / len(blur_areas), 1) if blur_areas else 0
            median_area = sorted(blur_areas)[len(blur_areas) // 2] if blur_areas else 0
            peak_memory = len(self.memory)

            lines = [
                f"Mello Reflection — {datetime.now().strftime('%B %d, %Y – %H:%M:%S')}",
                "-" * 60,
                f"Total objects tracked: {total_tracked}",
                f"Session frame count: {frame_count}",
                f"Average detection confidence: {avg_conf}",
                f"Top labels blurred: {label_summary}",
                f"Peak memory (objects tracked at once): {peak_memory}",
                f"Average blur target area: {avg_area} px²",
                f"Median blur target area: {median_area} px²",
                f"Frames with confidence dips: {len(conf_dips)} → {conf_dips if dev_mode else 'hidden'}"
            ]

            if avg_conf < 0.4:
                lines.append("⚠️ Low overall confidence. I recommend raising detection threshold.")
            elif avg_conf > 0.8:
                lines.append("👍 Strong confidence! Threshold may be relaxed slightly next time.")

            if mode == "fun":
                lines.append("\n✨ Takeaway:\nBlurred like a champ. I'm evolving every day 🐶💨")
            else:
                lines.append("\nTakeaway:\nReflection used to update defaults for next session.")

            journal_dir = os.path.join("outputs", "mello_journal")
            os.makedirs(journal_dir, exist_ok=True)
            journal_path = os.path.join(journal_dir, f"mello_reflection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")

            with open(journal_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            print(f"✅ Reflection saved: {journal_path}")

        except Exception as e:
            print(f"❌ Reflection generation failed: {e}")