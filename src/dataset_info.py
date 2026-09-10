from pathlib import Path

# حطي هنا المسار الحقيقي لمجلد 01
dataset_path = Path(
    r"C:\Users\User\PycharmProjects\JoNova\dataset\raw\KArSL\01"
)

classes = [folder for folder in dataset_path.iterdir() if folder.is_dir()]

print("Number of classes:", len(classes))

for cls in classes[:5]:
    videos = list(cls.glob("*.mp4"))
    print(f"{cls.name}: {len(videos)} videos")