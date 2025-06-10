import os
import shutil

def save_uploaded_files(upload_folder, files):
    os.makedirs(upload_folder, exist_ok=True)
    saved_paths = []
    for file in files:
        file_path = os.path.join(upload_folder, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(file_path)
    return saved_paths
