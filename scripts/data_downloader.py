# downlaoding the datasets

from huggingface_hub import snapshot_download

DATA_DIR = "/home/dilmurod/job/tts_project/data"

snapshot_download(
    repo_id="kaa-ml/karakalpak-audio-dataset",
    repo_type="dataset",
    local_dir=DATA_DIR,
)
# also kaa-ml/karakalpak-audio-dataset //requires to login into hugging face
# atikuwu/karakalpak-speech-corpus
