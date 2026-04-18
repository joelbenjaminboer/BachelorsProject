# Downloads the ENABL3S Dataset to data
from pathlib import Path
import zipfile

import hydra
from loguru import logger
from omegaconf import DictConfig
import requests
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_url = cfg.dataset.url
        self.raw_data_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.raw_data_dir))
        self.output_zip = Path(hydra.utils.to_absolute_path(cfg.dataset.output_zip))
        self.extract_dir = Path(hydra.utils.to_absolute_path(cfg.dataset.extract_dir))

    def download_url(self, url: str, output_path: Path):
        logger.info(f"Downloading {url} to {output_path}...")
        headers = {"User-Agent": "Mozilla/5.0"}

        if "figshare.com/ndownloader/articles/" in url:
            article_id = url.split("articles/")[1].split("/")[0]
            api_url = f"https://api.figshare.com/v2/articles/{article_id}"
            logger.info(f"Figshare URL detected. Fetching file list from API: {api_url}")

            api_res = requests.get(api_url, headers=headers)
            api_res.raise_for_status()
            files = api_res.json().get("files", [])

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    f_name = f["name"]
                    f_dl = f["download_url"]
                    f_size = f["size"]

                    logger.info(f"Downloading {f_name} ({f_size} bytes)...")
                    f_res = requests.get(f_dl, stream=True, headers=headers)
                    f_res.raise_for_status()

                    with (
                        zf.open(f_name, "w") as z_file,
                        tqdm(
                            desc=f_name,
                            total=f_size,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar,
                    ):
                        for data in f_res.iter_content(chunk_size=1024):
                            z_file.write(data)
                            bar.update(len(data))
        else:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(output_path, "wb") as file,
                tqdm(
                    desc=output_path.name,
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        logger.success("Download complete.")

    def extract_zip(self, zip_path: Path, extract_to: Path):
        logger.info(f"Extracting {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.success("Extraction complete.")

    def run(self):
        logger.info("Starting dataset download process...")

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        if not self.output_zip.exists() and not self.extract_dir.exists():
            self.download_url(self.dataset_url, self.output_zip)
        else:
            logger.info("Dataset zip already exists. Skipping download.")

        if self.output_zip.exists() and not self.extract_dir.exists():
            self.extract_zip(self.output_zip, self.extract_dir)

        logger.success("Dataset preparation finished.")


def run_download(cfg: DictConfig):
    downloader = DatasetDownloader(cfg)
    downloader.run()


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    run_download(cfg)


if __name__ == "__main__":
    main()
