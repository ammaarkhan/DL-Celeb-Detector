import ssl
from bing_image_downloader import downloader

# ssl._create_default_https_context = ssl._create_unverified_context

downloader.download('Mike Tyson', limit=50,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=60, verbose=True)