import matplotlib.pyplot as plt
import pandas as pd

def generate_images(images, link):
    link.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.savefig(link / name)
