import os
import io
import numpy as np

from PIL import Image, ImageOps

import pyspark.sql.functions as F
from pyspark.ml.linalg import DenseVector, VectorUDT


# Préprocessing des chemins d'accès => nom du fruit
def cat_extract(path):
    """
    Fonction donnant le nom du dossier indiqué par le chemin path.
    """
    path = os.path.dirname(path)
    return os.path.basename(path)


# Préprocessing des images
def preprocess(content, imgsize=[100, 100]):
    """
    Préprocessing des images : changement de taille, et correction de l'histogramme.
    Par défaut, taille d'image en 100px x 100px.
    """
    img = Image.open(io.BytesIO(content))
    if imgsize != None:
        img = img.resize(imgsize)
    img = ImageOps.autocontrast(img)
    img = ImageOps.equalize(img)
    arr = np.asarray(img)
    return arr