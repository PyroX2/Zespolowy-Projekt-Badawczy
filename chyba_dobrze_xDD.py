import os
import pandas as pd

csv_path = 'EMBED_OpenData_metada.csv' #sciezka do csv trza wpisac

dest_root = 'train/images' #sciezka wyjsciowa trzeba wpisac
os.makedirs(dest_root, exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    src = row['png_path'].strip()

    rel_path = src.split('/extracted-images/')[-1]
    rel_path = 'embed-dataset-open/png_images/cohort_1/' + rel_path #dodac mozna sciezke przed embed
    filename = row['png_filename'].strip() 
    dst = os.path.join(dest_root, filename)
    os.symlink(rel_path, dst)

    #bierze pliki z folderu np 
    #embed-dataset-open/
    #   png_images/
    #       cohort_1/
    #           fff854bd953019f15d559d1b4aa5f4fdda6fb63fef8c944a51efd2d4/
    #               1b1b705ddc3d9fd631d5bab9e4bd1a25716d450c74bbc7d46052725e/
    #                   dddd7da9dfd909e9803155028590ded7126a3ea99ebad84ccc97f56e.png
    #i wstawia do folderu link z nimi:
    #train/
    #   images/
    #       dddd7da9dfd909e9803155028590ded7126a3ea99ebad84ccc97f56e.png   