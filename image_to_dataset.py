import os
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class ImageToDataset:
    def __init__(self, data_dirs_labels, IMG_RESIZE, output_csv='dataset_images.csv', batch_size=500):
        self.data_dirs_labels = data_dirs_labels
        self.IMG_RESIZE = IMG_RESIZE
        self.batch_size = batch_size
        self.output_csv = output_csv
        self.first_batch = not os.path.exists(self.output_csv)

    def process_batch(self, batch_data):
        batch_array = np.array([item['image'] for item in batch_data], dtype=np.float32) / 255.0
        df = pd.DataFrame(batch_array)
        df['filename'] = [item['filename'] for item in batch_data]
        df['label'] = [item['label'] for item in batch_data]

        mode = 'w' if self.first_batch else 'a'
        df.to_csv(self.output_csv, mode=mode, index=False, header=self.first_batch)
        self.first_batch = False

    def process_image(self, img_path, label):
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_array is None:
                print(f"Erreur chargement {img_path}")
                return None

            resized_array = cv2.resize(img_array, (self.IMG_RESIZE, self.IMG_RESIZE),
                                       interpolation=cv2.INTER_AREA)  # INTER_AREA = rapide et qualité OK
            img_flattened = resized_array.flatten()

            return {
                'image': img_flattened,
                'filename': os.path.basename(img_path),
                'label': label
            }
        except Exception as e:
            print(f"Erreur traitement {img_path} : {e}")
            return None

    def convert_image_to_array(self):
        batch_data = []

        for data_dir, label in self.data_dirs_labels:
            if not os.path.exists(data_dir):
                print(f"Le dossier {data_dir} n'existe pas.")
                continue

            img_paths = [os.path.join(data_dir, img) for img in os.listdir(data_dir)]

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self.process_image, img_path, label) for img_path in img_paths]

                for future in futures:
                    result = future.result()
                    if result is not None:
                        batch_data.append(result)

                    if len(batch_data) >= self.batch_size:
                        self.process_batch(batch_data)
                        batch_data = []

        if batch_data:
            self.process_batch(batch_data)

        print(f"Dataset créé ou mis à jour dans {self.output_csv}")

    def image_to_dataset(self):
        self.convert_image_to_array()

if __name__ == "__main__":
    data_dirs_labels = [
        ("data/conformes", 0),
        ("data/non_conformes", 1),
    ]
    IMG_SIZE = 128
    BATCH_SIZE = 500  # Plus gros batch => moins d'accès disque

    print("--------------------Début------------------------")
    output_csv = input("Entrez le nom du fichier CSV de sortie (ex: mon_dataset.csv) : ")

    dataCreator = ImageToDataset(data_dirs_labels, IMG_SIZE, output_csv=output_csv, batch_size=BATCH_SIZE)
    dataCreator.image_to_dataset()
    print("--------------------Fin------------------------")
