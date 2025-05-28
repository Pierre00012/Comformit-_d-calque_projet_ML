import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


IMG_SIZE = 1024


def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def handle_missing_values(df):
    df = df.dropna()
    return df


def display_image(row):
    pixels = row[:-1].values.astype(float)   
    image = pixels.reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(image, cmap='gray') 
    plt.axis('off')  
    plt.show()


def plot_pixel_distribution(image):
    plt.figure(figsize=(8, 4))
    plt.hist(image.ravel(), bins=256, color='gray', alpha=0.7)
    plt.title('Distribution des pixels')
    plt.xlabel('Valeur des pixels')
    plt.ylabel('Nombre de pixels')
    plt.show()


def display_multiple_images(images, cols=5):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, rows * 3))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def resize_image(image, size=(IMG_SIZE, IMG_SIZE)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image


def detect_distortion(img_array, visualize=False, use_matplotlib=True):
    """
    Détecte une distorsion barillet ou coussinet 
    dans une image en niveaux de gris.
    Paramètres :
    - img_array : Image en niveaux de gris (NumPy array)
    - visualize : Booléen pour afficher les lignes détectées
    - use_matplotlib : Booléen pour utiliser matplotlib au lieu de cv2.imshow
    Retourne :
    - distortion_type : "barillet", "coussinet", ou "aucune"
    """
    if img_array is None or img_array.size == 0:
        raise ValueError("Image invalide ou vide")
    if img_array.dtype != np.uint8:
        raise ValueError("L'image doit être de type np.uint8")

    # Ajout d 'un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    # Détection des contours avec Canny 
    edges = cv2.Canny(blurred, 30, 120)  
    # Détection des lignes avec HoughLinesP 
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 
                            threshold=50, minLineLength=50, maxLineGap=20)
    # Analyse des lignes pour détecter une distorsion
    distortion_type = "aucune"
    # Pour visualisation
    img_with_lines = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)  
    if lines is not None:
        h, w = img_array.shape
        center_x, center_y = w // 2, h // 2
        outward_count = 0
        inward_count = 0
        total_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            dist2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            if dist2 > dist1:
                outward_count += 1
            elif dist2 < dist1:
                inward_count += 1
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if total_lines > 0:
            outward_ratio = outward_count / total_lines
            inward_ratio = inward_count / total_lines
            # Assouplir l'heuristique (60 % au lieu de 70 %)
            if outward_ratio > 0.6:
                distortion_type = "barillet"
            elif inward_ratio > 0.6:
                distortion_type = "coussinet"
    # Visualisation
    if visualize:
        if use_matplotlib:
            plt.figure(figsize=(10, 5))
            plt.imshow(img_with_lines[:, :, ::-1])  # Convertir BGR en RGB
            if lines is not None:
                plt.title(f"Type de distorsion : {distortion_type} ({total_lines} lignes détectées)")
            else:
                plt.title("Aucune ligne détectée")
            plt.axis('off')
            plt.show()
        else:
            cv2.imshow("Lignes détectées", img_with_lines)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return distortion_type


def rename_image(folder_path):
    # 📜 Extensions à considérer comme images
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # 🔍 Lister et trier les fichiers image
    files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_extensions])

    # 🔁 Parcourir et renommer
    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"image_{idx:03d}{ext}"
        
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
            print(f"✅ {filename} renommé en {new_name}")

    print("🎉 Renommage terminé.")
