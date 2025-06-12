# Projet de Vérification de Conformité de Décalques 

## But
Ce projet est une application Python utilisant le Machine Learning et le traitement d'image  pour automatiser la vérification de la conformité de décalques de montres .
Elle permet, à partir d'une photo, d’analyser des éléments afin de détecter toute anomalie visuelle par rapport aux spécifications attendues.

## Objectif : 
Automatiser un processus de contrôle qualité à partir d’images, pour :
- Éviter les erreurs humaines lors du contrôle visuel.
- Réduire les temps d’inspection.
- Standardiser la détection d’éléments non conformes (mauvaise orientation, mauvaise couleur).

## Fonctionnalités principales : 
- Vérifier si les couleurs présentent sur le décalque sont conformes
- Vérifier si sur un décalque un chiffre manque (conforme et non_conforme)
- Détecter les chiffres présents sur le décalque et dire lequel manque

## 🖥️ Pré-requis
- Avant de lancer l'application, assurez-vous d’avoir installé :
- Python 3.9.X
- Git
- Un éditeur de texte (comme VS Code ou PyCharm)
- Une connexion Internet pour récupérer les dépendances

## ✅ Installation de l'environnement virtuelle 

Ouvrir le terminal 
##  Installation

Ouvrir le terminal 
1. Cloner le dépôt `https://github.com/Pierre00012/Comformit-_d-calque_projet_ML.git`
2. Se placer dans le dossier du projet: `cd Conformit-d-calque`
3. Installer l'environnement virtuel Python (3.10.x): `python -m .venv .`
4. Activer l'environnement virtuel Python: `source bin/activate`
5. Installer les dépendances Python: `pip install -r requirements.txt`

## Fonction utiles : Fichier capture_des_rois.ipynb
### 🔹 `binarize_image(image_path: str, threshold: int = 100) → np.ndarray`

- **Paramètres :**
  - `image_path` : chemin vers l’image à binariser
  - `threshold` : seuil utilisé pour la binarisation (valeur entre 0 et 255)

- **Retourne :**
  - Un tuple `(retval, binary_image)` contenant l’image binarisée (`np.ndarray`)

- **Effet :**
  - Affiche l’image binaire dans une fenêtre Matplotlib

---

### 🔹 `draw_rectangle(event, x, y, flags, param) → None`

- **Paramètres :**
  - Utilisés automatiquement par OpenCV (`cv2.setMouseCallback`) :
    - `event` : type d'événement souris (clic, mouvement, relâchement…)
    - `x`, `y` : position du curseur
    - `flags`, `param` : paramètres internes

- **Retourne :**
  - Rien

- **Effet :**
  - Permet de dessiner interactivement des rectangles à la souris
  - Stocke les coordonnées dans la liste `rectangles`

---

### 🔹 `extract_and_display_regions(rectangles: list, image_path: str, cols: int = 5, figsize: tuple = (15, 20)) → matplotlib.figure.Figure`

- **Paramètres :**
  - `rectangles` : liste de dictionnaires avec les clés `x`, `y`, `width`, `height`
  - `image_path` : chemin vers l’image d’origine
  - `cols` : nombre de colonnes dans l'affichage Matplotlib
  - `figsize` : taille de la figure (en pouces)

- **Retourne :**
  - Une figure Matplotlib affichant toutes les ROIs

- **Effet :**
  - Affiche les régions extraites dans une grille

---

### 🔹 `save_extracted_rois(rectangles: list, image_path: str, output_folder: str = "extracted_rois") → list[str]`

- **Paramètres :**
  - `rectangles` : liste des rectangles à extraire
  - `image_path` : chemin vers l’image source
  - `output_folder` : dossier de sortie des images ROIs

- **Retourne :**
  - Liste des chemins des fichiers enregistrés

- **Effet :**
  - Sauvegarde chaque ROI comme image individuelle `.jpg`

---

### 🔹 `extract_display_and_save(rectangles: list, image_path: str, output_folder: str = "extracted_rois", cols: int = 5, figsize: tuple = (15, 20)) → list[str]`

- **Paramètres :**
  - `rectangles` : liste des ROIs
  - `image_path` : chemin de l’image à traiter
  - `output_folder` : dossier de sauvegarde des résultats
  - `cols` : nombre de colonnes à afficher (pour Matplotlib)
  - `figsize` : taille de la figure Matplotlib

- **Retourne :**
  - Liste des chemins vers les fichiers images extraits

- **Effet :**
  - Sauvegarde les ROIs et affiche un aperçu
