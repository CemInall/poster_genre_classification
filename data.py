import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import settings
import os
from PIL import Image
import numpy as np
import requests
from multiprocessing.pool import ThreadPool


def get_cleaned_movielabels():
    choices = set(("action", "comedy", "drama", "horror"))
    movielabels = pd.read_csv("data/duplicate_free_41K.csv")

    # remove unreadable files
    with open("data/unusable_rows.txt", "r") as file:
        unusable_rows = file.readline().split(",")
        unusable_rows = [int(row) for row in unusable_rows]

    movielabels = movielabels.drop(movielabels[movielabels["id"].isin(unusable_rows)].index)

    def clean(row):
        genres = set(row.replace(" ", "").split(","))
        include = len(choices.intersection(genres)) == 1
        return include

    mask = movielabels["genre"].apply(lambda row: clean(row))
    movielabels = movielabels[mask]

    # add target column
    movielabels['target'] = movielabels['genre'].apply(lambda x: list(choices.intersection(set(x.replace(" ", "").split(","))))[0])

    movielabels = movielabels.drop(
        columns=["adventure", "animation", "crime", "fantasy", "mystery", "romance", "sci-fi", "short", "thriller"])

    # get minimum quantity of the classes we're interesed in
    data_cutoff = min([movielabels[col].sum() for col in choices])

    # shuffle deterministically
    movielabels = movielabels.sample(frac=1, random_state=42).reset_index(drop=True)

    for col in choices:
        current_choice_indices = movielabels[movielabels[col] == 1].index
        indices_to_remove = current_choice_indices[data_cutoff:]
        movielabels = movielabels.drop(indices_to_remove)

    # shuffle deterministically
    movielabels = movielabels.sample(frac=1, random_state=42).reset_index(drop=True)

    print(movielabels.sum(axis=0))
    print(len(movielabels))

    movielabels.to_csv("data/cleaned_movielabels.csv", index=False)
    return movielabels

def download_posters(movielabels):
    image_urls = movielabels['poster']
    movie_ids = movielabels['id']
    folder_path = settings.poster_folder
    os.makedirs(folder_path, exist_ok=True)
    failed_files = []
    failed_movie_ids = []

    def download_image(url, movie_id):
        file_name = f'image_{movie_id}.jpg'
        if os.path.exists(os.path.join(folder_path, file_name)):
            return None

        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f'Error downloading image {movie_id}: {response.status_code}')
            failed_files.append(file_name)

    pool = ThreadPool()  # Create a thread pool for parallel processing
    pool.starmap(download_image, zip(image_urls, movie_ids))
    pool.close()
    pool.join()

    with open("data/failed_files.txt", "w") as file:
        file.write(",".join(failed_files))

    with open("data/failed_movie_ids.txt", "w") as file:
        file.write(",".join(failed_movie_ids))


def get_movielabels_without_dead_images(movielabels):
    ids = movielabels['id'].tolist()
    print(f'Number of elements: {len(ids)}')

    for id in ids:
        file_name = f'image_{id}.jpg'
        file_path = os.path.join(settings.poster_folder, file_name)
        if not os.path.isfile(file_path):
            movielabels = movielabels[movielabels['id'] != id]

    print(f'Number of elements after removing dead images: {len(movielabels)}')
    return movielabels


def get_preprocess_images(movielabels):
    ids = movielabels['id'].tolist()

    reshaped_images = []
    filename = f'{settings.data_folder}/preprocessed_images_{settings.IMG_SIZE[0]}x{settings.IMG_SIZE[1]}.npy'

    if not os.path.isfile(filename):
        for id in ids:
            file = f'{settings.poster_folder}/image_{id}.jpg'

            # normalize the image
            img = Image.open(file)
            img = img.convert('RGB')
            img = img.resize(settings.IMG_SIZE, resample=Image.BICUBIC)

            # Reshape the data to be 2-dimensional
            normalized_image_array = np.array(img) / 255.0
            reshaped_image = normalized_image_array.reshape(-1, 3)
            reshaped_images.append(reshaped_image)

        reshaped_images = np.array(reshaped_images)
        np.save(filename, reshaped_images)
    return np.load(filename)


def get_data():
    movielabels = get_cleaned_movielabels()
    # enable if need to download again
    #download_posters(movielabels)
    movielabels = get_movielabels_without_dead_images(movielabels)
    images = get_preprocess_images(movielabels)
    assert (len(movielabels == images.shape[0]))
    return movielabels, images
