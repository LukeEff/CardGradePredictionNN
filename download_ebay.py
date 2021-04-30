from ebaysdk.finding import Connection
import json
import pickle
import requests
import time
from PIL import Image
import os
import numpy as np

from file_management import FileManagement

"""
The purpose of this project is to quickly and easily query data from Ebay to use in training neural networks.

Fill the key.txt file with your API key.


"""


# TODO refactor

class EbayDataset:
    def __init__(self):
        self.file_management = FileManagement()

    def find_cards(self):
        api = Connection(appid=FileManagement.get_json_value(self.file_management.client_id), config_file=None)
        grades = range(1, 11)  # Get data for grades 1-10
        for grade in grades:
            for page_num in range(1, 100):
                parameters = {
                    "categoryName": "Baseball Cards",
                    "keywords": "psa {}".format(grade),
                    "paginationInput":
                        {
                            "entriesPerPage": 100,  # max 100.
                            "pageNumber": page_num
                        },
                    "outputSelector": "PictureURLLarge"  # important: if not selected, returned images are too small.
                }
            response = api.execute("findItemsAdvanced", parameters)
            j = response.json()

            # save
            of_p = "psa_grade_{}_page_{}.pkl".format(grade, page_num)
            of_p = os.path.join("page_jsons", of_p)
            print(of_p)
            with open(of_p, "wb") as of:
                pickle.dump(j, of)

    def download_images(self):
        for page_num in range(1, 100):
            for grade in range(1, 11):
                # Load appropriate json file.
                in_p = "page_jsons/psa_grade_{}_page_{}.pkl".format(grade, page_num)
                j = pickle.load(open(in_p, "rb"))
                j = json.loads(j)
                print(j.keys())

                # Get data.
                items = j["searchResult"]["item"]  # json[] length 100.
                for idx, item in enumerate(items):

                    try:

                        # Get picture URL.
                        # Note that this will fail on some items because only a small "thumbnail" picture is available.
                        # This is a desirable behavior: in pilot experiments, it was difficult for the model to learn from smaller images.
                        url = item["pictureURLLarge"]

                        # Download image.
                        of_p = "imgs/grade_{}_page_{}_id_{}.jpg".format(grade, page_num, idx)
                        res = requests.get(url).content
                        print(of_p)

                        # Write image to file.
                        with open(of_p, "wb") as of:
                            of.write(res)

                        # Sleep for a small period to be nice.
                        time.sleep(0.25)

                    except Exception as e:
                        # Expected exceptions will occur when a large image (pictureURLLarge) is available.

                        print(e)
                        time.sleep(0.25)

    def auto_crop_images(self):
        """
        The images have a grade at the top; automatically remove the grade.

        Parameters:
        -------
        None

        Returns:
        --------
        None

        Automatically crops each image in imgs/ directory, populating cropped_imgs/ directory.
        The current method is something of a hack that works extremely well:
            1. Crop top 1/3 off of each image.
            2. Only retain image if height is still > width.

        This works well because:
            -- Effectively removes nearly all number grades.
            -- Effectively removes bulk lots containing many cards, which typically have width > height after cropping and are thus rejected.

        A useful extension to this method would use contour detection to find and remove the red box (with white inside) containing the grade.
        """

        # Iterate over downloaded images in the imgs/ directory.
        counter = 0
        n = len(os.listdir("imgs"))
        for fname in os.listdir("imgs"):

            try:

                # Counter.
                counter += 1
                if counter % 100 == 0:
                    print("{}/{}".format(counter, n))

                # Get grade, and skip if not a legal grade.
                grade = fname.split("_")[1]
                legal_grades = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
                if grade not in legal_grades:  # not expected to occur...just for safety.
                    continue

                    # Load image as numpy array.
                p = os.path.join("imgs", fname)
                img = Image.open(p)
                img = np.array(img)

                # Crop, removing top 1/3.
                width, height, channels = img.shape
                new_top = int(height / 3)
                cropped = img[new_top:, :, :]

                of_p = os.path.join("cropped_imgs", fname)
                PIL_img = Image.fromarray(cropped)
                PIL_img.save(of_p)

            except Exception as e:

                print(e)
