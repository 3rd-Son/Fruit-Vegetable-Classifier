import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D


# Custom DepthwiseConv2D layer that ignores the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' from kwargs if present
        if "groups" in kwargs:
            del kwargs["groups"]
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config):
        # Remove 'groups' from config if present
        if "groups" in config:
            del config["groups"]
        return super().from_config(config)


# Modified model loading function
def load_model_with_custom_objects(model_path):
    custom_objects = {"DepthwiseConv2D": CustomDepthwiseConv2D}
    return load_model(model_path, custom_objects=custom_objects)


# Replace your current model loading line with this:
model = load_model_with_custom_objects("FruitModel.h5")
labels = {
    0: "apple",
    1: "banana",
    2: "beetroot",
    3: "bell pepper",
    4: "cabbage",
    5: "capsicum",
    6: "ccarrot",
    7: "cauliflower",
    8: "chilli pepper",
    9: "corn",
    10: "cucumber",
    11: "eggplant",
    12: "garlic",
    13: "ginger",
    14: "grapes",
    15: "jalepeno",
    16: "kiwi",
    17: "lemon",
    18: "lettuce",
    19: "mango",
    20: "onion",
    21: "orange",
    22: "paprika",
    23: "pear",
    24: "peas",
    25: "pineapple",
    26: "pomegranate",
    27: "potato",
    28: "raddish",
    29: "soy beans",
    30: "spinach",
    31: "sweetcorn",
    32: "sweetpotato",
    33: "tomato",
    34: "turnip",
    35: "watermelon",
}

fruits = [
    "apple",
    "banana",
    "kiwi",
    "grapes",
    "lemon",
    "mango",
    "orange",
    "pear",
    "pineapple",
    "pomegranate",
    "watermelon",
]

vegetables = [
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "jalepeno",
    "lettuce",
    "onion",
    "paprika",
    "peas",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
]


def fetch_calories(prediction):
    try:
        url = "https://www.google.com/search?q=calories in" + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, "html.parser")
        calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
        return calories
    except Exception as e:
        st.error("Sorry ! Calories not found")
        print(e)


def preprocessed_img(location):
    img = load_img(location, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res.capitalize()


def run():
    st.title("Fruits🍍 ~vegetable 🍎 Classification")
    img_file = st.file_uploader("choose an image", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img)
        save_image_path = "./upload_images/" + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = preprocessed_img(save_image_path)
            if result in vegetables:
                st.info("**Category: Vegetables**")
            else:
                st.info("**Category: Fruit**")
            st.success("**Predicted: " + result + "**")
            cal = fetch_calories(result)
            if cal:
                st.warning("**" + cal + "**")


st.write(
    """
    Dashboard created by [3rdSon](https://www.linkedin.com/in/victory-nnaji-8186231b7/), with [Streamlit](https://www.streamlit.io)
    """
)

run()
