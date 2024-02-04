import pandas as pd
import re
import random
from utils import TextTransformer, ImageEncoder, ImageCaptionSequence, ImageCaptionModel

# Flickr 30k dataset
# https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
# https://www.kaggle.com/code/ghazouanihaythem/image-captioninng-using-cnn-and-lstm
# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/22_Image_Captioning.ipynb


# Import captions dataset
captions_df = pd.read_csv("flickr8k/captions.txt")

# Preprocess Captions
# Lowercase, no single letter words, only alphabet letters, [STR] and [END] tokens in head and tail.
captions_df["caption"] = captions_df["caption"].apply(lambda caption: caption.lower())
captions_df["caption"] = captions_df["caption"].apply(lambda caption: re.sub(r"[^a-z ]", "", caption))
captions_df["caption"] = captions_df["caption"].apply(lambda caption: re.sub(r"\b\w\b", "", caption))
captions_df["caption"] = captions_df["caption"].apply(lambda caption: "[STR] " + caption + " [END]")
captions_df["caption"] = captions_df["caption"].apply(lambda caption: re.sub(r"\s+", " ", caption))
print(captions_df.head(10))


# Tokenize Captions
text_transformer = TextTransformer(captions_df["caption"].to_list(), 8000)
text_encoding = text_transformer.transform(["kid running in park"])
print("Text Encoding Shape: ", text_encoding.shape)
print(type(text_encoding))


# Image Encoding
image_encoder = ImageEncoder("flickr8k/images")
image_encoding = image_encoder("667626_18933d713e.jpg")
print("Image Encoding Shape: ", image_encoding.shape)
print(type(image_encoding))


# Model Creation
caption_generator = ImageCaptionModel(image_encoder, text_transformer)
caption_generator.summary()

# Training
train_split = 0.70
rng = random.Random(42)
image_references = captions_df["image"].unique().tolist()
rng.shuffle(image_references)
train_images = image_references[:round(len(image_references)*train_split)]
test_images = image_references[round(len(image_references)*train_split):]
train_captions = captions_df[captions_df["image"].isin(train_images)].groupby('image')['caption'].apply(list).to_dict()
test_captions = captions_df[captions_df["image"].isin(test_images)].groupby('image')['caption'].apply(list).to_dict()

train_loader = ImageCaptionSequence(train_captions, 64, image_encoder, text_transformer, cache="train_loader.pkl")
test_loader = ImageCaptionSequence(test_captions, 64, image_encoder, text_transformer, cache="test_loader.pkl")
#
# caption_generator.train(train_loader, test_loader, 100)

# Testing
caption_generator.load()
result = caption_generator.predict("flickr8k/images/17273391_55cfc7d3d4.jpg")
print(result)
result = caption_generator.predict("flickr8k/images/19212715_20476497a3.jpg")
print(result)
result = caption_generator.predict("flickr8k/images/23445819_3a458716c1.jpg")
print(result)


caption_generator.evaluate_bleu_score(test_loader)
