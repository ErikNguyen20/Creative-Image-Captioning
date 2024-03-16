import json
import os.path
from typing import Optional, List, Dict, Union
from nltk.translate.bleu_score import corpus_bleu

import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

from openai import OpenAI
from tqdm import tqdm
import math
import numpy as np
from json import loads


class ImageEncoder:
    def __init__(self, image_directory: Optional[str] = None):
        self.image_directory = image_directory

        self.interpolation = "bicubic"
        self.pretrained_model = tf.keras.applications.Xception(weights="imagenet")
        self.image_size = tf.keras.backend.int_shape(self.pretrained_model.input)[1:3]
        self.encoding_model = tf.keras.models.Model(inputs=self.pretrained_model.input,
                                                    outputs=self.pretrained_model.layers[-2].output)
        self.output_size = tf.keras.backend.int_shape(self.pretrained_model.layers[-2].output)[1]

    def __call__(self, image_path: str, from_directory: bool = True):
        if self.image_directory and from_directory:
            image_path = os.path.join(self.image_directory, image_path)

        image = tf.keras.utils.load_img(image_path, color_mode="rgb", target_size=self.image_size, interpolation=self.interpolation)
        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0

        # Converts grayscale image
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        # Creates image encoding
        result = self.encoding_model.predict(np.expand_dims(image, axis=0), verbose=0)
        return result


class TextTransformer:
    def __init__(self, captions: Union[List[str], str], max_vocabulary_size: int = 8000,
                 max_sequence_length: Optional[int] = None):
        # If the captions is a filepath
        if isinstance(captions, str):
            with open(captions, 'rb') as file:
                loaded_data = pickle.load(file)

            self.vectorize_model = tf.keras.layers.TextVectorization(
                max_tokens=loaded_data['config']['max_tokens'],
                output_mode=loaded_data['config']['output_mode'],
                output_sequence_length=loaded_data['config']['output_sequence_length'])
            self.vectorize_model.set_vocabulary(loaded_data['vocabulary'])
            self.vectorize_model.set_weights(loaded_data['weights'])

            self.max_sequence_length = loaded_data['config']['output_sequence_length']
            self.vocab_size = loaded_data['config']['max_tokens']
        else:
            # If the captions is a list of strings
            if max_sequence_length is None:
                max_sequence_length = len(max(captions, key=lambda x: len(x.split())).split()) + 1

            self.vectorize_model = tf.keras.layers.TextVectorization(
                max_tokens=max_vocabulary_size,  # Set the maximum vocabulary size
                output_mode='int',  # Output integers representing token indices
                output_sequence_length=max_sequence_length,  # Set the maximum sequence length for padding
                standardize=None
            )

            # Adapt the TextVectorization layer to the captions
            self.vectorize_model.adapt(captions)
            self.max_sequence_length = max_sequence_length
            self.vocab_size = max_vocabulary_size

        self.vocabulary = self.vectorize_model.get_vocabulary()

    def transform(self, text: Union[List[str], str]):
        if type(text) is str:
            return self.vectorize_model(np.array([text]))
        return self.vectorize_model(np.array(text))

    def inverse_transform(self, embedding: Union[tf.Tensor, int]) -> Union[str, List[str]]:
        if type(embedding) is int:
            return self.vocabulary[embedding]

        if tf.rank(embedding) == 2:
            results = []
            for sequence in embedding:
                results.append(' '.join([self.vocabulary[word] for word in sequence if word != 0]))
            return results
        else:
            return ' '.join([self.vocabulary[word] for word in embedding if word != 0])

    def save(self, file_path: str):
        with open(file_path, 'wb') as file:
            pickle.dump(
                {"weights": self.vectorize_model.get_weights(),
                 "config": self.vectorize_model.get_config(),
                 "vocabulary": self.vectorize_model.get_vocabulary()}, file)


class ImageCaptionSequence(tf.keras.utils.Sequence):
    def __init__(self, image_captions: Dict[str, List[str]], batch_size: int,
                 image_encoder: ImageEncoder, vocabulary: TextTransformer, cache: Optional[str] = None):
        """
        :param image_captions: key=file_reference, value=list_of_captions
        :param batch_size:
        :param image_encoder:
        """
        self.batch_size = batch_size
        self.image_encoder = image_encoder
        self.vocabulary = vocabulary
        self._caption_iteration = 0
        self._min_captions_count = len(min(list(image_captions.values()), key=lambda x: len(x)))
        self.image_captions = image_captions

        # Applies encoding transformation
        if cache and os.path.exists(cache):
            print("Loaded Encoded Images from cache")
            with open(cache, 'rb') as file:
                self.image_encodings = pickle.load(file)
                file.close()
        else:
            self.image_encodings = {}
            for image_ref in tqdm(list(image_captions.keys()), "Encoding Images"):
                self.image_encodings[image_ref] = self.image_encoder(image_ref)
            if cache:
                with open(cache, 'wb') as file:
                    pickle.dump(self.image_encodings, file)
                    file.close()

        # Applies text embedding to captions
        self.caption_embeddings = {}
        for image_ref, captions in tqdm(image_captions.items(), "Converting to Text Embeddings"):
            self.caption_embeddings[image_ref] = vocabulary.transform(captions)

    def on_epoch_end(self):
        self._caption_iteration += 1

    def __len__(self):
        return math.ceil(len(self.image_encodings.keys()) / self.batch_size)

    def __getitem__(self, idx):
        image_references = list(self.image_encodings.keys())
        selected_caption_index = self._caption_iteration % self._min_captions_count
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(image_references))

        caption_x_batch = []
        caption_y_batch = []
        image_x_batch = []
        for image_ref in image_references[low:high]:
            for i in range(1, self.vocabulary.max_sequence_length-1):
                if self.caption_embeddings[image_ref][selected_caption_index, i-1] == 0:
                    break
                caption_x_sequence = self.caption_embeddings[image_ref][selected_caption_index, :i]
                padding_length = self.vocabulary.max_sequence_length-1 - tf.shape(caption_x_sequence)[0]
                caption_x_sequence = tf.pad(caption_x_sequence, paddings=[[0, padding_length]])

                caption_y_sequence = self.caption_embeddings[image_ref][selected_caption_index, i]
                caption_y_sequence = tf.keras.utils.to_categorical(caption_y_sequence, num_classes=self.vocabulary.vocab_size)

                caption_x_batch.append(caption_x_sequence)
                caption_y_batch.append(caption_y_sequence)
                image_x_batch.append(self.image_encodings[image_ref][0, :])

        caption_x_batch = np.array(caption_x_batch)
        image_x_batch = np.array(image_x_batch)
        caption_y_batch = np.array(caption_y_batch)

        # # image_x_batch = (batch_size, image_encoding_size)
        # image_x_batch = np.array([self.image_encodings[image_ref] for image_ref in image_references[low:high]])
        # image_x_batch = image_x_batch.squeeze(1)
        #
        # # caption_x_batch shape: (batch_size, max_tokens-1)
        # caption_x_batch = np.array(
        #     [self.caption_embeddings[image_ref][selected_caption_index, :-1]
        #      for image_ref in image_references[low:high]]
        # )
        #
        # # Caption output is shifted by 1 and one-hot-encoded
        # # caption_y_batch shape: (batch_size, max_tokens-1, vocab_size)
        # caption_y_batch = np.array(
        #     [self.caption_embeddings[image_ref][selected_caption_index, 1:]
        #      for image_ref in image_references[low:high]]
        # )
        # caption_y_batch = caption_y_batch.reshape(caption_y_batch.shape[0], caption_y_batch.shape[1], 1)
        # caption_y_batch = tf.keras.utils.to_categorical(caption_y_batch, num_classes=self.vocabulary.vocab_size)

        return (image_x_batch, caption_x_batch), caption_y_batch


class ImageCaptionModel:
    def __init__(self, image_encoder: ImageEncoder, text_transformer: TextTransformer,
                 model_file_path: str = "model.h5"):
        self.image_encoder = image_encoder
        self.text_transformer = text_transformer
        self.model_file_path = model_file_path
        self.is_loaded = False

        # for dropout 0.4 and 2 additional dense layers at end (large_bilstm)
        # BLEU 1 - gram: 0.4534395718022018
        # BLEU 2 - gram: 0.2647884037811029
        # BLEU 3 - gram: 0.17946029058000412
        # BLEU 4 - gram: 0.08316229210719871
        LSTM_size = 512
        text_embedding_size = 256

        image_input_layer = tf.keras.layers.Input(shape=(image_encoder.output_size,), name="Image Features Input")
        pass_in = tf.keras.layers.Dropout(0.4)(image_input_layer)
        pass_in = tf.keras.layers.Dense(LSTM_size, activation="swish")(pass_in)

        text_input_layer = tf.keras.layers.Input(shape=(text_transformer.max_sequence_length - 1,),
                                                 name="Text Tokens Input")
        text_embedding_layer = tf.keras.layers.Embedding(input_dim=text_transformer.vocab_size,
                                                         output_dim=text_embedding_size, mask_zero=True)(text_input_layer)

        x1 = tf.keras.layers.LSTM(LSTM_size, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(text_embedding_layer)
        x1 = tf.keras.layers.Add()([pass_in, x1])
        x1 = tf.keras.layers.LSTM(LSTM_size, return_sequences=False, recurrent_dropout=0.2, dropout=0.2)(x1)

        x2 = tf.keras.layers.LSTM(LSTM_size, return_sequences=True, go_backwards=True, recurrent_dropout=0.2, dropout=0.2)(text_embedding_layer)
        x2 = tf.keras.layers.Add()([pass_in, x2])
        x2 = tf.keras.layers.LSTM(LSTM_size, return_sequences=False, go_backwards=True, recurrent_dropout=0.2, dropout=0.2)(x2)

        x = tf.keras.layers.Concatenate(axis=-1)([x1, x2])
        x = tf.keras.layers.Dense(LSTM_size * 4, activation="swish")(x)
        x = tf.keras.layers.Dense(LSTM_size * 4, activation="swish")(x)
        output_layer = tf.keras.layers.Dense(text_transformer.vocab_size, activation="softmax")(x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.caption_model = tf.keras.models.Model(inputs=[image_input_layer, text_input_layer], outputs=output_layer)
        self.caption_model.compile(optimizer=optimizer, loss="categorical_crossentropy")

    def summary(self):
        self.caption_model.summary()

    def train(self, train_loader: ImageCaptionSequence, validation_loader: ImageCaptionSequence, epochs: int):
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_file_path, save_weights_only=True,
                                                        save_best_only=True, verbose=1)
        earlystopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        history = self.caption_model.fit(train_loader, epochs=epochs,
                                         validation_data=validation_loader, callbacks=[checkpoint, earlystopping])
        self.is_loaded = True

        plt.figure(figsize=(20, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("loss_plot.png")
        # plt.show()

    def load(self):
        if self.is_loaded:
            return

        self.caption_model.load_weights(self.model_file_path)
        self.is_loaded = True

    def predict(self, image_path: str, from_directory=False):
        if not self.is_loaded:
            print("Model not trained or loaded!")
            return None

        image_vector = self.image_encoder(image_path, from_directory=from_directory)
        caption_embedding = np.zeros(shape=(1, self.text_transformer.max_sequence_length - 1), dtype=np.int32)

        token_count = 0
        current_token = ""
        output_caption = []
        while token_count < self.text_transformer.max_sequence_length-1 and current_token != "[END]":
            output_caption.append(current_token)

            prediction = self.caption_model.predict((image_vector, caption_embedding), verbose=0)
            token_int = int(np.argmax(prediction[0, :]))

            caption_embedding[0, token_count] = token_int
            current_token = self.text_transformer.inverse_transform(token_int)
            token_count += 1

        return ' '.join(output_caption).strip()

    def evaluate_bleu_score(self, loader: ImageCaptionSequence):
        actual = []
        predicted = []

        for image_ref, captions in tqdm(loader.image_captions.items(), "Predicting for Test Images"):
            caption_predict = self.predict(image_ref, from_directory=True).split()
            actual_caption = [caption.split() for caption in captions]

            actual.append(actual_caption)
            predicted.append(caption_predict)

        # Ranges from [0.0, 1.0], where 1.0 is best
        print("BLEU 1-gram: ", corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        print("BLEU 2-gram: ", corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
        print("BLEU 3-gram: ", corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
        print("BLEU 4-gram: ", corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


class ImageCaptionPipline:
    def __init__(self, text_transformer_file_path: str, model_file_path: str, system_prompt_file_path: str = "system_prompt.txt"):
        self.text_transformer = TextTransformer(text_transformer_file_path)
        self.image_encoder = ImageEncoder()
        self.caption_generator = ImageCaptionModel(self.image_encoder,
                                                   self.text_transformer,
                                                   model_file_path=model_file_path)
        self.caption_generator.load()
        self.gpt_client = OpenAI()

        self.system_prompt = None

        # Read message from text file
        with open(system_prompt_file_path, "r") as file:
            self.system_prompt = file.read().strip()

    def predict(self, image_path: str, context: str = None, options: int = 1) -> List[str]:
        options = 5 if options > 5 else options

        generated_caption = self.caption_generator.predict(image_path)
        if options <= 1 or self.system_prompt is None:
            return [self.caption_generator.predict(image_path)]
        else:
            context = context if context is not None else "None"
            user_input = f"(Caption): {generated_caption}\n(Context): {context}\n(Num Caption Responses): {options}"

            json_response = [generated_caption]
            try:
                response = self.gpt_client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_input}
                    ]
                )
                json_data = loads(response.choices[0].message.content)
                json_response = json_data["captions"]
            except Exception as e:
                print("Error: ", e)

            print(json_response, type(json_response))
            return json_response


