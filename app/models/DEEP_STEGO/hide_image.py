import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import imageio
import os
from app.models.DEEP_STEGO.Utils.preprocessing import normalize_batch, denormalize_batch
from app.utils.paths import get_model_path, get_output_path


def hide_image(cover_image_filepath, secret_image_filepath):
    try:
        # Use the utility function to get the model path
        model_path = get_model_path('DEEP_STEGO/models/hide.h5')
        model = load_model(model_path)

        secret_image_in = Image.open(secret_image_filepath).convert('RGB')
        print("secret image size : ", secret_image_in.size)
        cover_image_in = Image.open(cover_image_filepath).convert('RGB')
        print("cover image size : ", cover_image_in.size)

        # Resize if image to 224px*224px
        if secret_image_in.size != (224, 224):
            secret_image_in = secret_image_in.resize((224, 224))
            print("secret_image was resized to 224px * 224px")
        if cover_image_in.size != (224, 224):
            cover_image_in = cover_image_in.resize((224, 224))
            print("cover_image was resized to 224px * 224px")

        secret_image_in = np.array(secret_image_in).reshape(1, 224, 224, 3) / 255.0
        cover_image_in = np.array(cover_image_in).reshape(1, 224, 224, 3) / 255.0

        steg_image_out = model.predict([normalize_batch(secret_image_in), normalize_batch(cover_image_in)])

        steg_image_out = denormalize_batch(steg_image_out)
        steg_image_out = np.squeeze(steg_image_out) * 255.0
        steg_image_out = np.uint8(steg_image_out)

        # Use the utility function to get the output path
        output_path = get_output_path('steg_image.png')
        imageio.imsave(output_path, steg_image_out)
        print(f"Saved steg image to {output_path}")

        return output_path
    except Exception as e:
        print(f"Error in hide_image: {str(e)}")
        return None


