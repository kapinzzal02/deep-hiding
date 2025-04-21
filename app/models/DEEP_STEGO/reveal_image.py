import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import imageio
import os
from app.models.DEEP_STEGO.Utils.preprocessing import normalize_batch, denormalize_batch
from app.utils.paths import get_model_path, get_output_path


def reveal_image(stego_image_filepath):
    try:
        # Use the utility function to get the model path
        model_path = get_model_path('DEEP_STEGO/models/reveal.h5')
        model = load_model(model_path, compile=False)

        stego_image = Image.open(stego_image_filepath).convert('RGB')

        # Resize the image to 224px*224px
        if stego_image.size != (224, 224):
            stego_image = stego_image.resize((224, 224))
            print("stego_image was resized to 224px * 224px")

        stego_image = np.array(stego_image).reshape(1, 224, 224, 3) / 255.0

        secret_image_out = model.predict([normalize_batch(stego_image)])

        secret_image_out = denormalize_batch(secret_image_out)
        secret_image_out = np.squeeze(secret_image_out) * 255.0
        secret_image_out = np.uint8(secret_image_out)

        # Use the utility function to get the output path
        output_path = get_output_path('secret_out.png')
        imageio.imsave(output_path, secret_image_out)
        print(f"Saved revealed image to {output_path}")

        return output_path
    except Exception as e:
        print(f"Error in reveal_image: {str(e)}")
        return None





