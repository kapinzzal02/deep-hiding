import cv2
import numpy as np
import torch
import os
from app.models.ESRGAN import RRDBNet_arch as arch
from app.utils.paths import get_model_path, get_output_path


def upscale_image(image_filepath):
    try:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:400"
        
        # Use the utility function to get the model path
        model_path = get_model_path('ESRGAN/models/RRDB_ESRGAN_x4.pth')
        
        # Check if CUDA is available, otherwise use CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        model = model.to(device)

        print(f'Model path {model_path}. \nUp-scaling...')

        image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image from {image_filepath}")
            
        image = image * 1.0 / 255
        image = torch.from_numpy(np.transpose(image[:, :, [2, 1, 0]], (2, 0, 1))).float()
        image_low_res = image.unsqueeze(0)
        image_low_res = image_low_res.to(device)

        with torch.no_grad():
            image_high_res = model(image_low_res).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        image_high_res = np.transpose(image_high_res[[2, 1, 0], :, :], (1, 2, 0))
        image_high_res = (image_high_res * 255.0).round()

        # Use the utility function to get the output path
        output_filepath = get_output_path('upscaled.png')
        cv2.imwrite(output_filepath, image_high_res)
        print(f"Image saved as: {output_filepath}")

        return output_filepath
    except Exception as e:
        print(f"Error in upscale_image: {str(e)}")
        return None



