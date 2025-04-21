from base64 import b64encode, b64decode
import hashlib
import os
from Crypto.Cipher import Blowfish
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from app.utils.paths import get_output_path


def encrypt(image_path, key):
    try:
        # Load the image and convert it into bytes
        with open(image_path, 'rb') as f:
            image_data = f.read()

        # Encode the image data as a base64 string
        image_data = b64encode(image_data)

        # Create a SHA-256 hash of the key
        key = hashlib.sha256(key.encode()).digest()

        # Generate a random initialization vector
        iv = get_random_bytes(Blowfish.block_size)

        # Create a Blowfish Cipher object
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)

        # Encrypt the image data
        encrypted_image_data = cipher.encrypt(pad(image_data, Blowfish.block_size))

        # Generate output path
        output_path = get_output_path(os.path.basename(image_path) + '.enc')
        
        # Save the encrypted image data to a new file
        with open(output_path, 'wb') as f:
            f.write(iv + encrypted_image_data)
            
        return output_path
    except Exception as e:
        print(f"Error in Blowfish encryption: {str(e)}")
        return None


def decrypt(encrypted_image_path, key):
    try:
        # Load the encrypted image data
        with open(encrypted_image_path, 'rb') as f:
            encrypted_image_data = f.read()

        # Create a SHA-256 hash of the key
        key = hashlib.sha256(key.encode()).digest()

        # Extract the initialization vector from the encrypted image data
        iv = encrypted_image_data[:Blowfish.block_size]
        encrypted_image_data = encrypted_image_data[Blowfish.block_size:]

        # Create a Blowfish Cipher object
        cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv)

        try:
            # Decrypt the encrypted image data
            decrypted_image_data = unpad(cipher.decrypt(encrypted_image_data), Blowfish.block_size)

            # Decode the decrypted image data from a base64 string
            decrypted_image_data = b64decode(decrypted_image_data)

            # Generate output filename
            base_name = os.path.basename(encrypted_image_path)
            base_name = base_name.replace('.enc', '').replace('.png', '').replace('.jpg', '')
            output_path = get_output_path(f"{base_name}.dec.png")
            
            # Save the decrypted image data to a new file
            with open(output_path, 'wb') as f:
                f.write(decrypted_image_data)

        except ValueError:
            print("Wrong key")
            return -1, None
            
        return 0, output_path
    except Exception as e:
        print(f"Error in Blowfish decryption: {str(e)}")
        return -1, None
