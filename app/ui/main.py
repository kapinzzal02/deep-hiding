import os
import os
import sys

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QFile, QTextStream
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, \
    QMessageBox, QFileDialog, QDialog, QRadioButton, QButtonGroup

from app.models.DEEP_STEGO.hide_image import hide_image
from app.models.DEEP_STEGO.reveal_image import reveal_image
from app.models.ESRGAN import RRDBNet_arch as arch
from app.models.StableDiffusionAPI import StableDiffusionV2
from app.models.encryption import aes, blowfish
from app.ui.components.backgroundwidget import BackgroundWidget
from app.ui.components.customtextbox import CustomTextBox, CustomTextBoxForImageGen
from app.utils.paths import get_asset_path, get_output_path


class MainAppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # vars
        self.download_genimage_button = None
        self.gen_image_label = None
        self.text_desc_box = None
        self.main_content = None
        self.blowfish_radio_dec = None
        self.aes_radio_dec = None
        self.key_text_box_of_dec = None
        self.enc_filepath = None
        self.dec_display_label = None
        self.download_dec_button = None
        self.dec_img_text_label = None
        self.key_text_box = None
        self.blowfish_radio = None
        self.aes_radio = None
        self.image_tobe_enc_filepath = None
        self.download_enc_button = None
        self.enc_display_label = None
        self.container_image_filepath = None
        self.secret_out_display_label = None
        self.container_display_label = None
        self.download_revealed_secret_image_button = None
        self.download_steg_button = None
        self.secret_image_filepath = None
        self.cover_image_filepath = None
        self.steg_display_label = None
        self.secret_display_label = None
        self.cover_display_label = None
        self.low_res_image_text_label = None
        self.image_label = None
        self.low_res_image_filepath = None
        self.download_HR_button = None
        # Add variables for the new radio buttons
        self.cnn_radio_hide = None
        self.steganogan_radio_hide = None
        self.cnn_radio_reveal = None
        self.steganogan_radio_reveal = None


        # Set window properties
        self.setWindowTitle("Deep Hiding")
        self.setGeometry(200, 200, 1400, 800)
        self.setWindowIcon(QIcon(get_asset_path("icon.ico")))
        self.setStyleSheet("""
            background-color: #1a0033; 
            color: #ffffff;
            QLabel { color: #ffffff; }
            QPushButton { color: #ffffff; background-color: #430693; border: 1px solid #8844cc; border-radius: 4px; padding: 5px; }
            QPushButton:hover { background-color: #5a1cb3; }
            QRadioButton { color: #ffffff; }
        """)
        self.setFixedSize(self.size())
        # self.setWindowFlags(Qt.FramelessWindowHint)

        # Set up the main window layout
        main_layout = QHBoxLayout()

        # Create the side navigation bar
        side_navigation = BackgroundWidget()
        side_navigation.set_background_image(get_asset_path("components_backgrounds/sidebar_bg.jpg"))
        side_navigation.setObjectName("side_navigation")
        side_navigation.setFixedWidth(200)
        side_layout = QVBoxLayout()

        # label for logo
        logo_label = QLabel()
        logo_pixmap = QPixmap("logo.png").scaled(50, 50, Qt.KeepAspectRatio)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        # label for logo name
        name_label = QLabel()
        name_label.setText("<br><h2>Deep Hiding</h2><br>")
        name_label.setStyleSheet("color: #ffffff;")
        name_label.setAlignment(Qt.AlignCenter)

        # Create buttons for each option
        encryption_button = QPushButton("Encryption")
        decryption_button = QPushButton("Decryption")
        image_hiding_button = QPushButton("Image Hide")
        image_reveal_button = QPushButton("Image Reveal")
        super_resolution_button = QPushButton("Super Resolution")
        # Removed the imagegen_button

        # Connect button signals to their corresponding slots
        encryption_button.clicked.connect(self.show_encryption_page)
        decryption_button.clicked.connect(self.show_decryption_page)
        image_hiding_button.clicked.connect(self.show_image_hiding_page)
        image_reveal_button.clicked.connect(self.show_reveal_page)
        super_resolution_button.clicked.connect(self.show_super_resolution_page)
        # Removed the imagegen_button connection

        # Add buttons to the side navigation layout
        side_layout.addWidget(logo_label)
        side_layout.addWidget(name_label)
        side_layout.addWidget(image_hiding_button)
        side_layout.addWidget(encryption_button)
        side_layout.addWidget(decryption_button)
        side_layout.addWidget(image_reveal_button)
        side_layout.addWidget(super_resolution_button)
        # Removed the imagegen_button from layout

        # Add student names label
        student_names_label = QLabel(
            "<b>Project By:</b><br>"
            "Aryan Baruah (21BCI0199)<br>"
            "Kapinzzal Kashyap (21BCI0396)<br>"
            "Arindam Saikia (21BCE0692)"
        )
        student_names_label.setStyleSheet("color: #ffffff; font-size: 13px; padding-top: 20px; padding-bottom: 20 px") # Added padding top for spacing
        student_names_label.setAlignment(Qt.AlignCenter)
        student_names_label.setWordWrap(True) # Ensure text wraps within the sidebar width
        side_layout.addWidget(student_names_label)


        # Add a logout button
        logout_button = QPushButton("Exit")
        logout_button.setObjectName("logout_button")
        logout_button.clicked.connect(self.logout)
        side_layout.addStretch() # Add stretch *after* the student names
        side_layout.addWidget(logout_button)

        # Set the layout for the side navigation widget
        side_navigation.setLayout(side_layout)

        # Create the main content area
        self.main_content = BackgroundWidget()
        self.main_content.setObjectName("main_content")
        self.main_content.set_background_image(get_asset_path("components_backgrounds/main_window_welcome_bg.png"))
        self.main_layout = QVBoxLayout()
        self.main_content.setLayout(self.main_layout)

        # Add the side navigation and main content to the main window layout
        main_layout.addWidget(side_navigation)
        main_layout.addWidget(self.main_content)

        # Set the main window layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def show_encryption_page(self):
        self.main_content.set_background_image("/assets/components_backgrounds/main_window_bg.png")
        self.image_tobe_enc_filepath = None
        self.key_text_box = None
        self.enc_img_text_label = None
        # Clear the main window layout
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>Image Encryption</H2>")
        title_label.setStyleSheet("font-size: 24px; color: #ffffff;")
        title_label.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(title_label)

        # label layout
        label_layout = QHBoxLayout()

        method_text_label = QLabel("Select encryption method:")
        method_text_label.setAlignment(Qt.AlignVCenter)
        method_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(method_text_label)

        self.enc_img_text_label = QLabel("Select Image to be Encrypted:")
        self.enc_img_text_label.setAlignment(Qt.AlignCenter)
        self.enc_img_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(self.enc_img_text_label)

        label_layout_widget = QWidget()
        label_layout_widget.setLayout(label_layout)
        self.main_layout.addWidget(label_layout_widget)

        # Image  display layout
        image_display_layout = QHBoxLayout()

        radio_layout = QVBoxLayout()
        radio_layout.setAlignment(Qt.AlignLeft)
        self.aes_radio = QRadioButton("AES Encryption")
        self.aes_radio.setToolTip("Widely adopted symmetric-key block cipher with strong security and flexibility")

        self.blowfish_radio = QRadioButton("Blowfish Encryption")
        self.blowfish_radio.setToolTip("Fast, efficient symmetric-key block cipher with versatile key lengths")

        encryption_group = QButtonGroup()
        encryption_group.addButton(self.aes_radio)
        encryption_group.addButton(self.blowfish_radio)
        radio_layout.addWidget(self.blowfish_radio)
        radio_layout.addWidget(self.aes_radio)

        key_text_label = QLabel("<br><br><br>Enter the secret key")
        key_text_label.setStyleSheet("font-size: 18px; color: #ffffff; font-weight: bold;")
        radio_layout.addWidget(key_text_label)

        self.key_text_box = CustomTextBox()
        self.key_text_box.setFixedWidth(300)
        radio_layout.addWidget(self.key_text_box)

        radio_layout_widget = QWidget()
        radio_layout_widget.setLayout(radio_layout)
        image_display_layout.addWidget(radio_layout_widget)

        self.enc_display_label = QLabel()
        # self.enc_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("assets/dummy_images/image_dummy.png")
        self.enc_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_display_layout.addWidget(self.enc_display_label)

        image_display_layout_widget = QWidget()
        image_display_layout_widget.setLayout(image_display_layout)
        self.main_layout.addWidget(image_display_layout_widget)

        # button layout
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_encryption_page())
        button_layout.addWidget(clear_button)

        browse_enc_button = QPushButton("Browse image")
        browse_enc_button.clicked.connect(lambda: self.select_enc_image(self.enc_display_label))
        button_layout.addWidget(browse_enc_button)

        encrypt_button = QPushButton("Encrypt")
        encrypt_button.clicked.connect(lambda: self.perform_encryption(self.image_tobe_enc_filepath))
        button_layout.addWidget(encrypt_button)

        self.download_enc_button = QPushButton("Download🔽")
        self.download_enc_button.setEnabled(False)
        self.download_enc_button.clicked.connect(lambda: self.download_image())
        button_layout.addWidget(self.download_enc_button)

        button_layout_widget = QWidget()
        button_layout_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_layout_widget)

    def show_decryption_page(self):
        self.main_content.set_background_image("assets/components_backgrounds/main_window_bg.png")
        self.key_text_box_of_dec = None
        # Clear the main window layout
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>Image Decryption</H2>")
        title_label.setStyleSheet("font-size: 24px; color: #ffffff;")
        title_label.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(title_label)

        # label layout
        label_layout = QHBoxLayout()

        method_text_label = QLabel("Select Decryption method:")
        method_text_label.setAlignment(Qt.AlignVCenter)
        method_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(method_text_label)

        self.dec_img_text_label = QLabel("Select the file to be decrypted:")
        self.dec_img_text_label.setAlignment(Qt.AlignCenter)
        self.dec_img_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(self.dec_img_text_label)

        label_layout_widget = QWidget()
        label_layout_widget.setLayout(label_layout)
        self.main_layout.addWidget(label_layout_widget)

        # Image  display layout
        image_display_layout = QHBoxLayout()

        radio_layout = QVBoxLayout()
        radio_layout.setAlignment(Qt.AlignLeft)
        self.aes_radio_dec = QRadioButton("AES Decryption")
        self.aes_radio_dec.setToolTip("Widely adopted symmetric-key block cipher with strong security and flexibility")

        self.blowfish_radio_dec = QRadioButton("Blowfish Decryption")
        self.blowfish_radio_dec.setToolTip("Fast, efficient symmetric-key block cipher with versatile key lengths")

        encryption_group = QButtonGroup()
        encryption_group.addButton(self.aes_radio_dec)
        encryption_group.addButton(self.blowfish_radio_dec)
        radio_layout.addWidget(self.blowfish_radio_dec)
        radio_layout.addWidget(self.aes_radio_dec)

        key_text_label = QLabel("<br><br><br>Enter the secret key")
        key_text_label.setStyleSheet("font-size: 18px; color: #ffffff; font-weight: bold;")
        radio_layout.addWidget(key_text_label)

        self.key_text_box_of_dec = CustomTextBox()
        self.key_text_box_of_dec.setFixedWidth(300)
        radio_layout.addWidget(self.key_text_box_of_dec)

        radio_layout_widget = QWidget()
        radio_layout_widget.setLayout(radio_layout)
        image_display_layout.addWidget(radio_layout_widget)

        self.dec_display_label = QLabel()
        self.dec_display_label.setAlignment(Qt.AlignLeft)
        pixmap = QPixmap("assets/dummy_images/image_dummy.png")
        self.dec_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_display_layout.addWidget(self.dec_display_label)

        image_display_layout_widget = QWidget()
        image_display_layout_widget.setLayout(image_display_layout)
        self.main_layout.addWidget(image_display_layout_widget)

        # button layout
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_decryption_page())
        button_layout.addWidget(clear_button)

        browse_enc_button = QPushButton("Browse encrypted file")
        browse_enc_button.clicked.connect(lambda: self.select_dec_image(self.dec_display_label))
        button_layout.addWidget(browse_enc_button)

        decrypt_button = QPushButton("Decrypt")
        decrypt_button.clicked.connect(lambda: self.perform_decryption(self.enc_filepath))
        button_layout.addWidget(decrypt_button)

        self.download_dec_button = QPushButton("Download🔽")
        self.download_dec_button.setEnabled(False)
        self.download_dec_button.clicked.connect(lambda: self.download_image())
        button_layout.addWidget(self.download_dec_button)

        button_layout_widget = QWidget()
        button_layout_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_layout_widget)

    def show_image_hiding_page(self):
        self.main_content.set_background_image("components_backgrounds/main_window_bg.png") # Use get_asset_path
        self.secret_image_filepath = None
        self.cover_image_filepath = None
        # Clear the main window layout
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>Image Hiding</H2>") # Simplified title
        title_label.setStyleSheet("font-size: 24px; color: #ffffff;")
        title_label.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(title_label)

        # --- Add Model Selection Radio Buttons ---
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-size: 16px; color: #c6c6c6; font-weight: bold;")
        self.cnn_radio_hide = QRadioButton("CNN")
        self.cnn_radio_hide.setToolTip("Use Convolutional Neural Network model for hiding.")
        self.cnn_radio_hide.setChecked(True) # Default selection
        self.steganogan_radio_hide = QRadioButton("SteganoGAN (Dense)") # Changed text here
        self.steganogan_radio_hide.setToolTip("Use SteganoGAN model for hiding.")

        model_group_hide = QButtonGroup(self) # Pass parent to avoid early garbage collection
        model_group_hide.addButton(self.cnn_radio_hide)
        model_group_hide.addButton(self.steganogan_radio_hide)

        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.cnn_radio_hide)
        model_selection_layout.addWidget(self.steganogan_radio_hide)
        model_selection_layout.addStretch() # Push buttons to the left

        model_selection_widget = QWidget()
        model_selection_widget.setLayout(model_selection_layout)
        self.main_layout.addWidget(model_selection_widget)
        # --- End Model Selection ---


        # label layout
        label_layout = QHBoxLayout()
        cover_text_label = QLabel("Select cover image:")
        cover_text_label.setAlignment(Qt.AlignCenter)
        cover_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(cover_text_label)

        secret_text_label = QLabel("Select secret image:")
        secret_text_label.setAlignment(Qt.AlignCenter)
        secret_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(secret_text_label)

        steg_text_label = QLabel("Generated steg image:")
        steg_text_label.setAlignment(Qt.AlignCenter)
        steg_text_label.setStyleSheet("font-size: 16px; color: #00ff00; margin-bottom: 10px; font-weight: bold;")
        label_layout.addWidget(steg_text_label)

        label_layout_widget = QWidget()
        label_layout_widget.setLayout(label_layout)
        self.main_layout.addWidget(label_layout_widget)

        # Image  display layout
        image_display_layout = QHBoxLayout()
        self.cover_display_label = QLabel()
        self.cover_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(get_asset_path("dummy_images/cover_image_dummy.png")) # Use get_asset_path
        self.cover_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_display_layout.addWidget(self.cover_display_label)

        self.secret_display_label = QLabel()
        self.secret_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(get_asset_path("dummy_images/secret_image_dummy.png")) # Use get_asset_path
        self.secret_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_display_layout.addWidget(self.secret_display_label)

        self.steg_display_label = QLabel()
        self.steg_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(get_asset_path("dummy_images/steg_image_dummy.png")) # Use get_asset_path
        self.steg_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_display_layout.addWidget(self.steg_display_label)

        image_display_layout_widget = QWidget()
        image_display_layout_widget.setLayout(image_display_layout)
        self.main_layout.addWidget(image_display_layout_widget)

        # button layout
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_image_hiding_page())
        button_layout.addWidget(clear_button)

        browse_cover_button = QPushButton("Browse cover image")
        browse_cover_button.clicked.connect(lambda: self.select_cover_image(self.cover_display_label))
        button_layout.addWidget(browse_cover_button)

        browse_secret_button = QPushButton("Browse secret image")
        browse_secret_button.clicked.connect(lambda: self.select_secret_image(self.secret_display_label))
        button_layout.addWidget(browse_secret_button)

        hide_button = QPushButton("Hide")
        hide_button.clicked.connect(lambda: self.perform_hide(self.cover_image_filepath, self.secret_image_filepath))
        button_layout.addWidget(hide_button)

        self.download_steg_button = QPushButton("Download steg image🔽")
        self.download_steg_button.setEnabled(False)
        self.download_steg_button.clicked.connect(lambda: self.download_image())
        button_layout.addWidget(self.download_steg_button)

        button_layout_widget = QWidget()
        button_layout_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_layout_widget)

    def show_reveal_page(self):
        self.main_content.set_background_image("components_backgrounds/main_window_bg.png") # Use get_asset_path
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>Image Reveal</H2>") # Simplified title
        title_label.setStyleSheet("font-size: 24px; color: #ffffff;")
        title_label.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(title_label)

        # --- Add Model Selection Radio Buttons ---
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-size: 16px; color: #c6c6c6; font-weight: bold;")
        self.cnn_radio_reveal = QRadioButton("CNN")
        self.cnn_radio_reveal.setToolTip("Use Convolutional Neural Network model for revealing.")
        self.cnn_radio_reveal.setChecked(True) # Default selection
        self.steganogan_radio_reveal = QRadioButton("SteganoGAN (Dense)") # Changed text here
        self.steganogan_radio_reveal.setToolTip("Use SteganoGAN model for revealing.")

        model_group_reveal = QButtonGroup(self) # Pass parent
        model_group_reveal.addButton(self.cnn_radio_reveal)
        model_group_reveal.addButton(self.steganogan_radio_reveal)

        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.cnn_radio_reveal)
        model_selection_layout.addWidget(self.steganogan_radio_reveal)
        model_selection_layout.addStretch() # Push buttons to the left

        model_selection_widget = QWidget()
        model_selection_widget.setLayout(model_selection_layout)
        self.main_layout.addWidget(model_selection_widget)
        # --- End Model Selection ---

        # image text layout
        image_text_layout = QHBoxLayout()
        container_text_label = QLabel("Select steg image:")
        container_text_label.setAlignment(Qt.AlignCenter)
        container_text_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        image_text_layout.addWidget(container_text_label)

        secret_out_text_label = QLabel("Revealed secret image:")
        secret_out_text_label.setAlignment(Qt.AlignCenter)
        secret_out_text_label.setStyleSheet("font-size: 16px; color: #00ff00; margin-bottom: 10px; font-weight: bold;")
        image_text_layout.addWidget(secret_out_text_label)

        image_text_layout_widget = QWidget()
        image_text_layout_widget.setLayout(image_text_layout)
        self.main_layout.addWidget(image_text_layout_widget)
        
        # Image display layout
        image_layout = QHBoxLayout()
        self.container_display_label = QLabel()
        self.container_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(get_asset_path("dummy_images/steg_image_dummy.png")) # Use get_asset_path
        self.container_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_layout.addWidget(self.container_display_label)
        
        self.secret_out_display_label = QLabel()
        self.secret_out_display_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(get_asset_path("dummy_images/secret_image_dummy.png")) # Use get_asset_path
        self.secret_out_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
        image_layout.addWidget(self.secret_out_display_label)

        image_layout_widget = QWidget()
        image_layout_widget.setLayout(image_layout)
        self.main_layout.addWidget(image_layout_widget)

        # button layout
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_reveal_page())
        button_layout.addWidget(clear_button)

        browse_cover_button = QPushButton("Browse steg image")
        browse_cover_button.clicked.connect(lambda: self.select_container_image(self.container_display_label))
        button_layout.addWidget(browse_cover_button)

        reveal_button = QPushButton("Reveal")
        # Corrected the attribute name in the lambda function below
        reveal_button.clicked.connect(lambda: self.perform_reveal(self.container_image_filepath)) 
        button_layout.addWidget(reveal_button)

        self.download_revealed_secret_image_button = QPushButton("Download🔽")
        self.download_revealed_secret_image_button.setEnabled(False)
        self.download_revealed_secret_image_button.clicked.connect(lambda: self.download_image())
        button_layout.addWidget(self.download_revealed_secret_image_button)

        button_layout_widget = QWidget()
        button_layout_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_layout_widget)

    def show_super_resolution_page(self):
        self.main_content.set_background_image("components_backgrounds/main_window_bg.png")
        self.low_res_image_filepath = None
        # Clear the main window layout
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>Image Super Resolution using ESRGAN</H2>")
        title_label.setAlignment(Qt.AlignTop)
        title_label.setStyleSheet("font-size: 24px; color: #ffffff; margin-bottom: 20px;")
        self.main_layout.addWidget(title_label)

        # Low resolution image selection
        low_res_label = QLabel("Select Low Resolution Image:")
        low_res_label.setAlignment(Qt.AlignCenter)
        low_res_label.setStyleSheet("font-size: 16px; color: #c6c6c6; margin-bottom: 10px; font-weight: bold;")
        self.main_layout.addWidget(low_res_label)

        # image display
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("assets/dummy_images/lr_image_dummy.png")
        image_label.setPixmap(pixmap.scaled(384, 384, Qt.KeepAspectRatio))
        self.main_layout.addWidget(image_label)

        # defining button layout
        button_layout = QHBoxLayout()

        # Browse button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_super_resolution_page())
        button_layout.addWidget(clear_button)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self.select_low_resolution_image(image_label))
        button_layout.addWidget(browse_button)

        # Up-scale button
        upscale_button = QPushButton("UP-SCALE")
        upscale_button.clicked.connect(lambda: self.upscaleImage(image_label))
        button_layout.addWidget(upscale_button)

        # Download button
        download_button = QPushButton("Download🔽")
        download_button.setObjectName("download_button")
        download_button.setEnabled(False)
        download_button.clicked.connect(self.download_image)
        button_layout.addWidget(download_button)

        # add the button layout to the main layout
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_widget)

        # Set the image labels as attributes
        self.low_res_image_text_label = low_res_label
        self.image_label = image_label
        self.download_HR_button = download_button

    def show_imagegen_page(self):
        self.main_content.set_background_image("components_backgrounds/main_window_bg.png")
        self.clear_main_layout()

        # Add content to the super resolution page
        title_label = QLabel("<H2>StackGAN : Image generation using GenAI</H2>")
        title_label.setStyleSheet("font-size: 24px; color: #ffffff;")
        title_label.setAlignment(Qt.AlignTop)
        self.main_layout.addWidget(title_label)

        # image display
        self.gen_image_label = QLabel()
        self.gen_image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap("assets/dummy_images/imagegen_dummy.png")
        self.gen_image_label.setPixmap(pixmap.scaled(384, 384, Qt.KeepAspectRatio))
        self.main_layout.addWidget(self.gen_image_label)

        # Text description enter prompt
        image_desc_label = QLabel("Your image description here👇🏻")
        image_desc_label.setAlignment(Qt.AlignCenter)
        image_desc_label.setStyleSheet("font-size: 16px; color: #c6c6c6; font-weight: semi-bold;")
        self.main_layout.addWidget(image_desc_label)

        # defining layout for textbox and i'm feeling lucky
        textbox_layout = QHBoxLayout()

        # Create a QlineEdit widget
        self.text_desc_box = CustomTextBoxForImageGen()
        self.text_desc_box.setPlaceholderText("Type here....")
        textbox_layout.addWidget(self.text_desc_box)

        # i'm feeling lucky button
        lucky_button = QPushButton("I'm feeling lucky")
        lucky_button.setObjectName("luckyButton")
        lucky_button.setToolTip("Selects a random text prompt")
        lucky_button.clicked.connect(lambda: self.show_random_text(self.text_desc_box))
        textbox_layout.addWidget(lucky_button)

        # add the textbox layout to the main layout
        textbox_widget = QWidget()
        textbox_widget.setLayout(textbox_layout)
        self.main_layout.addWidget(textbox_widget)

        # defining button layout
        button_layout = QHBoxLayout()

        # clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.show_imagegen_page())
        button_layout.addWidget(clear_button)

        # generate button
        generate_button = QPushButton("Generate✨")
        generate_button.setObjectName("gen_button")
        generate_button.clicked.connect(lambda: self.generateImage(self.gen_image_label))
        button_layout.addWidget(generate_button)

        # Download button
        download_button = QPushButton("Download🔽")
        download_button.setObjectName("download_button")
        download_button.setEnabled(False)
        download_button.clicked.connect(self.download_image)
        button_layout.addWidget(download_button)
        self.download_genimage_button = download_button

        # add the button layout to the main layout
        button_widget = QWidget()
        button_widget.setLayout(button_layout)
        self.main_layout.addWidget(button_widget)

    def generateImage(self, label):
        print("Image gen")
        gen_image_path = 'C:/Users/asirw/PycharmProjects/InvisiCipher/app/generated_image.png'
        if self.text_desc_box.text() == "":
            QMessageBox.information(self, "Generation Error", "Please enter a description")
            return
        try:
            image = StableDiffusionV2.generate(text_prompt=self.text_desc_box.text())
            image.save(gen_image_path)
            pixmap = QPixmap(gen_image_path)
            label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.download_genimage_button.setEnabled(True)
        except:
            QMessageBox.critical(self, "Generating error", "Failed to generate the image")

    def show_random_text(self, textbox):
        import json, random
        with open("E:/Deep Hiding/app/ui/assets/json/lucky.json", "r") as f:
            text = random.choice(json.load(f))
            textbox.setText(text)
    def select_low_resolution_image(self, label):
        file_dialog = QFileDialog()
        low_res_image_filepath, _ = file_dialog.getOpenFileName(self, "Select Low Resolution Image")
        if low_res_image_filepath:
            self.low_res_image_filepath = low_res_image_filepath
            pixmap = QPixmap(low_res_image_filepath)
            label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))

    def upscaleImage(self, image_label):
        if self.low_res_image_filepath is None:
            QMessageBox.information(self, "Upscaling Error", "Please select the low-resolution image first.")
            return
        try:
            from app.models.ESRGAN.upscale_image import upscale_image
            output_path = upscale_image(self.low_res_image_filepath)
            
            if output_path:
                pixmap = QPixmap(output_path)
                image_label.setPixmap(pixmap.scaled(384, 384, Qt.KeepAspectRatio))
                self.low_res_image_text_label.setText("High Res Image:")
                self.low_res_image_text_label.setStyleSheet(
                    "font-size: 16px; color: #00ff00; margin-bottom: 10px; font-weight: bold;")
                self.download_HR_button.setEnabled(True)
                self.current_output_file = output_path
            else:
                QMessageBox.critical(self, "Upscaling Error", "Failed to upscale the image.")
        except Exception as e:
            QMessageBox.critical(self, "Upscaling Error", f"Failed to upscale the image: {str(e)}")

    def download_image(self):
        # Implement the logic to download the high resolution image
        QMessageBox.information(self, "Download", "Downloaded the image...")

    def clear_main_layout(self):
        # Remove all widgets from the main layout
        while self.main_layout.count():
            child = self.main_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def select_cover_image(self, label):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select cover Image")
        if filepath:
            self.cover_image_filepath = filepath
            pixmap = QPixmap(filepath)
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def select_secret_image(self, label):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select secret Image")
        if filepath:
            self.secret_image_filepath = filepath
            pixmap = QPixmap(filepath)
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def select_container_image(self, label):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select secret Image")
        if filepath:
            self.container_image_filepath = filepath
            pixmap = QPixmap(filepath)
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def select_enc_image(self, label):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select Image")
        if filepath:
            self.image_tobe_enc_filepath = filepath
            pixmap = QPixmap(filepath)
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def select_dec_image(self, label):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Select enc file")
        if filepath:
            self.enc_filepath = filepath
            pixmap = QPixmap("assets/dummy_images/locked_image_dummy.png")
            label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))

    def perform_hide(self, cover_filepath, secret_filepath):
        if cover_filepath is None or secret_filepath is None:
            QMessageBox.information(self, "Hiding Error", "Please select the images first.")
            return
        try:
            steg_image_path = hide_image(cover_filepath, secret_filepath)
            if steg_image_path:
                # Check which model is selected and resize accordingly
                if self.steganogan_radio_hide.isChecked():
                    # Resize to 1024x1024 for SteganoGAN
                    img = cv2.imread(steg_image_path)
                    if img is None:
                        raise Exception(f"Failed to read intermediate image: {steg_image_path}")
                    resized_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4) # Changed target size
                    resized_path = steg_image_path.replace('.png', '_gan.png') 
                    cv2.imwrite(resized_path, resized_img)
                    
                    # Delete the original 224x224 image
                    try:
                        os.remove(steg_image_path)
                    except OSError as e:
                        print(f"Error deleting intermediate file {steg_image_path}: {e}") # Log error but continue

                    self.current_output_file = resized_path
                    pixmap = QPixmap(resized_path)
                else:
                    # Use original 224x224 for CNN
                    self.current_output_file = steg_image_path
                    pixmap = QPixmap(steg_image_path)
                
                # Display the image
                self.steg_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
                self.download_steg_button.setEnabled(True)
            else:
                QMessageBox.critical(self, "Hiding Error", "Failed to hide the image.")
        except Exception as e:
            QMessageBox.critical(self, "Hiding Error", f"Failed to hide the image: {str(e)}")

    def perform_reveal(self, container_filepath):
        if container_filepath is None:
            QMessageBox.information(self, "Reveal Error", "Please select a stego image first.")
            return

        # --- Updated Warning Check Block ---
        filename_lower = container_filepath.lower()
        if self.cnn_radio_reveal.isChecked() and filename_lower.endswith('_gan.png'):
            QMessageBox.warning(self, "Model Mismatch Warning", 
                                "You have selected the CNN model, but the chosen image appears to be generated by SteganoGAN (Dense) ('_gan.png').\n\n"
                                "Revealing might not work correctly. It's recommended to use the SteganoGAN (Dense) model for this image.")
        elif self.steganogan_radio_reveal.isChecked() and not filename_lower.endswith('_gan.png'): # Added this check
             QMessageBox.warning(self, "Model Mismatch Warning", 
                                "You have selected the SteganoGAN (Dense) model, but the chosen image does not have the typical '_gan.png' suffix.\n\n"
                                "If this image was generated using the CNN model, revealing might not work correctly. It's recommended to use the CNN model for this image.")
        # --- End Warning Check Block ---
            
        try:
            secret_out_path = reveal_image(container_filepath)
            if secret_out_path:
                # Check which model is selected and resize accordingly
                if self.steganogan_radio_reveal.isChecked():
                    # Resize to 1024x1024 for SteganoGAN
                    img = cv2.imread(secret_out_path)
                    if img is None:
                         raise Exception(f"Failed to read intermediate image: {secret_out_path}")
                    resized_img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4) # Changed target size
                    resized_path = secret_out_path.replace('.png', '_gan.png') 
                    cv2.imwrite(resized_path, resized_img)

                    # Delete the original 224x224 image
                    try:
                        os.remove(secret_out_path)
                    except OSError as e:
                        print(f"Error deleting intermediate file {secret_out_path}: {e}") # Log error but continue
                        
                    self.current_output_file = resized_path
                    pixmap = QPixmap(resized_path)
                else:
                    # Use original 224x224 for CNN
                    self.current_output_file = secret_out_path
                    pixmap = QPixmap(secret_out_path)
                
                # Display the image
                self.secret_out_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
                self.download_revealed_secret_image_button.setEnabled(True)
            else:
                QMessageBox.critical(self, "Reveal Error", "Failed to reveal the secret image.")
        except Exception as e:
            QMessageBox.critical(self, "Reveal Error", f"Failed to reveal the secret image: {str(e)}")

    def perform_encryption(self, image_filepath):
        if image_filepath is None:
            QMessageBox.information(self, "Encryption Error", "Please select an image first.")
            return
        
        if not self.aes_radio.isChecked() and not self.blowfish_radio.isChecked():
            QMessageBox.information(self, "Encryption Error", "Please select an encryption method.")
            return
            
        key = self.key_text_box.text()
        if not key:
            QMessageBox.information(self, "Encryption Error", "Please enter a secret key.")
            return
            
        try:
            if self.aes_radio.isChecked():
                output_path = aes.encrypt(image_filepath, key)
            else:
                output_path = blowfish.encrypt(image_filepath, key)
                
            if output_path:
                QMessageBox.information(self, "Encryption Success", f"Image encrypted successfully and saved to {output_path}")
                self.download_enc_button.setEnabled(True)
                self.current_output_file = output_path
            else:
                QMessageBox.critical(self, "Encryption Error", "Failed to encrypt the image.")
        except Exception as e:
            QMessageBox.critical(self, "Encryption Error", f"Failed to encrypt the image: {str(e)}")

    def perform_decryption(self, encrypted_filepath):
        if encrypted_filepath is None:
            QMessageBox.information(self, "Decryption Error", "Please select an encrypted file first.")
            return
            
        if not self.aes_radio_dec.isChecked() and not self.blowfish_radio_dec.isChecked():
            QMessageBox.information(self, "Decryption Error", "Please select a decryption method.")
            return
            
        key = self.key_text_box_of_dec.text()
        if not key:
            QMessageBox.information(self, "Decryption Error", "Please enter a secret key.")
            return
            
        try:
            if self.aes_radio_dec.isChecked():
                status, output_path = aes.decrypt(encrypted_filepath, key)
            else:
                status, output_path = blowfish.decrypt(encrypted_filepath, key)
                
            if status == 0 and output_path:
                pixmap = QPixmap(output_path)
                self.dec_display_label.setPixmap(pixmap.scaled(256, 256, Qt.KeepAspectRatio))
                self.download_dec_button.setEnabled(True)
                self.current_output_file = output_path
            else:
                QMessageBox.critical(self, "Decryption Error", "Failed to decrypt the image. Check your key.")
        except Exception as e:
            QMessageBox.critical(self, "Decryption Error", f"Failed to decrypt the image: {str(e)}")

    def logout(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Exit")
        dialog.setMinimumSize(450, 100)

        layout = QVBoxLayout(dialog)
        msg_box = QMessageBox()
        msg_box.setText("<h3>Are you sure you want to Exit?</h3>")

        # Set custom font and size
        font = QFont("Arial", 12)  # Adjust the font and size as desired
        msg_box.setFont(font)

        button_layout = QHBoxLayout()
        layout.addWidget(msg_box)
        layout.addLayout(button_layout)

        # Remove the standard buttons
        msg_box.setStandardButtons(QMessageBox.NoButton)

        yes_button = QPushButton("Yes")
        yes_button.setStyleSheet("color: #000000;")
        yes_button.clicked.connect(lambda: QApplication.quit())

        no_button = QPushButton("No")
        no_button.setStyleSheet("color: #000000;")
        no_button.clicked.connect(dialog.reject)

        button_layout.addWidget(yes_button)
        button_layout.addWidget(no_button)

        dialog.exec_()

    def load_stylesheet(self):
        stylesheet = QFile("styles/style.qss")
        if stylesheet.open(QFile.ReadOnly | QFile.Text):
            stream = QTextStream(stylesheet)
            self.setStyleSheet(stream.readAll())


# Create the application
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
app = QApplication(sys.argv)
window = MainAppWindow()
window.load_stylesheet()
window.show()
sys.exit(app.exec_())
