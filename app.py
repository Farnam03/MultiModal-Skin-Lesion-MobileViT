import sys
import os
import base64
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QFileDialog,
    QProgressBar, QDialog, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QTextDocument
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtPrintSupport import QPrinter

# ---------------------------------------------------------
# 1. AI Core Architecture
# ---------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiModalMobileViT(nn.Module):
    def __init__(self, num_classes=4, num_metadata=21):
        super(MultiModalMobileViT, self).__init__()
        self.image_branch = timm.create_model('mobilevit_s', pretrained=False, num_classes=0)
        self.metadata_branch = nn.Sequential(
            nn.Linear(num_metadata, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        dummy_img = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            img_feat_size = self.image_branch(dummy_img).shape[1]
            
        self.fusion = nn.Sequential(
            nn.Linear(img_feat_size + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, metadata):
        img_features = self.image_branch(image)
        meta_features = self.metadata_branch(metadata)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.fusion(combined)

class ModelWrapperForCAM(nn.Module):
    def __init__(self, model, meta_data):
        super().__init__()
        self.model = model
        self.meta_data = meta_data
    def forward(self, x):
        return self.model(x, self.meta_data)

class_info = {
    'MEL': {'name': 'Melanoma (Critical)', 'dangerous': True},
    'BCC': {'name': 'Basal Cell Carcinoma (Malignant)', 'dangerous': True},
    'BKL': {'name': 'Benign Keratosis (Benign)', 'dangerous': False},
    'NV':  {'name': 'Melanocytic Nevus (Benign Mole)', 'dangerous': False}
}
class_keys = ['BCC', 'BKL', 'MEL', 'NV']

img_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 2. Asynchronous Worker Threads (Non-blocking Engine)
# ---------------------------------------------------------
class ModelLoaderThread(QThread):
    loaded_signal = Signal(object, object)
    error_signal = Signal(str)

    def run(self):
        try:
            model = MultiModalMobileViT(num_classes=4, num_metadata=21).to(device)
            model_path = 'best_multimodal_model.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                # Fast Warm-up pass
                with torch.no_grad():
                    dummy_img = torch.randn(1, 3, 256, 256).to(device)
                    dummy_meta = torch.zeros((1, 21), dtype=torch.float32).to(device)
                    _ = model(dummy_img, dummy_meta)
                target_layers = [model.image_branch.stages[-1]]
                self.loaded_signal.emit(model, target_layers)
            else:
                self.error_signal.emit(f"Model file '{model_path}' not found!")
        except Exception as e:
            self.error_signal.emit(str(e))

class InferenceThread(QThread):
    result_signal = Signal(dict, object)
    error_signal = Signal(str)

    def __init__(self, model, target_layers, cv_image, meta_features, temp_val):
        super().__init__()
        self.model = model
        self.target_layers = target_layers
        self.cv_image = cv_image
        self.meta_features = meta_features
        self.temp_val = temp_val

    def run(self):
        try:
            # 1. Transform Image
            img_resized = cv2.resize(self.cv_image, (256, 256))
            pil_img = Image.fromarray(img_resized)
            input_tensor = img_transforms(pil_img).unsqueeze(0).to(device)
            meta_tensor = torch.tensor(self.meta_features).unsqueeze(0).to(device)

            # 2. Forward Pass
            with torch.no_grad():
                logits = self.model(input_tensor, meta_tensor)
                scaled_logits = logits / self.temp_val
                probabilities = F.softmax(scaled_logits, dim=1)[0]

            pred_scores = {class_keys[i]: float(probabilities[i]) * 100 for i in range(4)}
            predicted_class = torch.argmax(probabilities).item()

            # 3. Grad-CAM Generation
            cam_model = ModelWrapperForCAM(self.model, meta_tensor)
            cam = GradCAM(model=cam_model, target_layers=self.target_layers)
            targets = [ClassifierOutputTarget(predicted_class)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            img_float = img_resized.astype(np.float32) / 255.0
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

            self.result_signal.emit(pred_scores, cam_image)
        except Exception as e:
            self.error_signal.emit(str(e))

# ---------------------------------------------------------
# 3. Modern Stylesheets (QSS)
# ---------------------------------------------------------
DARK_STYLE = """
QMainWindow { background-color: #0b111e; color: #f1f5f9; }
QFrame#card { background-color: #162032; border: 1px solid #1e293b; border-radius: 12px; }
QLabel { color: #f1f5f9; font-family: 'Segoe UI', Arial; }
QLabel#headerTitle { font-size: 20px; font-weight: bold; color: #38bdf8; }
QLabel#sectionTitle { font-size: 14px; font-weight: bold; color: #94a3b8; }
QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 8px; padding: 8px 16px; font-weight: bold; }
QPushButton:hover { background-color: #334155; }
QPushButton#btnPrimary { background-color: #0284c7; color: white; border: none; font-size: 14px; }
QPushButton#btnPrimary:hover { background-color: #0369a1; }
QPushButton#btnPrimary:disabled { background-color: #1e293b; color: #64748b; }
QPushButton#btnReset { background-color: #334155; color: #cbd5e1; border: none; }
QPushButton#btnReset:hover { background-color: #475569; }
QComboBox { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 6px; padding: 6px; }
QComboBox QAbstractItemView { background-color: #0f172a; color: #f8fafc; selection-background-color: #0284c7; }
QSlider::groove:horizontal { background: #334155; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #0284c7; border-radius: 3px; }
QSlider::handle:horizontal { background: #38bdf8; width: 16px; margin: -5px 0; border-radius: 8px; }
QProgressBar { border: 1px solid #334155; border-radius: 6px; background-color: #0f172a; text-align: center; color: white; font-weight: bold; }
QProgressBar::chunk { border-radius: 5px; }
"""

LIGHT_STYLE = """
QMainWindow { background-color: #f8fafc; color: #0f172a; }
QFrame#card { background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 12px; }
QLabel { color: #0f172a; font-family: 'Segoe UI', Arial; }
QLabel#headerTitle { font-size: 20px; font-weight: bold; color: #0284c7; }
QLabel#sectionTitle { font-size: 14px; font-weight: bold; color: #64748b; }
QPushButton { background-color: #e2e8f0; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 8px; padding: 8px 16px; font-weight: bold; }
QPushButton:hover { background-color: #cbd5e1; }
QPushButton#btnPrimary { background-color: #0284c7; color: white; border: none; font-size: 14px; }
QPushButton#btnPrimary:hover { background-color: #0369a1; }
QPushButton#btnPrimary:disabled { background-color: #e2e8f0; color: #94a3b8; }
QPushButton#btnReset { background-color: #e2e8f0; color: #475569; border: 1px solid #cbd5e1; }
QPushButton#btnReset:hover { background-color: #cbd5e1; }
QComboBox { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px; }
QComboBox QAbstractItemView { background-color: #ffffff; color: #0f172a; selection-background-color: #0284c7; selection-color: #ffffff; }
QSlider::groove:horizontal { background: #cbd5e1; height: 6px; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #0284c7; border-radius: 3px; }
QSlider::handle:horizontal { background: #0284c7; width: 16px; margin: -5px 0; border-radius: 8px; }
QProgressBar { border: 1px solid #cbd5e1; border-radius: 6px; background-color: #e2e8f0; text-align: center; color: #0f172a; font-weight: bold; }
QProgressBar::chunk { border-radius: 5px; }
"""

# ---------------------------------------------------------
# 4. Custom In-App Modal
# ---------------------------------------------------------
class ModernModal(QDialog):
    def __init__(self, parent, title, html_body, is_dark=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedWidth(460)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        bg_color = "#162032" if is_dark else "#ffffff"
        text_color = "#f8fafc" if is_dark else "#0f172a"
        border_color = "#334155" if is_dark else "#cbd5e1"
        btn_bg = "#0284c7"
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1.5px solid {border_color};
                border-radius: 14px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(25, 20, 25, 20)
        card_layout.setSpacing(15)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {btn_bg}; border: none;")
        card_layout.addWidget(lbl_title)

        lbl_body = QLabel()
        lbl_body.setTextFormat(Qt.RichText)
        lbl_body.setText(html_body)
        lbl_body.setWordWrap(True)
        lbl_body.setStyleSheet(f"font-size: 13px; color: {text_color}; border: none; line-height: 1.4;")
        card_layout.addWidget(lbl_body)

        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(36)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background-color: {btn_bg};
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: #0369a1;
            }}
        """)
        btn_close.clicked.connect(self.accept)
        card_layout.addWidget(btn_close)

        main_layout.addWidget(card)

# ---------------------------------------------------------
# 5. Main GUI Application
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dermatologist AI Pro - Clinical Diagnostic Suite")
        self.resize(1180, 820)
        self.is_dark = True
        self.setStyleSheet(DARK_STYLE)

        self.model = None
        self.target_layers = None
        self.cv_image = None
        self.cam_image = None
        self.last_results = None

        self.init_ui()
        self.start_background_model_loader()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 15, 20, 20)
        main_layout.setSpacing(15)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("🩺 Dermatologist AI Pro")
        title.setObjectName("headerTitle")
        header_layout.addWidget(title)
        
        header_layout.addStretch()

        self.theme_btn = QPushButton("🌙 Dark Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.theme_btn)

        about_btn = QPushButton("About Us")
        about_btn.clicked.connect(self.show_about)
        header_layout.addWidget(about_btn)

        credits_btn = QPushButton("Credits")
        credits_btn.clicked.connect(self.show_credits)
        header_layout.addWidget(credits_btn)

        main_layout.addLayout(header_layout)

        # Body Layout
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)

        # LEFT COLUMN
        left_card = QFrame()
        left_card.setObjectName("card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)

        lbl_input_sec = QLabel("01 | PATIENT & LESION DATA")
        lbl_input_sec.setObjectName("sectionTitle")
        left_layout.addWidget(lbl_input_sec)

        self.img_preview = QLabel("Click to Upload Dermoscopy Image")
        self.img_preview.setAlignment(Qt.AlignCenter)
        self.img_preview.setFixedHeight(180)
        self.img_preview.setStyleSheet("border: 2px dashed #475569; border-radius: 8px; cursor: pointer;")
        self.img_preview.mousePressEvent = self.upload_image
        left_layout.addWidget(self.img_preview)

        self.lbl_age = QLabel("Patient Age: 45")
        left_layout.addWidget(self.lbl_age)
        self.slider_age = QSlider(Qt.Horizontal)
        self.slider_age.setRange(0, 100)
        self.slider_age.setValue(45)
        self.slider_age.valueChanged.connect(lambda v: self.lbl_age.setText(f"Patient Age: {v}"))
        left_layout.addWidget(self.slider_age)

        left_layout.addWidget(QLabel("Biological Sex:"))
        self.combo_sex = QComboBox()
        self.combo_sex.addItems(["Unknown", "Male", "Female"])
        left_layout.addWidget(self.combo_sex)

        left_layout.addWidget(QLabel("Anatomical Site:"))
        self.combo_site = QComboBox()
        sites = ["Unknown", "Back", "Lower Extremity", "Trunk", "Upper Extremity", "Abdomen", "Face", "Chest", "Foot", "Neck", "Scalp", "Hand", "Ear", "Genital", "Acral"]
        self.combo_site.addItems(sites)
        left_layout.addWidget(self.combo_site)

        self.lbl_temp = QLabel("Calibration Temperature: 1.5")
        left_layout.addWidget(self.lbl_temp)
        self.slider_temp = QSlider(Qt.Horizontal)
        self.slider_temp.setRange(10, 30)
        self.slider_temp.setValue(15)
        self.slider_temp.valueChanged.connect(lambda v: self.lbl_temp.setText(f"Calibration Temperature: {v/10.0:.1f}"))
        left_layout.addWidget(self.slider_temp)

        left_layout.addStretch()

        btn_layout = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setObjectName("btnReset")
        self.btn_reset.clicked.connect(self.reset_form)
        btn_layout.addWidget(self.btn_reset, 1)

        self.btn_analyze = QPushButton("⚡ Initializing AI Engine...")
        self.btn_analyze.setObjectName("btnPrimary")
        self.btn_analyze.setFixedHeight(42)
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.clicked.connect(self.run_inference)
        btn_layout.addWidget(self.btn_analyze, 2)
        left_layout.addLayout(btn_layout)

        body_layout.addWidget(left_card, 1)

        # RIGHT COLUMN
        right_card = QFrame()
        right_card.setObjectName("card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(12)

        lbl_res_sec = QLabel("02 | DIAGNOSTIC RESULTS & EXPLAINABILITY (XAI)")
        lbl_res_sec.setObjectName("sectionTitle")
        right_layout.addWidget(lbl_res_sec)

        self.results_container = QVBoxLayout()
        self.result_widgets = []
        for i in range(4):
            lbl = QLabel(f"Rank {i+1}: -")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            self.results_container.addWidget(lbl)
            self.results_container.addWidget(bar)
            self.result_widgets.append((lbl, bar))
            
        right_layout.addLayout(self.results_container)

        right_layout.addWidget(QLabel("Grad-CAM Activation Heatmap:"))
        self.cam_display = QLabel("Heatmap will be displayed here after analysis")
        self.cam_display.setAlignment(Qt.AlignCenter)
        self.cam_display.setFixedHeight(220)
        self.cam_display.setStyleSheet("border: 1px solid #334155; border-radius: 8px; background-color: #0f172a;")
        right_layout.addWidget(self.cam_display)

        self.btn_export = QPushButton("📄 Print / Export PDF Report")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_report)
        right_layout.addWidget(self.btn_export)

        body_layout.addWidget(right_card, 1)
        main_layout.addLayout(body_layout)

    # ---------------------------------------------------------
    # Threading: Background Loader
    # ---------------------------------------------------------
    def start_background_model_loader(self):
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.loaded_signal.connect(self.on_model_loaded)
        self.loader_thread.error_signal.connect(self.on_model_load_error)
        self.loader_thread.start()

    def on_model_loaded(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.btn_analyze.setText("Analyze & Diagnose 🚀")
        self.btn_analyze.setEnabled(True)

    def on_model_load_error(self, err_msg):
        self.btn_analyze.setText("Model Load Failed ❌")
        dlg = ModernModal(self, "Engine Error", f"Failed to initialize PyTorch model:<br><br>{err_msg}", self.is_dark)
        dlg.exec()

    # ---------------------------------------------------------
    # UI Actions & Modals
    # ---------------------------------------------------------
    def toggle_theme(self):
        if self.is_dark:
            self.setStyleSheet(LIGHT_STYLE)
            self.theme_btn.setText("☀️ Light Mode")
            self.is_dark = False
        else:
            self.setStyleSheet(DARK_STYLE)
            self.theme_btn.setText("🌙 Dark Mode")
            self.is_dark = True
        if self.last_results:
            self.update_results_ui(self.last_results)

    def show_about(self):
        body = """
        <b>Dermatologist AI Pro</b> is a multi-modal deep learning framework utilizing 
        <b>MobileViT</b> and tabular clinical patient metadata (Late-Fusion architecture) 
        for trustworthy skin cancer screening.<br><br>
        • <b>Validation Accuracy:</b> 91.31%<br>
        • <b>Vision Backbone:</b> MobileViT-S<br>
        • <b>Inference Engine:</b> Multi-threaded PyTorch Edge
        """
        dlg = ModernModal(self, "About Project", body, self.is_dark)
        dlg.exec()

    def show_credits(self):
        body = """
        <b>Research & Software Engineering:</b><br><br>
        • <b>Student:</b> Behrooz Karbasi<br>
        • <b>Supervisor:</b> Dr. Mehdi Babagoli<br>
        • <b>Faculty:</b> Faculty of Engineering and Technology, University of Mazandaran<br>
        • <b>Contact:</b> behrooz.karbasi82@gmail.com
        """
        dlg = ModernModal(self, "Project Credits", body, self.is_dark)
        dlg.exec()

    def upload_image(self, event=None):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dermoscopy Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            try:
                img_data = np.fromfile(file_path, dtype=np.uint8)
                self.cv_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if self.cv_image is None:
                    raise ValueError("Could not decode image.")
                self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
                
                h, w, ch = self.cv_image.shape
                qimg = QImage(self.cv_image.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(self.img_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.img_preview.setPixmap(pix)
            except Exception as e:
                dlg = ModernModal(self, "Error", f"Failed to load image:<br>{str(e)}", self.is_dark)
                dlg.exec()

    def reset_form(self):
        self.cv_image = None
        self.cam_image = None
        self.last_results = None
        self.img_preview.clear()
        self.img_preview.setText("Click to Upload Dermoscopy Image")
        self.slider_age.setValue(45)
        self.combo_sex.setCurrentIndex(0)
        self.combo_site.setCurrentIndex(0)
        self.slider_temp.setValue(15)
        for i, (lbl, bar) in enumerate(self.result_widgets):
            lbl.setText(f"Rank {i+1}: -")
            lbl.setStyleSheet("")
            bar.setValue(0)
            bar.setStyleSheet("")
        self.cam_display.clear()
        self.cam_display.setText("Heatmap will be displayed here after analysis")
        self.btn_export.setEnabled(False)

    # ---------------------------------------------------------
    # Threaded Inference Execution
    # ---------------------------------------------------------
    def run_inference(self):
        if self.cv_image is None:
            dlg = ModernModal(self, "Input Warning", "Please upload a dermoscopy image first.", self.is_dark)
            dlg.exec()
            return

        if self.model is None:
            return

        self.btn_analyze.setText("Analyzing Image... ⏳")
        self.btn_analyze.setEnabled(False)

        # Prepare Metadata
        meta_features = np.zeros(21, dtype=np.float32)
        meta_features[0] = self.slider_age.value() / 100.0

        sex_val = self.combo_sex.currentText().lower()
        if sex_val == 'female': meta_features[1] = 1.0
        elif sex_val == 'male': meta_features[2] = 1.0
        elif sex_val == 'unknown': meta_features[3] = 1.0
        else: meta_features[4] = 1.0

        sites = ['abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand', 'lower extremity', 'neck', 'scalp', 'trunk', 'unknown', 'upper extremity']
        loc_val = self.combo_site.currentText().lower()
        if loc_val in sites:
            meta_features[sites.index(loc_val) + 5] = 1.0
        else:
            meta_features[20] = 1.0

        temp_val = self.slider_temp.value() / 10.0

        # Launch Inference Worker Thread
        self.infer_thread = InferenceThread(self.model, self.target_layers, self.cv_image, meta_features, temp_val)
        self.infer_thread.result_signal.connect(self.on_inference_finished)
        self.infer_thread.error_signal.connect(self.on_inference_error)
        self.infer_thread.start()

    def on_inference_finished(self, pred_scores, cam_image):
        self.last_results = pred_scores
        self.cam_image = cam_image

        self.update_results_ui(pred_scores)

        # Display CAM
        h, w, ch = self.cam_image.shape
        qimg_cam = QImage(self.cam_image.data, w, h, ch * w, QImage.Format_RGB888)
        pix_cam = QPixmap.fromImage(qimg_cam).scaled(self.cam_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_display.setPixmap(pix_cam)

        self.btn_export.setEnabled(True)
        self.btn_analyze.setText("Analyze & Diagnose 🚀")
        self.btn_analyze.setEnabled(True)

    def on_inference_error(self, err_msg):
        self.btn_analyze.setText("Analyze & Diagnose 🚀")
        self.btn_analyze.setEnabled(True)
        dlg = ModernModal(self, "Analysis Error", f"An error occurred during inference:<br>{err_msg}", self.is_dark)
        dlg.exec()

    def update_results_ui(self, pred_scores):
        sorted_results = sorted(pred_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (key, prob) in enumerate(sorted_results):
            lbl, bar = self.result_widgets[i]
            info = class_info[key]
            text = f"{key} - {info['name']}: {prob:.2f}%"
            lbl.setText(text)
            bar.setValue(int(prob))
            
            if i == 0:
                color = "#ef4444" if info['dangerous'] else "#22c55e"
                lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color};")
                bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; }}")
            else:
                default_text_color = "#f1f5f9" if self.is_dark else "#0f172a"
                lbl.setStyleSheet(f"font-size: 12px; font-weight: normal; color: {default_text_color};")
                bar.setStyleSheet("QProgressBar::chunk { background-color: #0284c7; }")

    # ---------------------------------------------------------
    # PDF Report Exporting
    # ---------------------------------------------------------
    def export_report(self):
        if not self.last_results or self.cam_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Diagnostic PDF Report", "Clinical_Report.pdf", "PDF Files (*.pdf)")
        if not file_path:
            return

        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.cam_image, cv2.COLOR_RGB2BGR))
        cam_base64 = base64.b64encode(buffer).decode('utf-8')

        sorted_results = sorted(self.last_results.items(), key=lambda x: x[1], reverse=True)
        rows_html = ""
        for i, (k, v) in enumerate(sorted_results):
            info = class_info[k]
            badge = "<span style='color:red; font-weight:bold;'>CRITICAL / MALIGNANT</span>" if info['dangerous'] else "<span style='color:green;'>BENIGN</span>"
            rows_html += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #cbd5e1;'><b>#{i+1} {k}</b> - {info['name']}</td>
                <td style='padding: 8px; border: 1px solid #cbd5e1; text-align: center;'><b>{v:.2f}%</b></td>
                <td style='padding: 8px; border: 1px solid #cbd5e1; text-align: center;'>{badge}</td>
            </tr>
            """

        html_content = f"""
        <html>
        <body style='font-family: Arial, sans-serif; color: #1e293b; padding: 20px;'>
            <h1 style='color: #0284c7; text-align: center; border-bottom: 2px solid #0284c7; padding-bottom: 10px;'>
                🩺 Dermatologist AI Pro - Diagnostic Report
            </h1>
            <p style='text-align: right; color: #64748b; font-size: 11px;'>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>📋 Patient Clinical Context:</h3>
            <ul>
                <li><b>Age:</b> {self.slider_age.value()} years</li>
                <li><b>Biological Sex:</b> {self.combo_sex.currentText()}</li>
                <li><b>Anatomical Site:</b> {self.combo_site.currentText()}</li>
                <li><b>Calibration Temperature:</b> {self.slider_temp.value()/10.0:.1f}</li>
            </ul>

            <h3>📊 Ranked Diagnostic Predictions:</h3>
            <table style='width: 100%; border-collapse: collapse; margin-top: 10px;'>
                <tr style='background-color: #f1f5f9;'>
                    <th style='padding: 8px; border: 1px solid #cbd5e1; text-align: left;'>Condition Class</th>
                    <th style='padding: 8px; border: 1px solid #cbd5e1;'>Probability</th>
                    <th style='padding: 8px; border: 1px solid #cbd5e1;'>Status</th>
                </tr>
                {rows_html}
            </table>

            <h3>🔍 Explainable AI (Grad-CAM Heatmap):</h3>
            <div style='text-align: center; margin-top: 10px;'>
                <img src='data:image/jpeg;base64,{cam_base64}' width='250' height='250' style='border: 1px solid #cbd5e1; border-radius: 8px;' />
                <p style='color: #64748b; font-size: 12px; margin-top: 5px;'>Visual attention map highlighting pathological border irregularity.</p>
            </div>

            <hr style='border: none; border-top: 1px solid #e2e8f0; margin-top: 30px;' />
            <p style='font-size: 10px; color: #94a3b8; text-align: center;'>
                Dermatologist AI Pro | Student: Behrooz Karbasi | Supervisor: Dr. Mehdi Babagoli | Faculty of Engineering & Technology, University of Mazandaran
            </p>
        </body>
        </html>
        """

        doc = QTextDocument()
        doc.setHtml(html_content)

        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(file_path)
        
        doc.print_(printer)

        dlg = ModernModal(self, "Export Success", f"Report saved successfully:<br><br><b>{file_path}</b>", self.is_dark)
        dlg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())