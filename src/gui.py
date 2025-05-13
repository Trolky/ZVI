
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QComboBox, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QGroupBox, QFormLayout, QCheckBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from thresholding import (manual_threshold, otsu_threshold,
                          su_local_max_min_threshold, recursive_otsu_threshold)


class ThresholdingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.original_image = None
        self.current_image = None
        self.history = []
        self.history_position = -1

        self.inital_width = 0
        self.initial_height = 0
        self.set_initial = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Thresholding Application')
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # Image loading section
        load_group = QGroupBox("Image")
        load_layout = QVBoxLayout()

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        self.save_button = QPushButton("Save Result")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setEnabled(False)

        load_layout.addWidget(self.load_button)
        load_layout.addWidget(self.save_button)
        load_group.setLayout(load_layout)

        # Thresholding method selection
        method_group = QGroupBox("Thresholding Method")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Manual Threshold", "Otsu Threshold", "SU Local Max-Min", "Recursive Otsu"])
        self.method_combo.currentIndexChanged.connect(self.method_changed)

        method_layout.addWidget(self.method_combo)
        method_group.setLayout(method_layout)

        # Parameters section
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout()

        # Manual threshold parameters
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(127)
        self.threshold_slider.valueChanged.connect(self.update_threshold_value)

        self.threshold_value_label = QLabel("127")

        # SU Local Max-Min parameters
        self.window_size_spin = QSpinBox()
        self.window_size_spin.setRange(3, 101)
        self.window_size_spin.setSingleStep(2)  # Ensure odd values
        self.window_size_spin.setValue(15)

        self.k_value_spin = QDoubleSpinBox()
        self.k_value_spin.setRange(0.01, 0.99)
        self.k_value_spin.setSingleStep(0.05)
        self.k_value_spin.setValue(0.5)

        self.r_value_spin = QSpinBox()
        self.r_value_spin.setRange(1, 255)
        self.r_value_spin.setValue(128)

        # Checkbox for scipy/custom filter
        self.use_scipy_checkbox = QCheckBox("Use scipy filters (faster)")
        self.use_scipy_checkbox.setChecked(True)
        self.use_scipy_checkbox.stateChanged.connect(self.su_filter_checkbox_changed)

        # Warning label for custom filter
        self.slow_warning_label = QLabel(
            "<span style='color: red;'>Warning: Custom implementation can be slow for large images.</span>")
        self.slow_warning_label.setWordWrap(True)
        self.slow_warning_label.hide()

        # Recursive Otsu parameters
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 5)
        self.max_depth_spin.setValue(3)

        # Add parameters to layout
        self.params_layout.addRow("Threshold:", self.threshold_slider)
        self.params_layout.addRow("Value:", self.threshold_value_label)
        self.params_layout.addRow("Window Size:", self.window_size_spin)
        self.params_layout.addRow("K Value:", self.k_value_spin)
        self.params_layout.addRow("R Value:", self.r_value_spin)
        self.params_layout.addRow(self.use_scipy_checkbox)
        self.params_layout.addRow(self.slow_warning_label)
        self.params_layout.addRow("Max Depth:", self.max_depth_spin)

        self.params_group.setLayout(self.params_layout)

        # Apply button
        self.apply_button = QPushButton("Apply Thresholding")
        self.apply_button.clicked.connect(self.apply_thresholding)
        self.apply_button.setEnabled(False)

        # Undo/Redo buttons
        history_layout = QHBoxLayout()

        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo)
        self.undo_button.setEnabled(False)

        self.redo_button = QPushButton("Redo")
        self.redo_button.clicked.connect(self.redo)
        self.redo_button.setEnabled(False)

        history_layout.addWidget(self.undo_button)
        history_layout.addWidget(self.redo_button)

        # Add all controls to left panel
        left_layout.addWidget(load_group)
        left_layout.addWidget(method_group)
        left_layout.addWidget(self.params_group)
        left_layout.addWidget(self.apply_button)
        left_layout.addLayout(history_layout)
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        left_panel.setFixedWidth(300)

        # Right panel for image display
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # Original image display
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_image_view = QLabel()
        self.original_image_view.setAlignment(Qt.AlignCenter)
        self.original_image_view.setMinimumSize(400, 300)
        self.original_image_view.setStyleSheet("border: 1px solid #cccccc;")

        # Result image display
        self.result_label = QLabel("Result Image")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_image_view = QLabel()
        self.result_image_view.setAlignment(Qt.AlignCenter)
        self.result_image_view.setMinimumSize(400, 300)
        self.result_image_view.setStyleSheet("border: 1px solid #cccccc;")

        right_layout.addWidget(self.original_label)
        right_layout.addWidget(self.original_image_view)
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.result_image_view)

        right_panel.setLayout(right_layout)

        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Initialize UI state
        self.method_changed()

    def su_filter_checkbox_changed(self):
        # Show/hide warning label based on checkbox state
        if self.use_scipy_checkbox.isChecked():
            self.slow_warning_label.hide()
        else:
            self.slow_warning_label.show()
            # Optionally, show a popup warning
            QMessageBox.warning(
                self,
                "Performance Warning",
                "Custom implementation can be very slow for large images.\n"
                "It is recommended to use scipy filters if possible."
            )

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")

        if file_path:
            # Load image
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            if self.original_image is None:
                return

            # Reset history
            self.history = []
            self.history_position = -1
            self.current_image = self.original_image.copy()
            self.set_initial = False

            # Display original image
            self.display_image(self.original_image, self.original_image_view)

            # Clear result image
            self.result_image_view.clear()

            # Enable apply button
            self.apply_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.redo_button.setEnabled(False)

    def save_image(self):
        if self.current_image is None:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                   "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff)")

        if file_path:
            cv2.imwrite(file_path, self.current_image)

    def method_changed(self):
        method = self.method_combo.currentText()

        # Hide all parameters
        for i in range(self.params_layout.rowCount()):
            for j in range(2):
                item = self.params_layout.itemAt(i, j)
                if item and item.widget():
                    item.widget().hide()
        # Hide single-widget rows (checkbox, warning)
        for i in range(self.params_layout.rowCount()):
            item = self.params_layout.itemAt(i, 0)
            if item and item.widget() and isinstance(item.widget(), (QCheckBox, QLabel)):
                item.widget().hide()

        # Show relevant parameters based on selected method
        if method == "Manual Threshold":
            self.params_layout.itemAt(0, 0).widget().show()  # Threshold label
            self.params_layout.itemAt(0, 1).widget().show()  # Threshold slider
            self.params_layout.itemAt(1, 0).widget().show()  # Value label
            self.params_layout.itemAt(1, 1).widget().show()  # Value display

        elif method == "Otsu Threshold":
            # No parameters needed
            pass

        elif method == "SU Local Max-Min":
            self.params_layout.itemAt(2, 0).widget().show()  # Window size label
            self.params_layout.itemAt(2, 1).widget().show()  # Window size spin
            self.params_layout.itemAt(3, 0).widget().show()  # K value label
            self.params_layout.itemAt(3, 1).widget().show()  # K value spin
            self.params_layout.itemAt(4, 0).widget().show()  # R value label
            self.params_layout.itemAt(4, 1).widget().show()  # R value spin
            # Show checkbox and warning label
            self.use_scipy_checkbox.show()
            if not self.use_scipy_checkbox.isChecked():
                self.slow_warning_label.show()
            else:
                self.slow_warning_label.hide()

        elif method == "Recursive Otsu":
            self.params_layout.itemAt(7, 0).widget().show()  # Max depth label
            self.params_layout.itemAt(7, 1).widget().show()  # Max depth spin

    def update_threshold_value(self):
        value = self.threshold_slider.value()
        self.threshold_value_label.setText(str(value))

    def apply_thresholding(self):
        if self.original_image is None:
            return

        method = self.method_combo.currentText()
        result = None

        if method == "Manual Threshold":
            threshold_value = self.threshold_slider.value()
            result = manual_threshold(self.original_image, threshold_value)

        elif method == "Otsu Threshold":
            result, _ = otsu_threshold(self.original_image)

        elif method == "SU Local Max-Min":
            window_size = self.window_size_spin.value()
            k_value = self.k_value_spin.value()
            r_value = self.r_value_spin.value()
            use_scipy = self.use_scipy_checkbox.isChecked()
            result = su_local_max_min_threshold(self.original_image, window_size, k_value, r_value, use_scipy)

        elif method == "Recursive Otsu":
            max_depth = self.max_depth_spin.value()
            result = recursive_otsu_threshold(self.original_image, max_depth)

        if result is not None:
            # Add to history
            if self.history_position < len(self.history) - 1:
                # If we're not at the end of history, truncate it
                self.history = self.history[:self.history_position + 1]

            self.history.append(result.copy())
            self.history_position = len(self.history) - 1

            # Update current image
            self.current_image = result.copy()

            # Display result
            self.display_image(result, self.result_image_view)

            # Enable save, undo buttons
            self.save_button.setEnabled(True)
            self.undo_button.setEnabled(self.history_position > 0)
            self.redo_button.setEnabled(False)

    def undo(self):
        if self.history_position > 0:
            self.history_position -= 1
            self.current_image = self.history[self.history_position].copy()
            self.display_image(self.current_image, self.result_image_view)

            # Update button states
            self.undo_button.setEnabled(self.history_position > 0)
            self.redo_button.setEnabled(self.history_position < len(self.history) - 1)

    def redo(self):
        if self.history_position < len(self.history) - 1:
            self.history_position += 1
            self.current_image = self.history[self.history_position].copy()
            self.display_image(self.current_image, self.result_image_view)

            # Update button states
            self.undo_button.setEnabled(self.history_position > 0)
            self.redo_button.setEnabled(self.history_position < len(self.history) - 1)

    def display_image(self, image, label):
        if image is None:
            return

        # Convert to RGB for display
        if len(image.shape) == 2:
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, c = image.shape
            bytes_per_line = w * c
            q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)

        # Only scale down, not up
        label_size = label.size()
        if not self.set_initial:
            self.inital_width = label_size.width()
            self.initial_height = label_size.height()
            self.set_initial = True

        scaled_pixmap = pixmap.scaled(self.inital_width, self.initial_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        label.setPixmap(scaled_pixmap)