import os
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QSlider,
    QComboBox,
    QFileDialog,
    QTextEdit,
    QScrollArea,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QCheckBox,
    QMessageBox,
    QSizePolicy,
    QScrollArea as QSubScrollArea,
    QSpinBox,
    QProgressBar,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QFont
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)  # for plots
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from gui.worker import Worker  # for running approximation in a separate thread
import traceback
import time
from core.metrics import AVAILABLE_METRICS
from core.shapes import get_total_shape_params  # used for validating loaded shapes
from utils.config_loader import ConfigLoader
from numba.typed import (
    List as NumbaList,
)
import core.optimizers as optimizers
from utils.shape_io import load_results_and_shapes


class Communicate(QObject):
    """
    A simple QObject subclass for custom signals.

    Used to communicate from worker threads or other parts of the application
    back to the main GUI thread in a thread-safe manner.
    """

    update_output = pyqtSignal(QImage)  # signal to update the output image preview
    update_results = pyqtSignal(
        str
    )  # signal to append a message to the log/results text area
    enable_plot_button = pyqtSignal()  # signal to enable the overall metric plot button
    enable_pso_plot_button = (
        pyqtSignal()
    )  # signal to enable the pso-specific plot button
    finished = pyqtSignal()  # signal indicating that a worker thread has finished
    progress_update = pyqtSignal(
        int, int, float, float, float
    )  # current_step, total_steps, percentage, elapsed_sec, remaining_sec


class ImageApproximationUI(QMainWindow):
    """
    Main window for the Image Approximation GUI.

    Handles user interactions, parameter selection, starting/stopping
    the approximation process, and displaying results.
    """

    def __init__(
        self,
        app_instance,  # the main App class instance
    ):
        """
        Initializes the ImageApproximationUI.

        :param app_instance: The main application instance (App class).
        :type app_instance: App
        """
        super().__init__()
        self.app_instance = app_instance
        self.setWindowTitle("Image Approximation with Shapes")
        self.resize(1200, 800)  # default window size
        self.config_loader = ConfigLoader()  # for loading shape/border configs
        self.available_metrics = AVAILABLE_METRICS  # from core.metrics

        self.c = Communicate()  # communication object for signals
        self.input_image_path = None
        self.output_directory = None
        self.is_running = False  # flag to track if approximation is ongoing
        # dictionary to store widgets for fixing non-coordinate parameters
        self.fix_param_widgets: Dict[str, Tuple[QCheckBox, QDoubleSpinBox]] = {}
        self.pso_pbest_scores_plot_data: Optional[List[np.ndarray]] = (
            None  # data for pso plot
        )
        self.loaded_shapes_data: Optional[Dict[str, Any]] = (
            None  # stores data from loaded shapes file
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(
            central_widget
        )  # main layout: left (controls) and right (display)

        # left panel for controls, made scrollable
        self.left_panel_scroll = QScrollArea()
        self.left_panel_scroll.setWidgetResizable(True)
        self.left_panel = QWidget()
        self.left_panel_scroll.setWidget(self.left_panel)
        left_layout = QVBoxLayout(self.left_panel)

        # right panel for image previews and logs
        self.right_panel = QFrame()
        right_layout = QVBoxLayout(self.right_panel)

        main_layout.addWidget(self.left_panel_scroll, 1)  # left panel takes 1/3 space
        main_layout.addWidget(self.right_panel, 2)  # right panel takes 2/3 space

        # create ui sections
        self.create_input_section(left_layout)
        self.create_load_shapes_section(left_layout)
        self.create_parameter_fixing_section(left_layout)  # for non-coordinate params
        self.create_control_section(left_layout)  # run/stop button
        self.create_optimization_section(left_layout)
        self.create_output_section(left_layout)  # results, saving, plots
        left_layout.addStretch(1)  # push all widgets to the top

        self.create_image_display(right_layout)
        self.create_results_section(right_layout)  # log text and progress bar

        self.reload_configs()  # initial population of shape/border combos

        # connect signals
        self.c.update_output.connect(self.set_output_image)
        self.c.update_results.connect(self.log_message)
        self.c.enable_plot_button.connect(self._enable_plot_button)
        self.c.enable_pso_plot_button.connect(self._enable_pso_plot_button)
        self.c.finished.connect(self._on_run_finished)
        self.c.progress_update.connect(self.update_progress_bar)

        self.run_stop_btn.setEnabled(
            False
        )  # initially disabled until inputs are selected

        self.update_method_parameters(
            self.opt_method_combo.currentText()
        )  # populate initial method params

        # timer to periodically check if run conditions are met (inputs selected, weights valid)
        self.weight_check_timer = QTimer(self)
        self.weight_check_timer.setInterval(500)  # check every 0.5 seconds
        self.weight_check_timer.timeout.connect(self.check_run_conditions)
        self.weight_check_timer.start()

        # connect signals for dynamic UI updates
        self.shape_type_combo.currentTextChanged.connect(
            self.update_fixable_parameters_display  # non-coord params
        )
        self.color_scheme_combo.currentTextChanged.connect(
            self.update_fixable_parameters_display  # non-coord params (color affects available params)
        )
        self.shape_type_combo.currentTextChanged.connect(
            self._populate_coord_fixing_widgets  # coord params
        )
        self.num_shapes_spinbox.valueChanged.connect(
            self.update_load_mismatch_widgets_state  # for loaded shapes options
        )

        self._populate_coord_fixing_widgets()  # initial population
        self.update_fixable_parameters_display()  # initial population

        self.app_instance.config_loader = (
            self.config_loader
        )  # pass config_loader to app instance
        self.check_run_conditions()  # initial check for run button state
        self.update_load_mismatch_widgets_state()  # initial state of loaded shapes ui

    def reload_configs(self):
        """
        Loads or reloads shape and border configurations from JSON files.

        Updates relevant UI elements like QComboBoxes for shape types,
        border types, and metrics. Also repopulates parameter fixing sections.

        :return: True if configurations were reloaded successfully, False otherwise.
        :rtype: bool
        """
        try:
            # store current selections to try and restore them after reload
            current_shape = (
                self.shape_type_combo.currentText()
                if hasattr(self, "shape_type_combo")
                else None
            )
            current_border = (
                self.border_type_combo.currentText()
                if hasattr(self, "border_type_combo")
                else None
            )
            current_metrics = []
            if hasattr(self, "metric_widgets"):
                for _, combo, _ in self.metric_widgets:
                    current_metrics.append(combo.currentText())

            # load available items from config files
            self.available_shapes = self.config_loader.get_available_shapes()
            self.available_borders = self.config_loader.get_available_borders()
            self.available_metrics = (
                AVAILABLE_METRICS  # this is usually static from core.metrics
            )

            # update shape type combo box
            if hasattr(self, "shape_type_combo"):
                self.shape_type_combo.blockSignals(
                    True
                )  # prevent signals during update
                self.shape_type_combo.clear()
                self.shape_type_combo.addItems(self.available_shapes)
                if current_shape in self.available_shapes:
                    self.shape_type_combo.setCurrentText(current_shape)
                elif (
                    self.available_shapes
                ):  # set to first if current not found or none selected
                    self.shape_type_combo.setCurrentIndex(0)
                self.shape_type_combo.blockSignals(False)

            # update border type combo box
            if hasattr(self, "border_type_combo"):
                self.border_type_combo.blockSignals(True)
                self.border_type_combo.clear()
                self.border_type_combo.addItems(self.available_borders)
                if current_border in self.available_borders:
                    self.border_type_combo.setCurrentText(current_border)
                elif (
                    "b_original" in self.available_borders
                ):  # prefer "b_original" as default
                    self.border_type_combo.setCurrentText("b_original")
                elif self.available_borders:
                    self.border_type_combo.setCurrentIndex(0)
                self.border_type_combo.blockSignals(False)

            # update metric combo boxes
            if hasattr(self, "metric_widgets"):
                for i, (_, combo, _) in enumerate(self.metric_widgets):
                    combo.blockSignals(True)
                    current_metric_val = (  # restore previous selection if possible
                        current_metrics[i] if i < len(current_metrics) else None
                    )
                    combo.clear()
                    combo.addItems(self.available_metrics)
                    if current_metric_val in self.available_metrics:
                        combo.setCurrentText(current_metric_val)
                    elif self.available_metrics:  # default to first available metric
                        combo.setCurrentIndex(0)
                    combo.blockSignals(False)

            # refresh parameter fixing UI sections as available params might change
            if hasattr(self, "fix_params_layout"):
                self.update_fixable_parameters_display()
            if hasattr(self, "coord_fix_layout"):
                self._populate_coord_fixing_widgets()

            # check if loaded shapes are still valid with new configs
            if self.loaded_shapes_data:
                loaded_shape_type_val = self.loaded_shapes_data.get("shape_type", None)
                loaded_color_scheme_val = self.loaded_shapes_data.get(
                    "color_scheme", None
                )
                # if current UI shape/color (after reload) mismatches loaded data, invalidate loaded data
                if (
                    loaded_shape_type_val
                    and loaded_shape_type_val != self.shape_type_combo.currentText()
                ) or (
                    loaded_color_scheme_val
                    and loaded_color_scheme_val != self.color_scheme_combo.currentText()
                ):
                    self.log_message(
                        "Warning: Config reload invalidated loaded shapes. Clearing loaded data."
                    )
                    self.loaded_shapes_data = None
                    self.load_status_label.setText("No shapes loaded.")
                    self.load_status_label.setStyleSheet("")  # reset color
                    self.update_load_mismatch_widgets_state()  # update ui accordingly

            self.log_message("Configuration reloaded successfully.")
            return True
        except Exception as e:
            error_msg = f"Failed to reload configurations: {e}"
            self.log_message(f"Error: {error_msg}")
            traceback.print_exc()
            QMessageBox.critical(self, "Config Reload Error", error_msg)
            return False

    def log_message(self, message):
        """
        Appends a message to the results_text (log output) text area.

        :param message: The message string to append.
        :type message: str
        """
        self.results_text.append(message)
        QApplication.processEvents()  # ensure gui updates, useful for long operations

    def create_input_section(self, parent_layout):
        """
        Creates the UI section for input image, output directory, and core shape/color/border settings.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        input_frame = QGroupBox("Input Settings")
        layout = QVBoxLayout(input_frame)

        # input image selection
        input_img_layout = QHBoxLayout()
        self.select_image_btn = QPushButton("Select Input Image")
        self.select_image_btn.clicked.connect(self.select_input_image)
        self.input_image_path_label = QLabel("No image selected.")
        self.input_image_path_label.setWordWrap(True)  # allow text to wrap
        input_img_layout.addWidget(self.select_image_btn)
        input_img_layout.addWidget(
            self.input_image_path_label, 1
        )  # label takes remaining space
        layout.addLayout(input_img_layout)

        # output directory selection
        output_dir_layout = QHBoxLayout()
        self.select_output_btn = QPushButton("Select Output Directory")
        self.select_output_btn.clicked.connect(self.select_output_directory)
        self.output_dir_path_label = QLabel("No directory selected.")
        self.output_dir_path_label.setWordWrap(True)
        output_dir_layout.addWidget(self.select_output_btn)
        output_dir_layout.addWidget(self.output_dir_path_label, 1)
        layout.addLayout(output_dir_layout)

        # shape type selection
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Shape Type:"))
        self.shape_type_combo = QComboBox()
        shape_layout.addWidget(self.shape_type_combo)
        self.reload_config_btn = QPushButton("R")  # small reload button
        self.reload_config_btn.setToolTip(
            "Reload shape/border configurations from JSON files"
        )
        self.reload_config_btn.setFixedSize(QSize(25, 25))  # make it compact
        self.reload_config_btn.clicked.connect(self.reload_configs)
        shape_layout.addWidget(self.reload_config_btn)
        layout.addLayout(shape_layout)

        # color scheme selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems(["rgb", "grayscale", "both"])
        self.color_scheme_combo.setCurrentText("rgb")  # default to rgb
        color_layout.addWidget(self.color_scheme_combo)
        layout.addLayout(color_layout)

        # border type selection
        border_layout = QHBoxLayout()
        border_layout.addWidget(QLabel("Border Type:"))
        self.border_type_combo = QComboBox()
        border_layout.addWidget(self.border_type_combo)
        layout.addLayout(border_layout)

        # parameter initialization mode (for non-coordinate, configurable params)
        param_init_layout = QHBoxLayout()
        param_init_layout.addWidget(QLabel("Param Init:"))
        self.param_init_combo = QComboBox()
        self.param_init_combo.addItems(["Midpoint", "Min", "Max", "Random"])
        self.param_init_combo.setCurrentText("Midpoint")
        param_init_layout.addWidget(self.param_init_combo)
        layout.addLayout(param_init_layout)

        # coordinate settings group
        coord_group = QGroupBox("Coordinate Settings")
        coord_layout = QVBoxLayout(coord_group)
        init_mode_layout = QHBoxLayout()
        init_mode_layout.addWidget(QLabel("Init Mode:"))
        self.coord_init_mode_combo = QComboBox()
        self.coord_init_mode_combo.addItems(
            [
                "Random",
                "Grid",
                "Zero",
                "Intensity PDF",
                "SSIM PDF",
            ]  # available coord init modes
        )
        self.coord_init_mode_combo.setCurrentText("Random")
        init_mode_layout.addWidget(self.coord_init_mode_combo)
        coord_layout.addLayout(init_mode_layout)

        # container for dynamically generated coordinate fixing widgets
        self.coord_fix_widgets_container = QWidget()
        self.coord_fix_layout = QVBoxLayout(self.coord_fix_widgets_container)
        self.coord_fix_layout.setContentsMargins(0, 5, 0, 0)  # small top margin
        coord_layout.addWidget(self.coord_fix_widgets_container)
        # dictionary to store coord fixing widgets: {name: (fix_cb, set_val_cb, val_spin)}
        self.coord_fix_widgets: Dict[
            str, Tuple[QCheckBox, QCheckBox, QDoubleSpinBox]
        ] = {}
        layout.addWidget(coord_group)

        parent_layout.addWidget(input_frame)

    def create_load_shapes_section(self, parent_layout):
        """
        Creates the UI section for loading shapes from existing files.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        load_group = QGroupBox("Load Shapes")
        load_layout = QVBoxLayout(load_group)

        hbox_load = QHBoxLayout()
        self.load_shapes_btn = QPushButton("Load Shapes from JSON")
        self.load_shapes_btn.clicked.connect(self._load_shapes)
        self.load_status_label = QLabel("No shapes loaded.")
        self.load_status_label.setWordWrap(True)
        hbox_load.addWidget(self.load_shapes_btn)
        hbox_load.addWidget(self.load_status_label, 1)
        load_layout.addLayout(hbox_load)

        self.render_canvas_checkbox = QCheckBox("Render Loaded Shapes as Canvas")
        self.render_canvas_checkbox.setToolTip(
            "If checked, render all loaded shapes as the background canvas, and optimize the specified 'Num Shapes' as new shapes on top. Mismatch/Truncation options below are ignored."
        )
        self.render_canvas_checkbox.toggled.connect(
            self.update_load_mismatch_widgets_state  # toggling this affects other options
        )
        load_layout.addWidget(self.render_canvas_checkbox)

        # container for options relevant when num_shapes (K) < num_loaded_shapes (N)
        self.mismatch_widgets_container = QWidget()
        mismatch_layout = QVBoxLayout(self.mismatch_widgets_container)
        mismatch_layout.setContentsMargins(10, 5, 0, 0)  # indent slightly
        self.mismatch_label = QLabel("Options for K < N (Num Shapes < Loaded):")
        mismatch_label_font = self.mismatch_label.font()
        mismatch_label_font.setItalic(True)  # make label italic
        self.mismatch_label.setFont(mismatch_label_font)
        mismatch_layout.addWidget(self.mismatch_label)

        # truncation mode if K < N
        hbox_truncate = QHBoxLayout()
        hbox_truncate.addWidget(QLabel("Truncation Mode:"))
        self.truncate_mode_combo = QComboBox()
        self.truncate_mode_combo.addItems(["Random", "First K", "Last K"])
        self.truncate_mode_combo.setCurrentText("Random")  # default truncation
        hbox_truncate.addWidget(self.truncate_mode_combo)
        mismatch_layout.addLayout(hbox_truncate)

        self.render_untruncated_checkbox = QCheckBox("Render Untruncated Shapes")
        self.render_untruncated_checkbox.setToolTip(
            "If truncating (K < N), render the shapes that were *not* selected for optimization onto the background canvas before optimization starts."
        )
        mismatch_layout.addWidget(self.render_untruncated_checkbox)

        self.mismatch_widgets_container.setVisible(False)  # initially hidden
        load_layout.addWidget(self.mismatch_widgets_container)
        parent_layout.addWidget(load_group)

    def _load_shapes(self):
        """
        Handles the action of the "Load Shapes from JSON" button.

        Opens a file dialog for the user to select a JSON results file,
        then attempts to load the results and associated .npy shape data.
        Updates UI elements based on the loaded data or displays an error.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Results/Shapes JSON",
            self.output_directory or "",  # start in output dir if set, else current
            "JSON files (*.json)",
        )
        if not file_path:  # user cancelled dialog
            return

        self.loaded_shapes_data = None  # reset previous
        self.load_status_label.setText("Loading...")
        self.load_status_label.setStyleSheet("")  # reset color
        QApplication.processEvents()  # update ui

        try:
            results_data, shapes_array = load_results_and_shapes(file_path)
            if results_data is None:
                raise ValueError("Failed to load or parse JSON file.")

            # validate essential metadata from the loaded json
            required_meta = [
                "shape_type",
                "num_shapes",  # num_shapes in json is N (number in .npy)
                "color_scheme",
                "param_names_config",  # names of configurable params for these shapes
            ]
            if not all(k in results_data for k in required_meta):
                raise ValueError("JSON file missing required metadata fields.")

            loaded_shape_type_val = results_data["shape_type"]
            loaded_num_shapes_meta = results_data["num_shapes"]  # N from metadata
            loaded_color_scheme_val = results_data["color_scheme"]
            loaded_param_names_val = results_data["param_names_config"]
            shape_data_file_ref = results_data.get(
                "shape_data_file"
            )  # name of .npy file

            # check if shape array was actually loaded
            if shapes_array is None and shape_data_file_ref:
                raise ValueError(
                    f"Shape data file '{shape_data_file_ref}' not found or failed to load."
                )
            elif shapes_array is None and not shape_data_file_ref:
                raise ValueError(
                    "JSON file does not reference any shape data (.npy file)."
                )
            elif shapes_array is None:  # generic fail
                raise ValueError("Failed to load shape data array.")

            # validate shape array dimensions and consistency with metadata
            if shapes_array.ndim != 2:
                raise ValueError(
                    f"Loaded shapes array has incorrect dimensions ({shapes_array.ndim} instead of 2)."
                )
            n_loaded_actual, p_loaded_actual = (
                shapes_array.shape
            )  # N, P from actual array
            if n_loaded_actual != loaded_num_shapes_meta:
                raise ValueError(
                    f"Loaded array shape ({n_loaded_actual} shapes) mismatch with JSON metadata ({loaded_num_shapes_meta} shapes)."
                )

            # validate number of parameters (P) against expected based on loaded metadata
            expected_p = get_total_shape_params(
                loaded_shape_type_val,
                loaded_param_names_val,  # use param names from loaded file
                loaded_color_scheme_val in ("rgb", "both"),
                True,  # assume alpha is always part of the shape data structure
            )
            if p_loaded_actual != expected_p:
                raise ValueError(
                    f"Loaded array parameters ({p_loaded_actual}) mismatch with expected ({expected_p}) based on loaded metadata."
                )

            self.log_message(
                f"Successfully loaded {n_loaded_actual} '{loaded_shape_type_val}' shapes ({loaded_color_scheme_val})."
            )
            self.load_status_label.setText(
                f"Loaded {n_loaded_actual} '{loaded_shape_type_val}' shapes."
            )
            self.load_status_label.setStyleSheet("color: green;")  # success color

            # store loaded data
            self.loaded_shapes_data = {
                "array": shapes_array.astype(np.float32),  # ensure float32
                "param_names": loaded_param_names_val,
                "num_loaded": n_loaded_actual,  # N
                "shape_type": loaded_shape_type_val,
                "color_scheme": loaded_color_scheme_val,
            }

            # update relevant UI controls to match loaded data, block signals to avoid cascades
            self.shape_type_combo.blockSignals(True)
            self.color_scheme_combo.blockSignals(True)
            self.num_shapes_spinbox.blockSignals(True)

            self.shape_type_combo.setCurrentText(loaded_shape_type_val)
            self.color_scheme_combo.setCurrentText(loaded_color_scheme_val)
            self.num_shapes_spinbox.setValue(n_loaded_actual)  # set K to N by default

            self.shape_type_combo.blockSignals(False)
            self.color_scheme_combo.blockSignals(False)
            self.num_shapes_spinbox.blockSignals(False)

            # refresh parameter fixing ui based on new (loaded) shape type and color
            self.update_fixable_parameters_display()
            self._populate_coord_fixing_widgets()
            self.update_load_mismatch_widgets_state()  # update visibility of K<N options

        except Exception as e:
            self.log_message(f"Error loading shapes: {e}")
            traceback.print_exc()
            self.load_status_label.setText("Load Failed!")
            self.load_status_label.setStyleSheet("color: red;")  # error color
            self.loaded_shapes_data = None  # clear any partial data
            QMessageBox.critical(
                self, "Load Shapes Error", f"Failed to load shapes:\n{e}"
            )
            self.update_load_mismatch_widgets_state()  # ensure ui reflects no loaded data

    def update_load_mismatch_widgets_state(self):
        """
        Updates the visibility and enabled state of UI elements related to loaded shapes.

        This includes the "Render Loaded as Canvas" checkbox and options for
        handling cases where the number of shapes to optimize (K) is different
        from the number of loaded shapes (N), particularly when K < N (truncation).
        """
        shapes_loaded = self.loaded_shapes_data is not None
        render_as_canvas_checked = (
            self.render_canvas_checkbox.isChecked() if shapes_loaded else False
        )

        # enable "render as canvas" only if shapes are loaded and not currently running
        self.render_canvas_checkbox.setEnabled(shapes_loaded and not self.is_running)

        show_mismatch_options = False
        if shapes_loaded and not render_as_canvas_checked:
            # only show truncation options if not rendering as canvas and K < N
            k_val = self.num_shapes_spinbox.value()  # K
            n_val = self.loaded_shapes_data.get("num_loaded", 0)  # N
            if k_val < n_val:
                show_mismatch_options = True

        self.mismatch_widgets_container.setVisible(show_mismatch_options)
        enabled_state_mismatch = show_mismatch_options and not self.is_running
        self.mismatch_label.setEnabled(enabled_state_mismatch)
        self.truncate_mode_combo.setEnabled(enabled_state_mismatch)
        self.render_untruncated_checkbox.setEnabled(enabled_state_mismatch)

        # manage "Save Optimized Shapes Only" checkbox state
        if hasattr(self, "save_optimized_only_checkbox"):
            can_enable_optimized_save = (
                self.save_shapes_checkbox.isEnabled()  # must also be saving shapes generally
                and shapes_loaded  # must have loaded shapes
                and not self.is_running  # not during a run
            )
            self.save_optimized_only_checkbox.setEnabled(can_enable_optimized_save)
            if not can_enable_optimized_save:  # if conditions not met, uncheck it
                self.save_optimized_only_checkbox.setChecked(False)

    def _populate_coord_fixing_widgets(self):
        """
        Dynamically populates the coordinate fixing section of the UI.

        Clears existing widgets and adds new ones (checkboxes, spinboxes)
        based on the currently selected shape type ('x', 'y' for circle).
        """
        # clear existing widgets in the layout
        while self.coord_fix_layout.count():
            item = self.coord_fix_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:  # it might be a sub-layout
                layout_item = item.layout()
                if layout_item:
                    self._clear_layout(layout_item)  # recursively clear sub-layouts
        self.coord_fix_widgets.clear()  # clear the storage dict

        shape_type_val = self.shape_type_combo.currentText()
        coord_param_names = []  # names of coordinate parameters for this shape
        if shape_type_val in ("circle", "rectangle", "triangle", "image"):
            coord_param_names = ["x", "y"]
        elif (
            shape_type_val == "line"
        ):  # line uses x1, y1 as its primary coords in UI/parser
            coord_param_names = ["x", "y"]  # these map to x1, y1 internally for line

        if (
            not coord_param_names
        ):  # if shape has no standard coords (hypothetical global shape)
            no_coords_label = QLabel("No fixable coordinate parameters for this shape.")
            no_coords_label.setStyleSheet("font-style: italic; color: gray;")
            self.coord_fix_layout.addWidget(no_coords_label)
            return

        # create widgets for each coordinate parameter
        for name in coord_param_names:
            hbox = QHBoxLayout()
            fix_checkbox = QCheckBox(f"Fix '{name}'")  # "Fix 'x'"
            fix_checkbox.setToolTip(
                f"Prevent '{name}' from changing during optimization (fixed to initial value)."
            )
            set_value_checkbox = QCheckBox(
                "Set Value:"
            )  # option to set a specific value
            set_value_checkbox.setToolTip(
                f"Requires 'Fix {name}' to be checked.\nSet ALL shapes to this specific '{name}' value."
            )
            set_value_checkbox.setEnabled(
                False
            )  # disabled until "Fix 'name'" is checked
            value_spin = QDoubleSpinBox()
            value_spin.setRange(0, 10000)  # arbitrary large range for coords
            value_spin.setValue(100.0)  # default specific value
            value_spin.setDecimals(1)
            value_spin.setEnabled(False)  # disabled until "Set Value" is checked

            # connect toggles for enabled states
            fix_checkbox.toggled.connect(set_value_checkbox.setEnabled)
            set_value_checkbox.toggled.connect(value_spin.setEnabled)

            hbox.addWidget(fix_checkbox)
            hbox.addWidget(set_value_checkbox)
            hbox.addWidget(value_spin, 1)  # spinbox takes available space
            self.coord_fix_layout.addLayout(hbox)
            self.coord_fix_widgets[name] = (  # store widgets for later access
                fix_checkbox,
                set_value_checkbox,
                value_spin,
            )

    def create_parameter_fixing_section(self, parent_layout):
        """
        Creates the UI section for fixing non-coordinate parameters (radius, color components).

        This section is checkable itself, and contains a scrollable area for
        dynamically populated parameter fixing widgets.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        self.fix_params_group = QGroupBox("Parameter Fixing (Non-Coordinate)")
        self.fix_params_group.setCheckable(True)  # group can be enabled/disabled
        self.fix_params_group.setChecked(False)  # default to disabled
        main_fix_layout = QVBoxLayout(self.fix_params_group)

        scroll = QSubScrollArea()  # use aliased QScrollArea for clarity
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.fix_params_layout = QVBoxLayout(scroll_widget)  # layout inside scroll area
        scroll.setWidget(scroll_widget)
        scroll.setMinimumHeight(100)  # ensure it's not too small
        main_fix_layout.addWidget(scroll)
        parent_layout.addWidget(self.fix_params_group)

    def update_fixable_parameters_display(self):
        """
        Dynamically populates the non-coordinate parameter fixing section.

        Clears existing widgets and adds new ones based on the currently
        selected shape type and color scheme, as these affect the available
        parameters ('radius' for circle, 'r', 'g', 'b' for RGB color).
        """
        # clear existing widgets
        while self.fix_params_layout.count():
            item = self.fix_params_layout.takeAt(0)
            widget = None
            sub_layout = None
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
                else:  # item might be a layout itself
                    sub_layout = item.layout()
                    if sub_layout:
                        self._clear_layout(sub_layout)
        self.fix_param_widgets.clear()  # clear storage

        shape_type_val = self.shape_type_combo.currentText()
        if not shape_type_val:  # need a shape type to know params
            self.log_message(
                "Warning: No shape type selected. Cannot update fixable params."
            )
            return

        color_scheme_val = self.color_scheme_combo.currentText()
        shape_params_include_rgb_val = color_scheme_val in ("rgb", "both")
        shape_params_include_alpha_val = True  # alpha is always structurally present

        try:
            shape_config = self.config_loader.load_shape_config(shape_type_val)
            config_params_from_json = shape_config.get(
                "parameters", []
            )  # [{"name":"radius", ...}]
            config_param_names_list = [p["name"] for p in config_params_from_json]

            # determine parameter names to use for the index map
            # if shapes are loaded, their param_names config might be different (older version)
            # but for UI display, we prefer the current config if not loading, or loaded config if loading that shape type
            temp_typed_list = NumbaList()  # for _get_param_index_map
            param_names_to_use_for_map = config_param_names_list
            if (
                self.loaded_shapes_data
                and self.loaded_shapes_data.get("shape_type") == shape_type_val
            ):
                # if loading shapes of the current type, use their param name config for consistency
                # this is subtle: fixable params UI should reflect what *can* be fixed for *current settings*
                # however, the _get_param_index_map needs the param_names that define the *structure*
                # for newly initialized shapes, this is config_param_names_list
                # for loaded shapes, it's self.loaded_shapes_data.get("param_names", [])
                # for UI display, we want to show all possible fixable params for the *selected* shape type
                # so, use config_param_names_list for determining what to display.
                # the full_param_index_map will use the *actual* param_names relevant to the data (new or loaded) later
                pass  # already set to config_param_names_list

            if param_names_to_use_for_map:  # effectively config_param_names_list here
                for name in param_names_to_use_for_map:
                    temp_typed_list.append(name)

            # get all possible parameter names for the current shape & color config
            full_param_index_map = optimizers._get_param_index_map(
                shape_type_val,
                temp_typed_list,  # use the current config's param names
                shape_params_include_rgb_val,
                shape_params_include_alpha_val,
            )

            coord_names = [
                "x",
                "y",
                "x1",
                "y1",
            ]  # list of coordinate names to exclude from this section
            # determine display order: config params first, then others (like color, alpha)
            display_order = param_names_to_use_for_map[:]  # start with config params
            for name in sorted(
                full_param_index_map.keys()
            ):  # add others if not already present
                if name not in display_order and name not in coord_names:
                    display_order.append(name)

            for name in display_order:
                if (
                    name not in full_param_index_map
                ):  # should not happen if display_order is correct
                    continue
                if name in coord_names:  # skip coordinate parameters, handled elsewhere
                    continue

                hbox = QHBoxLayout()
                checkbox = QCheckBox(name)  # checkbox to enable fixing this param
                value_spin = QDoubleSpinBox()  # spinbox for the fixed value
                value_spin.setRange(-1e6, 1e6)  # generic large range
                value_spin.setDecimals(3)  # default decimals
                value_spin.setEnabled(False)  # disabled until checkbox is checked

                # try to get min/max/initial from shape config for better defaults
                param_config_details = next(
                    (p for p in config_params_from_json if p["name"] == name), None
                )
                min_val, max_val = -1e6, 1e6  # fallback bounds
                default_val = 0.5  # fallback initial value
                if param_config_details:  # if parameter is in shape's json config
                    min_val = param_config_details.get("min", min_val)
                    max_val = param_config_details.get("max", max_val)
                    default_val = param_config_details.get(
                        "initial", (min_val + max_val) / 2.0
                    )
                elif (
                    name
                    in [  # special handling for color/alpha params if not in config
                        "r",
                        "g",
                        "b",
                        "gray",
                        "alpha",
                        "stroke_r",
                        "stroke_g",
                        "stroke_b",
                        "stroke_gray",
                    ]
                ):  # these are typically 0-1 range (before scaling to 0-255)
                    min_val, max_val = 0.0, 1.0
                    default_val = 0.5

                value_spin.setRange(min_val, max_val)
                value_spin.setValue(default_val)

                # lambda ensures spin arg is captured at definition time
                checkbox.toggled.connect(
                    lambda checked, spin=value_spin: spin.setEnabled(checked)
                )
                hbox.addWidget(checkbox)
                hbox.addWidget(value_spin, 1)  # spinbox takes available space
                self.fix_params_layout.addLayout(hbox)
                self.fix_param_widgets[name] = (
                    checkbox,
                    value_spin,
                )  # store for later access

        except FileNotFoundError:  # if shape config json is missing
            self.log_message(
                f"Warning: Shape config for '{shape_type_val}' not found. Cannot populate fixable params."
            )
        except Exception as e:
            self.log_message(f"Error updating fixable params display: {e}")
            traceback.print_exc()
        self.fix_params_layout.addStretch(1)  # push widgets to top

    def create_optimization_section(self, parent_layout):
        """
        Creates the UI section for optimization method, metrics, number of shapes, and method specific parameters.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        opt_frame = QGroupBox("Optimization Settings")
        layout = QVBoxLayout(opt_frame)

        # optimization method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.opt_method_combo = QComboBox()
        self.opt_method_combo.addItems(
            [
                "Simulated Annealing",
                "Particle Swarm",
                "Hill Climbing",
            ]  # available methods
        )
        if "Simulated Annealing" in [  # set a default if available
            self.opt_method_combo.itemText(i)
            for i in range(self.opt_method_combo.count())
        ]:
            self.opt_method_combo.setCurrentText("Simulated Annealing")
        self.opt_method_combo.currentTextChanged.connect(
            self.update_method_parameters
        )  # update params on change
        method_layout.addWidget(self.opt_method_combo)
        layout.addLayout(method_layout)

        # evaluation metrics group
        metric_group = QGroupBox(
            "Evaluation Metrics (Select 1-3, Weights must sum to 1.0)"
        )
        metric_layout = QVBoxLayout(metric_group)
        self.weight_sum_label = QLabel(
            "Current Weight Sum: 1.000"
        )  # displays sum of weights
        self.metric_widgets: List[Tuple[QCheckBox, QComboBox, QDoubleSpinBox]] = []
        for i in range(3):  # allow up to 3 metrics
            slot_layout = QHBoxLayout()
            checkbox = QCheckBox(f"Use Metric {i+1}")
            combo = QComboBox()
            combo.addItems(self.available_metrics)
            weight_spin = QDoubleSpinBox()
            weight_spin.setRange(0.0, 1.0)
            weight_spin.setDecimals(3)
            weight_spin.setSingleStep(0.05)
            weight_spin.setValue(0.0)  # default weight

            combo.setEnabled(False)  # disabled until checkbox checked
            weight_spin.setEnabled(False)

            checkbox.toggled.connect(
                self.update_metric_slot_state
            )  # enable/disable on toggle
            weight_spin.valueChanged.connect(
                self.update_weights_label
            )  # update sum label on change

            slot_layout.addWidget(checkbox)
            slot_layout.addWidget(combo, 1)  # combo takes more space
            slot_layout.addWidget(QLabel("Weight:"))
            slot_layout.addWidget(weight_spin)
            metric_layout.addLayout(slot_layout)
            self.metric_widgets.append((checkbox, combo, weight_spin))  # store widgets

        # set default for the first metric slot
        self.metric_widgets[0][0].setChecked(True)
        if "mse" in self.available_metrics:  # prefer mse as default if available
            self.metric_widgets[0][1].setCurrentText("mse")
        else:  # else first in list
            if self.available_metrics:
                self.metric_widgets[0][1].setCurrentIndex(0)
        self.metric_widgets[0][2].setValue(1.0)  # default weight for first metric

        self.update_metric_slot_state()  # initial enable/disable state
        metric_layout.addWidget(self.weight_sum_label)
        layout.addWidget(metric_group)

        # number of shapes
        shapes_layout = QHBoxLayout()
        shapes_layout.addWidget(QLabel("Num Shapes:"))
        self.num_shapes_spinbox = QSpinBox()
        self.num_shapes_spinbox.setRange(1, 10000)  # min 1 shape, max 10k
        self.num_shapes_spinbox.setValue(100)  # default
        self.num_shapes_spinbox.setSingleStep(10)
        shapes_layout.addWidget(self.num_shapes_spinbox)
        shapes_layout.addStretch(1)  # push to left
        layout.addLayout(shapes_layout)

        # gif generation options
        gif_layout = QHBoxLayout()
        gif_layout.addWidget(QLabel("Generate GIF:"))
        self.gif_options_combo = QComboBox()
        self.gif_options_combo.addItems(
            ["No GIF", "Single Loop", "Infinite Loop", "Both"]
        )
        gif_layout.addWidget(self.gif_options_combo)
        layout.addLayout(gif_layout)

        # frame for method specific parameters (temperature, swarm size)
        self.params_frame = QFrame()
        self.params_frame.setFrameShape(QFrame.Shape.StyledPanel)  # add a border
        self.params_layout = QVBoxLayout(self.params_frame)  # layout for these params
        layout.addWidget(self.params_frame)

        parent_layout.addWidget(opt_frame)
        self.update_method_parameters(
            self.opt_method_combo.currentText()
        )  # populate initial

    def update_metric_slot_state(self):
        """
        Updates the enabled state of metric combo boxes and weight spinboxes.

        Called when a "Use Metric X" checkbox is toggled. Ensures at least one
        metric remains selected.
        """
        any_checked = False
        sender_widget = self.sender()  # the checkbox that was toggled

        for checkbox, combo, spin in self.metric_widgets:
            is_checked = checkbox.isChecked()
            combo.setEnabled(is_checked)
            spin.setEnabled(is_checked)
            if not is_checked:  # if unchecked, reset weight to 0
                spin.blockSignals(True)  # prevent valueChanged signal
                spin.setValue(0.0)
                spin.blockSignals(False)
            if is_checked:
                any_checked = True

        # ensure at least one metric is always selected
        if not any_checked and isinstance(sender_widget, QCheckBox):
            # if user tried to uncheck the last one, recheck it
            sender_widget.blockSignals(True)
            sender_widget.setChecked(True)
            sender_widget.blockSignals(False)
            # reenable its corresponding combo and spin
            for checkbox, combo, spin in self.metric_widgets:
                if checkbox == sender_widget:
                    combo.setEnabled(True)
                    spin.setEnabled(True)
                    break
            self.update_weights_label()  # update sum label
            QMessageBox.warning(  # inform user
                self, "Metric Selection", "At least one metric must be selected."
            )
            return
        self.update_weights_label()  # update sum after any change

    def _get_weights_validity(self) -> bool:
        """
        Checks if the current metric weights are valid (at least one selected, sum is close to 1.0).

        :return: True if weights are valid, False otherwise.
        :rtype: bool
        """
        current_sum = 0.0
        selected_count = 0
        for checkbox, _, spin in self.metric_widgets:
            if checkbox.isChecked():
                current_sum += spin.value()
                selected_count += 1
        return selected_count > 0 and np.isclose(current_sum, 1.0)

    def update_weights_label(self):
        """
        Updates the label displaying the sum of current metric weights.

        Also changes the label color to red if the sum is not close to 1.0.
        """
        current_sum = 0.0
        selected_count = 0
        for checkbox, _, spin in self.metric_widgets:
            if checkbox.isChecked():
                current_sum += spin.value()
                selected_count += 1

        self.weight_sum_label.setText(f"Current Weight Sum: {current_sum:.3f}")
        is_valid = self._get_weights_validity()
        palette = self.weight_sum_label.palette()
        # set text color based on validity
        color = QColor("black") if is_valid else QColor("red")
        palette.setColor(self.weight_sum_label.foregroundRole(), color)
        self.weight_sum_label.setPalette(palette)
        self.check_run_conditions()  # run button state depends on weight validity

    def create_output_section(self, parent_layout):
        """
        Creates the UI section for results display options, plotting, and saving settings.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        output_frame = QGroupBox("Results & Saving")
        layout = QVBoxLayout(output_frame)

        # plot buttons
        plot_layout = QHBoxLayout()
        self.show_plot_btn = QPushButton("Show Overall Metric History")
        self.show_plot_btn.clicked.connect(self.show_metric_plot_with_data)
        self.show_plot_btn.setEnabled(False)  # enabled after run completes with data
        plot_layout.addWidget(self.show_plot_btn)

        self.show_pso_plot_btn = QPushButton("Show PSO Score History")
        self.show_pso_plot_btn.clicked.connect(self.show_pso_plot)
        self.show_pso_plot_btn.setEnabled(False)  # enabled for pso after run
        plot_layout.addWidget(self.show_pso_plot_btn)
        layout.addLayout(plot_layout)

        # saving options
        self.save_results_checkbox = QCheckBox("Save Detailed Results File (.json)")
        self.save_results_checkbox.setChecked(True)  # default to save
        self.save_results_checkbox.toggled.connect(self._update_save_checkboxes_state)
        layout.addWidget(self.save_results_checkbox)

        self.save_shapes_checkbox = QCheckBox("Save Shape Data (.npy)")
        self.save_shapes_checkbox.setToolTip(
            "Requires 'Save Detailed Results' to be checked."  # .npy path is stored in .json
        )
        self.save_shapes_checkbox.setChecked(True)  # default to save
        self.save_shapes_checkbox.toggled.connect(self._update_save_checkboxes_state)
        self.save_shapes_checkbox.setEnabled(
            self.save_results_checkbox.isChecked()
        )  # dependent
        layout.addWidget(self.save_shapes_checkbox)

        self.save_optimized_only_checkbox = QCheckBox("Save Optimized Shapes Only")
        self.save_optimized_only_checkbox.setToolTip(
            "If checked, saves only the K shapes that were actively optimized in this run.\nIf unchecked (and shapes were loaded): \n - K < N (truncation): Merges optimized K shapes back into original N shapes.\n - Render Canvas: Saves original N loaded shapes + K newly optimized shapes.\n - K >= N: Saves the K optimized shapes (same as checked).\n(Only relevant when 'Save Shape Data' is checked and shapes were initially loaded)"
        )
        self.save_optimized_only_checkbox.setChecked(False)  # default to merge/combine
        self.save_optimized_only_checkbox.setEnabled(
            False
        )  # dependent on loaded shapes & save_shapes
        hbox_optimized_save = QHBoxLayout()
        hbox_optimized_save.addSpacing(20)  # indent this option
        hbox_optimized_save.addWidget(self.save_optimized_only_checkbox)
        layout.addLayout(hbox_optimized_save)

        parent_layout.addWidget(output_frame)

    def _update_save_checkboxes_state(self):
        """
        Updates the enabled state of save-related checkboxes based on dependencies.
        """
        save_results_checked = self.save_results_checkbox.isChecked()
        # "save shapes" enabled only if "save results" is checked and not running
        self.save_shapes_checkbox.setEnabled(
            save_results_checked and not self.is_running
        )
        if (
            not save_results_checked
        ):  # if "save results" is off, "save shapes" must be off
            self.save_shapes_checkbox.setChecked(False)
        # state of "save optimized only" also depends on loaded shapes, handled in update_load_mismatch_widgets_state
        self.update_load_mismatch_widgets_state()

    def create_control_section(self, parent_layout):
        """
        Creates the main control section with the Run/Stop button.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        control_frame = QFrame()  # use a frame for potential future styling
        layout = QHBoxLayout(control_frame)
        layout.setContentsMargins(0, 0, 0, 0)  # no margins for this frame

        self.run_stop_btn = QPushButton("Run Approximation")
        self.run_stop_btn.setStyleSheet(
            "font-size: 16px; padding: 10px;"
        )  # make it prominent
        # initial connection is to start_approximation
        self.run_stop_btn.clicked.connect(self.start_approximation)
        layout.addWidget(self.run_stop_btn, 1)  # button takes full width of its cell
        parent_layout.addWidget(control_frame)

    def create_image_display(self, parent_layout):
        """
        Creates the UI section for displaying input and output image previews.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        display_frame = QGroupBox("Image Preview")
        layout = QHBoxLayout(display_frame)  # side-by-side input and output

        # input image display
        input_vbox = QVBoxLayout()
        input_vbox.addWidget(QLabel("Input Image"))
        self.input_image_label = QLabel("No image selected")
        self.input_image_label.setMinimumSize(200, 200)  # ensure it has some size
        self.input_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_image_label.setFrameStyle(
            QFrame.Shape.Box | QFrame.Shadow.Sunken
        )  # add border
        # ignored size policy allows it to shrink/grow with window, scaled pixmap handles content
        self.input_image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        input_vbox.addWidget(
            self.input_image_label, 1
        )  # label takes available vertical space
        layout.addLayout(input_vbox)

        # output image display
        output_vbox = QVBoxLayout()
        output_vbox.addWidget(QLabel("Output Image"))
        self.output_image_label = QLabel("No output yet")
        self.output_image_label.setMinimumSize(200, 200)
        self.output_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_image_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.output_image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        output_vbox.addWidget(self.output_image_label, 1)
        layout.addLayout(output_vbox)

        parent_layout.addWidget(display_frame)

    def create_results_section(self, parent_layout):
        """
        Creates the UI section for displaying log messages and the progress bar.

        :param parent_layout: The parent QVBoxLayout to add this section to.
        :type parent_layout: QVBoxLayout
        """
        results_frame = QGroupBox("Log Output")
        layout = QVBoxLayout(results_frame)

        self.results_text = QTextEdit()  # for log messages
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(100)  # ensure some space for logs
        layout.addWidget(self.results_text)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # percentage
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)  # show text on progress bar
        self.progress_bar.setFormat("Waiting... 0%")  # initial text
        layout.addWidget(self.progress_bar)

        parent_layout.addWidget(results_frame)

    def update_progress_bar(
        self, current_step, total_steps, percentage, elapsed_sec, remaining_sec
    ):
        """
        Updates the progress bar with current step, percentage, and time estimates.

        :param current_step: Current step of the optimization.
        :type current_step: int
        :param total_steps: Total estimated steps for the optimization.
        :type total_steps: int
        :param percentage: Current progress percentage.
        :type percentage: float
        :param elapsed_sec: Time elapsed in seconds.
        :type elapsed_sec: float
        :param remaining_sec: Estimated time remaining in seconds.
        :type remaining_sec: float
        """
        current_percent_val = self.progress_bar.value()
        new_percent_val = int(percentage)
        # avoid progress bar going backwards if updates arrive out of order or from different stages
        if new_percent_val < current_percent_val and self.is_running:
            return
        if not self.is_running:  # don't update if not running (after stop)
            return

        self.progress_bar.setValue(new_percent_val)
        elapsed_str = time.strftime(
            "%H:%M:%S", time.gmtime(elapsed_sec)
        )  # format H:M:S
        rem_str = "..."
        if remaining_sec < np.inf and remaining_sec >= 0:  # if calculable
            rem_str = time.strftime("%H:%M:%S", time.gmtime(remaining_sec))
        elif remaining_sec < 0:  # sometimes happens at very start or end
            rem_str = "Calculating..."
        progress_text = f"Step {current_step}/{total_steps} ({percentage:.1f}%) | Elapsed: {elapsed_str} | Remaining: {rem_str}"
        self.progress_bar.setFormat(progress_text)

    def request_stop(self):
        """
        Requests the running approximation process to stop.

        Sets the stop_flag in the app_instance and disables the Run/Stop button
        to prevent further clicks until the process acknowledges the stop.
        """
        if self.is_running:
            self.log_message("... Stop Requested ...")
            self.app_instance.stop_flag[0] = 1  # signal the optimizer to stop
            self.run_stop_btn.setEnabled(False)  # disable button temporarily

    def _enable_plot_button(self):
        """Enables the 'Show Overall Metric History' plot button."""
        self.show_plot_btn.setEnabled(True)

    def _enable_pso_plot_button(self):
        """Enables the 'Show PSO Score History' plot button."""
        self.show_pso_plot_btn.setEnabled(True)

    def _enable_controls(self, enabled=True):
        """
        Enables or disables various UI controls, typically based on whether an approximation is running.

        :param enabled: True to enable controls (idle state), False to disable (running state).
        :type enabled: bool
        """
        can_enable_idle = (
            enabled and not self.is_running
        )  # true only if idle and requested to enable

        # enable/disable general input controls
        self.reload_config_btn.setEnabled(can_enable_idle)
        self.load_shapes_btn.setEnabled(can_enable_idle)
        self.select_image_btn.setEnabled(can_enable_idle)
        self.select_output_btn.setEnabled(can_enable_idle)
        self.shape_type_combo.setEnabled(can_enable_idle)
        self.color_scheme_combo.setEnabled(can_enable_idle)
        self.border_type_combo.setEnabled(can_enable_idle)
        self.param_init_combo.setEnabled(can_enable_idle)
        self.coord_init_mode_combo.setEnabled(can_enable_idle)

        # enable/disable coordinate fixing widgets based on their own state and can_enable_idle
        for name, (fix_cb, set_val_cb, val_spin) in self.coord_fix_widgets.items():
            fix_cb.setEnabled(can_enable_idle)
            can_set_val_cb = fix_cb.isChecked()  # "set value" cb depends on "fix" cb
            set_val_cb.setEnabled(can_enable_idle and can_set_val_cb)
            can_input_val_spin = (
                set_val_cb.isChecked()
            )  # spin depends on "set value" cb
            val_spin.setEnabled(
                can_enable_idle and can_set_val_cb and can_input_val_spin
            )

        # optimization controls
        self.opt_method_combo.setEnabled(can_enable_idle)
        self.num_shapes_spinbox.setEnabled(can_enable_idle)
        self.gif_options_combo.setEnabled(can_enable_idle)
        self.save_results_checkbox.setEnabled(can_enable_idle)
        self._update_save_checkboxes_state()  # handles dependent save checkboxes

        # metric widgets
        for checkbox, combo, spin in self.metric_widgets:
            checkbox.setEnabled(can_enable_idle)
            is_checked_metric = checkbox.isChecked()
            combo.setEnabled(can_enable_idle and is_checked_metric)
            spin.setEnabled(can_enable_idle and is_checked_metric)

        # method specific parameter frame and its contents
        self.params_frame.setEnabled(can_enable_idle)
        for i in range(
            self.params_layout.count()
        ):  # iterate over widgets in params_layout
            item = self.params_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:  # if it's a direct widget
                    widget.setEnabled(can_enable_idle)
                else:  # if it's a sub-layout (QHBoxLayout for slider+label)
                    sub_layout = item.layout()
                    if sub_layout:
                        for j in range(sub_layout.count()):
                            layout_item = sub_layout.itemAt(j)
                            if layout_item:
                                sub_widget = layout_item.widget()
                                if sub_widget:
                                    sub_widget.setEnabled(can_enable_idle)

        # non-coordinate parameter fixing group and its contents
        self.fix_params_group.setEnabled(can_enable_idle)
        fix_group_is_checked_val = self.fix_params_group.isChecked()
        for name, (checkbox, spin) in self.fix_param_widgets.items():
            checkbox.setEnabled(can_enable_idle and fix_group_is_checked_val)
            is_checked_param_fix = checkbox.isChecked()
            spin.setEnabled(
                can_enable_idle and fix_group_is_checked_val and is_checked_param_fix
            )

        self.update_load_mismatch_widgets_state()  # handles loaded shape specific ui

    def _on_run_finished(self):
        """
        Handles cleanup and UI updates after an approximation run finishes or is stopped.

        Resets the Run/Stop button, enables controls, and updates progress bar status.
        """
        was_stopped = self.app_instance.stop_flag[0] == 1  # check if stopped by user
        self.is_running = False  # update running state
        self.app_instance.stop_flag[0] = 0  # reset stop flag for next run
        self._enable_controls(True)  # reenable all appropriate ui controls

        # reset run/stop button to "Run" mode
        try:
            self.run_stop_btn.clicked.disconnect(
                self.request_stop
            )  # remove "stop" action
        except TypeError:  # if not connected (first run or already disconnected)
            pass
        self.run_stop_btn.setText("Run Approximation")
        self.run_stop_btn.setToolTip("Run the approximation process")
        try:
            self.run_stop_btn.clicked.connect(
                self.start_approximation
            )  # add "run" action
        except (
            TypeError
        ):  # if already connected (should not happen if disconnect worked)
            pass

        self.log_message("Processing finished.")
        final_progress_val = self.progress_bar.value()
        # set final progress bar message
        if was_stopped:
            self.progress_bar.setFormat(f"Stopped at {final_progress_val}%")
        elif final_progress_val < 100:  # if finished early but not stopped (error)
            self.progress_bar.setFormat(f"Finished ({final_progress_val}%)")
        else:  # normal completion
            self.progress_bar.setFormat("Finished 100%")
        # revert to default message after a delay
        QTimer.singleShot(2500, lambda: self.progress_bar.setFormat("Waiting... 0%"))

        self.check_run_conditions()  # recheck run button enable state

    def update_method_parameters(self, method):
        """
        Dynamically updates the UI section for method specific hyperparameters.

        Clears existing parameter widgets and adds new ones (sliders, spinboxes)
        based on the selected optimization method.

        :param method: The display name of the selected optimization method ("Simulated Annealing").
        :type method: str
        """
        self._clear_layout(self.params_layout)  # clear previous method's params

        # define parameters for each method
        # format: {label: (min, max, default, widget_type, internal_name)}
        parameters = {
            "Simulated Annealing": {
                "Initial Temp": (0.1, 100.0, 1.0, "float", "init_temp"),
                "Cooling Rate": (0.80, 0.9999, 0.95, "float", "cooling_rate"),
                "Iterations": (1, 10000000, 10000, "int_spin", "iterations"),
            },
            "Particle Swarm": {
                "Swarm Size": (2, 1000, 50, "int_spin", "swarm_size"),
                "Cognitive Param": (0.1, 4.0, 1.5, "float", "cognitive_param"),
                "Social Param": (0.001, 4.0, 1.5, "float", "social_param"),
                "Inertia Weight": (0.1, 1.0, 0.7, "float", "inertia_weight"),
                "Iterations": (1, 10000000, 1000, "int_spin", "iterations"),
            },
            "Hill Climbing": {
                "Iterations": (1, 10000000, 10000, "int_spin", "iterations"),
            },
            # genetic algorithm and differential evolution are placeholders, not fully implemented in core optimizers
            "Genetic Algorithm": {
                "Population Size": (10, 1000, 100, "int", "population_size"),
                "Mutation Rate": (0.0, 1.0, 0.1, "float", "mutation_rate"),
                "Generations": (1, 100000, 100, "int_spin", "generations"),
            },
            "Differential Evolution": {
                "Population Size": (10, 1000, 100, "int", "population_size"),
                "Crossover Rate": (0.0, 1.0, 0.7, "float", "crossover_rate"),
                "Mutation Factor": (0.1, 2.0, 0.8, "float", "mutation_factor"),
                "Generations": (1, 100000, 100, "int_spin", "generations"),
            },
        }

        if method in parameters:
            for label_text, (
                min_val,
                max_val,
                default,
                ptype,  # widget type: "float", "int" (slider), "int_spin"
                internal_name,  # name used in params dict
            ) in parameters[method].items():
                param_widget = QWidget()  # container for label + input widget
                param_widget.setObjectName(
                    f"param_{internal_name}"
                )  # for later lookup if needed
                param_layout = QHBoxLayout(param_widget)
                param_layout.setContentsMargins(0, 0, 0, 0)  # compact layout
                label = QLabel(label_text + ":")
                value_widget = None  # the actual input widget (slider, spinbox)
                value_label = None  # for sliders, to display current value

                if ptype == "float":
                    value_widget = QDoubleSpinBox()
                    value_widget.setRange(min_val, max_val)
                    # adjust decimals and step for better usability
                    decimals = (
                        4 if ("Rate" in label_text or "Inertia" in label_text) else 2
                    )
                    step = 0.0001 if ("Rate" in label_text) else 0.05
                    value_widget.setDecimals(decimals)
                    value_widget.setSingleStep(step)
                    value_widget.setValue(default)
                elif ptype == "int":  # integer slider
                    value_widget = QSlider(Qt.Orientation.Horizontal)
                    value_widget.setRange(int(min_val), int(max_val))
                    value_widget.setValue(int(default))
                    value_label = QLabel(
                        str(int(default))
                    )  # label to show slider value
                    # lambda captures lbl at definition time
                    value_widget.valueChanged.connect(
                        lambda v, lbl=value_label: lbl.setText(str(v))
                    )
                elif ptype == "int_spin":  # integer spinbox
                    value_widget = QSpinBox()
                    value_widget.setRange(int(min_val), int(max_val))
                    value_widget.setValue(int(default))
                    # set appropriate step sizes for different parameters
                    step = 10  # default step
                    if internal_name == "swarm_size" and method == "Particle Swarm":
                        step = 5
                    elif internal_name == "iterations":
                        if method == "Particle Swarm":
                            step = 100
                        elif method in ["Simulated Annealing", "Hill Climbing"]:
                            step = 1000
                    value_widget.setSingleStep(step)
                    value_widget.setGroupSeparatorShown(True)  # 10,000

                if value_widget:
                    param_layout.addWidget(label)
                    param_layout.addWidget(value_widget, 1)  # input widget takes space
                if value_label:  # if it was a slider, add its value label
                    param_layout.addWidget(value_label)
                self.params_layout.addWidget(param_widget)  # add to main params layout

    def _clear_layout(self, layout):
        """
        Recursively clears all widgets and sub-layouts from a given QLayout.

        :param layout: The QLayout to clear.
        :type layout: QLayout
        """
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()  # remove and delete widget
            else:
                sub_layout = item.layout()
                if sub_layout:
                    self._clear_layout(sub_layout)  # recurse for sub-layouts

    def select_input_image(self):
        """
        Handles the "Select Input Image" button action.

        Opens a file dialog for image selection and updates the input image preview.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            "",  # start in current/last directory
            "Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)",  # filter for common image types
        )
        if file_path:
            self.input_image_path = file_path
            self.input_image_path_label.setText(
                os.path.basename(file_path)
            )  # show filename
            pixmap = None
            try:
                # use Pillow for robust image loading, then convert to QImage/QPixmap
                image = Image.open(file_path)
                # handle different PIL modes for QImage conversion
                if image.mode == "RGBA":
                    qimage = QImage(
                        image.tobytes("raw", "RGBA"),
                        image.width,
                        image.height,
                        QImage.Format.Format_RGBA8888,
                    )
                elif image.mode == "RGB":
                    qimage = QImage(
                        image.tobytes("raw", "RGB"),
                        image.width,
                        image.height,
                        QImage.Format.Format_RGB888,
                    )
                elif image.mode == "L":  # grayscale
                    qimage = QImage(
                        image.tobytes("raw", "L"),
                        image.width,
                        image.height,
                        QImage.Format.Format_Grayscale8,
                    )
                elif image.mode == "P":  # palette mode, convert to RGBA
                    image = image.convert("RGBA")
                    qimage = QImage(
                        image.tobytes("raw", "RGBA"),
                        image.width,
                        image.height,
                        QImage.Format.Format_RGBA8888,
                    )
                else:  # try converting other modes to RGBA as a fallback
                    try:
                        image = image.convert("RGBA")
                        qimage = QImage(
                            image.tobytes("raw", "RGBA"),
                            image.width,
                            image.height,
                            QImage.Format.Format_RGBA8888,
                        )
                    except Exception as conv_e:
                        raise ValueError(
                            f"Unsupported PIL mode '{image.mode}': {conv_e}"
                        )

                if qimage.isNull():  # check if QImage creation failed
                    raise ValueError("Failed QImage creation from PIL image.")
                pixmap = QPixmap.fromImage(qimage)
                if pixmap.isNull():  # check if QPixmap creation failed
                    raise ValueError("Failed QPixmap creation from QImage.")

                self._original_input_pixmap = pixmap  # store original for rescaling
                self._rescale_input_preview()  # display scaled preview
            except Exception as e:
                error_msg = f"Preview Error:\n{e}"
                self.input_image_label.setText(error_msg)
                self.log_message(f"Error loading/previewing input image: {e}")
                traceback.print_exc()
                self.input_image_path = None  # reset path on error
                self.input_image_path_label.setText("Error loading image.")
                self._original_input_pixmap = None
            self.check_run_conditions()  # update run button state

    def resizeEvent(self, event):
        """
        Handles window resize events to rescale image previews.

        :param event: The QResizeEvent.
        :type event: QResizeEvent
        """
        super().resizeEvent(event)
        self._rescale_input_preview()  # rescale input image
        self._rescale_output_preview()  # rescale output image

    def _rescale_input_preview(self):
        """Rescales the stored original input pixmap to fit the input_image_label."""
        if (
            hasattr(self, "_original_input_pixmap")
            and self._original_input_pixmap
            and not self._original_input_pixmap.isNull()
        ):
            label_size = self.input_image_label.size()
            if (
                label_size.width() > 0 and label_size.height() > 0
            ):  # ensure label has valid size
                scaled_pixmap = self._original_input_pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,  # maintain aspect ratio
                    Qt.TransformationMode.SmoothTransformation,  # better quality scaling
                )
                self.input_image_label.setPixmap(scaled_pixmap)
                self.input_image_label.setText("")  # clear any error text
        else:  # if no valid pixmap, show default text
            if not self.input_image_path:
                self.input_image_label.setText("No image selected")

    def _rescale_output_preview(self):
        """Rescales the stored original output pixmap to fit the output_image_label."""
        if (
            hasattr(self, "_original_output_pixmap")
            and self._original_output_pixmap
            and not self._original_output_pixmap.isNull()
        ):
            label_size = self.output_image_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = self._original_output_pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.output_image_label.setPixmap(scaled_pixmap)
                self.output_image_label.setText("")
        else:  # if no valid output, show default text
            self.output_image_label.setText("No output yet")

    def select_output_directory(self):
        """
        Handles the "Select Output Directory" button action.

        Opens a directory dialog and updates the output directory path label.
        """
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_directory = dir_path
            try:  # try to create a shorter display path for the label
                base_name = os.path.basename(dir_path)
                parent_name = os.path.basename(os.path.dirname(dir_path))
                display_path = (  # ".../parent/base"
                    os.path.join("...", parent_name, base_name)
                    if parent_name
                    else base_name
                )
            except Exception:  # fallback to full path if shortening fails
                display_path = dir_path
            self.output_dir_path_label.setText(display_path)
            self.output_dir_path_label.setToolTip(dir_path)  # full path in tooltip
            self.check_run_conditions()  # update run button state

    def get_selected_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Collects all selected parameters from the GUI into a dictionary.

        This dictionary is then passed to the `App.run_approximation` method.
        Performs validation for essential inputs like image path, output directory,
        and metric selection/weights.

        :return: A dictionary of parameters, or None if validation fails.
        :rtype: Optional[Dict[str, Any]]
        """
        if not self.input_image_path or not self.output_directory:
            self.log_message("Error: Input image or output directory not selected.")
            return None

        # collect selected metrics and their weights
        selected_metrics_list = []
        raw_weights_list = []
        for checkbox, combo, spin in self.metric_widgets:
            if checkbox.isChecked():
                selected_metrics_list.append(combo.currentText())
                raw_weights_list.append(spin.value())

        if not selected_metrics_list:  # must have at least one metric
            self.log_message("Error: No evaluation metrics selected.")
            return None

        # normalize weights if sum is not 1.0
        sum_raw_weights_val = sum(raw_weights_list)
        normalized_weights_list = [1.0 / len(selected_metrics_list)] * len(
            selected_metrics_list
        )  # default equal
        if sum_raw_weights_val > 1e-6:  # avoid division by zero
            normalized_weights_list = [
                w / sum_raw_weights_val for w in raw_weights_list
            ]
        if not np.isclose(sum_raw_weights_val, 1.0) and sum_raw_weights_val > 1e-6:
            self.log_message(
                f"Note: Metric weights normalized from sum {sum_raw_weights_val:.3f}."
            )
        elif sum_raw_weights_val <= 1e-6:  # if sum is near zero, use equal weights
            self.log_message(
                f"Warning: Metric weights sum to {sum_raw_weights_val:.3f}. Using equal weights."
            )

        # collect fixed non-coordinate parameters
        fixed_params_non_coord_list = []
        fixed_values_non_coord_list = []
        if self.fix_params_group.isChecked():  # only if the group is enabled
            for name, (checkbox, spin) in self.fix_param_widgets.items():
                if checkbox.isChecked():
                    fixed_params_non_coord_list.append(name)
                    fixed_values_non_coord_list.append(spin.value())

        # collect coordinate fixing details
        coord_fix_details_dict = {}
        for name, (fix_cb, set_val_cb, val_spin) in self.coord_fix_widgets.items():
            if fix_cb.isChecked():  # if "Fix 'name'" is checked
                if set_val_cb.isChecked():  # if "Set Value" is also checked
                    coord_fix_details_dict[name] = {
                        "mode": "specific",
                        "value": val_spin.value(),
                    }
                else:  # just "Fix 'name'" is checked, fix to initial
                    coord_fix_details_dict[name] = {"mode": "initial", "value": None}

        # map ui display names to internal optimizer/param names
        method_map_dict = {
            "Simulated Annealing": "sa",
            "Particle Swarm": "pso",
            "Hill Climbing": "hc",
        }
        gif_map_dict = {
            "No GIF": "none",
            "Single Loop": "single",
            "Infinite Loop": "infinite",
            "Both": "both",
        }
        coord_init_map_dict = {
            "Random": "random",
            "Grid": "grid",
            "Zero": "zero",
            "Intensity PDF": "intensity_pdf",
            "SSIM PDF": "ssim_pdf",
        }
        trunc_map_dict = {"Random": "random", "First K": "first", "Last K": "last"}

        coord_init_display_val = self.coord_init_mode_combo.currentText()
        coord_init_internal_val = coord_init_map_dict.get(
            coord_init_display_val, "random"
        )
        trunc_mode_display_val = self.truncate_mode_combo.currentText()
        trunc_mode_internal_val = trunc_map_dict.get(trunc_mode_display_val, "random")

        # determine if "save optimized only" is relevant and checked
        save_optimized_relevant_and_checked_val = False
        if self.loaded_shapes_data and self.save_shapes_checkbox.isChecked():
            save_optimized_relevant_and_checked_val = (
                self.save_optimized_only_checkbox.isChecked()
            )

        # construct the main parameters dictionary
        parameters_dict = {
            "input_image_path": self.input_image_path,
            "output_directory": self.output_directory,
            "shape_type": self.shape_type_combo.currentText(),
            "color_scheme": self.color_scheme_combo.currentText(),
            "border_type": self.border_type_combo.currentText(),
            "optimization_method": method_map_dict.get(
                self.opt_method_combo.currentText(), "sa"  # default "sa" if not found
            ),
            "evaluation_metrics": selected_metrics_list,
            "metric_weights": normalized_weights_list,
            "num_shapes": int(self.num_shapes_spinbox.value()),
            "gif_options": gif_map_dict.get(
                self.gif_options_combo.currentText(), "none"
            ),
            "method_params": {},  # for method specific hyperparams
            "param_init_type": self.param_init_combo.currentText().lower(),  # "midpoint"
            "coord_mode": coord_init_internal_val,
            "coord_fix_details": coord_fix_details_dict,
            "fixed_params_non_coord": fixed_params_non_coord_list,
            "fixed_values_non_coord": fixed_values_non_coord_list,
            "save_results_file": self.save_results_checkbox.isChecked(),
            "save_shape_data": self.save_shapes_checkbox.isChecked(),
            "save_optimized_only": save_optimized_relevant_and_checked_val,
            "use_loaded_shapes": self.loaded_shapes_data is not None,
            "loaded_shapes_array": (  # pass loaded array if exists
                self.loaded_shapes_data.get("array", None)
                if self.loaded_shapes_data
                else None
            ),
            "loaded_param_names_config": (  # pass loaded param names config if exists
                self.loaded_shapes_data.get("param_names", [])
                if self.loaded_shapes_data
                else []
            ),
            "render_loaded_as_canvas": (  # pass render as canvas flag
                self.render_canvas_checkbox.isChecked()
                if self.loaded_shapes_data
                else False
            ),
            "truncation_mode": (  # pass truncation mode
                trunc_mode_internal_val if self.loaded_shapes_data else "random"
            ),
            "render_untruncated_shapes": (  # pass render untruncated flag
                self.render_untruncated_checkbox.isChecked()
                if self.loaded_shapes_data
                else False
            ),
        }

        # collect method specific hyperparameters
        current_opt_method_display_val = self.opt_method_combo.currentText()
        # define which params are needed for each method
        opt_params_needed_dict = {
            "Simulated Annealing": ["init_temp", "cooling_rate", "iterations"],
            "Particle Swarm": [
                "swarm_size",
                "cognitive_param",
                "social_param",
                "inertia_weight",
                "iterations",
            ],
            "Hill Climbing": ["iterations"],
        }.get(
            current_opt_method_display_val, []
        )  # empty list if method not in dict

        for internal_name_val in opt_params_needed_dict:
            # find widget by objectName (set in update_method_parameters)
            widget = self.params_frame.findChild(QWidget, f"param_{internal_name_val}")
            value = None
            if widget:  # if widget found
                # try to find the specific input type within the widget's layout
                dbl_spin = widget.findChild(QDoubleSpinBox)
                int_spin = widget.findChild(QSpinBox)
                slider = widget.findChild(QSlider)
                if dbl_spin:
                    value = dbl_spin.value()
                elif int_spin:
                    value = int_spin.value()
                elif slider:
                    value = slider.value()

                if value is not None:
                    parameters_dict["method_params"][internal_name_val] = value
                else:  # should not happen if widgets are correctly named and contain inputs
                    self.log_message(
                        f"Warning: Could not retrieve value for parameter '{internal_name_val}'."
                    )
            else:  # widget for a required param not found
                self.log_message(
                    f"Warning: Could not find widget for parameter '{internal_name_val}'."
                )

        # ensure iterations is present if method is SA, PSO, or HC (as it's common)
        opt_method_internal_val = parameters_dict["optimization_method"]
        m_params_dict = parameters_dict["method_params"]
        if (
            opt_method_internal_val in ["sa", "pso", "hc"]
            and "iterations" not in m_params_dict
        ):
            # this is a fallback, should ideally be set by above loop or parser defaults
            widget = self.params_frame.findChild(QWidget, "param_iterations")
            if widget:
                int_spin = widget.findChild(QSpinBox)
                if int_spin:
                    m_params_dict["iterations"] = int_spin.value()
                    print(
                        "Note: Manually retrieved iterations value during get_selected_parameters."
                    )
                else:  # fallback if spinbox not found in widget
                    default_iters = (
                        10000 if opt_method_internal_val in ["sa", "hc"] else 1000
                    )
                    m_params_dict["iterations"] = default_iters
                    print(
                        f"Warning: Could not find spinbox for iterations, using default: {default_iters}"
                    )
            else:  # fallback if widget itself not found
                default_iters = (
                    10000 if opt_method_internal_val in ["sa", "hc"] else 1000
                )
                m_params_dict["iterations"] = default_iters
                print(
                    f"Warning: Could not find widget for iterations, using default: {default_iters}"
                )
        return parameters_dict

    def check_run_conditions(self):
        """
        Checks if all conditions to start an approximation run are met.

        Enables or disables the Run/Stop button accordingly. Conditions include
        having an input image, output directory, and valid metric weights.
        Also enables/disables config reload and shape load buttons based on running state.
        """
        conditions_met = True
        if not self.input_image_path:
            conditions_met = False
        if not self.output_directory:
            conditions_met = False
        weights_are_valid = self._get_weights_validity()
        if not weights_are_valid:
            conditions_met = False

        # run_stop_btn enabled if (conditions met AND not running) OR (is currently running - for stop action)
        self.run_stop_btn.setEnabled(
            (conditions_met and not self.is_running) or self.is_running
        )
        # reload and load shapes buttons only enabled when not running
        self.reload_config_btn.setEnabled(not self.is_running)
        self.load_shapes_btn.setEnabled(not self.is_running)

    def start_approximation(self):
        """
        Starts the image approximation process.

        If already running, this method (when connected to Run/Stop button's "Stop" state)
        will call `request_stop()`. Otherwise, it gathers parameters, sets up the UI
        for a running state, and starts the `Worker` thread.
        """
        if self.is_running:  # if button is in "Stop" state
            self.request_stop()
            return

        params = self.get_selected_parameters()
        if params is None:  # if parameter collection/validation failed
            self.log_message("Failed to gather parameters. Cannot start.")
            return

        self.is_running = True
        self.app_instance.stop_flag[0] = 0  # ensure stop flag is clear
        self._enable_controls(False)  # disable ui controls during run

        # change button to "Stop" mode
        self.run_stop_btn.setText("Stop")
        self.run_stop_btn.setToolTip("Stop the approximation process")
        try:  # disconnect "start" action, connect "stop" action
            self.run_stop_btn.clicked.disconnect(self.start_approximation)
        except TypeError:
            pass
        try:
            self.run_stop_btn.clicked.connect(self.request_stop)
        except TypeError:
            pass
        self.run_stop_btn.setEnabled(True)  # ensure stop button is enabled

        # reset ui elements for new run
        self.results_text.clear()
        self.show_plot_btn.setEnabled(False)  # disable plot buttons
        self.show_pso_plot_btn.setEnabled(False)
        self.app_instance.best_metrics_data = []  # clear previous metrics history
        self.pso_pbest_scores_plot_data = None  # clear previous pso data
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting... 0%")
        self.output_image_label.clear()  # clear previous output image
        self.output_image_label.setText("Running...")
        self._original_output_pixmap = None  # clear stored output pixmap

        self.log_message("Starting approximation process...")
        self.log_message("--- Parameters Summary")  # log selected parameters
        try:
            self.log_message(
                f" Input: {os.path.basename(params['input_image_path'])} -> Output Dir: {os.path.basename(params['output_directory'])}"
            )
            self.log_message(
                f" Shape: {params['shape_type']}, Num Shapes: {params['num_shapes']}, Color Scheme: {params['color_scheme']}"
            )
            if params["use_loaded_shapes"]:
                self.log_message(
                    f" Loading: YES ({len(params.get('loaded_shapes_array',[]))} shapes)"
                )
            else:
                self.log_message(" Loading: NO")
                self.log_message(f" Param Init: '{params['param_init_type']}'")
            self.log_message(f" Coord Init: '{params['coord_mode']}'")
            self.log_message(f" Method: {params['optimization_method'].upper()}")
            metric_strs = [
                f"{m}(w={w:.3f})"
                for m, w in zip(params["evaluation_metrics"], params["metric_weights"])
            ]
            self.log_message(f" Metrics: {', '.join(metric_strs)}")
            self.log_message("--------------------------")
        except Exception as log_e:  # catch errors during logging params
            self.log_message(f" (Error logging parameters: {log_e})")

        # create and start worker thread for approximation
        worker = Worker(
            self.app_instance.run_approximation, params
        )  # pass method and params
        worker.signals.result.connect(
            self.handle_approximation_result
        )  # connect result signal
        worker.signals.finished.connect(
            self.c.finished.emit
        )  # pass through finished signal
        worker.signals.error.connect(self.handle_error)  # connect error signal
        worker.signals.progress_update.connect(
            self.c.progress_update.emit
        )  # pass through progress
        self.app_instance.threadpool.start(worker)  # start worker in thread pool

    def handle_approximation_result(self, result_tuple):
        """
        Handles the result received from the approximation worker thread.

        Updates UI with final image, metrics history, and enables plot buttons.

        :param result_tuple: A tuple containing (final_image, metrics_history, final_shapes)
                             or None if an error occurred or process was stopped early.
        :type result_tuple: Optional[Tuple[Optional[np.ndarray], List[float], Optional[np.ndarray]]]
        """
        self.log_message("Approximation finished or stopped (worker result received).")

        # if shapes were loaded for this run, clear them now as the run is over
        if self.loaded_shapes_data:
            self.log_message("Clearing previously loaded shape data.")
            self.loaded_shapes_data = None
            self.load_status_label.setText("No shapes loaded.")
            self.load_status_label.setStyleSheet("")
            self.update_load_mismatch_widgets_state()  # reset load-related ui

        if result_tuple is None:  # indicates error or early stop without useful result
            self.log_message(
                "Result from worker was None (likely an error occurred or stopped early)."
            )
            if (  # check if app_instance has more specific error message
                hasattr(self.app_instance, "error_message")
                and self.app_instance.error_message
            ):
                self.log_message(f"Error detail: {self.app_instance.error_message}")
            else:  # generic message
                status = (
                    "stopped by user"
                    if self.app_instance.stop_flag[0] == 1  # check if stop flag was set
                    else "an error occurred"
                )
                self.log_message(f"Process may have been {status}. Check console/logs.")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Error or Stopped")
            return  # _on_run_finished will handle further ui reset

        final_image, metrics_history, final_shapes = result_tuple

        # handle overall metrics history for plotting
        if metrics_history is not None and isinstance(metrics_history, list):
            self.app_instance.best_metrics_data = metrics_history
            self.log_message(
                f"Collected {len(metrics_history)} overall best metric history points."
            )
            if metrics_history:  # if history is not empty
                self.log_message(
                    f"Final/Best overall combined metric: {metrics_history[-1]:.5f}"
                )
                self.c.enable_plot_button.emit()  # enable plot button
            else:
                self.log_message("No overall metric history data generated.")
                self.show_plot_btn.setEnabled(False)
        else:  # no valid history data
            self.app_instance.best_metrics_data = []
            self.log_message("No overall metrics history data received.")
            self.show_plot_btn.setEnabled(False)

        # handle pso-specific pbest scores history for plotting
        pso_history_data_val = getattr(
            self.app_instance, "pso_pbest_scores_history", None
        )
        if pso_history_data_val and isinstance(pso_history_data_val, list):
            if pso_history_data_val and all(  # ensure it's a list of numpy arrays
                isinstance(item, np.ndarray) for item in pso_history_data_val
            ):
                self.pso_pbest_scores_plot_data = pso_history_data_val
                self.log_message(
                    f"Collected {len(self.pso_pbest_scores_plot_data)} iterations of PSO pbest scores."
                )
                self.c.enable_pso_plot_button.emit()  # enable pso plot button
            else:  # incorrect format
                self.log_message(
                    "Warning: PSO history data format is incorrect. Cannot plot."
                )
                self.pso_pbest_scores_plot_data = None
                self.show_pso_plot_btn.setEnabled(False)
        else:  # no pso data (SA or HC was run)
            self.pso_pbest_scores_plot_data = None
            self.show_pso_plot_btn.setEnabled(False)

        # display the final image
        qimage = None
        if final_image is not None and isinstance(final_image, np.ndarray):
            try:
                h, w = final_image.shape[:2]
                final_image_u8 = final_image
                # ensure image is uint8 for QImage
                if final_image_u8.dtype != np.uint8:
                    temp_img = final_image_u8.copy()  # avoid modifying original
                    # clip and convert, assuming float is 0-1, int needs clipping
                    temp_img[temp_img < 0] = 0
                    temp_img[temp_img > 255] = 255
                    final_image_u8 = temp_img.astype(np.uint8)

                # create QImage based on dimensions (RGB or Grayscale)
                if final_image_u8.ndim == 3:  # RGB
                    qimage = QImage(
                        final_image_u8.data, w, h, 3 * w, QImage.Format.Format_RGB888
                    )
                elif final_image_u8.ndim == 2:  # Grayscale
                    qimage = QImage(
                        final_image_u8.data, w, h, w, QImage.Format.Format_Grayscale8
                    )
                else:  # unexpected shape
                    self.log_message(
                        f"Warning: Final display image has unexpected shape {final_image_u8.shape}."
                    )

                if qimage and not qimage.isNull():
                    self.set_output_image(qimage.copy())  # pass a copy to avoid issues
                else:
                    self.log_message(
                        "Error: Failed to create QImage from final display NumPy array."
                    )
                    self.output_image_label.setText("Display Error")
                    self.progress_bar.setFormat("Finished (Display Error)")
            except Exception as e:
                self.log_message(f"Error converting/displaying final image: {e}")
                traceback.print_exc()
                self.output_image_label.setText("Display Error")
                self.progress_bar.setFormat("Finished (Display Error)")
        else:  # no final image produced (stopped very early)
            self.log_message(
                "No final image available for display (possibly stopped early)."
            )
            self.output_image_label.setText("No Output Image")
            status_text = (
                "Stopped (No Image)"
                if self.app_instance.stop_flag[0] == 1  # check if it was a user stop
                else "Finished (No Image)"  # or other reason
            )
            self.progress_bar.setFormat(status_text)
            self.progress_bar.setValue(  # set progress to 0 if stopped, 100 if finished "naturally" without image
                0 if self.app_instance.stop_flag[0] == 1 else 100
            )

        # log final original (non-normalized) metrics
        final_metrics_orig_dict = getattr(self.app_instance, "final_metrics_orig", {})
        if final_metrics_orig_dict:
            self.log_message("\n--- Final Original Metric Values ---")
            for name, val in sorted(final_metrics_orig_dict.items()):
                # format value, handling inf/nan and numpy types
                format_str = (
                    f"{val:.6f}"
                    if isinstance(val, (int, float)) and np.isfinite(val)
                    else str(val)
                )
                self.log_message(f"  {name.upper()}: {format_str}")
            self.log_message("------------------------------------")
        else:
            self.log_message(
                "\n(No final original metrics were calculated or available)"
            )

        # update progress bar to 100% if run completed naturally
        if result_tuple is not None and self.app_instance.stop_flag[0] == 0:
            self.progress_bar.setValue(100)
            if not self.progress_bar.format().startswith(
                "Finished"
            ):  # ensure "Finished" prefix
                self.progress_bar.setFormat("Finished 100%")
        elif self.app_instance.stop_flag[0] == 1:  # if stopped by user
            current_percent_val = self.progress_bar.value()
            self.progress_bar.setFormat(f"Stopped at {current_percent_val}%")
        # _on_run_finished will handle further UI reset

    def handle_error(self, error_info):
        """
        Handles errors propagated from the worker thread.

        Logs the error and displays an error dialog.

        :param error_info: A tuple containing (exception_type, exception_value, traceback_string_list).
        :type error_info: tuple
        """
        # if shapes were loaded, clear them as run is now errored/finished
        if self.loaded_shapes_data:
            self.log_message(
                "Clearing previously loaded shape data (due to error or completion)."
            )
            self.loaded_shapes_data = None
            self.load_status_label.setText("No shapes loaded.")
            self.load_status_label.setStyleSheet("")
            self.update_load_mismatch_widgets_state()

        exctype, value, tb_str_list = error_info
        error_message_str = f"Error in worker thread: {exctype.__name__}: {value}"
        self.log_message(error_message_str)
        # append a portion of the traceback to the log for quick diagnosis
        self.results_text.append(
            "\nTraceback (most recent call last):\n"
            + "".join(tb_str_list[-15:])  # last 15 lines
        )
        self.progress_bar.setValue(0)  # reset progress
        self.progress_bar.setFormat(f"Error: {exctype.__name__}")

        # _on_run_finished (called via worker.signals.finished) will handle general UI control reset
        # display a user friendly error dialog
        error_dialog = QDialog(self)
        error_dialog.setWindowTitle("Worker Thread Error")
        layout = QVBoxLayout(error_dialog)
        scroll_area = QScrollArea()  # for potentially long error messages
        scroll_area.setWidgetResizable(True)
        error_label = QLabel(
            error_message_str + "\n\nSee log output panel for details."
        )
        error_label.setWordWrap(True)
        scroll_area.setWidget(error_label)
        layout.addWidget(scroll_area)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(error_dialog.accept)
        layout.addWidget(ok_button)
        error_dialog.resize(400, 200)  # reasonable default size
        error_dialog.exec()

    def set_output_image(self, qimage: QImage):
        """
        Sets the output image preview label with the given QImage.

        Stores the original QImage as a QPixmap for efficient rescaling on resize.

        :param qimage: The QImage to display.
        :type qimage: QImage
        """
        self._original_output_pixmap = None  # clear previous
        if qimage and not qimage.isNull():
            try:
                pixmap = QPixmap.fromImage(qimage)
                if pixmap.isNull():  # check conversion
                    raise ValueError("Failed to create QPixmap from QImage.")
                self._original_output_pixmap = pixmap  # store for rescaling
                self._rescale_output_preview()  # display scaled version
                self.output_image_label.setText("")  # clear any "No output" text
            except Exception as e:
                self.output_image_label.setText(f"Display Error:\n{e}")
                self.log_message(f"Error setting output image display: {e}")
                traceback.print_exc()
                self._original_output_pixmap = None  # clear on error
        else:  # if invalid qimage passed
            self.output_image_label.setText("Invalid image data")
            self.log_message("Error: Received invalid QImage for output display.")
            self._original_output_pixmap = None

    def show_metric_plot_with_data(self):
        """
        Displays a plot of the overall best combined metric history from the last run.
        """
        plot_data_ref = getattr(self.app_instance, "best_metrics_data", [])
        if not isinstance(plot_data_ref, list) or not plot_data_ref:
            QMessageBox.information(
                self, "Plot Data", "No overall metric history data available to plot."
            )
            return

        try:
            plt.switch_backend("Agg")  # use non-interactive backend for embedding
            fig, ax = plt.subplots(figsize=(8, 5))  # create figure and axes
            ax.plot(plot_data_ref, marker=".", linestyle="-", markersize=4)  # plot data
            ax.set_xlabel("Update Step (Callback Interval)")
            ax.set_ylabel("Overall Best Combined Score (Higher is Better)")
            ax.set_title(
                f"Overall Best Metric Score History ({len(plot_data_ref)} points)"
            )
            ax.grid(True)  # add grid
            ax.set_ylim([-0.05, 1.05])  # typical range for normalized scores (0-1)

            canvas = FigureCanvas(fig)  # create matplotlib canvas widget
            dialog = QDialog(self)  # embed canvas in a dialog
            dialog.setWindowTitle("Overall Metric History Plot")
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.addWidget(canvas)
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            dialog_layout.addWidget(close_button)
            dialog.resize(700, 500)  # set dialog size
            dialog.exec()  # show dialog modally
        except Exception as e:
            error_msg = f"Error generating metric plot: {e}"
            self.log_message(error_msg)
            traceback.print_exc()
            QMessageBox.critical(
                self, "Plot Error", f"Could not generate metric plot:\n{e}"
            )

    def show_pso_plot(self):
        """
        Displays a plot of Particle Swarm Optimization (PSO) particle best score history.

        Shows min, max, mean, and median scores of particles' personal bests over iterations.
        """
        plot_data_pbest_history_list = self.pso_pbest_scores_plot_data
        if (
            not isinstance(plot_data_pbest_history_list, list)
            or not plot_data_pbest_history_list
        ):
            QMessageBox.information(
                self,
                "Plot Data",
                "No PSO particle score history data available to plot.",
            )
            return
        if not all(
            isinstance(item, np.ndarray) for item in plot_data_pbest_history_list
        ):
            QMessageBox.warning(
                self,
                "Plot Data",
                "Invalid PSO history data format (expected list of NumPy arrays).",
            )
            return

        num_iterations = len(plot_data_pbest_history_list)
        if num_iterations == 0:
            QMessageBox.information(self, "Plot Data", "PSO history data is empty.")
            return

        try:  # determine number of particles from first history entry
            num_particles = (
                len(plot_data_pbest_history_list[0])
                if plot_data_pbest_history_list[0].size > 0
                else 0
            )
        except IndexError:  # if first entry is somehow malformed
            QMessageBox.warning(
                self,
                "Plot Data",
                "Could not determine number of particles from PSO data.",
            )
            return
        if num_particles == 0:
            QMessageBox.information(
                self, "Plot Data", "PSO history contains no particle data."
            )
            return

        iterations_axis = list(range(num_iterations))
        # calculate statistics for each iteration
        mean_scores_list = [  # mean pbest score at each iteration
            (
                np.mean(scores)
                if scores.size > 0 and np.all(np.isfinite(scores))  # handle empty/nan
                else np.nan
            )
            for scores in plot_data_pbest_history_list
        ]
        median_scores_list = [  # median pbest score
            (
                np.median(scores)
                if scores.size > 0 and np.all(np.isfinite(scores))
                else np.nan
            )
            for scores in plot_data_pbest_history_list
        ]
        min_scores_list = [  # min pbest score
            (
                np.min(scores)
                if scores.size > 0 and np.all(np.isfinite(scores))
                else np.nan
            )
            for scores in plot_data_pbest_history_list
        ]
        max_scores_list = [  # max pbest score (effectively gbest at that iteration)
            (
                np.max(scores)
                if scores.size > 0 and np.all(np.isfinite(scores))
                else np.nan
            )
            for scores in plot_data_pbest_history_list
        ]

        try:
            plt.switch_backend("Agg")  # use non-interactive backend
            fig, ax = plt.subplots(figsize=(9, 6))  # create figure

            # plot different statistics
            ax.plot(
                iterations_axis,
                max_scores_list,
                marker=".",
                linestyle="-",
                markersize=4,
                label="Max Particle Score (gbest)",
            )
            ax.plot(
                iterations_axis,
                mean_scores_list,
                marker=".",
                linestyle="--",
                markersize=3,
                alpha=0.8,
                label="Mean Particle Score",
            )
            ax.plot(
                iterations_axis,
                median_scores_list,
                marker=".",
                linestyle=":",
                markersize=3,
                alpha=0.8,
                label="Median Particle Score",
            )
            ax.plot(
                iterations_axis,
                min_scores_list,
                marker=".",
                linestyle="-.",
                markersize=3,
                alpha=0.8,
                label="Min Particle Score",
            )

            # fill between min and max scores to show range
            valid_range_mask = np.isfinite(min_scores_list) & np.isfinite(
                max_scores_list
            )
            if np.any(valid_range_mask):
                ax.fill_between(
                    np.array(iterations_axis)[valid_range_mask],
                    np.array(min_scores_list)[valid_range_mask],
                    np.array(max_scores_list)[valid_range_mask],
                    alpha=0.15,
                    label="Particle Score Range",
                    color="gray",
                )

            ax.set_xlabel("Iteration Step")
            ax.set_ylabel("Particle Best Combined Score (Higher is Better)")
            ax.set_title(
                f"PSO Particle Best Score Distribution ({num_iterations} iters, {num_particles} particles)"
            )
            ax.legend()
            ax.grid(True)
            ax.set_ylim([-0.05, 1.05])  # typical 0-1 score range

            canvas = FigureCanvas(fig)  # embed in dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("PSO Particle Score History Plot")
            dialog_layout = QVBoxLayout(dialog)
            dialog_layout.addWidget(canvas)
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            dialog_layout.addWidget(close_button)
            dialog.resize(800, 600)
            dialog.exec()
        except Exception as e:
            error_msg = f"Error generating PSO plot: {e}"
            self.log_message(error_msg)
            traceback.print_exc()
            QMessageBox.critical(
                self, "Plot Error", f"Could not generate PSO score plot:\n{e}"
            )
