import sys
import os
import glob
import json
import subprocess
import re
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QListWidget, QFileDialog, QScrollArea, QFrame
)
from PyQt6.QtGui import QPixmap, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QTimer

import process
from sokhan_processor import *


CONFIG_FILE = "config.json"



class sokhanGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dictionary OCR Viewer - PyQt6 (Auto Save)")
        self.setGeometry(200, 100, 1200, 800)

        self.sokhan_processor = SokhanProcessor(None,None)
        self.sokhan_processor.verbose = True

        self.image_files_list_entry = []
        self.current_entry_index = 0
        self.current_page_number = 0
        self.current_column_side = 0
        self.current_entry_index_in_column = 0
        self.current_entry_type = "Entry"

        self.current_entry_image_file = None
        self.current_headword_image_file = None
        self.current_headword_text_file = None

        self.current_page_image_file = None

        self.original_entry_pixmap = None
        self.original_page_pixmap = None

        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.setSingleShot(True)
        self.auto_save_timer.timeout.connect(self.save_text_silent)

        self.create_ui()
        self.create_shortcuts()

        self.load_configs()


    # ---------------- UI ----------------
    def create_ui(self):
        main_layout = QHBoxLayout()
        files_list_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        head_word_layout = QVBoxLayout()
        head_word_buttons_layout = QHBoxLayout()
        entry_layout = QVBoxLayout()
        entry_buttons_layout = QHBoxLayout()
        page_layout = QVBoxLayout()
        page_buttons_layout = QHBoxLayout()
        nav_layout = QHBoxLayout()

        # File list
        self.file_list = QListWidget()
        self.file_list.keyPressEvent = self.file_list_key_press
        self.file_list.itemSelectionChanged.connect(self.update_outputs_on_gui)

        # Head word image viewer
        self.scroll_area_head_word = QScrollArea()
        self.image_label_head_word = QLabel()
        self.image_label_head_word.setStyleSheet("background-color: cyan;")
        self.image_label_head_word.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        self.scroll_area_head_word.setWidgetResizable(True)
        self.scroll_area_head_word.setWidget(self.image_label_head_word)

        # Page image viewer
        self.scroll_area_page = QScrollArea()
        self.image_label_page = QLabel()
        self.image_label_page.setStyleSheet("background-color: cyan;")
        self.image_label_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area_page.setWidgetResizable(True)
        self.scroll_area_page.setWidget(self.image_label_page)

        # Entry image viewer
        self.scroll_area_entry = QScrollArea()
        self.image_label_entry = QLabel()
        self.image_label_entry.setStyleSheet("background-color: cyan;")
        self.image_label_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label_entry.background_color = Qt.GlobalColor.cyan
        self.scroll_area_entry.setWidgetResizable(True)
        self.scroll_area_entry.setWidget(self.image_label_entry)

        # OCR text (Persian / RTL)
        self.head_word_ocr_text_edit = FocusableTextEdit(self)
        self.head_word_ocr_text_edit.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.head_word_ocr_text_edit.setFontPointSize(32)
        # Set font name
        self.head_word_ocr_text_edit.setFontFamily("IRTitr")
        self.head_word_ocr_text_edit.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Connect text change to auto-save timer
        self.head_word_ocr_text_edit.textChanged.connect(self.schedule_auto_save)

        # Lables:
        self.output_folder_label = QLabel("Out: ")
        self.input_folder_label = QLabel("In: ")
        self.input_files_process_params_label = QLabel("Input files process params:")

        # Text Edits:
        self.input_files_process_params_text_edit = QTextEdit()
        self.process_column_params_text_edit = QTextEdit()
        self.process_entry_params_text_edit = QTextEdit()
        self.process_head_word_params_text_edit = QTextEdit()
        self.input_files_process_params_text_edit.textChanged.connect(self.save_configs)
        self.process_column_params_text_edit.textChanged.connect(self.save_configs)
        self.process_entry_params_text_edit.textChanged.connect(self.save_configs)
        self.process_head_word_params_text_edit.textChanged.connect(self.save_configs)


        # Buttons
        self.select_output_folder_btn = QPushButton("Select output folder")
        self.select_output_folder_btn.clicked.connect(self.select_output_folder)

        self.select_input_folder_btn = QPushButton("Select input folder")
        self.select_input_folder_btn.clicked.connect(self.select_input_folder)

        self.process_input_files_btn = QPushButton("Process input files")
        self.process_input_files_btn.clicked.connect(self.process_input_files)


        self.run_rules_on_all_btn = QPushButton("Run Rules on All")
        self.run_rules_on_all_btn.clicked.connect(self.run_rules_on_all)

        self.run_rules_on_selected_btn = QPushButton("Run Rules on Selected")
        self.run_rules_on_selected_btn.clicked.connect(self.run_rules_on_selected)

        # Button related to head word: Open headword image, open folder and process:
        self.open_head_word_img_btn = QPushButton("Image")
        self.open_head_word_img_btn.clicked.connect(self.open_head_word_image)
        self.open_head_word_folder_btn = QPushButton("Folder")
        self.open_head_word_folder_btn.clicked.connect(self.open_head_word_folder)
        self.process_head_word_btn = QPushButton("Process")
        self.process_head_word_btn.clicked.connect(self.process_head_word)

        # Button related to entry: Open entry image, open folder and process:
        self.open_entry_img_btn = QPushButton("Image")
        self.open_entry_img_btn.clicked.connect(self.open_entry_image)
        self.open_entry_folder_btn = QPushButton("Folder")
        self.open_entry_folder_btn.clicked.connect(self.open_entry_folder)
        self.process_entry_btn = QPushButton("Process")
        self.process_entry_btn.clicked.connect(self.process_entry)

        # Button related to page: Open page image, open folder and process:
        self.open_page_img_btn = QPushButton("Image")
        self.open_page_img_btn.clicked.connect(self.open_page_image)
        self.open_page_folder_btn = QPushButton("Folder")
        self.open_page_folder_btn.clicked.connect(self.open_page_folder)
        self.process_page_btn = QPushButton("Process")
        self.process_page_btn.clicked.connect(self.process_page)


        # ---------- Layouts ----------
        files_list_layout.addWidget(self.select_output_folder_btn)
        files_list_layout.addWidget(self.output_folder_label)
        files_list_layout.addWidget(self.select_input_folder_btn)
        files_list_layout.addWidget(self.input_folder_label)
        files_list_layout.addWidget(self.process_input_files_btn)
        files_list_layout.addWidget(self.input_files_process_params_label)
        files_list_layout.addWidget(self.input_files_process_params_text_edit, 1)
        files_list_layout.addWidget(self.file_list, 5)

        head_word_layout.addWidget(self.scroll_area_head_word, 3)
        head_word_layout.addWidget(self.head_word_ocr_text_edit, 3)
        head_word_layout.addWidget(self.process_head_word_params_text_edit, 1)
        head_word_buttons_layout.addWidget(self.open_head_word_img_btn)
        head_word_buttons_layout.addWidget(self.open_head_word_folder_btn)
        head_word_buttons_layout.addWidget(self.process_head_word_btn)
        head_word_layout.addLayout(head_word_buttons_layout)

        entry_layout.addWidget(self.scroll_area_entry, 6)
        entry_layout.addWidget(self.process_entry_params_text_edit, 1)
        entry_buttons_layout.addWidget(self.open_entry_img_btn)
        entry_buttons_layout.addWidget(self.open_entry_folder_btn)
        entry_buttons_layout.addWidget(self.process_entry_btn)
        entry_layout.addLayout(entry_buttons_layout)

        page_layout.addWidget(self.scroll_area_page, 6)
        page_layout.addWidget(self.process_column_params_text_edit, 1)
        page_buttons_layout.addWidget(self.open_page_img_btn)
        page_buttons_layout.addWidget(self.open_page_folder_btn)
        page_buttons_layout.addWidget(self.process_page_btn)
        page_layout.addLayout(page_buttons_layout)


        for btn in [
            self.run_rules_on_all_btn,
            self.run_rules_on_selected_btn,
            self.open_head_word_img_btn,
            self.open_head_word_folder_btn,
            self.process_head_word_btn
        ]:
            nav_layout.addWidget(btn)

        images_layout.addLayout(head_word_layout, 1)
        images_layout.addLayout(entry_layout, 1)
        images_layout.addLayout(page_layout, 1)

        right_layout.addLayout(images_layout, 1)
        right_layout.addLayout(nav_layout)
        main_layout.addLayout(files_list_layout, 1)

        main_layout.addLayout(right_layout, 5)

        self.setLayout(main_layout)


    # ---------------- Shortcuts ----------------
    def create_shortcuts(self):
        app = QApplication.instance()

        def add_shortcut(keys, func):
            sc = QShortcut(QKeySequence(keys), app)
            sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
            sc.activated.connect(func)

        add_shortcut("Ctrl+S", self.save_text_silent)
        add_shortcut("Ctrl+E", self.focus_text_edit)
        add_shortcut("Ctrl+L", self.focus_file_list)
        add_shortcut("Ctrl+R", self.run_rules_on_selected)


    # ---------------- Folder Handling ----------------
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select outputs folder")
        if folder:
            self.sokhan_processor.update_folders(output_folder=Path(folder))
            self.current_entry_index = 0
            self.save_configs()
            self.load_files_list()
            self.update_outputs_on_gui()


    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select input folder")
        if folder:
            self.sokhan_processor.update_folders(input_folder=Path(folder))
            self.save_configs()
            self.update_outputs_on_gui()


    def load_configs(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    last_output_folder = data.get("output_folder")
                    last_headword_index = data.get("current_entry_index")
                    last_input_folder = data.get("input_folder")
                    last_process_column_params = data.get("process_column_params")
                    last_process_entry_params = data.get("process_entry_params")
                    last_process_head_word_params = data.get("process_head_word_params")
                    last_input_files_process_params = data.get("input_files_process_params")

                    if last_output_folder and Path(last_output_folder).exists():
                        self.sokhan_processor.update_folders(output_folder=Path(last_output_folder))
                    if last_input_folder and Path(last_input_folder).exists():
                        self.sokhan_processor.update_folders(input_folder=Path(last_input_folder))
                    if last_headword_index is not None:
                        self.current_entry_index = last_headword_index
                    if last_process_column_params is not None:
                        self.process_column_params_text_edit.setText(last_process_column_params)
                        self.sokhan_processor.set_configs_from_string(last_process_column_params)
                    if last_process_entry_params is not None:
                        self.process_entry_params_text_edit.setText(last_process_entry_params)
                        self.sokhan_processor.set_configs_from_string(last_process_entry_params)
                    if last_process_head_word_params is not None:
                        self.process_head_word_params_text_edit.setText(last_process_head_word_params)
                        self.sokhan_processor.set_configs_from_string(last_process_head_word_params)
                    if last_input_files_process_params is not None:
                        self.input_files_process_params_text_edit.setText(last_input_files_process_params)

                    self.load_files_list()
                    self.update_outputs_on_gui()

            except Exception:
                pass


    def save_configs(self):
        data = {"output_folder": str(self.sokhan_processor.output_folder),
                "input_folder": str(self.sokhan_processor.input_folder),
                "current_entry_index": self.current_entry_index,
                "process_column_params": self.process_column_params_text_edit.toPlainText(),
                "process_entry_params": self.process_entry_params_text_edit.toPlainText(),
                "process_head_word_params": self.process_head_word_params_text_edit.toPlainText(),
                "input_files_process_params": self.input_files_process_params_text_edit.toPlainText()
                }

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    # ---------------- Update GUI ----------------
    def load_files_list(self):
        if self.sokhan_processor.output_folder.exists():
            # Load entries images and update the list
            self.image_files_list_entry = sorted(list(self.sokhan_processor.output_entries_folder.glob("*.png")))

            self.file_list.clear()
            for f in self.image_files_list_entry:
                self.file_list.addItem(f.stem)

            if self.image_files_list_entry:
                self.file_list.setCurrentRow(self.current_entry_index)
                self.update_outputs_on_gui()


    def update_outputs_on_gui(self):
        self.current_entry_index = self.file_list.currentRow()

        if self.sokhan_processor.output_folder is not None:
            self.output_folder_label.setText(f"Out: {self.sokhan_processor.output_folder}")

        if self.sokhan_processor.input_folder is not None:
            self.input_folder_label.setText(f"In: {self.sokhan_processor.input_folder}")

        if len(self.image_files_list_entry) == 0:
            self.current_entry_index = 0
            self.current_headword_image_file = None
            self.current_headword_text_file = None
            self.current_entry_image_file = None
            self.current_page_image_file = None

        else:
            if self.current_entry_index >= len(self.image_files_list_entry):
                self.current_entry_index = len(self.image_files_list_entry) - 1

            # For example, for 0126_1_04.png, we have:
            # current_page_number = 126
            # current_column_side = 1 (left)
            # current_entry_index_in_column = 4
            self.current_entry_image_file = self.image_files_list_entry[self.current_entry_index]
            self.current_page_number= int(self.current_entry_image_file.stem.split("_")[0])
            self.current_column_side = int(self.current_entry_image_file.stem.split("_")[1])
            self.current_entry_index_in_column = int(self.current_entry_image_file.stem.split("_")[2])


            if self.current_entry_index_in_column != 0:
                # If entry file name doesn't end with 00, it means it is a new entry and has a headword image.
                self.current_headword_image_file = self.sokhan_processor.output_head_words_folder / self.current_entry_image_file.name
                self.current_headword_text_file = self.current_headword_image_file.with_suffix(".gt.txt")
            else:
                # If entry file name ends with 00, it means it is a continuation entry and doesn't have a headword image.
                self.current_headword_image_file = None
                self.current_headword_text_file = None

            self.current_page_image_file = self.sokhan_processor.output_guiding_lines_folder / f"{self.current_page_number:04d}_{self.current_column_side}.png"



        # Display headword image
        if self.current_headword_image_file is not None:
            self.image_label_head_word.setPixmap(QPixmap(str(self.current_headword_image_file)))
            self.image_label_head_word.adjustSize()
        else:
            self.image_label_head_word.setPixmap(QPixmap())

        # Display headword text
        if self.current_headword_text_file is not None:
            with open(str(self.current_headword_text_file), "r", encoding="utf-8") as f:
                self.head_word_ocr_text_edit.blockSignals(True)
                self.head_word_ocr_text_edit.setText(f.read())
                self.head_word_ocr_text_edit.blockSignals(False)
        else:
            self.head_word_ocr_text_edit.blockSignals(True)
            self.head_word_ocr_text_edit.setText("")
            self.head_word_ocr_text_edit.blockSignals(False)



        # Display page image
        if self.current_page_image_file is not None:
            self.original_page_pixmap = QPixmap(str(self.current_page_image_file))
            self.update_scaled_page_image()

        # Display entry image
        if self.current_entry_image_file is not None:
            self.original_entry_pixmap = QPixmap(str(self.current_entry_image_file))
            self.update_scaled_entry_image()

        self.save_configs()


    # ---------------- Processing ----------------
    def process_input_files(self):

        text = self.input_files_process_params_text_edit.toPlainText().strip()

        # Case: "4-7"
        if "-" in text:
            start_str, end_str = text.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            process_pages = list(range(start, end + 1))
        else:
            # Case: "6"
            process_pages = [int(text)]
        self.sokhan_processor.update_folders()
        self.sokhan_processor.process_pages(process_pages)
        self.load_files_list()


    def process_head_word(self):
        self.sokhan_processor.set_configs_from_string(self.process_column_params_text_edit.toPlainText())
        self.sokhan_processor.process_pages(self.current_page_number)
        self.sokhan_processor.set_configs_from_string(self.process_head_word_params_text_edit.toPlainText())
        self.sokhan_processor.process_single_entry((self.current_column_side, self.current_entry_index_in_column))
        self.save_configs()
        self.load_files_list()
        self.update_outputs_on_gui()


    def process_entry(self):
        self.sokhan_processor.set_configs_from_string(self.process_entry_params_text_edit.toPlainText())
        self.sokhan_processor.process_single_entry((self.current_column_side, self.current_entry_index_in_column))
        self.save_configs()
        self.load_files_list()
        self.update_outputs_on_gui()


    def process_page(self):
        self.sokhan_processor.set_configs_from_string(self.process_column_params_text_edit.toPlainText())
        print()
        self.sokhan_processor.remove_single_page_output(self.current_page_number)
        self.sokhan_processor.process_pages(self.current_page_number)
        self.save_configs()
        self.load_files_list()
        self.update_outputs_on_gui()


    # ---------------- Text Processing ----------------
    def modify_text(self, input_text):
        text = input_text
        text = text.replace("\n", "")
        text = text.replace("\r", "")
        text = text.replace("\t", "")
        text = text.replace(" ", "‌")
        text = text.replace("،", "، ")
        text = text.replace("»", "، ")
        return text


    def run_rules_on_all(self):
        # Open all ocr text files and replace newline in them with space:
        for image_file in self.image_files_list_entry:
            text_file = os.path.splitext(image_file)[0] + ".gt.txt"
            with open(text_file, "r", encoding="utf-8") as fr:
                text = fr.read()
                text = clean_text(text)

                with open(text_file, "w", encoding="utf-8") as fw:
                    fw.write(text)
        self.update_outputs_on_gui()


    def run_rules_on_selected(self):
        image_file = self.image_files_list_entry[self.current_entry_index]
        text_file = os.path.splitext(image_file)[0] + ".gt.txt"
        with open(text_file, "r", encoding="utf-8") as fr:
            text = fr.read()
            text = clean_text(text)

            with open(text_file, "w", encoding="utf-8") as fw:
                fw.write(text)
        self.update_outputs_on_gui()


    # ---------------- Auto Save ----------------
    def schedule_auto_save(self):
        """Delay saving by 1 second after typing stops."""
        self.auto_save_timer.start(1000)


    def save_text_silent(self):
        """Save OCR text to file silently (no popups)."""
        if not self.image_files_list_entry:
            return
        txt_path = os.path.splitext(self.image_files_list_entry[self.current_entry_index])[0] + ".gt.txt"
        text = self.head_word_ocr_text_edit.toPlainText()
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"Error saving file {txt_path}: {e}")


    # ---------------- Open files and folders using OS ----------------
    def open_with_os(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform.startswith("darwin"):
                subprocess.call(["open", path])
            else:
                subprocess.call(["xdg-open", path])
        except Exception as e:
            print(f"Cannot open {path}: {e}")


    def open_head_word_image(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(self.image_files_list_entry[self.current_entry_index])


    def open_head_word_folder(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(str(self.sokhan_processor.output_head_words_folder))


    def open_entry_image(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(self.current_entry_image_file)


    def open_entry_folder(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(str(self.sokhan_processor.output_entries_folder))


    def open_page_image(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(self.current_page_image_file)


    def open_page_folder(self):
        if not self.image_files_list_entry:
            return
        self.open_with_os(str(self.sokhan_processor.output_guiding_lines_folder))

    # ----------------------------------
    def focus_text_edit(self):
        self.head_word_ocr_text_edit.setFocus()


    def focus_file_list(self):
        self.file_list.setFocus()


    def file_list_key_press(self, event):
        """When Enter is pressed on the file list, move focus to the text editor."""
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.head_word_ocr_text_edit.setFocus()
        else:
            # Call default handler so navigation and other keys still work
            QListWidget.keyPressEvent(self.file_list, event)


    # ---------------- Image Scaling ----------------

    def resizeEvent(self, event):
        """Automatically scale the image when window/label is resized."""
        super().resizeEvent(event)
        self.update_scaled_entry_image()


    def update_scaled_entry_image(self):
        """Scale the image to fit the QLabel while keeping aspect ratio."""
        if self.original_entry_pixmap is None:
            return
        label_size = self.scroll_area_entry.viewport().size()
        scaled_pixmap = self.original_entry_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label_entry.setPixmap(scaled_pixmap)


    def update_scaled_page_image(self):
        """Scale the image to fit the QLabel while keeping aspect ratio."""
        if self.original_page_pixmap is None:
            return
        label_size = self.scroll_area_page.viewport().size()
        scaled_pixmap = self.original_page_pixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label_page.setPixmap(scaled_pixmap)



class FocusableTextEdit(QTextEdit):
    """Custom QTextEdit that switches focus back to the list when Esc is pressed."""
    def __init__(self, parent_viewer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = parent_viewer

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.viewer.file_list.setFocus()
        elif event.key() == Qt.Key.Key_Return:
            self.viewer.show_next_image()
        else:
            super().keyPressEvent(event)


def clean_text(input_text):
    text = input_text

    text = text.replace("\n", "")
    text = text.replace("‌", " ")
    text = text.replace("  ", " ")
    text = text.replace(" ، ", "، ")

    # Remove all spaces at the end of the text:
    text = text.rstrip()

    # Remove all spaces at the beginning of the text:
    text = text.lstrip()

    # --- Basic Arabic → Persian letter normalization ---
    arabic_to_persian = {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "آ": "آ",
        "٠": "۰",
        "١": "۱",
        "٢": "۲",
        "٣": "۳",
        "٤": "۴",
        "٥": "۵",
        "٦": "۶",
        "٧": "۷",
        "٨": "۸",
        "٩": "۹",
    }
    # for ar, fa in arabic_to_persian.items():
    #    text = text.replace(ar, fa)

    # Remove zero-width characters (optional) ---
    text = re.sub(r"[\u200c\u200d\u202b\u202c\u202d]", " ", text)

    # --- Remove Tatweel ---
    text = text.replace("ـ", "")

    # ----- Farsi specific alphabet rules------
    only_connect_to_before = "اآأإدذرزژوؤة"
    connect_to_before_and_after = "ئبپتثجچحخسشصضطظعغفقکگلمنهیي"
    not_connect = "ء۱۲۳۴۵۶۷۸۹۰.\[\]"  # note: escape [ and ]
    # 1. Remove any space BEFORE or AFTER not_connect
    #    (space before: X <space> not_connect)
    #    (space after:  not_connect <space> X)
    text = re.sub(fr'\s*([{not_connect}])\s*', r'\1', text)

    # 2. Remove space after only_connect_to_before
    text = re.sub(fr'([{only_connect_to_before}])\s+', r'\1', text)

    # Remove space before comma
    text = re.sub(r'\s+،', '،', text)

    text = re.sub(r'\](.)\[', r'[\1]', text)
    text = re.sub(r'\](.)\]', r'[\1]', text)
    text = re.sub(r'\[(.)\[', r'[\1]', text)
    return text


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = sokhanGUI()
    viewer.show()
    sys.exit(app.exec())
