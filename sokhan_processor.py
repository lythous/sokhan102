import ast

from process import *

from pathlib import Path
import re
import cv2
import pytesseract

class SokhanProcessor:

    def __init__(self, input_folder, output_folder):

        # -------- Processing configs --------
        self.page_index = None
        self.verbose = False

        # Entries detection configs
        self.cfg_entries_detection_search_width_band = 40
        self.cfg_dividing_lines_offset_y_start = 0
        self.cfg_dividing_lines_offset_y_end = 0

        # Head word detection configs
        self.cfg_head_word_min_area = 160
        self.cfg_head_word_bold_thresh = 30
        self.cfg_head_word_boldness_thresh = 1
        self.cfg_head_word_width_thresh = 10
        self.cfg_entry_vertical_offset = (0, 0)
        self.cfg_head_word_offset = (0, 0, 0, 0)

        # Tesseract configs
        self.cfg_tesseract_config = r'-l sokhan_ocr_model --tessdata-dir ./tessdata/ --psm 7  --oem 3'

        # -------- Internal variables (results of processing) --------
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.input_page_image_path = None # To be set in the page process function when the input page index is given by the user.
        self.output_columns_folder = None
        self.output_entries_folder = None
        self.output_head_words_folder = None
        self.output_guiding_lines_folder = None

        self.image_page_input = None
        self.image_page_preprocessed = None

        # -------- Output variables (results of processing) --------
        # In all the tuples below, index 0 is for right column and index 1 is for left column.
        self.out_img_columns = [None, None]
        self.out_img_columns_with_guiding_lines =  [None, None]
        self.out_column_divider_lines_y = [[], []]

        self.out_entries_start_top_y =  [[], []]
        self.out_rect_entries =  [[], []]
        self.out_entries_divider_lines_y =  [[], []]
        self.out_typ_entries =  [[], []]
        self.out_img_entries =  [[], []]

        self.out_img_headwords =  [[], []]
        self.out_rect_headwords =  [[], []]
        self.out_txt_headwords =  [[], []]

        self.update_folders(input_folder, output_folder)


    def process_pages(self, pages):
        if self.input_folder is None:
            warnings.warn("Input folder is not set.")
            return
        if self.output_folder is None:
            warnings.warn("Output folder is not set.")
            return

        if type(pages) is list:
            for page_index in pages:
                if self.verbose:
                    print(f"Processing page {page_index} ...")
                self.page_index = page_index
                self.input_page_image_path = self.input_folder / f"{self.page_index}.pbm"
                self.do_all_processes_on_a_single_page()
        if type(pages) is int:
            if self.verbose:
                print(f"Processing page {pages} ...")
            self.page_index = pages
            self.input_page_image_path = self.input_folder / f"{self.page_index}.pbm"
            self.do_all_processes_on_a_single_page()


    def do_all_processes_on_a_single_page(self):
        self.preprocess_page()
        self.save_both_columns()

        self.process_both_columns()
        self.process_all_entries()

        self.draw_guiding_lines()
        self.save_guiding_lines()


    def preprocess_page(self):
        self.image_page_input = cv2.imread(str(self.input_page_image_path))
        self.image_page_preprocessed = preprocess(self.image_page_input)
        self.out_img_columns = separate_columns(self.image_page_preprocessed)
        if self.verbose:
            print(f"Page Preprocessed: {self.page_index:04d}")


    def process_both_columns(self):
        for i, image_column in enumerate(self.out_img_columns):
            self.out_column_divider_lines_y[i], self.out_entries_start_top_y[i], self.out_rect_entries[i], self.out_entries_divider_lines_y[i], self.out_typ_entries[i] = (
                process_column(image_column, self.cfg_entries_detection_search_width_band,
                               self.cfg_dividing_lines_offset_y_start, self.cfg_dividing_lines_offset_y_end))
            if self.verbose:
                print(f"Column processed: {self.page_index}_{i}")
        self.initialize_output_entry_and_headword_variables()


    def initialize_output_entry_and_headword_variables(self):
        for i in range(2):
            self.out_img_entries[i] = [None] * len(self.out_rect_entries[i])
            self.out_img_headwords[i] = [None] * len(self.out_rect_entries[i])
            self.out_rect_headwords[i] = [None] * len(self.out_rect_entries[i])
            self.out_txt_headwords[i] = [None] * len(self.out_rect_entries[i])


    def process_all_entries(self):
        for i, image_column in enumerate(self.out_img_columns):
            for j, entry_rect in enumerate(self.out_rect_entries[i]):
                self.process_single_entry((i, j), draw_guiding_lines = False)

        self.draw_guiding_lines()


    def process_single_entry(self, entry_address, draw_guiding_lines = True):
        column_side, entry_index = entry_address
        (self.out_img_entries[column_side][entry_index], self.out_img_headwords[column_side][entry_index],
         self.out_rect_entries[column_side][entry_index], self.out_rect_headwords[column_side][entry_index]) = (
            process_single_entry(self.out_img_columns[column_side], self.out_rect_entries[column_side][entry_index],
                                 self.out_typ_entries[column_side][entry_index],
                                 self.out_entries_divider_lines_y[column_side][entry_index], self.cfg_head_word_min_area,
                                 self.cfg_head_word_bold_thresh, self.cfg_head_word_boldness_thresh, self.cfg_head_word_width_thresh, self.cfg_entry_vertical_offset, self.cfg_head_word_offset))
        if self.verbose:
            print(f"Entry Processed:  {self.page_index:04d}_{column_side}_{entry_index:02d}")
            print("------------------------")

        self.out_txt_headwords[column_side][entry_index] = self.ocr_single_head_word(entry_address)

        self.save_single_entry_and_headword(entry_address)
        if draw_guiding_lines:
            self.draw_guiding_lines()


    def ocr_single_head_word(self, entry_address):
        column_side, entry_index = entry_address
        head_word_image = self.out_img_headwords[column_side][entry_index]
        if head_word_image is not None:
            text = pytesseract.image_to_string(head_word_image, config=self.cfg_tesseract_config)
            return text
        else:
            return None

        
    def draw_guiding_lines(self):
        for i, image_column in enumerate(self.out_img_columns):
            self.out_img_columns_with_guiding_lines[i] = (
                draw_guiding_lines(image_column, self.out_column_divider_lines_y[i], self.out_entries_start_top_y[i],
                                   self.out_rect_headwords[i], self.cfg_entries_detection_search_width_band))


    def save_both_columns(self):
        for i, image_column in enumerate(self.out_img_columns):
            if image_column is not None:
                cv2.imwrite(str(self.output_columns_folder / f"{self.page_index:04d}_{i}.png"), image_column)


    def save_guiding_lines(self):
        for i, image_column in enumerate(self.out_img_columns_with_guiding_lines):
            if image_column is not None:
                cv2.imwrite(str(self.output_guiding_lines_folder / f"{self.page_index:04d}_{i}.png"), image_column)


    def save_single_entry_and_headword(self, entry_address):
        column_side, entry_index = entry_address

        # Save entry image
        if self.out_img_entries[column_side][entry_index] is not None:
            # In each page entry index 0 is reserved for a continuation entry if exists.
            cv2.imwrite(str(self.output_entries_folder / f"{self.page_index:04d}_{column_side}_{entry_index:02d}.png"),
                        self.out_img_entries[column_side][entry_index])

            # Save head word image
            if self.out_img_headwords[column_side][entry_index] is not None:
                cv2.imwrite(str(self.output_head_words_folder / f"{self.page_index:04d}_{column_side}_{entry_index:02d}.png"),
                            self.out_img_headwords[column_side][entry_index])
                with open(str(self.output_head_words_folder / f"{self.page_index:04d}_{column_side}_{entry_index:02d}.gt.txt"), "w") as f:
                    f.write(str(self.out_txt_headwords[column_side][entry_index]))


    def save_all_entries_and_headwords(self):
        for i, column_entries in enumerate(self.out_img_entries):
            for j, entry_image in enumerate(column_entries):
                self.save_single_entry_and_headword((i, j))


    def update_folders(self, input_folder = None, output_folder = None):
        if input_folder is not None:
            self.input_folder = input_folder

        if output_folder is not None:
            self.output_folder = output_folder

        if self.output_folder is not None:

            self.output_columns_folder = self.output_folder / "columns"
            self.output_entries_folder = self.output_folder / "entries"
            self.output_head_words_folder = self.output_folder / "head_words"
            self.output_guiding_lines_folder = self.output_folder / "guiding_lines"

            self.output_folder.mkdir(parents=True, exist_ok=True)
            self.output_columns_folder.mkdir(parents=True, exist_ok=True)
            self.output_entries_folder.mkdir(parents=True, exist_ok=True)
            self.output_head_words_folder.mkdir(parents=True, exist_ok=True)
            self.output_guiding_lines_folder.mkdir(parents=True, exist_ok=True)



    def remove_outputs(self, pages):
        # Remove all output files related to the page
        if type(pages) is list:
            for page_index in pages:
                self.remove_single_page_output(page_index)
        if type(pages) is int:
            self.remove_single_page_output(pages)


    def remove_single_page_output(self, page_index):
        for folder in [self.output_columns_folder, self.output_entries_folder, self.output_head_words_folder, self.output_guiding_lines_folder]:
            for file in folder.glob(f"{page_index:04d}_*.png"):
                file.unlink()
                print(f"Removed successfully: {file}")


    def set_configs_from_string(self, text):
        """
        Parse lines like:
          cfg_head_word_min_area = 160
          cfg_head_word_bold_thresh: 30
          cfg_entry_vertical_offset = (0, 0)
          cfg_head_word_offset = (0, 0, 0, 0)
        and set them as attributes on `self`.
        """
        # Capture key and the rest of the line as value (multiline mode)
        pattern = r'(?m)^\s*(\w+)\s*[:=]\s*(.+)$'

        for key, raw_value in re.findall(pattern, text):
            # drop trailing commas/spaces and strip inline comments starting with # or //
            cleaned = re.sub(r'[,\s]+$', '', raw_value)
            cleaned = re.split(r'\s*(?:#|//)\s*', cleaned)[0].strip()

            # Try to parse as Python literal (int, tuple, list, string, float, etc.)
            parsed = None
            try:
                parsed = ast.literal_eval(cleaned)
            except Exception:
                # If literal_eval fails, try integer conversion
                try:
                    parsed = int(cleaned)
                except Exception:
                    # final fallback: unquote if quoted, otherwise raw string
                    parsed = cleaned.strip('\'"')

            setattr(self, key, parsed)


def test():
    dir_image_page_input = Path("/mnt/E0B80D7EB80D5506/sokhan/sokhan-img/")
    dir_image_page_output = Path("/mnt/E0B80D7EB80D5506/sokhan/out/")
    sokhan_processor = SokhanProcessor(dir_image_page_input, dir_image_page_output)
    sokhan_processor.verbose = True
    sokhan_processor.remove_outputs(127)
    sokhan_processor.process_pages(127)

    config = """
    cfg_entries_detection_search_width_band = 43
    cfg_dividing_lines_offset_y_start = 17
    """

    print(sokhan_processor.cfg_entries_detection_search_width_band)
    print(sokhan_processor.cfg_dividing_lines_offset_y_start)
    sokhan_processor.set_configs_from_string(config)

    print(sokhan_processor.cfg_entries_detection_search_width_band)
    print(sokhan_processor.cfg_dividing_lines_offset_y_start)

if __name__ == '__main__':
    print("Sokhan Processor Test")
    image = cv2.imread("/mnt/E0B80D7EB80D5506/sokhan/out/columns/0150_1.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    find_height_of_text_divider_lines(image)
