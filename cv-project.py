import os
import tkinter as tk
from tkinter import Label, Button, filedialog, Frame, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Computer Vision Project - Najeebullah Khan (023-20-0058)  Sec-C"
        )
        self.create_widgets()

        self.img_path = "img.jpg"
        self.original_image = None
        self.processed_image = None
        self.reference_image = None  # Reference image for histogram matching

        if os.path.exists(self.img_path):
            self.original_image = self.load_image(self.img_path)
            self.processed_image = self.original_image.copy()
            self.display_image(self.label_original_img, self.original_image)
            self.display_image(self.label_processed_img, self.processed_image)
            self.print_stats(self.original_image)

    def create_widgets(self):
        self.controls_frame = Frame(self.root, padx=10, pady=10)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)

        Button(
            self.controls_frame, text="Upload Image", command=self.upload_image
        ).pack(side=tk.LEFT, padx=10)
        Button(
            self.controls_frame,
            text="Apply Gaussian Filter",
            command=lambda: self.apply_filter(self.gaussian_filter),
        ).pack(side=tk.LEFT, padx=10)
        Button(
            self.controls_frame,
            text="Apply Butterworth Filter",
            command=lambda: self.apply_filter(self.butterworth_filter),
        ).pack(side=tk.LEFT, padx=10)
        Button(
            self.controls_frame,
            text="Apply Laplacian Filter",
            command=lambda: self.apply_filter(self.laplacian_filter),
        ).pack(side=tk.LEFT, padx=10)
        Label(
            self.controls_frame,
            text="Add Reference Image First to Apply Histogram Matching:",
            fg="red",
        ).pack(side=tk.LEFT, padx=10)
        Button(
            self.controls_frame,
            text="Upload Reference Image",
            command=self.upload_reference_image,
        ).pack(side=tk.LEFT, padx=10)
        Button(
            self.controls_frame,
            text="Histogram Matching",
            command=self.apply_histogram_matching,
        ).pack(side=tk.LEFT, padx=10)

        self.images_frame = Frame(self.root)
        self.images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.label_original_img = Label(self.images_frame, text="Original Image")
        self.label_original_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label_processed_img = Label(self.images_frame, text="Processed Image")
        self.label_processed_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.label_reference_img = Label(self.images_frame, text="Reference Image")
        self.label_reference_img.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.stats_label = Label(self.root, text="", relief=tk.SUNKEN, anchor=tk.W)
        self.stats_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Unable to load image at {image_path}.")
            return None
        return image

    def display_image(self, label, image):
        if image is not None:
            image_to_display = ImageTk.PhotoImage(image=Image.fromarray(image))
            label.config(image=image_to_display)
            label.image = image_to_display

    def print_stats(self, image):
        if image is not None:
            mean_val = np.mean(image)
            std_dev = np.std(image)
            self.stats_label.config(
                text=f"Standard Deviation: {std_dev:.2f} , Mean: {mean_val:.2f}  "
            )

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img_path = file_path
            self.original_image = self.load_image(self.img_path)
            self.processed_image = self.original_image.copy()
            self.display_image(self.label_original_img, self.original_image)
            self.display_image(self.label_processed_img, self.processed_image)
            self.print_stats(self.original_image)

    def upload_reference_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.reference_image = self.load_image(file_path)
            if self.reference_image is None:
                messagebox.showerror("Error", "Failed to load reference image.")
            else:
                self.display_image(self.label_reference_img, self.reference_image)
                print("Reference image loaded successfully.")

    def apply_filter(self, filter_function):
        if self.original_image is not None:
            self.processed_image = filter_function(self.original_image)
            self.display_image(self.label_processed_img, self.processed_image)
            self.print_stats(self.processed_image)
        else:
            messagebox.showerror("Error", "Load an image first.")

    def gaussian_filter(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def butterworth_filter(self, image):
        d = 30
        crow, ccol = int(image.shape[0] / 2), int(image.shape[1] / 2)
        mask = np.zeros((image.shape[0], image.shape[1]), np.float32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j] = 1 / (1 + ((i - crow) ** 2 + (j - ccol) ** 2) / d**2)
        fshift = np.fft.fftshift(np.fft.fft2(image))
        fshift_filtered = fshift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        img_back = np.abs(img_back)
        img_back_normalized = (img_back / np.max(img_back) * 255).astype(np.uint8)
        return img_back_normalized

    def laplacian_filter(self, image):
        laplacian_kernel = np.array(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32
        )
        return cv2.filter2D(image, -1, laplacian_kernel)

    def apply_histogram_matching(self):
        if self.original_image is not None and self.reference_image is not None:
            self.processed_image = self.histogram_matching(
                self.original_image, self.reference_image
            )
            self.display_image(self.label_processed_img, self.processed_image)
            self.print_stats(self.processed_image)
        else:
            messagebox.showerror("Error", "Original or reference image not loaded.")

    def histogram_matching(self, original, reference):
        original_hist, bins = np.histogram(
            original.flatten(), 256, [0, 256], density=True
        )
        original_cdf = original_hist.cumsum()

        reference_hist, ref_bins = np.histogram(
            reference.flatten(), 256, [0, 256], density=True
        )
        reference_cdf = reference_hist.cumsum()

        lookup_table = np.interp(original_cdf, reference_cdf, bins[:-1])

        matched = np.interp(original.flatten(), bins[:-1], lookup_table)
        matched_image = matched.reshape(original.shape)
        matched_image = np.uint8(matched_image)
        return matched_image


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.geometry("1400x700")  # Adjusted for potentially larger layout
    root.mainloop()
