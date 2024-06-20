# Image Processor App

## Description
This project is a computer vision application developed using Python and the Tkinter library for the GUI. It allows users to upload images and apply various image processing filters such as Gaussian, Butterworth, and Laplacian. Additionally, users can perform histogram matching using a reference image. The application provides a user-friendly interface to visualize the original, processed, and reference images along with displaying basic image statistics.

## Features
- **Upload Image**: Allows users to upload an image from their local filesystem.
- **Apply Gaussian Filter**: Applies a Gaussian blur to the uploaded image.
- **Apply Butterworth Filter**: Applies a Butterworth filter to the uploaded image.
- **Apply Laplacian Filter**: Applies a Laplacian filter to the uploaded image.
- **Upload Reference Image**: Allows users to upload a reference image for histogram matching.
- **Histogram Matching**: Matches the histogram of the uploaded image to that of the reference image.
- **Display Image Statistics**: Displays the mean and standard deviation of the uploaded and processed images.

## Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```sh
    cd image-processor-app
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Dependencies
- Python 3.x
- Tkinter
- OpenCV
- NumPy
- Pillow

## Usage
1. Run the application:
    ```sh
    python image_processor_app.py
    ```
2. Use the provided buttons to upload images and apply filters.

## File Structure
- `image_processor_app.py`: Main application file containing the GUI and image processing logic.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Author
Najeebullah Khan (023-20-0058), Sec-C

For any inquiries or issues, please contact the author at [your-email@example.com].

---
