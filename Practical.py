from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid

app = Flask(__name__)

# ============================================================
# SETTINGS
# ============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Your USN / Roll Number
ROLL_NO = "CS24103"


# ============================================================
# PRACTICAL LIST
# Website numbering starts from 01
# ============================================================

PRACTICALS = {
    1: "RGB / Grayscale, Arithmetic & Bitwise Operations",
    2: "2-D Geometric Transformations",
    3: "Histogram Equalization, Spatial Enhancement & Thresholding",
    4: "Spatial Domain Filters",
    5: "Image Inpainting",
    6: "Lossless Image Compression",
    7: "Morphological Operations",
    8: "Object Detection using Correlation",
    9: "Top-Hat Transformation",
    10: "Colour Space Conversion",
    11: "Edge Detection"
}


# ============================================================
# ADD USN TO OUTPUT IMAGE
# ============================================================

def add_roll_no(image):
    """
    Adds CS24103 to the bottom-right corner
    of every generated output image.
    """

    if image is None:
        return image

    # If output is grayscale, convert it to BGR
    # so that the text can be drawn properly.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image.copy()

    height, width = image.shape[:2]

    # Text size changes according to image width
    font_scale = max(0.6, min(1.2, width / 900))

    thickness = max(
        1,
        int(round(font_scale * 2))
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        ROLL_NO,
        font,
        font_scale,
        thickness
    )

    text_width, text_height = text_size

    margin = max(
        15,
        int(width * 0.02)
    )

    # Bottom-right position
    x = width - text_width - margin
    y = height - margin

    # White background rectangle
    cv2.rectangle(
        image,
        (
            x - 10,
            y - text_height - 10
        ),
        (
            x + text_width + 10,
            y + 10
        ),
        (255, 255, 255),
        -1
    )

    # Write USN
    cv2.putText(
        image,
        ROLL_NO,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    return image


# ============================================================
# READ IMAGE
# ============================================================

def read_image(file_storage, grayscale=False):

    if file_storage is None:
        return None

    if file_storage.filename == "":
        return None

    data = np.frombuffer(
        file_storage.read(),
        np.uint8
    )

    if data.size == 0:
        return None

    if grayscale:
        return cv2.imdecode(
            data,
            cv2.IMREAD_GRAYSCALE
        )

    return cv2.imdecode(
        data,
        cv2.IMREAD_COLOR
    )


# ============================================================
# RESIZE SECOND IMAGE
# ============================================================

def resize_to_match(image1, image2):

    if image1.shape[:2] != image2.shape[:2]:

        image2 = cv2.resize(
            image2,
            (
                image1.shape[1],
                image1.shape[0]
            )
        )

    return image2


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        practicals=PRACTICALS
    )


# ============================================================
# PRACTICAL PAGE
# ============================================================

@app.route("/practical/<int:number>")
def practical(number):

    if number not in PRACTICALS:
        return "Practical not found", 404

    return render_template(
        "practical.html",
        number=number,
        title=PRACTICALS[number]
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

@app.route(
    "/process/<int:number>",
    methods=["POST"]
)
def process(number):

    if number not in PRACTICALS:

        return jsonify({
            "error": "Invalid practical number."
        }), 400

    # --------------------------------------------------------
    # Read main input image
    # --------------------------------------------------------

    image = read_image(
        request.files.get("image")
    )

    if image is None:

        return jsonify({
            "error": "Please select a valid input image."
        }), 400

    operation = request.form.get(
        "operation",
        ""
    ).strip()

    if not operation:

        return jsonify({
            "error": "Please select an operation."
        }), 400

    result = None
    extra = {}

    try:

        # ====================================================
        # PRACTICAL 01
        # RGB / GRAYSCALE / ARITHMETIC / BITWISE
        # ====================================================

        if number == 1:

            # -------------------------------
            # Grayscale
            # -------------------------------

            if operation == "grayscale":

                result = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

            # -------------------------------
            # RGB
            # -------------------------------

            elif operation == "rgb":

                rgb_image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                # Convert back for correct image saving
                result = cv2.cvtColor(
                    rgb_image,
                    cv2.COLOR_RGB2BGR
                )

            # -------------------------------
            # Addition
            # -------------------------------

            elif operation == "add":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.add(
                    image,
                    image2
                )

            # -------------------------------
            # Subtraction
            # -------------------------------

            elif operation == "subtract":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.subtract(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise NOT
            # -------------------------------

            elif operation == "bitwise_not":

                result = cv2.bitwise_not(
                    image
                )

            # -------------------------------
            # Bitwise AND
            # -------------------------------

            elif operation == "and":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_and(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise OR
            # -------------------------------

            elif operation == "or":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_or(
                    image,
                    image2
                )

            # -------------------------------
            # Bitwise XOR
            # -------------------------------

            elif operation == "xor":

                image2 = read_image(
                    request.files.get("image2")
                )

                if image2 is None:

                    return jsonify({
                        "error":
                        "Please select the second image."
                    }), 400

                image2 = resize_to_match(
                    image,
                    image2
                )

                result = cv2.bitwise_xor(
                    image,
                    image2
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 01 operation."
                }), 400


        # ====================================================
        # PRACTICAL 02
        # GEOMETRIC TRANSFORMATIONS
        # ====================================================

        elif number == 2:

            height, width = image.shape[:2]

            # -------------------------------
            # Translation
            # -------------------------------

            if operation == "translation":

                tx = float(
                    request.form.get(
                        "tx",
                        100
                    )
                )

                ty = float(
                    request.form.get(
                        "ty",
                        50
                    )
                )

                matrix = np.float32([
                    [1, 0, tx],
                    [0, 1, ty]
                ])

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height)
                )

            # -------------------------------
            # Rotation
            # -------------------------------

            elif operation == "rotation":

                angle = float(
                    request.form.get(
                        "value",
                        30
                    )
                )

                center = (
                    width // 2,
                    height // 2
                )

                matrix = cv2.getRotationMatrix2D(
                    center,
                    angle,
                    1.0
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, height)
                )

            # -------------------------------
            # Scaling
            # -------------------------------

            elif operation == "scaling":

                scale = float(
                    request.form.get(
                        "value",
                        1.5
                    )
                )

                if scale <= 0:

                    return jsonify({
                        "error":
                        "Scaling value must be greater than 0."
                    }), 400

                result = cv2.resize(
                    image,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_LINEAR
                )

            # -------------------------------
            # X Shearing
            # -------------------------------

            elif operation == "shearing_x":

                shear = float(
                    request.form.get(
                        "value",
                        0.3
                    )
                )

                matrix = np.float32([
                    [1, shear, 0],
                    [0, 1, 0]
                ])

                new_width = int(
                    width +
                    abs(shear) * height
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (new_width, height)
                )

            # -------------------------------
            # Y Shearing
            # -------------------------------

            elif operation == "shearing_y":

                shear = float(
                    request.form.get(
                        "value",
                        0.3
                    )
                )

                matrix = np.float32([
                    [1, 0, 0],
                    [shear, 1, 0]
                ])

                new_height = int(
                    height +
                    abs(shear) * width
                )

                result = cv2.warpAffine(
                    image,
                    matrix,
                    (width, new_height)
                )

            # -------------------------------
            # Reflection
            # -------------------------------

            elif operation == "reflection":

                result = cv2.flip(
                    image,
                    1
                )

            # -------------------------------
            # Cropping
            # -------------------------------

            elif operation == "cropping":

                x1 = int(width * 0.20)
                x2 = int(width * 0.80)

                y1 = int(height * 0.20)
                y2 = int(height * 0.80)

                result = image[
                    y1:y2,
                    x1:x2
                ].copy()

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 02 operation."
                }), 400


        # ====================================================
        # PRACTICAL 03
        # IMAGE ENHANCEMENT
        # ====================================================

        elif number == 3:

            # -------------------------------
            # Histogram Equalization
            # -------------------------------

            if operation == "histogram_equalization":

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                result = cv2.equalizeHist(
                    gray
                )

            # -------------------------------
            # Smoothing
            # -------------------------------

            elif operation == "smoothing":

                result = cv2.GaussianBlur(
                    image,
                    (5, 5),
                    0
                )

            # -------------------------------
            # Sharpening
            # -------------------------------

            elif operation == "sharpening":

                kernel = np.array([
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ])

                result = cv2.filter2D(
                    image,
                    -1,
                    kernel
                )

            # -------------------------------
            # Thresholding
            # -------------------------------

            elif operation == "threshold":

                threshold_value = int(
                    float(
                        request.form.get(
                            "value",
                            127
                        )
                    )
                )

                threshold_value = max(
                    0,
                    min(
                        255,
                        threshold_value
                    )
                )

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                _, result = cv2.threshold(
                    gray,
                    threshold_value,
                    255,
                    cv2.THRESH_BINARY
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 03 operation."
                }), 400


        # ====================================================
        # PRACTICAL 04
        # SPATIAL FILTERS
        # ====================================================

        elif number == 4:

            # -------------------------------
            # Averaging
            # -------------------------------

            if operation == "averaging":

                result = cv2.blur(
                    image,
                    (5, 5)
                )

            # -------------------------------
            # Gaussian
            # -------------------------------

            elif operation == "gaussian":

                result = cv2.GaussianBlur(
                    image,
                    (5, 5),
                    0
                )

            # -------------------------------
            # Median
            # -------------------------------

            elif operation == "median":

                result = cv2.medianBlur(
                    image,
                    5
                )

            # -------------------------------
            # Bilateral
            # -------------------------------

            elif operation == "bilateral":

                result = cv2.bilateralFilter(
                    image,
                    9,
                    75,
                    75
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 04 operation."
                }), 400


        # ====================================================
        # PRACTICAL 05
        # IMAGE INPAINTING
        # ====================================================

        elif number == 5:

            mask = read_image(
                request.files.get("mask"),
                grayscale=True
            )

            if mask is None:

                return jsonify({
                    "error":
                    "Please select an inpainting mask."
                }), 400

            mask = cv2.resize(
                mask,
                (
                    image.shape[1],
                    image.shape[0]
                )
            )

            _, mask = cv2.threshold(
                mask,
                127,
                255,
                cv2.THRESH_BINARY
            )

            # -------------------------------
            # Telea
            # -------------------------------

            if operation == "telea":

                result = cv2.inpaint(
                    image,
                    mask,
                    3,
                    cv2.INPAINT_TELEA
                )

            # -------------------------------
            # Navier-Stokes
            # -------------------------------

            elif operation == "ns":

                result = cv2.inpaint(
                    image,
                    mask,
                    3,
                    cv2.INPAINT_NS
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 05 operation."
                }), 400


        # ====================================================
        # PRACTICAL 06
        # LOSSLESS COMPRESSION
        # ====================================================

        elif number == 6:

            if operation != "lossless":

                return jsonify({
                    "error":
                    "Invalid Practical 06 operation."
                }), 400

            # PNG with no compression
            success, original_png = cv2.imencode(
                ".png",
                image,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    0
                ]
            )

            if not success:

                return jsonify({
                    "error":
                    "Could not create PNG image."
                }), 500

            # PNG with maximum compression
            success, compressed_png = cv2.imencode(
                ".png",
                image,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    9
                ]
            )

            if not success:

                return jsonify({
                    "error":
                    "Could not compress image."
                }), 500

            original_size = len(
                original_png.tobytes()
            )

            compressed_size = len(
                compressed_png.tobytes()
            )

            if original_size > 0:

                saving_percent = round(
                    (
                        (
                            original_size -
                            compressed_size
                        )
                        /
                        original_size
                    ) * 100,
                    2
                )

            else:

                saving_percent = 0

            result = cv2.imdecode(
                compressed_png,
                cv2.IMREAD_COLOR
            )

            extra = {
                "original_size":
                    original_size,

                "compressed_size":
                    compressed_size,

                "saving_percent":
                    saving_percent
            }


        # ====================================================
        # PRACTICAL 07
        # MORPHOLOGICAL OPERATIONS
        # ====================================================

        elif number == 7:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            _, binary = cv2.threshold(
                gray,
                127,
                255,
                cv2.THRESH_BINARY
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (5, 5)
            )

            # -------------------------------
            # Erosion
            # -------------------------------

            if operation == "erosion":

                result = cv2.erode(
                    binary,
                    kernel,
                    iterations=1
                )

            # -------------------------------
            # Dilation
            # -------------------------------

            elif operation == "dilation":

                result = cv2.dilate(
                    binary,
                    kernel,
                    iterations=1
                )

            # -------------------------------
            # Opening
            # -------------------------------

            elif operation == "opening":

                result = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_OPEN,
                    kernel
                )

            # -------------------------------
            # Closing
            # -------------------------------

            elif operation == "closing":

                result = cv2.morphologyEx(
                    binary,
                    cv2.MORPH_CLOSE,
                    kernel
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 07 operation."
                }), 400


        # ====================================================
        # PRACTICAL 08
        # CORRELATION OBJECT DETECTION
        # ====================================================

        elif number == 8:

            if operation != "correlation":

                return jsonify({
                    "error":
                    "Invalid Practical 08 operation."
                }), 400

            template = read_image(
                request.files.get("template")
            )

            if template is None:

                return jsonify({
                    "error":
                    "Please select a template image."
                }), 400

            if (
                template.shape[0] >
                image.shape[0]
                or
                template.shape[1] >
                image.shape[1]
            ):

                return jsonify({
                    "error":
                    "Template image must be smaller than input image."
                }), 400

            gray_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            gray_template = cv2.cvtColor(
                template,
                cv2.COLOR_BGR2GRAY
            )

            result_map = cv2.matchTemplate(
                gray_image,
                gray_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_value, _, max_location = (
                cv2.minMaxLoc(result_map)
            )

            result = image.copy()

            template_height, template_width = (
                gray_template.shape[:2]
            )

            # Threshold for successful detection
            threshold = 0.8

            if max_value >= threshold:

                top_left = max_location

                bottom_right = (
                    top_left[0] + template_width,
                    top_left[1] + template_height
                )

                # Green rectangle around detected object
                cv2.rectangle(
                    result,
                    top_left,
                    bottom_right,
                    (0, 255, 0),
                    3
                )

            extra = {
                "correlation_score":
                    round(
                        float(max_value),
                        4
                    )
            }


        # ====================================================
        # PRACTICAL 09
        # TOP-HAT TRANSFORMATION
        # ====================================================

        elif number == 9:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (15, 15)
            )

            result = cv2.morphologyEx(
                gray,
                cv2.MORPH_TOPHAT,
                kernel
            )


        # ====================================================
        # PRACTICAL 10
        # COLOUR SPACE CONVERSION
        # ====================================================

        elif number == 10:

            # -------------------------------
            # RGB
            # -------------------------------

            if operation == "rgb":

                rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                result = cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR
                )

            # -------------------------------
            # HSV
            # -------------------------------

            elif operation == "hsv":

                hsv = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2HSV
                )

                result = cv2.normalize(
                    hsv,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

            # -------------------------------
            # YCrCb
            # -------------------------------

            elif operation == "ycrcb":

                ycrcb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2YCrCb
                )

                result = cv2.normalize(
                    ycrcb,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

            # -------------------------------
            # Lab
            # -------------------------------

            elif operation == "lab":

                lab = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2LAB
                )

                result = cv2.normalize(
                    lab,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 10 operation."
                }), 400


        # ====================================================
        # PRACTICAL 11
        # EDGE DETECTION
        # ====================================================

        elif number == 11:

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            # -------------------------------
            # Canny
            # -------------------------------

            if operation == "canny":

                result = cv2.Canny(
                    gray,
                    100,
                    200
                )

            # -------------------------------
            # Sobel
            # -------------------------------

            elif operation == "sobel":

                sobel_x = cv2.Sobel(
                    gray,
                    cv2.CV_64F,
                    1,
                    0,
                    ksize=3
                )

                sobel_y = cv2.Sobel(
                    gray,
                    cv2.CV_64F,
                    0,
                    1,
                    ksize=3
                )

                magnitude = cv2.magnitude(
                    sobel_x.astype(np.float32),
                    sobel_y.astype(np.float32)
                )

                result = cv2.convertScaleAbs(
                    magnitude
                )

            # -------------------------------
            # Prewitt
            # -------------------------------

            elif operation == "prewitt":

                kernel_x = np.array([
                    [-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]
                ], dtype=np.float32)

                kernel_y = np.array([
                    [-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]
                ], dtype=np.float32)

                prewitt_x = cv2.filter2D(
                    gray,
                    cv2.CV_32F,
                    kernel_x
                )

                prewitt_y = cv2.filter2D(
                    gray,
                    cv2.CV_32F,
                    kernel_y
                )

                magnitude = cv2.magnitude(
                    prewitt_x,
                    prewitt_y
                )

                result = cv2.convertScaleAbs(
                    magnitude
                )

            else:

                return jsonify({
                    "error":
                    "Invalid Practical 11 operation."
                }), 400

        else:

            return jsonify({
                "error":
                "Practical not implemented."
            }), 400


        # ====================================================
        # IMPORTANT:
        # ADD CS24103 TO EVERY OUTPUT IMAGE
        # ====================================================

        result = add_roll_no(result)


        # ====================================================
        # CREATE UNIQUE FILE NAMES
        # ====================================================

        unique_id = uuid.uuid4().hex

        input_filename = (
            "input_" +
            unique_id +
            ".png"
        )

        output_filename = (
            "output_" +
            unique_id +
            ".png"
        )

        input_path = os.path.join(
            UPLOAD_FOLDER,
            input_filename
        )

        output_path = os.path.join(
            UPLOAD_FOLDER,
            output_filename
        )


        # ====================================================
        # SAVE IMAGES
        # ====================================================

        cv2.imwrite(
            input_path,
            image
        )

        cv2.imwrite(
            output_path,
            result
        )


        # ====================================================
        # SEND RESULT TO FRONTEND
        # ====================================================

        return jsonify({
            "input":
                f"/uploads/{input_filename}",

            "output":
                f"/uploads/{output_filename}",

            "extra":
                extra,

            "roll_no":
                ROLL_NO
        })


    except Exception as e:

        print(
            "Processing error:",
            e
        )

        return jsonify({
            "error":
            f"Processing failed: {str(e)}"
        }), 500


# ============================================================
# SERVE GENERATED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )



# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
