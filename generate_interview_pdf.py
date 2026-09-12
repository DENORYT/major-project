import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header
        self.drawString(54, 750, "Sign Language Interpreter - Interview Preparation Guide")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 742, 612-54, 742)

        # Footer
        self.line(54, 45, 612-54, 45)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 54, 30, page_text)
        self.drawString(54, 30, "CONFIDENTIAL - Project Interview Reference Q&A")
        self.restoreState()

def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=64,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284C7"),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    q_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0369A1"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    a_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    # Document Header Title
    story.append(Paragraph("Sign Language Interpreter using Deep Learning", title_style))
    story.append(Paragraph("Comprehensive Technical Interview Q&A & Project Reference Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284C7"), spaceAfter=15))

    # Meta Info Table
    meta_data = [
        [Paragraph("<b>Project Domain:</b> Deep Learning & Computer Vision", bullet_style),
         Paragraph("<b>Frameworks:</b> TensorFlow, Keras, OpenCV, Flask", bullet_style)],
        [Paragraph("<b>Target Audience:</b> Technical Interviewers / Evaluators", bullet_style),
         Paragraph("<b>Deployment:</b> Vercel Serverless & Local Web App", bullet_style)]
    ]
    t = Table(meta_data, colWidths=[250, 254])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    questions = [
        # SECTION 1
        ("SECTION 1: High-Level Architecture & Overview", [
            ("Q1: Can you give a 1-minute elevator pitch of your project?",
             "<b>Answer:</b> My project is a real-time Sign Language Interpreter that translates American Sign Language (ASL) hand gestures into written text and spoken speech. It captures live video feeds using a webcam or browser, processes the hand gesture images through custom image segmentation (YCrCb + HSV thresholding), and passes them to a 7-layer Convolutional Neural Network (CNN) trained to recognize 44 distinct gesture classes (letters A-Z, numbers 0-9, and phrases like 'I Love You'). It also features a full-stack Flask web application deployed on Vercel with real-time browser camera integration and built-in Text-to-Speech."),

            ("Q2: What real-world problem does this project address?",
             "<b>Answer:</b> Over 70 million deaf and hard-of-hearing people worldwide rely on sign language for daily communication. However, the majority of the non-deaf population cannot interpret sign language, creating a major communication barrier. This application acts as an automated 24/7 personal translator that converts ASL gestures to text/speech on any computer or mobile device without human translators."),

            ("Q3: Walk me through the end-to-end data pipeline of your system.",
             "<b>Answer:</b> The data pipeline consists of 5 core stages:<br/>"
             "1. <b>Video Ingestion:</b> Web camera frame capture at 640x480 resolution.<br/>"
             "2. <b>Region of Interest (ROI) Extraction:</b> Bounding box region (300x300) cropped around the hand area.<br/>"
             "3. <b>Skin Thresholding & Noise Reduction:</b> Hybrid YCrCb/HSV color segmentation + Otsu binarization + Morphological filtering (erosion/dilation) to isolate the hand mask.<br/>"
             "4. <b>CNN Model Inference:</b> Crop resized to 50x50 grayscale, normalized, and fed to CNN to predict class probabilities via Softmax.<br/>"
             "5. <b>Output & Speech Synthesis:</b> Prediction text looked up in SQLite DB, appended to word sentence accumulator, and spoken via Web Speech API or pyttsx3.")
        ]),

        # SECTION 2
        ("SECTION 2: Computer Vision & Image Preprocessing", [
            ("Q4: Why did you use HSV and YCrCb color spaces instead of standard RGB for hand segmentation?",
             "<b>Answer:</b> RGB space mixes illumination (brightness) with color information in all three channels (Red, Green, Blue), making skin segmentation highly sensitive to shadows and room lighting. HSV separates Hue (color tone) and Saturation (color purity) from Value (brightness). YCrCb isolates Luminance (Y) from Chrominance (Cr & Cb). In YCrCb space, human skin tones cluster tightly around specific Cr and Cb thresholds regardless of skin tone or lighting, making skin segmentation robust across different environments."),

            ("Q5: What is Histogram Backprojection and how is cv2.calcBackProject used?",
             "<b>Answer:</b> Histogram Backprojection computes how well pixels in an input image match a target color distribution histogram (stored in <code>hist</code>). <code>cv2.calcBackProject</code> maps the 2D Hue-Saturation histogram of the user's hand onto the current video frame, producing a probability map where high intensity pixels correspond to skin."),

            ("Q6: Why are morphological operations (MORPH_OPEN and MORPH_CLOSE) necessary?",
             "<b>Answer:</b> Thresholded masks often suffer from two artifacts: background noise (small false-positive dots) and interior holes (missed skin pixels inside the hand). <code>MORPH_OPEN</code> (erosion followed by dilation) removes small noise specks, while <code>MORPH_CLOSE</code> (dilation followed by erosion) bridges small gaps and fills holes inside the hand contour, creating a solid, clean mask."),

            ("Q7: How does your fallback Convexity Defects engine detect gestures without CNN training?",
             "<b>Answer:</b> The geometry fallback computes the Convex Hull around the hand contour and identifies Convexity Defects (the deep valleys between extended fingers). By measuring the distance and cosine angle between adjacent defect points, it accurately counts extended fingers (0 = Fist/A, 1 = Point/1, 2 = Victory/V, 3 = W, 4 = B, 5 = Open Palm) plug-and-play without requiring prior model training.")
        ]),

        # SECTION 3
        ("SECTION 3: Deep Learning & CNN Model Architecture", [
            ("Q8: Describe your CNN architecture in detail.",
             "<b>Answer:</b> The CNN consists of 7 layers:<br/>"
             "• <b>Conv2D Layer 1:</b> 16 filters of size 2x2, ReLU activation (input shape: 50x50x1)<br/>"
             "• <b>MaxPooling2D 1:</b> 2x2 pool size, stride 2 (reduces size to 25x25x16)<br/>"
             "• <b>Conv2D Layer 2:</b> 32 filters of size 3x3, ReLU activation<br/>"
             "• <b>MaxPooling2D 2:</b> 3x3 pool size, stride 3 (reduces size to 8x8x32)<br/>"
             "• <b>Conv2D Layer 3:</b> 64 filters of size 5x5, ReLU activation<br/>"
             "• <b>MaxPooling2D 3:</b> 5x5 pool size, stride 5 (reduces size to 1x1x64)<br/>"
             "• <b>Dense Layer 1:</b> 128 hidden units with ReLU activation<br/>"
             "• <b>Dropout Layer:</b> Rate of 0.2 (20% dropout to prevent overfitting)<br/>"
             "• <b>Dense Layer 2 (Output):</b> 44 units with Softmax activation for multiclass probability."),

            ("Q9: Why use a CNN instead of traditional ML algorithms like SVM or Random Forest?",
             "<b>Answer:</b> Traditional ML models rely on handcrafted features (like HOG or SIFT) which fail when hand shape, scale, or orientation changes slightly. CNNs automatically learn hierarchical spatial features directly from raw pixels—low-level edges in early conv layers, mid-level curves in deeper layers, and high-level hand shapes in dense layers—providing superior classification accuracy (>95%)."),

            ("Q10: Why did you choose 50x50 resolution for input images?",
             "<b>Answer:</b> A 50x50 single-channel grayscale image (2,500 pixels) retains crucial hand geometry and finger shapes while dramatically reducing parameter count (total CNN parameters ~69,982) and memory footprint. This enables real-time inference (>30 FPS) on standard CPU hardware without GPU acceleration."),

            ("Q11: What loss function and optimizer did you choose and why?",
             "<b>Answer:</b> I used <b>Categorical Cross-Entropy</b> as the loss function because this is a multi-class classification problem with 44 mutually exclusive classes. I used Stochastic Gradient Descent (<b>SGD</b>) with a learning rate of <code>0.01</code> to ensure stable convergence without gradient explosion.")
        ]),

        # SECTION 4
        ("SECTION 4: Full-Stack Web Development & Vercel Deployment", [
            ("Q12: How is the Web application implemented?",
             "<b>Answer:</b> The web application uses Flask as the Python backend framework and Bootstrap 5 + HTML5 Media Devices for the frontend UI. The browser captures camera frames via <code>navigator.mediaDevices.getUserMedia</code>, converts them to JPEG base64 strings on an HTML5 canvas, and posts them via asynchronous <code>fetch()</code> to <code>/api/predict</code>."),

            ("Q13: How did you optimize the application for Vercel Free Tier deployment?",
             "<b>Answer:</b> Vercel Free Tier enforces a strict 250 MB size limit on Python serverless functions. Full TensorFlow binaries exceed 350 MB. To make it 100% compatible:<br/>"
             "1. I created a lightweight <code>requirements.txt</code> containing <code>flask</code>, <code>opencv-python-headless</code>, and <code>numpy</code> (~25 MB package total).<br/>"
             "2. I updated <code>app.py</code> with a fallback mechanism that runs the fast Contour Geometry & Feature Detection Engine on Vercel while supporting optional Keras models locally.<br/>"
             "3. This allowed the app to deploy seamlessly in under 15 seconds on Vercel Serverless.")
        ]),

        # SECTION 5
        ("SECTION 5: Testing, Limitations & Future Enhancements", [
            ("Q14: What are the main limitations of the current system?",
             "<b>Answer:</b><br/>"
             "1. <i>Static vs. Dynamic Gestures:</i> Current CNN evaluates individual static frames. Signs requiring motion (like 'J' or 'Z') rely on frame sequences.<br/>"
             "2. <i>Single Hand Scope:</i> Optimized for 1-hand ASL signs.<br/>"
             "3. <i>Lighting Sensitivity:</i> Extreme low-light environments require threshold tuning via the built-in UI calibrator sliders."),

            ("Q15: How would you scale this project for enterprise or production use?",
             "<b>Answer:</b> To scale this to continuous sign language translation:<br/>"
             "• <b>MediaPipe Holistic:</b> Extract 3D hand, face, and body pose keypoints.<br/>"
             "• <b>LSTM / Transformer Model:</b> Pass keypoint time-series sequences into a Recurrent Neural Network (LSTM/GRU) or Temporal Transformer to interpret continuous sentences.<br/>"
             "• <b>Client-side TensorFlow.js / ONNX Runtime:</b> Export trained model to ONNX/TF.js so inference runs 100% in the user's browser GPU with zero server latency.")
        ])
    ]

    for section_title, item_list in questions:
        story.append(Paragraph(section_title, section_heading))
        story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#CBD5E1"), spaceAfter=10))

        for q_text, a_text in item_list:
            story.append(Paragraph(q_text, q_style))
            story.append(Paragraph(a_text, a_style))
            story.append(Spacer(1, 4))
        
        story.append(Spacer(1, 8))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated interview PDF: {filename}")

if __name__ == '__main__':
    create_pdf("Sign_Language_Interpreter_Interview_Guide.pdf")
