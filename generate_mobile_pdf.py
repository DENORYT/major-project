from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'ISL Interpreter - Offline Mobile App Guide', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, 'How to install and run the AI completely offline on your smartphone', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0, 102, 204) # Blue
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def body_text(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(4)

pdf = PDF()
pdf.add_page()

pdf.section_title('1. How is this possible?')
pdf.body_text("""This specific version of the app does not use Python. The entire custom mathematical 3D AI engine was translated into JavaScript. Because of this, your phone downloads the AI directly into its own memory (cache) and processes the camera feed using its own internal processor. Once installed, it requires zero internet connection.""")

pdf.section_title('2. Step 1: Host the App for Free (GitHub Pages)')
pdf.body_text("""1. Open your computer browser and go to your GitHub repository: https://github.com/DENORYT/major-project
2. Click on "Settings" (the gear icon near the top right).
3. On the left sidebar menu, click on "Pages".
4. Look for the "Build and deployment" section.
5. Change the "Source" drop-down to "Deploy from a branch".
6. Under "Branch", select "master", keep the folder as "/(root)", and click "Save".
7. Wait exactly 2 minutes, then refresh the page. At the top, GitHub will give you a live URL link (e.g., https://denoryt.github.io/major-project/).""")

pdf.section_title('3. Step 2: Install it on your Mobile Phone')
pdf.body_text("""1. Ensure your mobile phone is connected to Wi-Fi.
2. Open your phone's web browser (Chrome for Android, Safari for iOS).
3. Type in the link GitHub gave you, but add "/offline_app/index.html" to the end of it.
   Example: https://denoryt.github.io/major-project/offline_app/index.html
4. Tap the "Share" icon (on iOS) or the "3-dots menu" (on Android).
5. Tap "Add to Home Screen".
6. A new app icon will appear on your phone's home screen!""")

pdf.section_title('4. Step 3: The Offline Magic Setup')
pdf.body_text("""1. Open the new App from your home screen while still connected to Wi-Fi.
2. You will see a yellow badge saying "Loading AI...".
3. Wait a few seconds until the badge turns green and says "Ready!".
4. In the background, the app just securely downloaded the Google MediaPipe models into your phone's permanent cache.
5. You can now turn on Airplane Mode or turn off your Wi-Fi/Data completely. The app will continue to work perfectly forever.""")

pdf.add_page()
pdf.section_title('5. App Gestures and Controls Refresher')
pdf.body_text("""- Hello: Put both palms together (Namaste).
- Bye: Give a Double Thumbs Up.
- Space: Give a 1-Handed Thumbs Up.
- Delete: Point your Pinky finger straight up.
- Double Letters: Briefly drop your hand between the signs.
- Speak: Tap the Speak button to hear your built sentence.""")

pdf.output('Offline_Mobile_App_Guide.pdf')
