let handLandmarker = undefined;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let sentence = "";
let lastPred = "";
let sameCount = 0;
let lastAddedLetter = "";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const startBtn = document.getElementById("startBtn");
const statusBadge = document.getElementById("statusBadge");
const predText = document.getElementById("predText");
const sentenceBox = document.getElementById("sentenceBox");

// Import MediaPipe
const { HandLandmarker, FilesetResolver, DrawingUtils } = window;

async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
    });
    
    statusBadge.innerText = "Ready!";
    statusBadge.className = "badge bg-success";
    startBtn.disabled = false;
}

createHandLandmarker();

// Start Camera
startBtn.addEventListener("click", () => {
    if (!handLandmarker) return;
    
    if (webcamRunning) {
        webcamRunning = false;
        startBtn.innerHTML = '<i class="bi bi-camera-video"></i> Start';
        startBtn.classList.replace("btn-danger", "btn-success");
        video.srcObject.getTracks().forEach(t => t.stop());
    } else {
        webcamRunning = true;
        startBtn.innerHTML = '<i class="bi bi-stop-circle"></i> Stop';
        startBtn.classList.replace("btn-success", "btn-danger");
        
        navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } }).then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        });
    }
});

function getDistance(lm1, lm2) {
    return Math.sqrt(Math.pow(lm1.x - lm2.x, 2) + Math.pow(lm1.y - lm2.y, 2) + Math.pow(lm1.z - lm2.z, 2));
}

function getFingerStates(lms) {
    let states = [false, false, false, false, false];
    const tips = [4, 8, 12, 16, 20];
    const dips = [3, 6, 10, 14, 18];
    const palm = lms[0];
    
    // Thumb is special
    states[0] = getDistance(lms[4], lms[17]) > getDistance(lms[3], lms[17]);
    
    // Other fingers
    for(let i=1; i<5; i++) {
        states[i] = getDistance(lms[tips[i]], palm) > getDistance(lms[dips[i]], palm);
    }
    return states;
}

function getPalmSize(lms) {
    return getDistance(lms[0], lms[9]);
}

// Translate Python Geometry to JS
function classifyISL(results) {
    const hands = results.landmarks;
    const handedness = results.handednesses;
    
    if (hands.length === 0) return "Unknown";
    
    let fStates = hands.map(h => getFingerStates(h));
    
    if (hands.length === 1) {
        let fs = fStates[0];
        let upCount = fs.filter(Boolean).length;
        
        if (upCount === 1 && fs[0]) return "SPACE";
        if (upCount === 1 && fs[4]) return "DEL";
        
        if (upCount === 1 && fs[1]) return "1";
        if (upCount === 2 && fs[1] && fs[2]) {
            if (getDistance(hands[0][8], hands[0][12]) > 0.05) return "2";
        }
        if (upCount === 3 && fs[0] && fs[1] && fs[2]) return "3";
        if (upCount === 4 && !fs[0]) return "4";
        if (upCount === 5) return "5";
        if (upCount === 0) return "0";
        
        const distC = getDistance(hands[0][4], hands[0][8]);
        if (upCount === 2 && fs[0] && fs[1] && distC > 0.15) return "L";
        if (upCount === 0 && distC > 0.1) return "C";
        
        return "Unknown";
    }
    
    // Two hands
    for(let i=0; i<2; i++) {
        let base_lms = hands[i];
        let ptr_lms = hands[1-i];
        let base_f = fStates[i];
        let ptr_f = fStates[1-i];
        
        let ptr_tip = ptr_lms[8];
        let palm_size = getPalmSize(base_lms) || 0.01;
        
        function rel_dist(lm1, lm2) { return getDistance(lm1, lm2) / palm_size; }
        let sum = (arr) => arr.filter(Boolean).length;
        
        if (sum(base_f) === 5) {
            let p_up = sum(ptr_f);
            if (p_up===1) return "6";
            if (p_up===2) return "7";
            if (p_up===3) return "8";
            if (p_up===4) return "9";
        }
        
        if (rel_dist(ptr_lms[0], base_lms[0]) < 1.2 && rel_dist(ptr_lms[12], base_lms[12]) < 1.0 && sum(base_f)>=4 && sum(ptr_f)>=4) return "HELLO";
        if (sum(base_f)===1 && base_f[0] && sum(ptr_f)===1 && ptr_f[0]) return "BYE";
        
        if (rel_dist(ptr_tip, base_lms[4]) < 0.7) return "A";
        
        let ptr_c = rel_dist(ptr_lms[8], ptr_lms[4]) < 0.8;
        let base_c = rel_dist(base_lms[8], base_lms[4]) < 0.8;
        if (ptr_c && base_c && rel_dist(ptr_lms[8], base_lms[8]) < 1.2) return "Q";
        if (ptr_c && rel_dist(ptr_lms[8], base_lms[8]) < 0.8) return "P";
        
        if (rel_dist(ptr_tip, base_lms[8]) < 0.7) return "E";
        if (rel_dist(ptr_tip, base_lms[12]) < 0.7) return "I";
        if (rel_dist(ptr_tip, base_lms[16]) < 0.7) return "O";
        if (rel_dist(ptr_tip, base_lms[20]) < 0.7) return "U";
        
        let base_palm = base_lms[9];
        let d_idx = rel_dist(ptr_lms[8], base_palm);
        let d_mid = rel_dist(ptr_lms[12], base_palm);
        let d_ring = rel_dist(ptr_lms[16], base_palm);
        
        if (d_idx < 1.2 && d_mid < 1.2 && d_ring < 1.2) return "M";
        else if (d_idx < 1.2 && d_mid < 1.2) {
            if (rel_dist(ptr_lms[8], ptr_lms[12]) > 0.5) return "V";
            else return "N";
        }
        
        if (rel_dist(ptr_lms[8], base_lms[8]) < 1.0 && rel_dist(ptr_lms[4], base_lms[5]) < 1.0) return "D";
        if (rel_dist(base_lms[5], ptr_lms[5]) < 0.8 && sum(base_f)>=4 && sum(ptr_f)>=4) return "B";
        if (rel_dist(ptr_lms[8], base_lms[12]) < 0.7 && ptr_f[1] && base_f[2]) return "F";
        if (sum(base_f)<=1 && sum(ptr_f)<=1 && rel_dist(ptr_lms[5], base_lms[5]) < 1.2) return "G";
        if (sum(base_f)>=4 && sum(ptr_f)>=4 && rel_dist(ptr_lms[8], base_lms[0]) < 1.5 && rel_dist(ptr_lms[0], base_lms[0]) > 2.0) return "H";
        if (rel_dist(ptr_lms[20], base_palm) < 0.8 && ptr_f[4] && !ptr_f[1]) return "J";
        if (rel_dist(ptr_lms[8], base_lms[5]) < 0.8 && ptr_f[1] && base_f[1]) return "K";
        if (rel_dist(ptr_lms[8], base_lms[0]) < 0.8 && ptr_f[1]) return "R";
        if (rel_dist(ptr_lms[4], base_lms[8]) < 0.8 && ptr_f[0] && base_f[1]) return "Y";
        if (sum(base_f)>=4 && sum(ptr_f)>=4 && rel_dist(ptr_lms[0], base_lms[9]) < 1.5 && rel_dist(ptr_lms[8], base_lms[9]) > 1.5) return "Z";
        
        if (rel_dist(ptr_lms[8], base_lms[8]) < 0.8 && ptr_f[1] && base_f[1] && !ptr_f[2] && !base_f[2]) return "X";
        if (rel_dist(ptr_lms[20], base_lms[20]) < 0.8 && ptr_f[4] && base_f[4]) return "S";
        if (rel_dist(ptr_lms[8], base_palm) < 0.8 && ptr_f[1] && sum(base_f)>=4) return "T";
        if (rel_dist(ptr_lms[8], base_lms[8]) < 0.8 && rel_dist(ptr_lms[12], base_lms[12]) < 0.8) return "W";
    }
    
    return "Unknown";
}

let lastCallTime = Date.now();

async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        let results = handLandmarker.detectForVideo(video, startTimeMs);
        
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        const drawingUtils = new DrawingUtils(canvasCtx);
        if (results.landmarks) {
            for (const landmarks of results.landmarks) {
                drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 3 });
                drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
            }
        }
        canvasCtx.restore();
        
        let pred = classifyISL(results);
        
        if (pred !== "Unknown") {
            predText.innerText = pred;
            if (pred === lastPred) {
                sameCount++;
            } else {
                sameCount = 0;
                lastPred = pred;
            }
            
            if (sameCount >= 8) { // About 1.2 seconds at normal frame rates
                if (lastAddedLetter !== pred) {
                    lastAddedLetter = pred;
                    if (pred === 'SPACE') {
                        if (!sentence.endsWith(' ')) sentence += ' ';
                    } else if (pred === 'DEL') {
                        sentence = sentence.slice(0, -1);
                    } else if (pred === 'HELLO' || pred === 'BYE') {
                        if (sentence.length > 0 && !sentence.endsWith(' ')) sentence += ' ';
                        sentence += pred + ' ';
                    } else {
                        sentence += pred;
                    }
                    sentenceBox.innerText = sentence;
                    sentenceBox.style.borderColor = '#3b82f6';
                    setTimeout(() => sentenceBox.style.borderColor = '#374151', 400);
                }
                sameCount = 0;
            }
        } else {
            predText.innerText = "—";
            sameCount = 0;
            lastPred = "";
            lastAddedLetter = "";
        }
    }
    
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

document.getElementById("speakBtn").addEventListener("click", () => {
    if (sentence.trim() && window.speechSynthesis) {
        const u = new SpeechSynthesisUtterance(sentence);
        window.speechSynthesis.speak(u);
    }
});
document.getElementById("spaceBtn").addEventListener("click", () => { sentence += ' '; sentenceBox.innerText = sentence; });
document.getElementById("clearBtn").addEventListener("click", () => { sentence = ''; sentenceBox.innerText = ''; });
