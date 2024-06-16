import {
    HandLandmarker,
    FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";
const demosSection = document.getElementById("demos");

let handLandmarker = undefined;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
let capturedPoses = [];

let knnClassifier = ml5.KNNClassifier();



const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });
    demosSection.classList.remove("invisible");
};
createHandLandmarker();

const imageContainers = document.getElementsByClassName("detectOnClick");

for (let i = 0; i < imageContainers.length; i++) {
    // Add event listener to the child element which is the img element.
    imageContainers[i].children[0].addEventListener("click", handleClick);
}

async function handleClick(event) {
    if (!handLandmarker) {
        console.log("Wait for handLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        await handLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // Remove all landmarks drawn before
    const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
    for (var i = allCanvas.length - 1; i >= 0; i--) {
        const n = allCanvas[i];
        n.parentNode.removeChild(n);
    }
    const handLandmarkerResult = handLandmarker.detect(event.target);
    const canvas = document.createElement("canvas");
    canvas.setAttribute("class", "canvas");
    canvas.setAttribute("width", canvas.offsetWidth + "px");
    canvas.setAttribute("height", canvas.offsetHeight + "px");
    canvas.style =
        "left: 0px;" +
        "top: 0px;" +
        "width: " +
        event.target.width +
        "px;" +
        "height: " +
        event.target.height +
        "px;";

    event.target.parentNode.appendChild(canvas);
    const cxt = canvas.getContext("2d");
    for (const landmarks of handLandmarkerResult.landmarks) {
        drawConnectors(cxt, landmarks, HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5
        });
        drawLandmarks(cxt, landmarks, { color: "#FF0000", lineWidth: 1 });
    }
}

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

// Check if webcam access is supported.
const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableWebcamButton = document.getElementById("webcamButton");
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    if (webcamRunning === true) {
        webcamRunning = false;
        enableWebcamButton.innerText = "ENABLE PREDICTIONS";
    } else {
        webcamRunning = true;
        enableWebcamButton.innerText = "DISABLE PREDICTIONS";
    }

    // getUsermedia parameters.
    const constraints = {
        video: true
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

let lastVideoTime = -1;
let results = undefined;
async function predictWebcam() {
    canvasElement.style.width = video.videoWidth;
    canvasElement.style.height = video.videoHeight;
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    // Start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.landmarks && results.landmarks.length > 0) {
        for (const landmarks of results.landmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
                color: "#00FF00",
                lineWidth: 5
            });

            drawLandmarks(canvasCtx, landmarks, { color: "#FF0000", lineWidth: 2 });

            const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
            knnClassifier.classify(pose, (error, result) => {
                if (error) {
                    console.error('Error classifying pose:', error);
                    return;
                } else {
                    const { label, confidencesByLabel } = result;
                    const confidence = confidencesByLabel[label] * 100; // Zet om naar percentage
                    displayPose(label, confidence);
                }
            });
        }
    } else {
        // If no landmarks are detected, reset the displayed pose
        displayPose("empty", 0);
    }
    canvasCtx.restore();

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function displayPose(poseLabel, confidence) {
    // Display the current pose label somewhere on the page with confidence score
    document.getElementById('currentPoseDisplay').textContent = `Current Pose: ${poseLabel} (${confidence.toFixed(2)}%)`;
}

function detectScrollGesture(landmarks) {
    const indexFingerTip = landmarks[8];

    const screenHeight = video.videoHeight;
    const pointingUpThreshold = screenHeight * 0.3;
    const pointingDownThreshold = screenHeight * 0.7;

    if (indexFingerTip.y < pointingUpThreshold) {
        scrollIframe('up');
    } else if (indexFingerTip.y > pointingDownThreshold) {
        scrollIframe('down');
    }
}

function scrollIframe(direction) {
    const iframe = document.getElementById('if1').contentWindow;
    const scrollAmount = 100;

    try {
        if (direction === 'up') {
            iframe.scrollBy(0, -scrollAmount);
        } else if (direction === 'down') {
            iframe.scrollBy(0, scrollAmount);
        }
    } catch (error) {
        console.error("Error scrolling within iframe:", error);
    }
}

// Logic to add poses:

let savedPoses = new Set(); // This will store unique pose labels
let trainingsData = [];

document.getElementById('capturePoseButton').addEventListener('click', () => {
    const currentEmoji = document.getElementById('capturePoseButton').textContent;

    if (results && results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);

        trainingsData.push({ pose, label: currentEmoji });

        localStorage.setItem('trainingData', JSON.stringify(trainingsData));

        knnClassifier.addExample(pose, currentEmoji);
        console.log(`Pose for ${currentEmoji} added to the classifier after verification.`);
        generatePoseCanvasses();

    } else {
        console.log("No pose detected or webcam not running.");
    }
});

document.getElementById('capturePoseButton1').addEventListener('click', () => {
    const currentEmoji = document.getElementById('capturePoseButton1').textContent;

    if (results && results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
        trainingsData.push({ pose, label: currentEmoji });

        localStorage.setItem('trainingData', JSON.stringify(trainingsData));

        knnClassifier.addExample(pose, currentEmoji);
        generatePoseCanvasses();
        console.log(`Pose for ${currentEmoji} added to the classifier after verification.`);
    } else {
        console.log("No pose detected or webcam not running.");
    }
});

document.getElementById('capturePoseButton2').addEventListener('click', () => {
    const currentEmoji = document.getElementById('capturePoseButton2').textContent;

    if (results && results.landmarks && results.landmarks.length > 0) {
        const landmarks = results.landmarks[0];
        const pose = landmarks.flatMap(({ x, y, z }) => [x, y, z ?? 0]);
        trainingsData.push({ pose, label: currentEmoji });

        localStorage.setItem('trainingData', JSON.stringify(trainingsData));

        knnClassifier.addExample(pose, currentEmoji);
        generatePoseCanvasses();
        console.log(`Pose for ${currentEmoji} added to the classifier after verification.`);
    } else {
        console.log("No pose detected or webcam not running.");
    }
});

document.addEventListener('DOMContentLoaded', (event) => {
    rebuildClassifierFromLocalStorage();
});

function rebuildClassifierFromLocalStorage() {
    const storedData = JSON.parse(localStorage.getItem('trainingData'));
    if (storedData) {
        storedData.forEach(({ pose, label }) => {
            knnClassifier.addExample(pose, label);
        });
        trainingsData = storedData;
        generatePoseCanvasses();
        updateSavedPosesUI();
    }
}

const generatePoseCanvasses = () => {
    const poseData = trainingsData;
    let div = '';
    poseData.map((pose, index) => {
        const canvas = document.createElement('canvas');
        const xyValues = [];
        for (let i = 0; i < pose.pose.length; i += 2) {
            xyValues.push({ x: pose.pose[i], y: pose.pose[i + 1] });
        }
        let canvasId = ''

        switch (pose.label) {
            case '👆':
                div = document.getElementById('resultUp');
                canvasId = `poseChartUp${index}`;
                canvas.className = 'poseCanvas';
                canvas.id = canvasId;
                canvas.style = { height: 200, width: 200 }; // Adjust size as needed
                if (document.getElementById(canvasId)) {
                    return;
                }
                div.appendChild(canvas);
                break;
            case '👇':
                div = document.getElementById('resultDown');
                canvasId = `poseChartDown${index}`;
                canvas.className = 'poseCanvas';
                canvas.id = canvasId;
                canvas.style = { height: 200, width: 200 }; // Adjust size as needed
                if (document.getElementById(canvasId)) {
                    return;
                }
                div.appendChild(canvas);
                break;
            case '👌':
                div = document.getElementById('resultOkay');
                canvasId = `poseChartOkay${index}`;
                canvas.className = 'poseCanvas';
                canvas.id = canvasId;
                canvas.style = { height: 200, width: 200 }; // Adjust size as needed
                if (document.getElementById(canvasId)) {
                    return;
                }
                div.appendChild(canvas);
                break;
        }
        if (canvasId) {
            const ctx = document.getElementById(canvasId).getContext('2d');

            new Chart(ctx, {
                type: "scatter",
                data: {
                    datasets: [{
                        pointRadius: 4,
                        pointBackgroundColor: "rgba(0,0,255,1)",
                        data: xyValues
                    }]
                },
                options: {
                    scales: {
                        x: { beginAtZero: true, title: { display: true, text: 'X' } },
                        y: { beginAtZero: true, title: { display: true, text: 'Y' } }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Pose Coordinates'
                        }
                    }
                }
            });
        }
    });
};

document.getElementById('saveModelButton').addEventListener('click', () => {
    const jsonStr = JSON.stringify(trainingsData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = "poseData.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
});

document.getElementById('fileInput').addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const poseData = JSON.parse(e.target.result);
                console.log('Pose data ingelezen:', poseData);

                // Reset de huidige trainingsdata en classifier
                trainingsData = [];
                knnClassifier.clearAllLabels();

                poseData.forEach(({ pose, label }) => {
                    console.log(`Voorbeeld toegevoegd: label = ${label}, pose = ${pose}`);
                    trainingsData.push({ pose, label });
                    knnClassifier.addExample(pose, label);
                });
                localStorage.setItem('trainingData', JSON.stringify(trainingsData));
                generatePoseCanvasses();

                // Reset het inputveld
                event.target.value = '';
            } catch (error) {
                console.error('Error parsing JSON:', error);
                alert('Fout bij het inlezen van het bestand. Controleer of het bestand correct is.');
            }
        };
        reader.readAsText(file);
    }
});

const calculateConfusionMatrix = (actualLabels, predictedLabels, uniqueLabels) => {
    const matrix = Array(uniqueLabels.length).fill(null).map(() => Array(uniqueLabels.length).fill(0));

    actualLabels.forEach((actual, i) => {
        const actualIndex = uniqueLabels.indexOf(actual);
        const predictedIndex = uniqueLabels.indexOf(predictedLabels[i]);
        matrix[actualIndex][predictedIndex] += 1;
    });

    return matrix;
};

const drawConfusionMatrix = (matrix, labels) => {
    const canvas = document.getElementById('confusionMatrixCanvas');
    const ctx = canvas.getContext('2d');
    const cellSize = canvas.width / labels.length;
    const maxVal = Math.max(...matrix.flat());

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '14px Arial';

    // Draw grid
    for (let i = 0; i <= labels.length; i++) {
        ctx.moveTo(i * cellSize, 0);
        ctx.lineTo(i * cellSize, canvas.height);
        ctx.moveTo(0, i * cellSize);
        ctx.lineTo(canvas.width, i * cellSize);
    }
    ctx.strokeStyle = '#000';
    ctx.stroke();

    // Labels aanmaken
    for (let i = 0; i < labels.length; i++) {
        ctx.fillText(labels[i], i * cellSize + cellSize / 2 - 10, canvas.height - 5);
        ctx.fillText(labels[i], 5, i * cellSize + cellSize / 2 + 5);
    }

    // Cellen inkleuren
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[i].length; j++) {
            const value = matrix[i][j];
            const intensity = value / maxVal;
            ctx.fillStyle = `rgba(0, 0, 255, ${intensity})`;
            ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            ctx.fillStyle = 'black';
            ctx.fillText(value, j * cellSize + cellSize / 2 - 10, i * cellSize + cellSize / 2 + 5);
        }
    }
};

const evaluateModel = async () => {
    const actualLabels = [];
    const predictedLabels = [];
    const uniqueLabels = ['👇', '👆', '👌'];

    for (const { pose, label } of trainingsData) {
        actualLabels.push(label);

        await new Promise((resolve) => {
            knnClassifier.classify(pose, (error, result) => {
                if (error) {
                    console.error('Error classifying pose:', error);
                    predictedLabels.push('error');
                } else {
                    predictedLabels.push(result.label);
                }
                resolve();
            });
        });
    }

    const confusionMatrix = calculateConfusionMatrix(actualLabels, predictedLabels, uniqueLabels);
    drawConfusionMatrix(confusionMatrix, uniqueLabels);
};

document.getElementById('evaluateModelButton').addEventListener('click', evaluateModel);

document.getElementById('deleteModelButton').addEventListener('click', () => {
    localStorage.setItem('trainingData', JSON.stringify([])); // Sla een lege array op in localStorage
    knnClassifier.clearAllLabels();

    // Leegmaken van de divs
    document.getElementById('resultDown').innerHTML = '<h2>Label 1: 👇</h2>';
    document.getElementById('resultUp').innerHTML = '<h2>Label 2: 👆</h2>';
    document.getElementById('resultOkay').innerHTML = '<h2>Label 3: 👌</h2>';
});
