# Hand Pose Detection Project

Dit project detecteert handposes met behulp van een webcam en maakt gebruik van het Mediapipe Hand Landmarker-model en ML5.js KNN Classifier. De applicatie herkent verschillende handposes zoals 👆, 👇, en 👌, en toont de voorspelling samen met een confidence score.

## Inhoudsopgave

- [Installatie](#installatie)
- [Gebruik](#gebruik)
- [Projectstructuur](#projectstructuur)
- [Belangrijke functies](#belangrijke-functies)
- [Toekomstige verbeteringen](#toekomstige-verbeteringen)
- [Auteurs](#auteurs)
- [Licentie](#licentie)

## Installatie

1. Clone de repository naar je lokale machine:
    ```bash
    git clone <repository-url>
    ```

2. Open het project in je favoriete code-editor.
    ```

## Gebruik

1. Open `index.html` in je webbrowser via je lokale webserver.

2. Klik op de knop **ENABLE PREDICTIONS** om de webcam te activeren.

3. Gebruik de knoppen **👇**, **👆**, en **👌** om verschillende handposes vast te leggen en toe te voegen aan de trainingsdata.

4. Klik op **SAVE TRAINING** om de trainingsdata op te slaan in een JSON-bestand.

5. Gebruik de knop **EVALUATE MODEL** om de prestaties van het model te evalueren en een confusion matrix te genereren.

6. Gebruik de knop **DELETE TRAINING** om de trainingsdata te verwijderen.

## Projectstructuur

- **index.html**: De hoofd-HTML-pagina voor de applicatie.
- **styles.css**: Bevat de styling voor de applicatie.
- **script.js**: De hoofd-JavaScript-bestand met alle functionaliteit.
- **README.md**: Dit bestand.
- **poseData.json**: Voorbeeld van een opgeslagen trainingsdata bestand.

## Belangrijke functies

### createHandLandmarker

```javascript
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
Initialiseert het Hand Landmarker-model met behulp van Mediapipe.

enableCam
javascript
Code kopiëren
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

    const constraints = {
        video: true
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

function classifyPose(pose) {
    neuralNetwork.classify(pose, (error, results) => {
        if (error) {
            console.error(error);
            return;
        }
        const result = results[0];
        const label = result.label;
        const confidence = result.confidencesByLabel[label] * 100;
        displayPose(label, confidence);
    });
}

function displayPose(poseLabel, confidence) {
    document.getElementById('currentPoseDisplay').textContent = `Current Pose: ${poseLabel} (${confidence.toFixed(2)}%)`;
}
```
Toekomstige verbeteringen
Integratie van meer handposes voor uitgebreidere detectie.
Verbeterde gebruikersinterface voor een gebruiksvriendelijkere ervaring.
Optimalisatie van het model voor snellere en nauwkeurigere voorspellingen.


Auteur
Chris Moerman


