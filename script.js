const video = document.getElementById('video');
const messageBox = document.getElementById('messageBox');

const knownPersons = {
  'ab': 'Hello Amitabh Bachhan',
  'akd': 'Hello Ankan Kumar Daw ',
  'arpita': 'Chagol chana spotted ! , pak u !',
  'arun': 'Welcome back Mr Arun Kumar Daw ',
  'kaushik': 'sakh mah dik bebs',
  'kheyalee': 'Didimoniii je ',
  'madhu': 'Great things takes time to grow !, iykyk......cho ebar modhu khawa tor',
  'rina': 'Hey Rina!  ',
  'suor': 'Suorgiri kora bondho kor ebar ipsu',
  'unknown': ' Unidentified'
};

// Load face-api models
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('models/tiny_face_detector'),
  faceapi.nets.faceRecognitionNet.loadFromUri('models/face_recognition'),
  faceapi.nets.faceLandmark68Net.loadFromUri('models/face_landmark_68')
])
  .then(startVideo)
  .catch(err => {
    console.error("Model loading error:", err);
    messageBox.textContent = 'Error loading models';
  });

async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 1280, height: 720 }
    });

    video.srcObject = stream;

    video.onloadedmetadata = () => {
      video.width = video.videoWidth;
      video.height = video.videoHeight;
      runFaceDetection();
    };
  } catch (err) {
    console.error("Camera error:", err);
    messageBox.textContent = 'Error accessing camera';
  }
}

async function runFaceDetection() {
  const labeledDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.4);

  const canvas = faceapi.createCanvasFromMedia(video);
  const container = document.getElementById('videoContainer');
  canvas.setAttribute('id', 'overlay');
  container.appendChild(canvas);

  canvas.width = video.width;
  canvas.height = video.height;

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    let messages = [];

    resizedDetections.forEach(detection => {
      const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
      const name = bestMatch.label.toLowerCase();
      const message = knownPersons[name] || knownPersons['unknown'];
      messages.push(message);

      const box = detection.detection.box;
      const x = Math.round(video.width - box.x - box.width); // <-- precise mirroring
      
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, box.y, box.width, box.height);
      
      ctx.fillStyle = '#00FF00';
      ctx.font = '16px Arial';
      ctx.fillText(message.length > 25 ? message.slice(0, 25) + '...' : message, x, box.y - 10);
      
    });

    messageBox.textContent = messages.length > 0 ? messages.join(' | ') : 'No face detected';
  }, 1000);
}

function loadLabeledImages() {
  const labels = Object.keys(knownPersons).filter(name => name !== 'unknown');

  return Promise.all(labels.map(async label => {
    const descriptors = [];

    for (let i = 1; i <= 3; i++) {
      const imgUrl = `known_faces/${label}/${i}.jpg`;

      try {
        const img = await faceapi.fetchImage(imgUrl);
        const detection = await faceapi
          .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detection) {
          descriptors.push(detection.descriptor);
        } else {
          console.warn(`No face detected in ${label}/${i}.jpg`);
        }
      } catch (error) {
        console.warn(`Could not load image: ${label}/${i}.jpg`, error);
      }
    }

    return descriptors.length
      ? new faceapi.LabeledFaceDescriptors(label.toLowerCase(), descriptors)
      : null;
  })).then(descriptors => descriptors.filter(Boolean));
}
