let model;

async function loadModel() {
  // Load the TensorFlow.js model
  model = await tf.loadLayersModel('model.json');
}

async function preprocessImage(image) {
  // Create a tensor from the image
  const tfImage = tf.browser.fromPixels(image).toFloat();
  
  // Resize the image to 28x28 using bilinear interpolation
  const resizedImage = tf.image.resizeBilinear(tfImage, [28, 28]);

  // Convert the image to grayscale
  const grayImage = resizedImage.mean(2);

  // Normalize the image pixels to the range [0, 1]
  const normalizedImage = grayImage.div(255.0);

  // Reshape the image to match the model's expected shape (28, 28, 1)
  const reshapedImage = normalizedImage.reshape([1, 28, 28, 1]);

  return reshapedImage;
}

async function classifyImage() {
  const imageInput = document.getElementById('imageInput');
  const inputImage = imageInput.files[0];

  if (!inputImage) {
    alert('Please select an image.');
    return;
  }

  const reader = new FileReader();
  reader.onload = async function(event) {
    const image = new Image();
    image.src = event.target.result;
    await image.decode();

    // Preprocess the image
    const preprocessedImage = await preprocessImage(image);

    // Perform prediction
    const predictions = model.predict(preprocessedImage);
    // Process predictions as needed

    // Display the prediction result
    const predictionResultElement = document.getElementById('predictionResult');
    predictionResultElement.innerText = `Prediction: ${predictions}`;
  };

  reader.readAsDataURL(inputImage);
}

async function setup() {
  // Load the pre-trained model
  await loadModel();
}

setup();






// let model;

// async function loadModel() {
//   // Load the TensorFlow.js model
//   model = await tf.loadLayersModel('C:\Data_science_projects\Detection-and-Classification-of-Kidney-Diseases-Using-CT-Scanned-Image\model.json');
// }

// async function preprocessImage(image) {
//   // Preprocess the image: resize and normalize
//   const processedImage = tf.browser.fromPixels(image)
//     .resizeBilinear([224, 224])
//     .toFloat()
//     .div(tf.scalar(255))
//     .expandDims();

//   return processedImage;
// }

// async function classifyImage() {
//   const imageInput = document.getElementById('imageInput');
//   const inputImage = imageInput.files[0];

//   if (!inputImage) {
//     alert('Please select an image.');
//     return;
//   }

//   const reader = new FileReader();
//   reader.onload = async function(event) {
//     const image = new Image();
//     image.src = event.target.result;

//     // Ensure the image is loaded before processing
//     image.onload = async function() {
//       const processedImage = await preprocessImage(image);

//       // Perform prediction
//       const predictions = await model.predict(processedImage);
//       // Process predictions as needed

//       // Display the prediction result
//       const predictionResultElement = document.getElementById('predictionResult');
//       predictionResultElement.innerText = `Prediction: ${predictions}`;
//     };
//   };

//   reader.readAsDataURL(inputImage);
// }

// async function setup() {
//   // Load the pre-trained model
//   await loadModel();
// }

// setup();
