document.getElementById("imageInput").addEventListener("change", function(event) {
    let file = event.target.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        // Display Original Image
        let originalImg = document.getElementById("uploadedImage");
        originalImg.src = data.original_url;
        originalImg.style.display = "block";

        // Display Preprocessed Image
        let preprocessedImg = document.getElementById("preprocessedImage");
        preprocessedImg.src = data.preprocessed_url;
        preprocessedImg.style.display = "block";

        // Show Predict button after images are displayed
        document.getElementById("predictButton").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
});

function predictImage() {
    fetch("/predict", {
        method: "POST"
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        document.getElementById("prediction-output").innerHTML = `
            <h3>Predictions:</h3>
            <p><strong>CNN:</strong> ${data.cnn_prediction.stage} (Confidence: ${data.cnn_prediction.confidence.toFixed(2)})</p>
            <p><strong>GoogleNet-SVM:</strong> ${data.googlenet_svm_prediction.stage} (Confidence: ${data.googlenet_svm_prediction.confidence.toFixed(2)})</p>
            <p><strong>ResNet-SVM:</strong> ${data.resnet_svm_prediction.stage} (Confidence: ${data.resnet_svm_prediction.confidence.toFixed(2)})</p>
        `;

        document.getElementById("result-section").style.display = "block";

        document.getElementById("gradCamButton").style.display = "block";
    })
    .catch(error => console.error("Error:", error));
}
function generateGradCAM() {
    fetch("/generate_grad_cam", { method: "POST" })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
            return;
        }
        console.log("Grad-CAM data received:", data);

        // Ensure all Grad-CAM images are updated
        let timestamp = new Date().getTime(); // Prevent caching issues

        // ✅ Custom CNN Grad-CAM
        let cnnMessageElement = document.getElementById("grad-cam-cnn-message");
        let cnnImageElement = document.getElementById("grad-cam-cnn");

        if (!data.grad_cam_cnn || data.grad_cam_cnn.startsWith("Error:")) {
            cnnImageElement.style.display = "none"; // Hide image
            cnnMessageElement.innerText = data.grad_cam_cnn || "Error: Grad-CAM not available for CNN.";
            cnnMessageElement.style.display = "block"; // Show error message
        } else {
            cnnImageElement.src = data.grad_cam_cnn + "?t=" + timestamp; // Prevent caching
            cnnImageElement.style.display = "block"; // Show image
            cnnMessageElement.style.display = "none"; // Hide error message
        }

        // ✅ GoogleNet Grad-CAM
        let googlenetImg = document.getElementById("grad-cam-googlenet");
        if (data.grad_cam_googlenet) {
            googlenetImg.src = data.grad_cam_googlenet + "?t=" + timestamp; // Prevent caching
            googlenetImg.style.display = "block";
        } else {
            googlenetImg.style.display = "none";
        }

        // ✅ ResNet Grad-CAM
        let resnetImg = document.getElementById("grad-cam-resnet");
        if (data.grad_cam_resnet) {
            resnetImg.src = data.grad_cam_resnet + "?t=" + timestamp; // Prevent caching
            resnetImg.style.display = "block";
        } else {
            resnetImg.style.display = "none";
        }

        // ✅ Ensure Grad-CAM Section is visible
        document.getElementById("grad-cam-section").style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        alert("A network error occurred while generating Grad-CAM.");
    });
}

