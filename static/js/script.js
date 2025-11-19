document.getElementById('uploadForm').addEventListener('submit', function(event) {
    var fileInput = document.getElementById('fileInput');

    // Manual check: Is the file empty?
    if (fileInput.files.length === 0) {
        event.preventDefault(); // Stop submission
        alert("Please select an image first!"); // Show your own error
        return;
    }

    // If file exists, let the code continue...
});