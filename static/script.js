document.getElementById("fileInput").addEventListener("change", function(event) {
    const file = event.target.files[0];
    const previewContainer = document.getElementById("preview");
    
    previewContainer.innerHTML = '';

    if (file) {
        const fileType = file.type;
        const message = document.createElement("p");
        if (fileType.startsWith('image/')) {
            message.textContent = `Image "${file.name}" has been uploaded.`;
        } else if (fileType.startsWith('video/')) {
            message.textContent = `Video "${file.name}" has been uploaded.`;
        } else {
            message.textContent = `File "${file.name}" has been uploaded.`;
        }
        previewContainer.appendChild(message);
    } else {
        previewContainer.textContent = 'No file selected.';
    }
});
