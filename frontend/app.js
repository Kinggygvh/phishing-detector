// Wait for the DOM to be fully loaded before adding event listeners
document.addEventListener("DOMContentLoaded", function () {
    // Get the form and result elements
    const urlInput = document.getElementById("url-input");
    const resultMessage = document.getElementById("result-message");
    const spinner = document.getElementById("spinner");
    const submitButton = document.getElementById("submit-button");

    // Add event listener to the submit button
    submitButton.addEventListener("click", function (event) {
        event.preventDefault(); // Prevent the form from submitting in the traditional way

        const url = urlInput.value.trim(); // Get the value from the input

        // Validate the URL format
        if (!isValidUrl(url)) {
            showMessage("Please enter a valid URL.", "error");
            return;
        }

        // Show the loading spinner
        showSpinner(true);

        // Simulate URL analysis (replace this with actual backend API call)
        analyzeUrl(url).then(result => {
            showSpinner(false); // Hide the spinner
            showMessage(result.message, result.type); // Show the result message
        }).catch(err => {
            showSpinner(false); // Hide the spinner on error
            showMessage("Something went wrong. Please try again.", "error");
        });
    });

    // Function to validate if the URL format is correct
    function isValidUrl(url) {
        const regex = /^(http|https):\/\/[^ "]+$/; // Basic regex to check for valid URLs
        return regex.test(url);
    }

    // Function to simulate URL analysis (can be replaced with actual API call)
    function analyzeUrl(url) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // Simulate a random outcome (for testing purposes)
                const isPhishing = Math.random() < 0.5; // 50% chance of phishing

                if (isPhishing) {
                    resolve({
                        message: "Warning: This URL is potentially a phishing site.",
                        type: "warning"
                    });
                } else {
                    resolve({
                        message: "Safe: This URL appears to be legitimate.",
                        type: "success"
                    });
                }
            }, 3000); // Simulating a 3-second delay for processing
        });
    }

    // Function to show the result message and style it accordingly
    function showMessage(message, type) {
        resultMessage.textContent = message;
        if (type === "warning") {
            resultMessage.style.color = "red";
        } else if (type === "success") {
            resultMessage.style.color = "green";
        } else {
            resultMessage.style.color = "black";
        }
    }

    // Function to show or hide the loading spinner
    function showSpinner(show) {
        if (show) {
            spinner.style.display = "block";
        } else {
            spinner.style.display = "none";
        }
    }
});
