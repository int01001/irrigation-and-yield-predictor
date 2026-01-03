document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("form");
    const button = document.getElementById("btn");

    // Smooth button loading animation
    if (form && button) {
        form.addEventListener("submit", () => {
            button.innerText = "Predicting...";
            button.style.opacity = "0.8";
            button.disabled = true;
        });
    }
});
