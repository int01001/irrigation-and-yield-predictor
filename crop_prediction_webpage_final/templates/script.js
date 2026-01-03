document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictForm");
    const button = document.getElementById("predictBtn");

    if (form && button) {
        form.addEventListener("submit", () => {
            button.innerText = "Analyzing...";
            button.disabled = true;
        });
    }
});
