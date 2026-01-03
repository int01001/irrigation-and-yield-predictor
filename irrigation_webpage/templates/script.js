document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("irrigationForm");
    const button = document.getElementById("predictBtn");

    if (form && button) {
        form.addEventListener("submit", () => {
            button.innerText = "Calculating...";
            button.disabled = true;
        });
    }
});
