document.addEventListener("DOMContentLoaded", () => {

  if (window.cropSoilData) {
    new Chart(document.getElementById("cropChart"), {
      type: "radar",
      data: {
        labels: ["pH","Moisture","N","P","K","Temp","Humidity"],
        datasets: [{
          data: window.cropSoilData,
          backgroundColor: "rgba(34,197,94,0.3)",
          borderColor: "#22c55e"
        }]
      },
      options: { animation: { duration: 1500 } }
    });
  }

  if (window.irrigationValue) {
    new Chart(document.getElementById("irrigationChart"), {
      type: "bar",
      data: {
        labels: ["Water Needed"],
        datasets: [{
          data: [window.irrigationValue],
          backgroundColor: "#38bdf8"
        }]
      },
      options: { animation: { duration: 1200 } }
    });
  }

  if (window.yieldValue) {
    new Chart(document.getElementById("yieldChart"), {
      type: "line",
      data: {
        labels: ["Expected Yield"],
        datasets: [{
          data: [window.yieldValue],
          borderColor: "#facc15",
          fill: false
        }]
      },
      options: { animation: { duration: 1500 } }
    });
  }

});
