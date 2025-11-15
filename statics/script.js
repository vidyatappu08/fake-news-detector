function checkNews() {
  const text = document.getElementById("newsInput").value.trim();
  const resultBox = document.getElementById("result");

  if (!text) {
    resultBox.innerHTML = "⚠️ Please enter some news text!";
    return;
  }

  // Simulated prediction (you can later connect Flask here)
  const random = Math.random();
  if (random > 0.5) {
    resultBox.innerHTML = "🟢 This news looks REAL!";
    resultBox.style.boxShadow = "0 0 25px #00ff99";
  } else {
    resultBox.innerHTML = "🔴 This news might be FAKE!";
    resultBox.style.boxShadow = "0 0 25px #ff0066";
  }
}
