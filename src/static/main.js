document.getElementById('fraud-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const form = e.target;
  const data = {};
  for (const el of form.elements) {
    if (el.name) {
      data[el.name] = el.type === 'number' ? Number(el.value) : el.value;
    }
  }
  document.getElementById('result').innerHTML = 'Predicting...';
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Prediction failed');
    }
    const result = await response.json();
    document.getElementById('result').innerHTML = `<div class="prob-box">Fraud Probability: <b>${(result.fraud_probability * 100).toFixed(2)}%</b></div>`;
  } catch (err) {
    document.getElementById('result').innerHTML = `<span class="error">${err.message}</span>`;
  }
}); 