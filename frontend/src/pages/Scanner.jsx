import React, { useState } from 'react';

function Scanner() {
  const [emailText, setEmailText] = useState('');
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState(null);

  const handleScan = async () => {
    if (!emailText.trim()) return;
    
    setIsScanning(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to analyze email: ${response.statusText}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      alert('Error connecting to the backend. Is the model API running?');
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className="page-wrapper">
      <div className="card">
        <div className="form-group">
          <label htmlFor="email-input">Paste Email Content</label>
          <textarea
            id="email-input"
            className="textarea-input"
            placeholder="Dear User, You have won $1,000,000..."
            value={emailText}
            onChange={(e) => setEmailText(e.target.value)}
          />
        </div>
        
        <button 
          className={`btn btn-primary ${isScanning || !emailText.trim() ? 'btn-disabled' : ''}`}
          onClick={handleScan}
          disabled={isScanning || !emailText.trim()}
          style={{ width: '100%' }}
        >
          {isScanning ? 'Scanning...' : 'Analyze Email'}
        </button>

        {result && (
          <div className={`result-banner ${result.type}`}>
            <h2>{result.type === 'spam' ? 'SPAM DETECTED' : 'SAFE (HAM)'}</h2>
            <p>Model Confidence: {result.confidence}%</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default Scanner;
