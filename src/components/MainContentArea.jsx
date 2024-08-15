import React, { useState } from 'react';
import axios from 'axios';

export const MainContentArea = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    try {
      const response = await axios.post('http://localhost:8000/analyze', { text: inputText });
      setResult(response.data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className='main-container'>
      <div className='input-area'>
        <p className='info-text'>Please type/paste the text that you would like to analyse</p>
        <br />
        <textarea 
          id="" 
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <br />
        <button className='analyse-btn' onClick={handleAnalyze}>Analyse</button>
      </div>
      <br />
      <div className='output-area'>
        <h2 className='Results-title'>Results</h2>
        {result && (
          <div>
            <p>Sentiment: {result.sentiment}</p>
            <p>Confidence: {result.confidence}%</p>
          </div>
        )}
      </div>
    </div>
  );
};
