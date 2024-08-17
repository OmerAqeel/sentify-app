import React, { useState } from 'react';
import axios from 'axios';

export const MainContentArea = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);

  const handleAnalyze = async () => {

    if(file){
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await axios.post('http://localhost:8000/analyze_csv', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        setResult(response.data);
        console.log('response:', response.data);
      } catch (error) {
        console.error('Error:', error);
      }
    }else if (inputText) {
      try {
        const response = await axios.post('http://localhost:8000/analyze', { text: inputText });
        setResult(response.data);
        console.log('response:', response.data);
      } catch (error) {
        console.error('Error:', error);
      }
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
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
        <div className='or-line'>
        <hr className='line-1'/>
        OR
        <hr className='line-2'/>
        </div>
        <div className='file-upload'>
          <input type='file'  onChange={handleFileChange}/>
        </div>
        <br />
        <button className='analyse-btn' onClick={handleAnalyze}>Analyse</button>
      </div>
      <br />
      <div className='output-area'>
        <h2 className='Results-title'>Results</h2>
        {result && (
          <div>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};
