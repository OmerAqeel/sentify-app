import React, { useState } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

// Register the required components
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

export const MainContentArea = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [file, setFile] = useState(null);

  const handleAnalyze = async () => {
    if (file) {
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
    } else if (inputText) {
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

  const renderChart = () => {
    if (!result || !result.product_summary || !result.product_summary.POSITIVE || !result.product_summary.NEGATIVE) {
      return null;
    }
  
    const products = Object.keys(result.product_summary.POSITIVE || {}); // Safely get the product keys
    const data = {
      labels: products,
      datasets: [
        {
          label: 'Positive Feedbacks',
          data: products.map(product => result.product_summary.POSITIVE[product] || 0),
          backgroundColor: 'rgba(75, 192, 192, 0.6)',
        },
        {
          label: 'Negative Feedbacks',
          data: products.map(product => result.product_summary.NEGATIVE[product] || 0),
          backgroundColor: 'rgba(255, 99, 132, 0.6)',
        }
      ]
    };
  
    const options = {
      scales: {
        y: {
          beginAtZero: true,
          min: 0,
          max: Math.max(
            ...data.datasets[0].data.concat(data.datasets[1].data)
          ) + 1,
        },
      },
    };
  
    return <Bar data={data} options={options} height={100} width={400}/>;
  };
  
  return (
    <div className='main-container'>
      <div className='input-area'>
        <p className='info-text'>Please type/paste the text that you would like to analyse</p>
        <br />
        <textarea
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
          <input type='file' onChange={handleFileChange} />
        </div>
        <br />
        <button className='analyse-btn' onClick={handleAnalyze}>Analyse</button>
      </div>
      <br />
      <div className='output-area'>
        <h2 className='Results-title'>Results</h2>
        {result && (
          <div>
            {renderChart()}
          </div>
        )}
      </div>
    </div>
  );
};
