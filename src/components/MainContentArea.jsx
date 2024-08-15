import React from 'react'

export const MainContentArea = () => {
  return (
    <div className='main-container'>
        <div className='input-area'>
        <p className='info-text'>Please type/paste the text that you would like to analyse</p>
        <br />
        <textarea  id=""></textarea>
        <br />
        <button className='analyse-btn'>Analyse</button>
        </div>
        <br />
        <div className='output-area'>
            <h2 className='Results-title'>Results</h2>
        </div>
    </div>
  )
}
