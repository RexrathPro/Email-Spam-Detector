import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Home from './pages/Home';
import Scanner from './pages/Scanner';

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <header>
          <Link to="/" className="logo">
            <div className="logo-icon">S</div>
            Spam Detector
          </Link>
          <nav>
            <Link to="/">Home</Link>
            <Link to="/scan">Scanner</Link>
          </nav>
        </header>

        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/scan" element={<Scanner />} />
          </Routes>
        </main>

        <footer>
          Developed by Pratham Sharma | 
          <a href="https://github.com/rexrathpro" target="_blank" rel="noopener noreferrer"> GitHub</a> | 
          <a href="https://linkedin.com/in/theprathamsharma" target="_blank" rel="noopener noreferrer"> LinkedIn</a>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
