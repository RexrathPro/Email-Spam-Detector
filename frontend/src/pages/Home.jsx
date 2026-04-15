import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <div className="page-wrapper hero">
      <h1>Detect Spam with High Accuracy</h1>
      <p>
        A sophisticated machine learning model to classify your emails as Spam or Ham.
        Paste your email content and get instant, reliable results.
      </p>
      <Link to="/scan" className="btn btn-primary">
        Start Scanning Now
      </Link>
    </div>
  );
}

export default Home;
