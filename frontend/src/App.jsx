import React, { useState } from "react";
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const apiUrl = import.meta.env.VITE_BACKEND_API || "http://192.168.28.90:8000/";
  console.log("FETCH API HERE: ", apiUrl);
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setFile(file);
    setPrediction("");
    if (file) {
      setPreview(URL.createObjectURL(file));
    } else {
      setPreview(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setPrediction("");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setPrediction(data.predicted_class || "No prediction");
    } catch (err) {
      setPrediction("Error: " + err.message);
    }
    setLoading(false);
  };

  return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">      <div className="bg-white p-8 rounded shadow-md w-full max-w-md">
        <h1 className="text-2xl font-bold mb-4 text-center">Mushroom Classifier</h1>
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:bg-blue-50 file:text-blue-700 file:hover:bg-blue-200"
          />
          {preview && (
          <div className="mt-4 flex justify-center">
            <img
              src={preview}
              alt="Preview"
              className="max-h-48 rounded shadow"
            />
          </div>
          )}
          <button
            type="submit"
            className={`py-2 rounded transition duration-200
              ${(!file || loading)
                ? "bg-gray-400 text-white cursor-not-allowed"
                : "bg-blue-600 text-white hover:bg-blue-700 hover:scale-105 active:bg-blue-800 cursor-pointer"}
            `}
            disabled={loading || !file}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>
        {prediction && (
          <div className="mt-6 text-center">
            <span className="font-semibold">Prediction:</span>
            <span className="ml-2 text-red-700">{prediction}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;