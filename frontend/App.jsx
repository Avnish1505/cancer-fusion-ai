import { useState, useMemo } from 'react'
import './App.css'

// Env se URL lo. Agar env na ho to fallback use hoga.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'https://cancer-fusion-ai-production.up.railway.app'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const [age, setAge] = useState(50)
  const [sex, setSex] = useState('male')
  const [localization, setLocalization] = useState('back')

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    setSelectedFile(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Pehle ek image select karo')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('age', age)
      formData.append('sex', sex)
      formData.append('localization', localization)

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Kuch galat ho gaya, backend check karo')
    } finally {
      setLoading(false)
    }
  }

  const sortedProbabilities = useMemo(() => {
    if (!result?.all_probabilities) return []
    return Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)
  }, [result])

  return (
    <div style={{ maxWidth: '700px', margin: '0 auto', padding: '2rem', textAlign: 'center' }}>
      <h1>🔬 Cancer Fusion AI</h1>
      <p style={{ color: '#888' }}>Skin Lesion Classifier — HAM10000</p>

      <div style={{ margin: '2rem 0' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '2rem',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
        }}
      >
        {preview && (
          <div>
            <h4>Original Image</h4>
            <img
              src={preview}
              alt="preview"
              style={{ width: '250px', height: '250px', objectFit: 'cover', borderRadius: '12px' }}
            />
          </div>
        )}
        {result?.gradcam_overlay_base64 && (
          <div>
            <h4>Grad-CAM Overlay</h4>
            <img
              src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
              alt="Grad-CAM Overlay"
              style={{ width: '250px', height: '250px', objectFit: 'cover', borderRadius: '12px' }}
            />
          </div>
        )}
      </div>

      <div
        style={{
          margin: '2rem 0',
          display: 'flex',
          justifyContent: 'center',
          gap: '1rem',
          alignItems: 'center',
        }}
      >
        <div>
          <label>Age: </label>
          <input
            type="number"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            style={{ width: '60px' }}
          />
        </div>
        <div>
          <label>Sex: </label>
          <select value={sex} onChange={(e) => setSex(e.target.value)}>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="unknown">Unknown</option>
          </select>
        </div>
      </div>

      {preview && (
        <div style={{ marginTop: '1rem' }}>
          <button onClick={handlePredict} disabled={loading}>
            {loading ? 'Analyzing...' : 'Predict'}
          </button>
        </div>
      )}

      {error && <p style={{ color: 'red', marginTop: '1rem' }}>{error}</p>}

      {result && (
        <div
          style={{
            marginTop: '2rem',
            textAlign: 'left',
            background: '#1a1a1a',
            padding: '1.5rem',
            borderRadius: '12px',
          }}
        >
          <h3>
            Prediction: {result.prediction_full_name} ({result.prediction.toUpperCase()})
          </h3>
          <p style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#4CAF50' }}>
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
          <h4 style={{ marginTop: '1.5rem', marginBottom: '0.5rem' }}>All Probabilities:</h4>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {sortedProbabilities.map(([name, prob]) => (
              <li key={name} style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>{name.toUpperCase()}</span> <span>{(prob * 100).toFixed(2)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
