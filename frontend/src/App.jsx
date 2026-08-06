import { useState, useMemo, useRef, useEffect } from 'react'
import './App.css'

const API_BASE_URL = "https://cancer-fusion-ai-production.up.railway.app"

const LOADING_STAGES = [
  'Preprocessing image',
  'Running model inference',
  'Generating Grad-CAM heatmap',
  'Finalizing report',
]
const STAGE_INTERVAL_MS = 1500

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [age, setAge] = useState(50)
  const [sex, setSex] = useState('male')
  const [localization, setLocalization] = useState('back')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)
  const stageIntervalRef = useRef(null)

  useEffect(() => {
    return () => {
      if (stageIntervalRef.current) clearInterval(stageIntervalRef.current)
    }
  }, [])

  const processFile = (file) => {
    if (!file) return
    setSelectedFile(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleFileChange = (e) => {
    processFile(e.target.files[0])
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    if (loading) return
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    if (loading) return
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (loading) return
    setIsDragging(false)
    processFile(e.dataTransfer.files?.[0])
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Pehle ek image select karo')
      return
    }
    setLoading(true)
    setError(null)
    setLoadingStage(0)
    stageIntervalRef.current = setInterval(() => {
      setLoadingStage((prev) => (prev < LOADING_STAGES.length - 1 ? prev + 1 : prev))
    }, STAGE_INTERVAL_MS)

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

      if (!response.ok) throw new Error(`Server error: ${response.status}`)
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Kuch galat ho gaya, backend check karo')
    } finally {
      clearInterval(stageIntervalRef.current)
      stageIntervalRef.current = null
      setLoading(false)
      setLoadingStage(0)
    }
  }

  const sortedProbabilities = useMemo(() => {
    if (!result?.all_probabilities) return []
    return Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)
  }, [result])

  return (
    <div className="page">
      <header className="hero">
        <h1 className="hero-title">🔬 Cancer Fusion AI</h1>
        <p className="hero-subtitle">Skin Lesion Classifier — HAM10000 Dataset</p>
      </header>

      <main className="card upload-card">
        <fieldset className="upload-fields" disabled={loading}>
          <div
            className={`dropzone${isDragging ? ' dropzone--active' : ''}${preview ? ' dropzone--has-preview' : ''}${loading ? ' dropzone--disabled' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => {
              if (loading) return
              fileInputRef.current?.click()
            }}
            role="button"
            tabIndex={loading ? -1 : 0}
            aria-disabled={loading}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="dropzone-input"
              aria-label="Upload lesion image"
            />
            {preview ? (
              <div className="preview-wrap">
                <div className={`preview-scan${loading ? ' preview-scan--active' : ''}`}>
                  <img src={preview} alt="Selected lesion" className="preview-image" />
                  {loading && <span className="scan-line" aria-hidden="true" />}
                </div>
                <span className="preview-hint">
                  {loading ? 'Analyzing image…' : 'Click or drop to replace image'}
                </span>
              </div>
            ) : (
              <div className="dropzone-empty">
                <div className="dropzone-icon" aria-hidden="true">📤</div>
                <p className="dropzone-title">Drag &amp; drop a lesion image</p>
                <p className="dropzone-subtitle">or click to browse — JPG, PNG</p>
              </div>
            )}
          </div>

          <div className="patient-fields">
            <div className="field">
              <label htmlFor="age">Age</label>
              <input id="age" type="number" value={age} onChange={(e) => setAge(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="sex">Sex</label>
              <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>
          </div>

          {preview && (
            <button className="predict-button" onClick={handlePredict} disabled={loading}>
              {loading ? 'Analyzing…' : 'Predict'}
            </button>
          )}
        </fieldset>

        {error && <p className="error-message">{error}</p>}
      </main>

      {loading && (
        <section className="card loading-card" role="status" aria-live="polite" aria-busy="true">
          <div className="loading-scan" aria-hidden="true">
            <span className="loading-scan-line" />
          </div>
          <ul className="stage-list">
            {LOADING_STAGES.map((stage, index) => {
              const isComplete = index < loadingStage
              const isActive = index === loadingStage
              const isFinalActive = isActive && index === LOADING_STAGES.length - 1
              return (
                <li
                  key={stage}
                  className={`stage-item${isComplete ? ' stage-item--complete' : ''}${isActive ? ' stage-item--active' : ''}${isFinalActive ? ' stage-item--looping' : ''}`}
                >
                  <span className="stage-dot" aria-hidden="true" />
                  <span className="stage-label">{stage}</span>
                </li>
              )
            })}
          </ul>
          <p className="sr-only">
            Step {loadingStage + 1} of {LOADING_STAGES.length}: {LOADING_STAGES[loadingStage]}
          </p>
        </section>
      )}

      {result && (
        <section className="card result-card">
          <div className="result-images">
            {preview && (
              <div className="result-image-block">
                <h4 className="result-heading">Original Image</h4>
                <img src={preview} alt="Original lesion" className="result-image" />
              </div>
            )}
            {result?.gradcam_overlay_base64 && (
              <div className="result-image-block">
                <h4 className="result-heading">Grad-CAM Overlay</h4>
                <img
                  src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                  alt="Grad-CAM Overlay"
                  className="result-image"
                />
              </div>
            )}
          </div>

          <h3 className="result-heading">
            Prediction: {result.prediction_full_name} ({result.prediction.toUpperCase()})
          </h3>
          <p className="confidence">Confidence: {(result.confidence * 100).toFixed(2)}%</p>

          <h4 className="result-heading">All Probabilities:</h4>
          <ul className="probabilities-list">
            {sortedProbabilities.map(([name, prob]) => (
              <li key={name}>
                <span>{name.toUpperCase()}</span>
                <span>{(prob * 100).toFixed(2)}%</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}

export default App
