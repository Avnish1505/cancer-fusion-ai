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

// Coarse clinical risk grouping for HAM10000 diagnosis codes, used only to
// color-code the presentation (does not affect the prediction itself).
const RISK_BY_CLASS = {
  mel: 'high',
  bcc: 'high',
  akiec: 'high',
  bkl: 'medium',
  nv: 'low',
  df: 'low',
  vasc: 'low',
}

const getRiskLevel = (className) => RISK_BY_CLASS[className?.toLowerCase()] || 'medium'

const LOCALIZATION_OPTIONS = [
  { value: 'back', label: 'Back' },
  { value: 'lower extremity', label: 'Lower Extremity' },
  { value: 'trunk', label: 'Trunk' },
  { value: 'upper extremity', label: 'Upper Extremity' },
  { value: 'abdomen', label: 'Abdomen' },
  { value: 'face', label: 'Face' },
  { value: 'chest', label: 'Chest' },
  { value: 'foot', label: 'Foot' },
  { value: 'neck', label: 'Neck' },
  { value: 'scalp', label: 'Scalp' },
  { value: 'hand', label: 'Hand' },
  { value: 'ear', label: 'Ear' },
  { value: 'genital', label: 'Genital' },
  { value: 'acral', label: 'Acral' },
  { value: 'unknown', label: 'Unknown' },
]

// Client-side only: keeps invalid/empty ages from being submitted.
// Does not change what gets sent to the API when valid.
const getAgeError = (value) => {
  if (value === '' || value === null || value === undefined) return 'Age is required'
  const num = Number(value)
  if (Number.isNaN(num)) return 'Enter a valid number'
  if (num < 0 || num > 120) return 'Enter an age between 0 and 120'
  return null
}

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

  const ageError = getAgeError(age)

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Pehle ek image select karo')
      return
    }
    if (ageError) {
      setError('Please fix the highlighted field before predicting')
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
            onKeyDown={(e) => {
              if (loading) return
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                fileInputRef.current?.click()
              }
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
              tabIndex={-1}
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
            <div className={`field${ageError ? ' field--error' : ''}`}>
              <label htmlFor="age">Age</label>
              <input
                id="age"
                type="number"
                inputMode="numeric"
                min="0"
                max="120"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                aria-invalid={Boolean(ageError)}
                aria-describedby={ageError ? 'age-error' : undefined}
              />
              {ageError ? (
                <span className="field-error" id="age-error" role="alert">
                  {ageError}
                </span>
              ) : (
                <span className="field-hint">Years, 0–120</span>
              )}
            </div>

            <div className="field">
              <label htmlFor="sex">Sex</label>
              <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>

            <div className="field">
              <label htmlFor="localization">Lesion Location</label>
              <select id="localization" value={localization} onChange={(e) => setLocalization(e.target.value)}>
                {LOCALIZATION_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
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

          <div className={`prediction-summary risk-${getRiskLevel(result.prediction)}`}>
            <span className="prediction-badge">Top Prediction</span>
            <h3 className="prediction-name">{result.prediction_full_name}</h3>
            <p className="prediction-code">{result.prediction.toUpperCase()}</p>
            <p className="confidence">
              Confidence <span className="confidence-value">{(result.confidence * 100).toFixed(2)}%</span>
            </p>
          </div>

          <div className="card-divider" />

          <div className="probabilities-section">
            <h4 className="result-heading">All Probabilities</h4>
            <ul className="probabilities-list">
              {sortedProbabilities.map(([name, prob], index) => {
                const isTop = index === 0
                const risk = getRiskLevel(name)
                return (
                  <li
                    key={name}
                    className={`probability-row risk-${risk}${isTop ? ' probability-row--top' : ''}`}
                    style={{ '--bar-width': `${(prob * 100).toFixed(2)}%`, '--row-index': index }}
                  >
                    <div className="probability-row-header">
                      <span className="probability-name">
                        {name.toUpperCase()}
                        {isTop && <span className="probability-top-badge">Top match</span>}
                      </span>
                      <span className="probability-value">{(prob * 100).toFixed(2)}%</span>
                    </div>
                    <div className="probability-bar-track">
                      <div className="probability-bar-fill" />
                    </div>
                  </li>
                )
              })}
            </ul>
          </div>
        </section>
      )}
    </div>
  )
}

export default App
