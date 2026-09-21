import { useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://cancer-fusion-ai-production.up.railway.app'

// Mirrors src/dataset.py's DX_FULL_NAMES — kept in sync manually, same as
// LOCALIZATION_OPTIONS below.
const DX_FULL_NAMES = {
  akiec: "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
  bcc: 'Basal cell carcinoma',
  bkl: 'Benign keratosis-like lesions',
  df: 'Dermatofibroma',
  mel: 'Melanoma',
  nv: 'Melanocytic nevi',
  vasc: 'Vascular lesions',
}

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

// Display labels for the response's timings_ms keys — a factual, post-hoc
// breakdown, not a progress simulation. Order here is the render order.
const TIMING_LABELS = [
  ['image_decode', 'Image decode'],
  ['preprocess', 'Preprocess'],
  ['forward_and_backward', 'Forward + backward pass'],
  ['heatmap_render', 'Heatmap render'],
  ['serialize', 'Serialize response'],
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

function SectionHeader({ number, label, helper }) {
  return (
    <div className="section-header">
      <span className="section-heading">
        <span className="section-number">{number}</span>
        {label}
      </span>
      <span className="section-helper">{helper}</span>
    </div>
  )
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [age, setAge] = useState(50)
  const [sex, setSex] = useState('male')
  const [localization, setLocalization] = useState('back')
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

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
      setLoading(false)
    }
  }

  const sortedProbabilities = useMemo(() => {
    if (!result?.all_probabilities) return []
    return Object.entries(result.all_probabilities).sort(([, a], [, b]) => b - a)
  }, [result])

  const isRejected = result?.status === 'rejected'
  const isAccepted = result?.status === 'ok'

  const statusText = loading ? 'Running' : error ? 'Error' : isRejected ? 'Rejected' : result ? 'Complete' : 'Idle'

  return (
    <div className="page">
      <div className="sheet">
        <header className="sheet-header">
          <div className="sheet-header-left">
            <span className="product-name">Cancer Fusion AI</span>
            <span className="product-descriptor">Skin lesion classifier — HAM10000</span>
          </div>
          <span className={`sheet-status sheet-status--${statusText.toLowerCase()}`}>{statusText}</span>
        </header>
        <hr className="header-rule" />

        {/* ---- Section 01 — Input ---- */}
        <section className="report-section">
          <SectionHeader
            number="01"
            label="INPUT"
            helper="Upload a dermoscopic image and patient details, then run the model."
          />
          <div className="section-box input-box">
            <div className="input-col input-col--dropzone">
              <div
                className={`dropzone${isDragging ? ' dropzone--active' : ''}${loading ? ' dropzone--disabled' : ''}`}
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
                {selectedFile ? (
                  <div className="dropzone-filled">
                    <span className="dropzone-filename">{selectedFile.name}</span>
                    <span className="dropzone-hint">click or drop to replace</span>
                  </div>
                ) : (
                  <div className="dropzone-empty">
                    <p className="dropzone-title">Drop a lesion image here</p>
                    <p className="dropzone-subtitle">or click to browse — JPG, PNG</p>
                  </div>
                )}
              </div>
              <p className="input-col-helper">Used only for this prediction. Not stored.</p>
            </div>

            <div className="input-col input-col--fields">
              <div className="field">
                <label htmlFor="age">AGE</label>
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
                  <span className="field-hint">years, 0–120</span>
                )}
              </div>

              <div className="field">
                <label htmlFor="sex">SEX</label>
                <select id="sex" value={sex} onChange={(e) => setSex(e.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>
            </div>

            <div className="input-col input-col--fields">
              <div className="field">
                <label htmlFor="localization">LESION LOCATION</label>
                <select
                  id="localization"
                  value={localization}
                  onChange={(e) => setLocalization(e.target.value)}
                >
                  {LOCALIZATION_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="input-col input-col--predict">
              <button className="predict-button" onClick={handlePredict} disabled={loading || !selectedFile}>
                {loading ? 'Analyzing…' : 'Predict'}
              </button>
            </div>
          </div>
          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}
          <p className="upload-note">
            This model was trained only on dermatoscope images, mostly of light skin. An ordinary
            phone photo can pass the image check and still produce a meaningless result.
          </p>
        </section>

        {/* ---- Section 02 — Evidence ---- */}
        <section className="report-section">
          <SectionHeader
            number="02"
            label="EVIDENCE"
            helper="What the model saw, and where it looked."
          />
          <div className="section-box evidence-box">
            {result ? (
              <>
                <div className="evidence-images">
                  <div className="evidence-image-block">
                    <span className="evidence-image-label">Original</span>
                    {preview && <img src={preview} alt="Original lesion" className="evidence-image" />}
                  </div>
                  {!isRejected && (
                    <div className="evidence-image-block">
                      <span className="evidence-image-label">Grad-CAM heatmap</span>
                      {result.gradcam_overlay_base64 && (
                        <img
                          src={`data:image/png;base64,${result.gradcam_overlay_base64}`}
                          alt="Grad-CAM heatmap overlay"
                          className="evidence-image"
                        />
                      )}
                    </div>
                  )}
                </div>
                {isRejected ? (
                  <p className="evidence-explanation">
                    No heatmap — the image was rejected before the classifier ran. See the Result section.
                  </p>
                ) : (
                  <p className="evidence-explanation">
                    The heatmap marks the regions of the image that most influenced the model's prediction —
                    warmer areas contributed more.
                  </p>
                )}
                {result.timings_ms && (
                  <div className="timings">
                    <span className="timings-label">LATENCY (ms)</span>
                    <ul className="timings-list">
                      {TIMING_LABELS.filter(([key]) => key in result.timings_ms).map(([key, label]) => (
                        <li key={key} className="timings-row">
                          <span className="timings-name">{label}</span>
                          <span className="timings-value">{result.timings_ms[key].toFixed(1)}</span>
                        </li>
                      ))}
                      <li className="timings-row timings-row--total">
                        <span className="timings-name">Total</span>
                        <span className="timings-value">
                          {Object.values(result.timings_ms)
                            .reduce((sum, v) => sum + v, 0)
                            .toFixed(1)}
                        </span>
                      </li>
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <p className="empty-state">
                The original image, Grad-CAM heatmap, and per-stage latency will appear here after a run.
              </p>
            )}
          </div>
        </section>

        {/* ---- Section 03 — Result ---- */}
        <section className="report-section">
          <SectionHeader
            number="03"
            label="RESULT"
            helper="What the model produced — not medical advice."
          />
          <div className={`section-box result-box${result ? ' result-box--filled' : ''}`}>
            {isRejected ? (
              <p className="rejected-message" role="alert">
                {result.message}
              </p>
            ) : isAccepted ? (
              <>
                <div
                  className={`malignant-flag${result.malignant.flagged ? ' malignant-flag--flagged' : ''}`}
                >
                  <span className="malignant-flag-label">MALIGNANT PATTERN CHECK</span>
                  <span className="malignant-flag-text">
                    {result.malignant.flagged
                      ? 'Pattern associated with malignant lesions — worth showing to a dermatologist.'
                      : 'No malignant pattern at this setting.'}
                  </span>
                  <span className="malignant-flag-operating-point">
                    This check is tuned to catch about 95% of malignant lesions, at the cost of
                    flagging many benign lesions too.
                  </span>
                </div>

                <div className="plausible-diagnoses">
                  <span className="plausible-diagnoses-label">CLASSES THE MODEL CAN'T RULE OUT</span>
                  <ul className="plausible-diagnoses-list">
                    {result.prediction_set.map((code) => (
                      <li key={code} className="plausible-diagnoses-row">
                        <span className="plausible-diagnoses-code">{code.toUpperCase()}</span>
                        <span className="plausible-diagnoses-name">
                          {DX_FULL_NAMES[code] ?? code}
                        </span>
                      </li>
                    ))}
                  </ul>
                  {result.prediction_set.length > 1 && (
                    <p className="plausible-diagnoses-note">
                      The model could not narrow this down to a single class at its target
                      confidence level.
                    </p>
                  )}
                </div>

                <div className="top-prediction">
                  <span className="top-prediction-label">Top prediction</span>
                  <span className="top-prediction-name">{result.prediction_full_name}</span>
                  <span className="top-prediction-code">{result.prediction.toUpperCase()}</span>
                  <span className="top-prediction-confidence">
                    {(result.confidence * 100).toFixed(2)}% calibrated confidence
                  </span>
                </div>
                <div className="probabilities">
                  <span className="probabilities-label">ALL 7 CLASSES</span>
                  <ul className="probabilities-list">
                    {sortedProbabilities.map(([name, prob]) => (
                      <li key={name} className="probability-row">
                        <span className="probability-name">{name.toUpperCase()}</span>
                        <span className="probability-bar-track">
                          <span
                            className="probability-bar-fill"
                            style={{ width: `${(prob * 100).toFixed(4)}%` }}
                          />
                        </span>
                        <span className="probability-value">{(prob * 100).toFixed(2)}%</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </>
            ) : (
              <p className="empty-state">Prediction results will appear here after a run.</p>
            )}
          </div>
        </section>

        <footer className="sheet-footer">
          <span className="footer-meta">
            MODEL: resnet50 fusion (image + metadata) · DATASET: HAM10000
          </span>
          <p className="footer-disclaimer">
            Research prototype. Not a medical device. Not for clinical or diagnostic use.
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App
