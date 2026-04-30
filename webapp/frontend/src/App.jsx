import { useState, useEffect, useRef, useCallback } from 'react'
import ModelViewer from './components/ModelViewer.jsx'

const API = '' // proxied via Vite → http://localhost:8000

const EDITABLE_FIELDS = [
  { key: 'length_mm',        label: 'Foot Length',       unit: 'mm' },
  { key: 'heel_width_mm',    label: 'Heel Width',        unit: 'mm' },
  { key: 'ball_width_mm',    label: 'Ball Width',        unit: 'mm' },
  { key: 'ball_perimeter_mm',label: 'Ball Perimeter',    unit: 'mm' },
  { key: 'ball_height_mm',   label: 'Ball Height',       unit: 'mm' },
  { key: 'arch_height_mm',   label: 'Arch Height',       unit: 'mm' },
  { key: 'toe_box_width_mm', label: 'Toe Box Width',     unit: 'mm' },
  { key: 'toe_box_height_mm',label: 'Toe Box Height',    unit: 'mm' },
]

const PIPELINE_STAGES = [
  { label: 'Loading template',       threshold: 10 },
  { label: 'Warping to measurements', threshold: 35 },
  { label: 'NRICP refinement',       threshold: 75 },
  { label: 'Exporting STL',          threshold: 95 },
  { label: 'Done',                   threshold: 100 },
]

function StepsBar({ step }) {
  const nodes = ['Upload', 'Review', 'Generate', 'Done']
  return (
    <div className="steps">
      {nodes.map((label, i) => {
        const nodeStep = i + 1
        const isDone = step > nodeStep
        const isActive = step === nodeStep
        return (
          <div key={label} className="step-node">
            {i > 0 && <div className={`step-connector ${step > nodeStep ? 'done' : ''}`} />}
            <div className={`step-bubble ${isDone ? 'done' : isActive ? 'active' : ''}`}>
              {isDone ? '✓' : nodeStep}
            </div>
            <span className={`step-label ${isDone ? 'done' : isActive ? 'active' : ''}`}>{label}</span>
          </div>
        )
      })}
    </div>
  )
}

function ShoeSizeBanner({ shoeSize }) {
  if (!shoeSize) return null
  const sizes = [
    { sys: 'EU', val: shoeSize.eu },
    { sys: 'US Men', val: shoeSize.us_mens },
    { sys: 'US Women', val: shoeSize.us_womens },
    { sys: 'UK', val: shoeSize.uk },
  ]
  return (
    <div className="shoe-size-banner">
      <span className="label">Predicted Size</span>
      <div className="size-chips">
        {sizes.map(({ sys, val }) => (
          <div key={sys} className="size-chip">
            <span className="size-val">{val}</span>
            <span className="size-sys">{sys}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function PipelineProgress({ progress, message }) {
  const stageIndex = PIPELINE_STAGES.findIndex((s) => progress < s.threshold)
  const activeIdx = stageIndex === -1 ? PIPELINE_STAGES.length - 1 : stageIndex

  return (
    <div className="progress-section">
      <div className="progress-header">
        <span className="progress-stage">{message}</span>
        <span className="progress-pct">{progress}%</span>
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${progress}%` }} />
      </div>
      <div className="progress-steps">
        {PIPELINE_STAGES.map((s, i) => {
          const done = i < activeIdx
          const active = i === activeIdx && progress < 100
          return (
            <div key={s.label} className={`progress-step-item ${done ? 'done' : active ? 'active' : ''}`}>
              <div className={`step-dot ${done ? 'done' : active ? 'active' : ''}`} />
              {s.label}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function App() {
  const [step, setStep] = useState(1)           // 1=upload, 2=review, 3=generate, 4=done
  const [jobId, setJobId] = useState(null)
  const [jobData, setJobData] = useState(null)  // full status from /api/status
  const [measurements, setMeasurements] = useState({})
  const [allowanceMm, setAllowanceMm] = useState(3.0)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileInputRef = useRef(null)
  const pollRef = useRef(null)

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const startPolling = useCallback((id) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/api/status/${id}`)
        if (!res.ok) return
        const data = await res.json()
        setJobData(data)

        if (data.status === 'measured') {
          stopPolling()
          setMeasurements({ ...data.measurements })
          setStep(2)
        }
        if (data.status === 'done') {
          stopPolling()
          setStep(4)
        }
        if (data.status === 'error') {
          stopPolling()
        }
      } catch (_) {}
    }, 1200)
  }, [])

  useEffect(() => () => stopPolling(), [])

  async function handleFile(file) {
    if (!file || !file.name.toLowerCase().endsWith('.obj')) {
      alert('Please select an OBJ file.')
      return
    }
    setUploading(true)
    setStep(1)
    setJobData(null)
    try {
      const form = new FormData()
      form.append('scan', file)
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      const { job_id } = await res.json()
      setJobId(job_id)
      setJobData({ status: 'measuring', progress: 10, message: 'Aligning and measuring foot scan…' })
      startPolling(job_id)
    } catch (err) {
      alert('Upload failed: ' + err.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleGenerate() {
    if (!jobId) return
    setStep(3)
    const res = await fetch(`${API}/api/generate/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ measurements, allowance_mm: allowanceMm }),
    })
    if (!res.ok) { alert('Generate request failed'); return }
    startPolling(jobId)
  }

  const measuring = jobData?.status === 'measuring'
  const generating = jobData?.status === 'generating'
  const hasError = jobData?.status === 'error'
  const stlUrl = step === 4 && jobId ? `${API}/api/download/${jobId}/stl` : null

  return (
    <div className="app">
      <header className="header">
        <span className="header-logo">👟</span>
        <h1>Custom Shoe Tree Generator</h1>
        <span className="header-sub">FABRIC-581</span>
      </header>

      <main className="main">
        <StepsBar step={step} />

        {/* ── Step 1: Upload ──────────────────────────────────────── */}
        {step === 1 && (
          <div className="card">
            <div className="card-header">
              <h2>Upload Foot Scan</h2>
              <p>Drop your foot scan OBJ file. The pipeline will align it and extract measurements automatically.</p>
            </div>
            <div className="card-body">
              {!measuring && !uploading && (
                <label
                  className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <span className="upload-icon">📁</span>
                  <h3>Drop OBJ file here or click to browse</h3>
                  <p>Foot scan in OBJ format (m, cm, or mm — auto-detected)</p>
                  <span className="upload-hint">Accepts .obj</span>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".obj"
                    onChange={(e) => handleFile(e.target.files[0])}
                  />
                </label>
              )}

              {(measuring || uploading) && (
                <div className="measuring-state">
                  <div className="spinner" />
                  <p className="measuring-label">{jobData?.message || 'Uploading…'}</p>
                </div>
              )}

              {hasError && (
                <div className="error-banner" style={{ marginTop: 16 }}>
                  <span>⚠</span>
                  <span>{jobData?.message}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Step 2: Review & Edit Measurements ──────────────────── */}
        {step === 2 && (
          <div className="card">
            <div className="card-header">
              <h2>Review Measurements</h2>
              <p>Edit any values if needed, then confirm the predicted size and generate your shoe tree.</p>
            </div>
            <div className="card-body">
              <ShoeSizeBanner shoeSize={jobData?.shoe_size} />

              <div className="section-title">Editable Dimensions</div>
              <div className="measurements-grid">
                {EDITABLE_FIELDS.map(({ key, label, unit }) => (
                  <div className="m-field" key={key}>
                    <label htmlFor={key}>{label}</label>
                    <div className="m-field-row">
                      <input
                        id={key}
                        type="number"
                        step="0.1"
                        value={measurements[key] ?? ''}
                        onChange={(e) =>
                          setMeasurements((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))
                        }
                      />
                      <span className="unit">{unit}</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="section-divider" />
              <div className="section-title">Morphology</div>
              <div className="measurements-grid" style={{ marginBottom: 0 }}>
                <div className="m-field">
                  <label>Arch Type</label>
                  <div style={{ paddingTop: 6 }}>
                    <span className="badge badge-accent">{measurements.arch_type}</span>
                  </div>
                </div>
                <div className="m-field">
                  <label>Toe Box Type</label>
                  <div style={{ paddingTop: 6 }}>
                    <span className="badge badge-neutral">{measurements.toe_box_type}</span>
                  </div>
                </div>
                <div className="m-field">
                  <label>Toe Angle</label>
                  <div style={{ paddingTop: 6 }}>
                    <span className="badge badge-neutral">{measurements.toe_angle_deg}°</span>
                  </div>
                </div>
              </div>

              <div className="section-divider" />
              <div className="section-title">Shoe-Tree Allowance</div>
              <div className="allowance-row">
                <label>Upper-surface offset</label>
                <input
                  type="range"
                  min="0"
                  max="10"
                  step="0.5"
                  value={allowanceMm}
                  onChange={(e) => setAllowanceMm(parseFloat(e.target.value))}
                />
                <span className="allowance-val">{allowanceMm} mm</span>
              </div>

              <div className="btn-row">
                <button className="btn btn-secondary" onClick={() => { stopPolling(); setStep(1); setJobData(null) }}>
                  ← New Scan
                </button>
                <button className="btn btn-primary" onClick={handleGenerate}>
                  Generate Shoe Tree →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── Step 3: Generating ──────────────────────────────────── */}
        {step === 3 && (
          <div className="card">
            <div className="card-header">
              <h2>Generating Shoe Tree</h2>
              <p>The full pipeline is running in the background. This typically takes 2–5 minutes.</p>
            </div>
            <div className="card-body">
              {hasError ? (
                <>
                  <div className="error-banner">
                    <span>⚠</span>
                    <span>{jobData?.message}</span>
                  </div>
                  <div className="btn-row" style={{ marginTop: 20 }}>
                    <button className="btn btn-secondary" onClick={() => { setStep(1); setJobData(null) }}>
                      Start Over
                    </button>
                  </div>
                </>
              ) : (
                <PipelineProgress
                  progress={generating ? jobData?.progress ?? 0 : 0}
                  message={jobData?.message ?? 'Starting…'}
                />
              )}
            </div>
          </div>
        )}

        {/* ── Step 4: Done ────────────────────────────────────────── */}
        {step === 4 && (
          <div className="card">
            <div className="card-header">
              <h2>Your Shoe Tree is Ready</h2>
              <p>Download the STL and import it directly into BambuLab Studio.</p>
            </div>
            <div className="card-body">
              <div className="success-banner">
                <span className="success-icon">✅</span>
                <span>Pipeline complete — fabrication-ready STL exported.</span>
              </div>

              <div className="result-layout">
                <div className="result-actions">
                  <a
                    href={`${API}/api/download/${jobId}/stl`}
                    download
                    className="btn btn-success"
                    style={{ textDecoration: 'none' }}
                  >
                    ⬇ Download STL
                  </a>
                  <a
                    href={`${API}/api/download/${jobId}/obj`}
                    download
                    className="btn btn-secondary"
                    style={{ textDecoration: 'none' }}
                  >
                    ⬇ Download OBJ
                  </a>
                  <button
                    className="btn btn-secondary"
                    style={{ marginTop: 8 }}
                    onClick={() => { setStep(1); setJobData(null); setJobId(null) }}
                  >
                    ← Process Another Scan
                  </button>

                  <div className="section-divider" />
                  <div className="section-title">Output Info</div>
                  {[
                    ['Scan ID', jobData?.scan_id],
                    ['Allowance', `${allowanceMm} mm`],
                    ['EU Size', jobData?.shoe_size?.eu],
                    ['US Men\'s', jobData?.shoe_size?.us_mens],
                  ].map(([k, v]) => (
                    <div key={k} className="result-stat">
                      <span>{k}</span>
                      <span className="val">{v ?? '—'}</span>
                    </div>
                  ))}
                </div>

                {stlUrl ? (
                  <ModelViewer url={stlUrl} />
                ) : (
                  <div className="viewer-placeholder">
                    <span className="icon">🥿</span>
                    <span>3D preview loading…</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
