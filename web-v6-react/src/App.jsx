
import { useEffect, useRef, useState } from 'react'
import {
  Activity, Camera, Image as ImageIcon, Eye, BrainCircuit,
  ShieldCheck, ShieldAlert, Cpu, Settings, Menu, RefreshCw,
  Upload, Play, Square, Zap, Gauge, LayoutDashboard,
  ChartNoAxesCombined, Radio, CheckCircle2, AlertCircle,
  ChevronRight, ScanFace, Sparkles
} from 'lucide-react'
import './App.css'
import { supabase } from './lib/supabase'

const API = `${(import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')}/api/v6`
const navigation = [
  { name: 'Overview', icon: LayoutDashboard },
  { name: 'Live Vision', icon: Camera },
  { name: 'Object Tracking', icon: ScanFace },
  { name: 'Visual Assistant', icon: BrainCircuit },
  { name: 'PAD Research', icon: ShieldCheck },
  { name: 'Intelligence Centre', icon: Sparkles },
  { name: 'Analytics & History', icon: ChartNoAxesCombined },
  { name: 'GPU & System Health', icon: Cpu },
  { name: 'Settings & Privacy', icon: Settings },
]

export default function App() {
  const [active, setActive] = useState('Overview')
  const [backend, setBackend] = useState('checking')
  const [system, setSystem] = useState(null)
  const [result, setResult] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [cameraOn, setCameraOn] = useState(false)
  const [sidebar, setSidebar] = useState(true)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const fileRef = useRef(null)
useEffect(() => {
  if (!cameraOn) return

  const video = videoRef.current
  const stream = streamRef.current

  if (!video || !stream) return

  video.srcObject = stream

  const playVideo = async () => {
    try {
      await video.play()
      setError('')
    } catch (err) {
      setError(`Camera playback failed: ${err.message}`)
    }
  }

  if (video.readyState >= 1) {
    playVideo()
  } else {
    video.onloadedmetadata = playVideo
  }

  return () => {
    video.onloadedmetadata = null

    if (video.srcObject === stream) {
      video.srcObject = null
    }
  }
}, [cameraOn])
  async function checkBackend() {
    setBackend('checking')
    try {
      const response = await fetch(`${API}/health`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = await response.json()
      setSystem(data)
      setBackend('online')
    } catch (err) {
      setBackend('offline')
      setError(`Backend connection failed: ${err.message}`)
    }
  }

  useEffect(() => {
    checkBackend()
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  async function analyzeImage(file) {
    if (!file) return

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setError('Please select a JPEG, PNG or WebP image.')
      return
    }

    if (file.size > 50 * 1024 * 1024) {
      setError('Image must be smaller than 50 MB.')
      return
    }

    setError('')
    setLoading(true)
    setResult(null)

    const imageURL = URL.createObjectURL(file)
    setPreview(imageURL)

    try {
      const form = new FormData()
      form.append('frame', file, file.name || 'capture.jpg')

      const { data: sessionData, error: sessionError } =
  await supabase.auth.getSession()

if (sessionError) {
  throw new Error('Unable to verify your login session.')
}

const accessToken = sessionData?.session?.access_token

if (!accessToken) {
  throw new Error('Your login session has expired. Please sign in again.')
}

const response = await fetch(`${API}/analysis?include_scene=true`, {
  method: 'POST',
  headers: {
    Authorization: `Bearer ${accessToken}`,
  },
  body: form,
})

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || `Analysis failed: HTTP ${response.status}`)
      }

      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function startCamera() {
  try {
    setError('')

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    })

    streamRef.current = stream
    setCameraOn(true)

  } catch (err) {
    setCameraOn(false)
    setError(`Camera unavailable: ${err.message}`)
  }
}

function stopCamera() {
  if (streamRef.current) {
    streamRef.current.getTracks().forEach(track => track.stop())
    streamRef.current = null
  }

  setCameraOn(false)
}

  async function captureCamera() {
    const video = videoRef.current
    if (!video || !video.videoWidth) {
      setError('Camera is not ready yet.')
      return
    }

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    const context = canvas.getContext('2d')
    context.drawImage(video, 0, 0)

    canvas.toBlob(async blob => {
      if (blob) {
        const file = new File([blob], 'camera-capture.jpg', {
          type: 'image/jpeg',
        })
        await analyzeImage(file)
      }
    }, 'image/jpeg', 0.92)
  }

  const scene = result?.scene
  const sceneText =
    typeof scene === 'string'
      ? scene
      : scene?.caption || scene?.description || scene?.text || ''

  function renderOverview() {
    return (
      <>
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow"><Sparkles size={15} /> UNIFIED VISION INTELLIGENCE</div>
            <h1>See beyond the <span>ordinary.</span></h1>
            <p>
              Your unified workspace for computer vision, object tracking,
              scene understanding and experimental PAD research.
            </p>
            <button className="primary-button" onClick={() => setActive('Live Vision')}>
              <Camera size={17} /> Open Live Vision
            </button>
          </div>
          <div className="hero-decoration"><Eye size={120} /></div>
        </section>

        <section className="section">
          <div className="section-heading">
            <div>
              <h2>System Overview</h2>
              <p>Live backend connectivity</p>
            </div>
            <button className="icon-button" onClick={checkBackend}>
              <RefreshCw size={17} />
            </button>
          </div>

          <div className="metrics">
            <Metric icon={Activity} label="Backend" value={backend.toUpperCase()} status={backend === 'online'} />
            <Metric icon={Cpu} label="Compute Engine" value={system ? 'RESPONDING' : 'UNKNOWN'} status={!!system} />
            <Metric icon={Eye} label="Vision Pipeline" value="API CONNECTED" status={backend === 'online'} />
            <Metric icon={ShieldAlert} label="PAD Research" value="EXPERIMENTAL" status={false} />
          </div>
        </section>

        <section className="section">
          <div className="section-heading">
            <div>
              <h2>Intelligence Modules</h2>
              <p>Open a workspace</p>
            </div>
          </div>
          <div className="module-grid">
            {navigation.slice(1, 7).map(item => {
              const Icon = item.icon
              return (
                <button className="module-card" key={item.name} onClick={() => setActive(item.name)}>
                  <Icon size={23} />
                  <h3>{item.name}</h3>
                  <span>Open workspace <ChevronRight size={14} /></span>
                </button>
              )
            })}
          </div>
        </section>
      </>
    )
  }

  function renderVision() {
    return (
      <section className="section">
        <div className="section-heading">
          <div>
            <h2>Live Vision Workspace</h2>
            <p>Analyze an uploaded image or capture a camera frame.</p>
          </div>
          <span className={`status-pill ${backend}`}>
            <span className="status-dot" /> {backend}
          </span>
        </div>

        <div className="vision-grid">
          <div className="panel">
            <h3><Camera size={18} /> Input Source</h3>

            <div className="preview">
              {cameraOn ? (
                <video ref={videoRef} autoPlay muted playsInline />
              ) : preview ? (
                <img src={preview} alt="Selected analysis input" />
              ) : (
                <div className="empty-preview">
                  <ImageIcon size={48} />
                  <p>Camera feed or uploaded image will appear here</p>
                </div>
              )}
            </div>

            <div className="button-row">
              {!cameraOn ? (
                <button className="secondary-button" onClick={startCamera}>
                  <Play size={16} /> Start Camera
                </button>
              ) : (
                <>
                  <button className="primary-button" onClick={captureCamera}>
                    <Camera size={16} /> Capture & Analyze
                  </button>
                  <button className="secondary-button" onClick={stopCamera}>
                    <Square size={16} /> Stop
                  </button>
                </>
              )}

              <button className="secondary-button" onClick={() => fileRef.current?.click()}>
                <Upload size={16} /> Upload Image
              </button>

              <input
                ref={fileRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                hidden
                onChange={event => analyzeImage(event.target.files?.[0])}
              />
            </div>

            {loading && <div className="loading-line"><span /> Analyzing image with LiveShield...</div>}
          </div>

          <div className="panel results-panel">
            <h3><BrainCircuit size={18} /> Analysis Results</h3>

            {!result && !loading && (
              <div className="empty-results">
                <Activity size={38} />
                <p>Analysis results will appear here.</p>
              </div>
            )}

            {result && (
              <>
                <div className="result-summary">
                  <div><span>Faces</span><strong>{result.face_count ?? '—'}</strong></div>
                  <div><span>Objects</span><strong>{result.object_count ?? '—'}</strong></div>
                  <div><span>People</span><strong>{result.person_count ?? '—'}</strong></div>
                </div>

                <div className="result-block">
                  <label>SCENE DESCRIPTION</label>
                  <p>{sceneText || 'No scene description returned.'}</p>
                </div>

                <div className="result-block">
                  <label>DETECTION DETAILS</label>
                  <pre>{JSON.stringify({
                    faces: result.face_boxes,
                    objects: result.objects,
                    quality: result.quality,
                    pad: result.pad,
                    performance: result.performance,
                    warnings: result.warnings,
                  }, null, 2)}</pre>
                </div>
              </>
            )}
          </div>
        </div>
      </section>
    )
  }

  function renderOtherPage() {
    const descriptions = {
      'Object Tracking': 'Live object tracking integration is the next development stage.',
      'Visual Assistant': 'Your existing V6 Visual Assistant remains available on the original dashboard. We will connect it to this React interface next.',
      'PAD Research': 'Experimental presentation-attack research indicators. These are not validated authentication decisions.',
      'Intelligence Centre': 'A unified workspace for vision analysis, detection and scene understanding.',
      'Analytics & History': 'Performance charts and detection history will be connected after live analysis is verified.',
      'GPU & System Health': 'Hardware and compute diagnostics will be connected to the V6 system endpoint.',
      'Settings & Privacy': 'Biometric frame storage remains disabled by default. Face recognition is not implemented.',
    }

    return (
      <section className="panel placeholder-panel">
        <Sparkles size={42} />
        <h2>{active}</h2>
        <p>{descriptions[active] || 'Workspace under development.'}</p>
        <button className="primary-button" onClick={() => setActive('Live Vision')}>
          <Camera size={16} /> Go to Live Vision
        </button>
      </section>
    )
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebar ? '' : 'collapsed'}`}>
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={23} /></div>
          {sidebar && <div><strong>LIVESHIELD</strong><small>VISION INTELLIGENCE</small></div>}
        </div>

        <div className="nav-label">{sidebar && 'WORKSPACE'}</div>
        <nav>
          {navigation.map(item => {
            const Icon = item.icon
            return (
              <button
                key={item.name}
                title={!sidebar ? item.name : ''}
                className={`nav-item ${active === item.name ? 'active' : ''}`}
                onClick={() => setActive(item.name)}
              >
                <Icon size={19} />
                {sidebar && <span>{item.name}</span>}
              </button>
            )
          })}
        </nav>

        {sidebar && (
          <div className="sidebar-bottom">
            <Radio size={15} /> LIVE V6 DEVELOPMENT
          </div>
        )}
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button" onClick={() => setSidebar(!sidebar)}><Menu size={20} /></button>
            <div><strong>{active}</strong><small>LiveShield / V6 / {active}</small></div>
          </div>
          <div className="topbar-right">
            <span className={`status-pill ${backend}`}><span className="status-dot" /> API {backend}</span>
            <button className="icon-button" onClick={checkBackend}><RefreshCw size={17} /></button>
          </div>
        </header>

        <div className="page-content">
          {error && (
            <div className="error-banner">
              <AlertCircle size={18} />
              <span>{error}</span>
              <button onClick={() => setError('')}>Dismiss</button>
            </div>
          )}

          {active === 'Overview' && renderOverview()}
          {active === 'Live Vision' && renderVision()}
          {!['Overview', 'Live Vision'].includes(active) && renderOtherPage()}

          <footer>LiveShield V6 · MOSIP BioSentinels · Development Interface</footer>
        </div>
      </main>
    </div>
  )
}

function Metric({ icon: Icon, label, value, status }) {
  return (
    <div className="metric-card">
      <div className="metric-top"><span>{label}</span><Icon size={19} /></div>
      <strong className={status ? 'good' : 'neutral'}>{value}</strong>
      <small>{status ? 'Service responding' : 'Not verified'}</small>
    </div>
  )
}
