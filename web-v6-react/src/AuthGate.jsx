import { useEffect, useState } from 'react'
import { supabase } from './lib/supabase'
import './AuthGate.css'

export default function AuthGate({ children }) {
  const [session, setSession] = useState(null)
  const [checking, setChecking] = useState(true)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let mounted = true

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      if (mounted) {
        setSession(currentSession)
        setChecking(false)
      }
    })

    supabase.auth.getSession().then(({ data, error: sessionError }) => {
      if (!mounted) return

      if (sessionError) {
        setError(sessionError.message)
      }

      setSession(data?.session ?? null)
      setChecking(false)
    })

    return () => {
      mounted = false
      subscription.unsubscribe()
    }
  }, [])

  async function handleLogin(event) {
    event.preventDefault()
    setError('')
    setLoading(true)

    const { error: loginError } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    })

    if (loginError) {
      setError(loginError.message)
    }

    setLoading(false)
  }

  async function handleLogout() {
    setError('')

    const { error: logoutError } = await supabase.auth.signOut()

    if (logoutError) {
      setError(logoutError.message)
    }
  }

  if (checking) {
    return (
      <div className="auth-loading">
        <div className="auth-spinner" />
        <p>Verifying secure session...</p>
      </div>
    )
  }

  if (!session) {
    return (
      <main className="auth-page">
        <section className="auth-card">
          <div className="auth-brand">
            <div className="auth-brand-icon">LS</div>
            <span>BIOSENTINELS</span>
          </div>

          <p className="auth-eyebrow">PRIVATE RESEARCH PLATFORM</p>
          <h1>Welcome to LiveShield</h1>
          <p className="auth-description">
            Sign in with your authorized team account to access the
            LiveShield V6 research dashboard.
          </p>

          <form onSubmit={handleLogin}>
            <label htmlFor="auth-email">Email address</label>
            <input
              id="auth-email"
              type="email"
              autoComplete="username"
              placeholder="name@team.com"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />

            <label htmlFor="auth-password">Password</label>
            <input
              id="auth-password"
              type="password"
              autoComplete="current-password"
              placeholder="Enter your password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              required
            />

            {error && <div className="auth-error">{error}</div>}

            <button className="auth-submit" type="submit" disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in securely'}
            </button>
          </form>

          <div className="auth-footer">
            <span className="auth-status-dot" />
            Authorized BioSentinels members only
          </div>
        </section>
      </main>
    )
  }

  return (
    <>
      <div className="auth-session-bar">
        <span title={session.user.email}>
          Signed in: {session.user.email}
        </span>
        <button onClick={handleLogout}>Sign out</button>
      </div>
      {children}
      {error && <div className="auth-session-error">{error}</div>}
    </>
  )
}
