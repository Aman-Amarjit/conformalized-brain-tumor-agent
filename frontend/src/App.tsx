import React, { useState, useEffect, useRef } from 'react';

const API_BASE = typeof window !== 'undefined' && window.location.origin.includes('localhost')
  ? 'http://localhost:8000'
  : '';

/* ── helpers ── */
const TUMOR_INFO: Record<string, { emoji: string; color: string; bg: string; border: string; what: string; action: string }> = {
  'Glioma': {
    emoji: '🔴',
    color: '#dc2626',
    bg: '#fef2f2',
    border: '#fca5a5',
    what: 'A type of brain tumor that grows in the brain or spinal cord tissue. It can range from slow-growing to fast-growing.',
    action: 'Please consult a neurologist or neuro-oncologist immediately. Early treatment greatly improves outcomes.',
  },
  'Meningioma': {
    emoji: '🟠',
    color: '#d97706',
    bg: '#fffbeb',
    border: '#fcd34d',
    what: 'A tumor that forms on the membranes covering the brain. Most meningiomas are not cancerous and grow slowly.',
    action: 'Schedule an appointment with a neurologist. Many meningiomas are monitored rather than treated immediately.',
  },
  'Pituitary Tumor': {
    emoji: '🟡',
    color: '#ca8a04',
    bg: '#fefce8',
    border: '#fde047',
    what: 'A tumor on the pituitary gland at the base of the brain. Most are benign and affect hormone levels.',
    action: 'See an endocrinologist or neurosurgeon. Treatments are highly effective, including medication and minimally invasive surgery.',
  },
  'Normal Scan': {
    emoji: '🟢',
    color: '#16a34a',
    bg: '#f0fdf4',
    border: '#86efac',
    what: 'No signs of a brain tumor were detected in this scan.',
    action: 'Your scan looks clear! If you still have symptoms, discuss them with your doctor for further evaluation.',
  },
};

function getTumorInfo(label: string) {
  return TUMOR_INFO[label] || {
    emoji: '⚪',
    color: '#6366f1',
    bg: '#eef2ff',
    border: '#a5b4fc',
    what: label,
    action: 'Please consult your doctor for further guidance.',
  };
}

interface PredictResponse {
  prediction_set: string[];
  is_confident: boolean;
  set_size: number;
  target_coverage: number;
  alpha: number;
  quantile_applied: number;
  softmax_probabilities: Record<string, number>;
  abstention_triage_flag: boolean;
  triage_message: string;
  inference_time_ms: number;
}

interface CuratedSample {
  id: string;
  filename: string;
  type: string;
  label: string;
  true_class: string;
  expected_set: string[];
  image_url: string;
}

/* ── main component ── */
export const App: React.FC = () => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [samples, setSamples] = useState<CuratedSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<CuratedSample | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/curated-samples`)
      .then(r => r.json())
      .then(setSamples)
      .catch(() => {});
  }, []);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPG, PNG, etc.)');
      return;
    }
    setUploadedFile(file);
    setSelectedSample(null);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = e => setImagePreview(e.target?.result as string);
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleSelectSample = (s: CuratedSample) => {
    setSelectedSample(s);
    setUploadedFile(null);
    setResult(null);
    setError(null);
    setImagePreview(`${API_BASE}${s.image_url}`);
  };

  const runAnalysis = async () => {
    if (!uploadedFile && !selectedSample) {
      setError('Please upload a brain scan image first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);
    try {
      let url = `${API_BASE}/api/predict?alpha=0.05`;
      const opts: RequestInit = { method: 'POST' };
      if (selectedSample) {
        url += `&sample_id=${selectedSample.id}`;
      } else if (uploadedFile) {
        const fd = new FormData();
        fd.append('file', uploadedFile);
        opts.body = fd;
      }
      const res = await fetch(url, opts);
      if (!res.ok) throw new Error(await res.text());
      const data: PredictResponse = await res.json();
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e: any) {
      setError('Analysis failed. Please try again or use a different image.');
    } finally {
      setIsLoading(false);
    }
  };

  const reset = () => {
    setUploadedFile(null);
    setImagePreview(null);
    setResult(null);
    setError(null);
    setSelectedSample(null);
  };

  /* top probability label */
  const topLabel = result
    ? Object.entries(result.softmax_probabilities).sort((a, b) => b[1] - a[1])[0]?.[0]
    : null;

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%)', display: 'flex', flexDirection: 'column' }}>

      {/* ── HEADER ── */}
      <header style={{ background: '#fff', borderBottom: '1px solid #e2e8f0', padding: '0 24px', height: 64, display: 'flex', alignItems: 'center', justifyContent: 'space-between', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 40, height: 40, borderRadius: 12, background: 'linear-gradient(135deg, #6366f1, #8b5cf6)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>🧠</div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 16, color: '#1e293b' }}>BrainScan AI</div>
            <div style={{ fontSize: 11, color: '#64748b' }}>Brain Tumor Detection Assistant</div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 20, padding: '4px 12px' }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#16a34a', animation: 'pulse 2s infinite' }} />
          <span style={{ fontSize: 12, color: '#16a34a', fontWeight: 600 }}>AI Online</span>
        </div>
      </header>

      <main style={{ flex: 1, maxWidth: 720, margin: '0 auto', width: '100%', padding: '32px 16px 60px' }}>

        {/* ── HERO ── */}
        <div style={{ textAlign: 'center', marginBottom: 36 }}>
          <h1 style={{ fontSize: 'clamp(26px, 5vw, 38px)', fontWeight: 800, color: '#1e293b', lineHeight: 1.2, marginBottom: 12 }}>
            Is your brain scan normal?
          </h1>
          <p style={{ fontSize: 16, color: '#64748b', maxWidth: 500, margin: '0 auto', lineHeight: 1.6 }}>
            Upload your MRI scan and our AI will analyse it in seconds — in plain English, not medical jargon.
          </p>
        </div>

        {/* ── DISCLAIMER ── */}
        <div style={{ background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 12, padding: '12px 16px', marginBottom: 28, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
          <span style={{ fontSize: 18, flexShrink: 0 }}>⚠️</span>
          <p style={{ fontSize: 13, color: '#92400e', lineHeight: 1.5, margin: 0 }}>
            <strong>This tool is for informational purposes only.</strong> It is not a medical diagnosis. Always consult a qualified doctor before making any health decisions.
          </p>
        </div>

        {/* ── UPLOAD CARD ── */}
        {!result && (
          <div style={{ background: '#fff', borderRadius: 20, boxShadow: '0 4px 20px rgba(0,0,0,0.08)', overflow: 'hidden', marginBottom: 24 }}>
            <div style={{ padding: '24px 24px 0' }}>
              <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>Step 1 — Upload your MRI scan</h2>
              <p style={{ fontSize: 14, color: '#64748b', marginBottom: 20 }}>Accepted formats: JPG, PNG, DICOM images</p>

              {/* Drop zone */}
              <div
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  border: `2px dashed ${dragOver ? '#6366f1' : imagePreview ? '#6366f1' : '#cbd5e1'}`,
                  borderRadius: 16,
                  background: dragOver ? '#eef2ff' : imagePreview ? '#fafbff' : '#f8fafc',
                  padding: '32px 24px',
                  textAlign: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  position: 'relative',
                  minHeight: imagePreview ? 260 : 180,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                  gap: 12,
                }}
              >
                {imagePreview ? (
                  <>
                    <img src={imagePreview} alt="Brain scan preview" style={{ maxHeight: 220, maxWidth: '100%', borderRadius: 10, objectFit: 'contain' }} />
                    <span style={{ fontSize: 13, color: '#6366f1', fontWeight: 500 }}>
                      {uploadedFile ? `📁 ${uploadedFile.name}` : selectedSample ? `🔬 ${selectedSample.label}` : ''} — click to change
                    </span>
                  </>
                ) : (
                  <>
                    <div style={{ fontSize: 48 }}>🩻</div>
                    <div>
                      <div style={{ fontWeight: 600, color: '#475569', fontSize: 15 }}>Drag & drop your scan here</div>
                      <div style={{ color: '#94a3b8', fontSize: 13, marginTop: 4 }}>or click to browse files</div>
                    </div>
                  </>
                )}
              </div>
              <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
            </div>

            {/* Sample scans */}
            {samples.length > 0 && (
              <div style={{ padding: '20px 24px' }}>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 12 }}>Or try a sample scan:</div>
                <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                  {samples.slice(0, 4).map(s => {
                    const info = getTumorInfo(s.true_class);
                    const active = selectedSample?.id === s.id;
                    return (
                      <button
                        key={s.id}
                        onClick={() => handleSelectSample(s)}
                        style={{
                          padding: '8px 14px',
                          borderRadius: 10,
                          border: `2px solid ${active ? info.color : '#e2e8f0'}`,
                          background: active ? info.bg : '#f8fafc',
                          color: active ? info.color : '#475569',
                          fontWeight: 600,
                          fontSize: 13,
                          transition: 'all 0.15s',
                          cursor: 'pointer',
                        }}
                      >
                        {info.emoji} {s.true_class}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Analyse button */}
            <div style={{ padding: '0 24px 24px' }}>
              {error && (
                <div style={{ background: '#fef2f2', border: '1px solid #fca5a5', borderRadius: 10, padding: '10px 14px', color: '#dc2626', fontSize: 14, marginBottom: 14 }}>
                  ❌ {error}
                </div>
              )}
              <button
                onClick={runAnalysis}
                disabled={isLoading || (!uploadedFile && !selectedSample)}
                style={{
                  width: '100%',
                  padding: '15px 24px',
                  borderRadius: 14,
                  background: isLoading || (!uploadedFile && !selectedSample)
                    ? '#cbd5e1'
                    : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  color: '#fff',
                  fontWeight: 700,
                  fontSize: 16,
                  cursor: isLoading || (!uploadedFile && !selectedSample) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s',
                  boxShadow: isLoading || (!uploadedFile && !selectedSample) ? 'none' : '0 4px 14px rgba(99,102,241,0.4)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                }}
              >
                {isLoading ? (
                  <>
                    <Spinner />
                    Analysing your scan…
                  </>
                ) : (
                  '🔍 Analyse My Brain Scan'
                )}
              </button>
            </div>
          </div>
        )}

        {/* ── RESULTS ── */}
        {result && (
          <div ref={resultRef}>
            {/* Main verdict */}
            {result.is_confident && topLabel ? (
              <VerdictCard label={topLabel} probs={result.softmax_probabilities} />
            ) : (
              <UncertainCard predictionSet={result.prediction_set} probs={result.softmax_probabilities} />
            )}

            {/* Image + probability bars side by side */}
            <div style={{ display: 'grid', gridTemplateColumns: imagePreview ? '1fr 1fr' : '1fr', gap: 16, marginBottom: 16 }}>
              {imagePreview && (
                <div style={{ background: '#fff', borderRadius: 16, boxShadow: '0 2px 12px rgba(0,0,0,0.07)', padding: 16 }}>
                  <div style={{ fontWeight: 600, fontSize: 14, color: '#475569', marginBottom: 10 }}>📷 Your Scan</div>
                  <img src={imagePreview} alt="Brain scan" style={{ width: '100%', borderRadius: 10, objectFit: 'contain', maxHeight: 200 }} />
                </div>
              )}
              <div style={{ background: '#fff', borderRadius: 16, boxShadow: '0 2px 12px rgba(0,0,0,0.07)', padding: 16 }}>
                <div style={{ fontWeight: 600, fontSize: 14, color: '#475569', marginBottom: 12 }}>📊 Probability Breakdown</div>
                {Object.entries(result.softmax_probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([label, prob]) => {
                    const info = getTumorInfo(label);
                    const pct = Math.round(prob * 100);
                    return (
                      <div key={label} style={{ marginBottom: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: '#334155' }}>{info.emoji} {label}</span>
                          <span style={{ fontSize: 13, fontWeight: 700, color: info.color }}>{pct}%</span>
                        </div>
                        <div style={{ height: 8, borderRadius: 99, background: '#f1f5f9', overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${pct}%`, borderRadius: 99, background: info.color, transition: 'width 0.8s ease' }} />
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* What should I do next */}
            <NextStepsCard isConfident={result.is_confident} predictionSet={result.prediction_set} />

            {/* Try again */}
            <button
              onClick={reset}
              style={{ width: '100%', padding: '13px', borderRadius: 12, background: '#f1f5f9', color: '#475569', fontWeight: 600, fontSize: 15, marginTop: 8, cursor: 'pointer', border: '1px solid #e2e8f0', transition: 'background 0.2s' }}
            >
              ↩ Analyse Another Scan
            </button>
          </div>
        )}

        {/* ── HOW IT WORKS ── */}
        {!result && (
          <div style={{ marginTop: 8 }}>
            <h2 style={{ fontSize: 16, fontWeight: 700, color: '#475569', textAlign: 'center', marginBottom: 16 }}>How does it work?</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {[
                { icon: '📤', title: 'Upload', desc: 'Upload your MRI scan image from your device' },
                { icon: '🤖', title: 'AI Analysis', desc: 'Our AI examines patterns in the scan in seconds' },
                { icon: '📋', title: 'Plain Report', desc: 'Get a simple, clear result with next steps' },
              ].map(step => (
                <div key={step.title} style={{ background: '#fff', borderRadius: 14, padding: 16, textAlign: 'center', boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}>
                  <div style={{ fontSize: 28, marginBottom: 8 }}>{step.icon}</div>
                  <div style={{ fontWeight: 700, fontSize: 14, color: '#1e293b', marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{step.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <footer style={{ textAlign: 'center', padding: '20px 16px', color: '#94a3b8', fontSize: 12, borderTop: '1px solid #e2e8f0', background: '#fff' }}>
        BrainScan AI — For educational purposes only. Not a substitute for professional medical advice.
      </footer>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
};

/* ── sub-components ── */

function Spinner() {
  return <div style={{ width: 18, height: 18, border: '2px solid rgba(255,255,255,0.4)', borderTopColor: '#fff', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />;
}

function VerdictCard({ label, probs }: { label: string; probs: Record<string, number> }) {
  const info = getTumorInfo(label);
  const isNormal = label === 'Normal Scan';
  const topProb = Math.round((probs[label] || 0) * 100);

  return (
    <div style={{ background: info.bg, border: `2px solid ${info.border}`, borderRadius: 20, padding: '28px 24px', marginBottom: 16, animation: 'fadeSlideUp 0.5s ease' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 44 }}>{isNormal ? '✅' : '🔬'}</span>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: info.color, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>
            {isNormal ? 'Good News' : 'Finding Detected'}
          </div>
          <h2 style={{ fontSize: 26, fontWeight: 800, color: info.color, lineHeight: 1.1 }}>{label}</h2>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right' }}>
          <div style={{ fontSize: 32, fontWeight: 800, color: info.color }}>{topProb}%</div>
          <div style={{ fontSize: 11, color: info.color, opacity: 0.7 }}>confidence</div>
        </div>
      </div>
      <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 12, padding: '14px 16px', marginBottom: 12 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 6 }}>📌 What this means</div>
        <p style={{ fontSize: 14, color: '#334155', lineHeight: 1.6, margin: 0 }}>{info.what}</p>
      </div>
      <div style={{ background: 'rgba(255,255,255,0.7)', borderRadius: 12, padding: '14px 16px' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#475569', marginBottom: 6 }}>👨‍⚕️ What you should do</div>
        <p style={{ fontSize: 14, color: '#334155', lineHeight: 1.6, margin: 0 }}>{info.action}</p>
      </div>
    </div>
  );
}

function UncertainCard({ predictionSet, probs }: { predictionSet: string[]; probs: Record<string, number> }) {
  return (
    <div style={{ background: '#fffbeb', border: '2px solid #fcd34d', borderRadius: 20, padding: '28px 24px', marginBottom: 16, animation: 'fadeSlideUp 0.5s ease' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <span style={{ fontSize: 44 }}>🔎</span>
        <div>
          <div style={{ fontSize: 13, fontWeight: 600, color: '#d97706', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 2 }}>Needs Expert Review</div>
          <h2 style={{ fontSize: 22, fontWeight: 800, color: '#92400e', lineHeight: 1.2 }}>The AI is not certain enough</h2>
        </div>
      </div>
      <div style={{ background: 'rgba(255,255,255,0.8)', borderRadius: 12, padding: '14px 16px', marginBottom: 12 }}>
        <p style={{ fontSize: 14, color: '#78350f', lineHeight: 1.6, margin: 0 }}>
          The AI detected signs that could point to one of several conditions. This happens with complex or ambiguous scans.
          The possible findings are:
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
          {predictionSet.map(label => {
            const info = getTumorInfo(label);
            const pct = Math.round((probs[label] || 0) * 100);
            return (
              <span key={label} style={{ background: info.bg, border: `1px solid ${info.border}`, color: info.color, borderRadius: 8, padding: '5px 12px', fontWeight: 600, fontSize: 13 }}>
                {info.emoji} {label} ({pct}%)
              </span>
            );
          })}
        </div>
      </div>
      <div style={{ background: 'rgba(255,255,255,0.8)', borderRadius: 12, padding: '14px 16px' }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#92400e', marginBottom: 6 }}>👨‍⚕️ What you should do</div>
        <p style={{ fontSize: 14, color: '#78350f', lineHeight: 1.6, margin: 0 }}>
          Please consult a neurologist or radiologist to review your scan in person. Bring this result and your original scan images to your appointment.
        </p>
      </div>
    </div>
  );
}

function NextStepsCard({ isConfident, predictionSet }: { isConfident: boolean; predictionSet: string[] }) {
  const isNormal = isConfident && predictionSet.includes('Normal Scan');

  const steps = isNormal
    ? [
        { icon: '😌', text: 'Breathe easy — your scan looks normal.' },
        { icon: '📋', text: 'If you still have symptoms like headaches or vision changes, discuss them with your GP.' },
        { icon: '🔄', text: 'Consider scheduling a follow-up scan if symptoms persist.' },
      ]
    : [
        { icon: '📞', text: 'Call your doctor today and tell them the AI result.' },
        { icon: '🏥', text: 'Ask for a referral to a neurologist or neuro-oncologist.' },
        { icon: '🖨️', text: 'Bring your original MRI images and this report to the appointment.' },
        { icon: '❓', text: 'Ask your doctor about next steps — treatment, monitoring, or more tests.' },
      ];

  return (
    <div style={{ background: '#fff', borderRadius: 16, boxShadow: '0 2px 12px rgba(0,0,0,0.07)', padding: '20px 20px', marginBottom: 16 }}>
      <div style={{ fontWeight: 700, fontSize: 16, color: '#1e293b', marginBottom: 14 }}>📋 Recommended Next Steps</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {steps.map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 12, padding: '10px 12px', background: '#f8fafc', borderRadius: 10 }}>
            <span style={{ fontSize: 20, flexShrink: 0 }}>{s.icon}</span>
            <p style={{ fontSize: 14, color: '#334155', lineHeight: 1.5, margin: 0 }}>{s.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
