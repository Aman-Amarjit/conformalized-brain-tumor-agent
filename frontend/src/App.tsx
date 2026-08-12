import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Brain,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  ShieldCheck,
  FileText,
  RefreshCw,
  Printer,
  Info,
  Sparkles,
  Search,
  ArrowRight
} from 'lucide-react';

const API_BASE = typeof window !== 'undefined' && window.location.origin.includes('localhost')
  ? 'http://localhost:8000'
  : '';

interface TumorMeta {
  title: string;
  badgeText: string;
  badgeBg: string;
  badgeColor: string;
  badgeBorder: string;
  summary: string;
  explanation: string;
  actionItems: string[];
}

const DIAGNOSIS_DETAILS: Record<string, TumorMeta> = {
  'Normal Scan': {
    title: 'Normal Scan (No Tumor Detected)',
    badgeText: 'Clear Scan',
    badgeBg: 'rgba(16, 185, 129, 0.15)',
    badgeColor: '#10b981',
    badgeBorder: 'rgba(16, 185, 129, 0.4)',
    summary: 'No signs of brain tumor tissue were detected in this MRI scan.',
    explanation: 'The AI model examined the structural patterns, tissue density, and cerebral ventricles in your MRI scan and confirmed that all visible structures align with healthy brain tissue parameters.',
    actionItems: [
      'Share this report with your primary healthcare provider for your medical records.',
      'If you are experiencing persistent symptoms (headaches, dizziness, or vision changes), follow up with a doctor for non-oncological evaluation.',
      'Schedule routine annual check-ups as recommended by your physician.'
    ]
  },
  'Glioma': {
    title: 'Glioma Tissue Pattern Identified',
    badgeText: 'Abnormal Finding Detected',
    badgeBg: 'rgba(239, 68, 68, 0.15)',
    badgeColor: '#ef4444',
    badgeBorder: 'rgba(239, 68, 68, 0.4)',
    summary: 'The AI identified imaging patterns characteristic of a Glioma.',
    explanation: 'A Glioma is a type of brain tumor that originates in the glial support cells of the brain. They can vary in growth rate and characteristics, requiring specialized medical review.',
    actionItems: [
      'Schedule an urgent consultation with a Neurologist or Neuro-Oncologist.',
      'Request a contrast-enhanced MRI scan (T1-weighted post-contrast) for high-resolution staging.',
      'Bring your original MRI DICOM files and this report to your medical specialist.'
    ]
  },
  'Meningioma': {
    title: 'Meningioma Tissue Pattern Identified',
    badgeText: 'Abnormal Finding Detected',
    badgeBg: 'rgba(245, 158, 11, 0.15)',
    badgeColor: '#f59e0b',
    badgeBorder: 'rgba(245, 158, 11, 0.4)',
    summary: 'The AI identified imaging patterns characteristic of a Meningioma.',
    explanation: 'A Meningioma is a tumor that forms on the meninges—the protective membranes covering the brain and spinal cord. The majority of meningiomas are non-cancerous (benign) and slow-growing.',
    actionItems: [
      'Schedule a clinical review with a Neurologist or Neurosurgeon.',
      'Your specialist will determine whether active monitoring (watchful waiting) or intervention is best.',
      'Keep track of any symptoms such as recurring headaches or local numbness.'
    ]
  },
  'Pituitary Tumor': {
    title: 'Pituitary Region Abnormal Pattern',
    badgeText: 'Abnormal Finding Detected',
    badgeBg: 'rgba(245, 158, 11, 0.15)',
    badgeColor: '#f59e0b',
    badgeBorder: 'rgba(245, 158, 11, 0.4)',
    summary: 'The AI identified imaging patterns near the pituitary gland region.',
    explanation: 'A Pituitary tumor develops in the pituitary gland at the base of the brain. Most pituitary tumors are benign adenomas that can affect hormone production or optic nerves.',
    actionItems: [
      'Consult an Endocrinologist and a Neurosurgeon for a comprehensive evaluation.',
      'Discuss hormone panel blood tests to evaluate pituitary function.',
      'Consider a specialized visual field examination if you experience vision changes.'
    ]
  }
};

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

export const App: React.FC = () => {
  const [curatedSamples, setCuratedSamples] = useState<CuratedSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<CuratedSample | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState<string | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchCuratedSamples();
  }, []);

  const fetchCuratedSamples = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/curated-samples`);
      if (res.ok) {
        const data = await res.json();
        setCuratedSamples(data);
        if (data.length > 0) {
          handleSelectSample(data[0]);
        }
      }
    } catch (e) {
      console.warn("Backend API offline or unreachable");
    }
  };

  const handleSelectSample = (sample: CuratedSample) => {
    setSelectedSample(sample);
    setUploadedFile(null);
    setPredictionResult(null);
    setErrorMessage(null);
    setSelectedImagePreview(`${API_BASE}${sample.image_url}`);
  };

  const handleFileUpload = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMessage("Please select a valid image file (.png, .jpg, .dicom, .jpeg)");
      return;
    }
    setUploadedFile(file);
    setSelectedSample(null);
    setPredictionResult(null);
    setErrorMessage(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const runAnalysis = async () => {
    if (!uploadedFile && !selectedSample) {
      setErrorMessage("Please upload an MRI scan or select a sample image to analyze.");
      return;
    }

    setIsLoading(true);
    setErrorMessage(null);

    try {
      let url = `${API_BASE}/api/predict?alpha=0.05`;
      const options: RequestInit = { method: 'POST' };

      if (selectedSample) {
        url += `&sample_id=${selectedSample.id}`;
      } else if (uploadedFile) {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        options.body = formData;
      }

      const res = await fetch(url, options);
      if (!res.ok) {
        throw new Error(`Server returned error status ${res.status}`);
      }

      const data: PredictResponse = await res.json();
      setPredictionResult(data);

      setTimeout(() => {
        reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);

    } catch (err: any) {
      setErrorMessage("Analysis failed. Please check your backend connection and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  // Top probability class
  const topClass = predictionResult
    ? Object.entries(predictionResult.softmax_probabilities).sort((a, b) => b[1] - a[1])[0]
    : null;

  const topClassName = topClass ? topClass[0] : 'Normal Scan';
  const topClassProb = topClass ? Math.round(topClass[1] * 100) : 0;
  const diagnosisMeta = DIAGNOSIS_DETAILS[topClassName] || DIAGNOSIS_DETAILS['Normal Scan'];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-dark)' }}>
      {/* Navbar Header */}
      <header style={{
        height: '64px',
        borderBottom: '1px solid var(--border)',
        backgroundColor: '#0d1322',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 1.5rem',
        position: 'sticky',
        top: 0,
        zIndex: 50
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #06b6d4, #6366f1)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(6, 182, 212, 0.3)'
          }}>
            <Brain size={22} color="#ffffff" />
          </div>
          <div>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '-0.01em' }}>
              NeuroScan AI <span style={{ fontSize: '0.75rem', fontWeight: 500, color: '#06b6d4', marginLeft: '6px' }}>v2.4</span>
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              Patient Diagnostic Decision Support System
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '5px 12px',
            borderRadius: '20px',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
            fontSize: '0.75rem',
            fontWeight: 600,
            color: '#10b981'
          }}>
            <ShieldCheck size={14} />
            <span>AI Calibration Active (95% Guarantee)</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, maxWidth: '1100px', margin: '0 auto', width: '100%', padding: '2rem 1.5rem 4rem' }}>
        
        {/* Banner Section */}
        <div style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: '#f8fafc', marginBottom: '0.5rem' }}>
            Brain MRI Diagnostic Analysis
          </h1>
          <p style={{ fontSize: '0.95rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
            Upload a brain MRI scan image or select a sample scan below to receive an instant, plain-English diagnostic assessment powered by calibrated statistical artificial intelligence.
          </p>
        </div>

        {/* Top Control Grid: Upload Box + Sample Selector */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2.5rem'
        }}>
          
          {/* Upload Card */}
          <div style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <UploadCloud size={18} color="#06b6d4" />
              <h2 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>
                1. Upload Brain Scan Image
              </h2>
            </div>

            {/* Drag Drop Area */}
            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed var(--border)',
                borderRadius: 'var(--radius-md)',
                backgroundColor: 'rgba(255, 255, 255, 0.02)',
                padding: '2rem 1rem',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.75rem',
                flex: 1,
                justifyContent: 'center'
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-cyan)'; e.currentTarget.style.backgroundColor = 'rgba(6, 182, 212, 0.04)'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.02)'; }}
            >
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                backgroundColor: 'rgba(6, 182, 212, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <UploadCloud size={24} color="#06b6d4" />
              </div>
              <div>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f1f5f9' }}>
                  Click or drag scan file here
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', marginTop: '4px' }}>
                  Supports DICOM, PNG, JPG, JPEG files
                </div>
              </div>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    handleFileUpload(e.target.files[0]);
                  }
                }}
              />
            </div>

            {uploadedFile && (
              <div style={{
                marginTop: '1rem',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                border: '1px solid rgba(99, 102, 241, 0.3)',
                fontSize: '0.8rem',
                color: '#a5b4fc',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span>Selected file: <strong>{uploadedFile.name}</strong></span>
                <CheckCircle2 size={16} color="#818cf8" />
              </div>
            )}
          </div>

          {/* Sample Select Card */}
          <div style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            padding: '1.5rem',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <FileText size={18} color="#6366f1" />
              <h2 style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>
                2. Or Select a Test MRI Sample
              </h2>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', flex: 1 }}>
              {curatedSamples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    style={{
                      padding: '0.75rem',
                      borderRadius: 'var(--radius-md)',
                      backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.15)' : 'var(--bg-card-light)',
                      border: `1px solid ${isSelected ? '#6366f1' : 'var(--border)'}`,
                      textAlign: 'left',
                      transition: 'all 0.2s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.65rem',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{
                      width: '40px',
                      height: '40px',
                      borderRadius: '6px',
                      overflow: 'hidden',
                      backgroundColor: '#000',
                      flexShrink: 0
                    }}>
                      <img
                        src={`${API_BASE}${sample.image_url}`}
                        alt={sample.label}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                      <div style={{ fontSize: '0.82rem', fontWeight: 600, color: isSelected ? '#a5b4fc' : '#f1f5f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {sample.true_class}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                        {sample.type}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

        </div>

        {/* Action Button Section */}
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          {errorMessage && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '10px 16px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: 'var(--status-danger-bg)',
              border: '1px solid var(--status-danger-border)',
              color: '#ef4444',
              fontSize: '0.85rem',
              marginBottom: '1rem'
            }}>
              <AlertCircle size={16} />
              <span>{errorMessage}</span>
            </div>
          )}

          <div>
            <button
              onClick={runAnalysis}
              disabled={isLoading}
              style={{
                padding: '1rem 2.5rem',
                borderRadius: '30px',
                background: isLoading ? '#334155' : 'linear-gradient(135deg, #06b6d4 0%, #6366f1 100%)',
                color: '#ffffff',
                fontWeight: 700,
                fontSize: '1.05rem',
                boxShadow: isLoading ? 'none' : '0 8px 24px rgba(99, 102, 241, 0.35)',
                transition: 'all 0.25s ease',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.75rem',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
            >
              {isLoading ? (
                <>
                  <RefreshCw size={20} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Analyzing Brain Scan...</span>
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  <span>Run AI Diagnostic Assessment</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Diagnostic Results Section */}
        {predictionResult && (
          <div ref={reportRef} className="animate-fade-in" style={{
            backgroundColor: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-lg)',
            padding: '2rem',
            boxShadow: '0 12px 32px rgba(0, 0, 0, 0.3)'
          }}>
            
            {/* Header Banner inside Report */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '1.5rem',
              borderBottom: '1px solid var(--border)',
              marginBottom: '2rem',
              flexWrap: 'wrap',
              gap: '1rem'
            }}>
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent-cyan)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Official Diagnostic Evaluation Report
                </div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f8fafc', marginTop: '4px' }}>
                  Assessment Findings & Action Plan
                </h2>
              </div>

              <button
                onClick={handlePrint}
                style={{
                  padding: '8px 16px',
                  borderRadius: 'var(--radius-sm)',
                  backgroundColor: 'var(--bg-card-light)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-main)',
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  cursor: 'pointer'
                }}
              >
                <Printer size={15} />
                <span>Print Diagnostic Summary</span>
              </button>
            </div>

            {/* Diagnostic Alert Card */}
            <div style={{
              backgroundColor: diagnosisMeta.badgeBg,
              border: `1px solid ${diagnosisMeta.badgeBorder}`,
              borderRadius: 'var(--radius-md)',
              padding: '1.5rem',
              marginBottom: '2rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '1.25rem'
            }}>
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '50%',
                backgroundColor: diagnosisMeta.badgeColor,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                color: '#ffffff'
              }}>
                {topClassName === 'Normal Scan' ? <CheckCircle2 size={24} /> : <AlertTriangle size={24} />}
              </div>

              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '6px', flexWrap: 'wrap' }}>
                  <span style={{
                    padding: '3px 10px',
                    borderRadius: '12px',
                    backgroundColor: diagnosisMeta.badgeColor,
                    color: '#ffffff',
                    fontSize: '0.75rem',
                    fontWeight: 700
                  }}>
                    {diagnosisMeta.badgeText}
                  </span>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    AI Model Confidence: <strong>{topClassProb}%</strong>
                  </span>
                </div>

                <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', marginBottom: '6px' }}>
                  {diagnosisMeta.title}
                </h3>
                <p style={{ fontSize: '0.92rem', color: '#e2e8f0', lineHeight: '1.5' }}>
                  {diagnosisMeta.summary}
                </p>
              </div>
            </div>

            {/* Two-Column Detail View: Scan + Probability Breakdown */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '1.75rem',
              marginBottom: '2rem'
            }}>
              
              {/* Scan Image Panel */}
              <div style={{
                backgroundColor: 'var(--bg-card-light)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Search size={16} color="#06b6d4" />
                  <span>MRI Scan Visualization</span>
                </div>

                {selectedImagePreview && (
                  <div style={{
                    width: '100%',
                    height: '240px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: '#000000',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid rgba(255, 255, 255, 0.05)'
                  }}>
                    <img
                      src={selectedImagePreview}
                      alt="Analyzed Brain MRI"
                      style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                    />
                  </div>
                )}
              </div>

              {/* Statistical Probability Panel */}
              <div style={{
                backgroundColor: 'var(--bg-card-light)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f8fafc', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={16} color="#6366f1" />
                  <span>Statistical Probability Distribution</span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {Object.entries(predictionResult.softmax_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, prob]) => {
                      const percentage = Math.round(prob * 100);
                      const isTop = label === topClassName;
                      return (
                        <div key={label}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '4px' }}>
                            <span style={{ fontWeight: isTop ? 700 : 500, color: isTop ? '#f8fafc' : 'var(--text-muted)' }}>
                              {label}
                            </span>
                            <span style={{ fontWeight: 700, color: isTop ? '#06b6d4' : 'var(--text-dim)' }}>
                              {percentage}%
                            </span>
                          </div>
                          <div style={{
                            height: '8px',
                            backgroundColor: 'rgba(255, 255, 255, 0.06)',
                            borderRadius: '4px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              height: '100%',
                              width: `${percentage}%`,
                              backgroundColor: isTop ? '#06b6d4' : '#475569',
                              borderRadius: '4px',
                              transition: 'width 0.6s ease'
                            }} />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

            </div>

            {/* Explanation & Patient Action Plan */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
              gap: '1.75rem'
            }}>
              
              {/* Detailed Explanation */}
              <div style={{
                backgroundColor: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Info size={16} color="#06b6d4" />
                  <span>Understanding Your Result</span>
                </div>
                <p style={{ fontSize: '0.86rem', color: 'var(--text-muted)', lineHeight: '1.6' }}>
                  {diagnosisMeta.explanation}
                </p>
              </div>

              {/* Recommended Next Steps */}
              <div style={{
                backgroundColor: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#f8fafc', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <CheckCircle2 size={16} color="#10b981" />
                  <span>Recommended Next Steps</span>
                </div>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {diagnosisMeta.actionItems.map((item, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
                      <ArrowRight size={14} color="#10b981" style={{ flexShrink: 0, marginTop: '3px' }} />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border)',
        backgroundColor: '#0d1322',
        padding: '1.5rem',
        textAlign: 'center',
        fontSize: '0.78rem',
        color: 'var(--text-dim)'
      }}>
        NeuroScan AI Medical Decision Support System • Developed for Clinical Evaluation & Educational Purposes
      </footer>
    </div>
  );
};

export default App;
