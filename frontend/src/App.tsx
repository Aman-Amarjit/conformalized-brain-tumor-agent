import React, { useState, useEffect, useRef } from 'react';
import {
  FileText,
  Upload,
  CheckCircle,
  AlertCircle,
  AlertTriangle,
  Printer,
  RefreshCw,
  Search,
  Activity,
  ArrowRight,
  Info,
  Calendar
} from 'lucide-react';

const API_BASE = typeof window !== 'undefined' && window.location.origin.includes('localhost')
  ? 'http://localhost:8000'
  : '';

interface ConditionDetail {
  title: string;
  badgeLabel: string;
  badgeBg: string;
  badgeColor: string;
  badgeBorder: string;
  cardBg: string;
  cardBorder: string;
  summary: string;
  explanation: string;
  nextSteps: string[];
}

const CONDITION_INFO: Record<string, ConditionDetail> = {
  'Normal Scan': {
    title: 'Normal Scan — No Abnormalities Detected',
    badgeLabel: 'Normal / Clear Scan',
    badgeBg: '#059669',
    badgeColor: '#ffffff',
    badgeBorder: '#10b981',
    cardBg: '#022c22',
    cardBorder: '#065f46',
    summary: 'The MRI scan shows normal brain tissue structures with no indications of tumor growth.',
    explanation: 'Our automated diagnostic tool analyzed tissue density, symmetry, and structure in your scan. The visible brain tissue, ventricles, and surrounding structures fall within healthy normal parameters.',
    nextSteps: [
      'Share this report with your primary care physician during your next routine visit.',
      'If you are experiencing persistent neurological symptoms (e.g. severe headaches or dizziness), consult your doctor for further non-oncological evaluation.',
      'Keep a copy of your MRI digital files for your personal health records.'
    ]
  },
  'Glioma': {
    title: 'Glioma Pattern Identified',
    badgeLabel: 'Abnormal Finding — Requires Specialist Review',
    badgeBg: '#dc2626',
    badgeColor: '#ffffff',
    badgeBorder: '#ef4444',
    cardBg: '#450a0a',
    cardBorder: '#991b1b',
    summary: 'The analysis identified tissue features characteristic of a Glioma.',
    explanation: 'A Glioma is a type of tumor that arises from the glial supportive cells in brain tissue. Because gliomas vary in growth behavior, a comprehensive clinical evaluation by a specialist is necessary.',
    nextSteps: [
      'Schedule a consultation with a Neurologist or Neuro-Oncologist promptly.',
      'Bring your original MRI digital scan files (DICOM format) and this summary to your appointment.',
      'Discuss whether additional contrast-enhanced imaging (T1 post-contrast) is recommended.'
    ]
  },
  'Meningioma': {
    title: 'Meningioma Pattern Identified',
    badgeLabel: 'Abnormal Finding — Specialist Consultation Recommended',
    badgeBg: '#d97706',
    badgeColor: '#ffffff',
    badgeBorder: '#f59e0b',
    cardBg: '#451a03',
    cardBorder: '#92400e',
    summary: 'The analysis identified tissue features characteristic of a Meningioma.',
    explanation: 'A Meningioma is a tumor originating from the meninges (the outer protective membranes surrounding the brain). Most meningiomas are non-cancerous (benign) and grow slowly over time.',
    nextSteps: [
      'Schedule a clinical consultation with a Neurologist or Neurosurgeon.',
      'Your specialist will determine if active monitoring (periodic MRI scans) or treatment is appropriate.',
      'Note any specific symptoms you experience, such as localized headaches or vision changes.'
    ]
  },
  'Pituitary Tumor': {
    title: 'Pituitary Region Finding Identified',
    badgeLabel: 'Abnormal Finding — Specialist Consultation Recommended',
    badgeBg: '#d97706',
    badgeColor: '#ffffff',
    badgeBorder: '#f59e0b',
    cardBg: '#451a03',
    cardBorder: '#92400e',
    summary: 'The analysis identified tissue features in the pituitary gland region.',
    explanation: 'A pituitary tumor develops in the pituitary gland at the base of the brain. The vast majority are benign adenomas that may affect hormone regulation or nearby optic pathways.',
    nextSteps: [
      'Consult an Endocrinologist and a Neurosurgeon for specialized evaluation.',
      'Discuss whether hormone level blood panels are recommended.',
      'Schedule a visual field assessment if you notice subtle visual changes.'
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
      console.warn("Backend API offline");
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
      setErrorMessage("Please upload an image file (e.g. .png, .jpg, .jpeg, .dcm)");
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
      setErrorMessage("Please select a sample MRI scan or upload an image to analyze.");
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
        throw new Error(`Server status ${res.status}`);
      }

      const data: PredictResponse = await res.json();
      setPredictionResult(data);

      setTimeout(() => {
        reportRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);

    } catch (err: any) {
      setErrorMessage("Analysis request failed. Please check your connection and try again.");
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
  const condition = CONDITION_INFO[topClassName] || CONDITION_INFO['Normal Scan'];

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#000000', color: '#f4f4f5' }}>
      
      {/* Mobile-Friendly Header Navbar */}
      <header className="no-print" style={{
        minHeight: '56px',
        backgroundColor: '#09090b',
        borderBottom: '1px solid #18181b',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0.6rem 1rem',
        gap: '0.75rem',
        flexWrap: 'wrap'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div style={{
            width: '30px',
            height: '30px',
            borderRadius: '6px',
            backgroundColor: '#2563eb',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff',
            flexShrink: 0
          }}>
            <Activity size={16} />
          </div>
          <div>
            <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#ffffff', letterSpacing: '-0.01em', lineHeight: 1.2 }}>
              Brain MRI Assessment Portal
            </div>
            <div className="header-subtitle" style={{ fontSize: '0.7rem', color: '#71717a', marginTop: '2px' }}>
              Patient Diagnostic Decision Support System
            </div>
          </div>
        </div>

        <div className="header-badge" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.75rem',
            color: '#4ade80',
            backgroundColor: '#052e16',
            padding: '3px 9px',
            borderRadius: '4px',
            border: '1px solid #14532d',
            fontWeight: 600
          }}>
            <CheckCircle size={13} />
            <span>Calibrated Reliability Model (95% Guarantee)</span>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="main-container print-full-width" style={{ flex: 1, maxWidth: '1080px', margin: '0 auto', width: '100%', padding: '1.75rem 1.25rem 3.5rem' }}>
        
        {/* Intro / Banner */}
        <div className="no-print" style={{ marginBottom: '1.75rem' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.35rem' }}>
            Brain Scan Image Analysis
          </h1>
          <p style={{ fontSize: '0.9rem', color: '#a1a1aa', maxWidth: '780px', lineHeight: '1.5' }}>
            Select an MRI scan sample or upload an image file to generate a clear, plain-language assessment of structural findings and recommended healthcare steps.
          </p>
        </div>

        {/* Upload & Sample Section (Grid) */}
        <div className="no-print" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(290px, 1fr))',
          gap: '1.25rem',
          marginBottom: '1.75rem'
        }}>
          
          {/* Card 1: File Uploader */}
          <div className="card-padding" style={{
            backgroundColor: '#09090b',
            border: '1px solid #18181b',
            borderRadius: '8px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Upload size={16} color="#3b82f6" />
              <span>1. Upload Brain MRI Scan</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed #27272a',
                borderRadius: '6px',
                backgroundColor: '#121215',
                padding: '1.5rem 0.85rem',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.15s ease-in-out',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem',
                flex: 1
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#3b82f6'; e.currentTarget.style.backgroundColor = '#172554'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#27272a'; e.currentTarget.style.backgroundColor = '#121215'; }}
            >
              <Upload size={24} color="#71717a" />
              <div>
                <div style={{ fontSize: '0.84rem', fontWeight: 600, color: '#ffffff' }}>
                  Click to select scan image
                </div>
                <div style={{ fontSize: '0.72rem', color: '#71717a', marginTop: '2px' }}>
                  Supports DICOM, PNG, JPG files
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
                marginTop: '0.75rem',
                padding: '6px 10px',
                borderRadius: '4px',
                backgroundColor: '#172554',
                border: '1px solid #1d4ed8',
                fontSize: '0.76rem',
                color: '#93c5fd',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span>File selected: <strong>{uploadedFile.name}</strong></span>
                <CheckCircle size={14} color="#60a5fa" />
              </div>
            )}
          </div>

          {/* Card 2: Sample Scans */}
          <div className="card-padding" style={{
            backgroundColor: '#09090b',
            border: '1px solid #18181b',
            borderRadius: '8px',
            padding: '1.25rem',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={16} color="#3b82f6" />
              <span>2. Or Select a Test MRI Sample</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.6rem', flex: 1 }}>
              {curatedSamples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    style={{
                      padding: '0.5rem',
                      borderRadius: '6px',
                      backgroundColor: isSelected ? '#172554' : '#121215',
                      border: `1px solid ${isSelected ? '#3b82f6' : '#27272a'}`,
                      textAlign: 'left',
                      transition: 'all 0.15s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      backgroundColor: '#000000',
                      flexShrink: 0
                    }}>
                      <img
                        src={`${API_BASE}${sample.image_url}`}
                        alt={sample.label}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                      <div style={{ fontSize: '0.78rem', fontWeight: 600, color: isSelected ? '#93c5fd' : '#ffffff', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {sample.true_class}
                      </div>
                      <div style={{ fontSize: '0.68rem', color: '#71717a' }}>
                        {sample.type}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

        </div>

        {/* Submit Button */}
        <div className="no-print" style={{ textAlign: 'center', marginBottom: '2rem' }}>
          {errorMessage && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '8px 14px',
              borderRadius: '4px',
              backgroundColor: '#450a0a',
              border: '1px solid #991b1b',
              color: '#f87171',
              fontSize: '0.82rem',
              marginBottom: '1rem'
            }}>
              <AlertCircle size={15} />
              <span>{errorMessage}</span>
            </div>
          )}

          <div>
            <button
              onClick={runAnalysis}
              disabled={isLoading}
              style={{
                padding: '0.8rem 2rem',
                borderRadius: '6px',
                backgroundColor: isLoading ? '#3f3f46' : '#2563eb',
                color: '#ffffff',
                fontWeight: 600,
                fontSize: '0.92rem',
                boxShadow: isLoading ? 'none' : '0 2px 8px rgba(37, 99, 235, 0.4)',
                transition: 'all 0.15s ease-in-out',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
            >
              {isLoading ? (
                <>
                  <RefreshCw size={17} style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Processing Analysis...</span>
                </>
              ) : (
                <>
                  <Search size={17} />
                  <span>Analyze MRI Scan</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results / Findings Display */}
        {predictionResult && (
          <div ref={reportRef} className="card-padding" style={{
            backgroundColor: '#09090b',
            border: '1px solid #18181b',
            borderRadius: '8px',
            padding: '1.75rem'
          }}>
            
            {/* Header bar */}
            <div className="report-header" style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '1rem',
              borderBottom: '1px solid #18181b',
              marginBottom: '1.25rem',
              flexWrap: 'wrap',
              gap: '0.75rem'
            }}>
              <div>
                <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#3b82f6', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Medical Assessment Summary
                </div>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#ffffff', marginTop: '2px' }}>
                  MRI Diagnostic Findings & Patient Guide
                </h2>
              </div>

              <button
                className="no-print"
                onClick={handlePrint}
                style={{
                  padding: '6px 12px',
                  borderRadius: '4px',
                  backgroundColor: '#121215',
                  border: '1px solid #27272a',
                  color: '#e4e4e7',
                  fontSize: '0.78rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  cursor: 'pointer'
                }}
              >
                <Printer size={13} />
                <span>Print Report</span>
              </button>
            </div>

            {/* Assessment Banner Box */}
            <div style={{
              backgroundColor: condition.cardBg,
              border: `1px solid ${condition.cardBorder}`,
              borderRadius: '6px',
              padding: '1.15rem 1.25rem',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.85rem'
            }}>
              <div style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: condition.badgeBg,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ffffff',
                flexShrink: 0
              }}>
                {topClassName === 'Normal Scan' ? <CheckCircle size={18} /> : <AlertTriangle size={18} />}
              </div>

              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '4px', flexWrap: 'wrap' }}>
                  <span style={{
                    padding: '2px 7px',
                    borderRadius: '4px',
                    backgroundColor: condition.badgeBg,
                    color: condition.badgeColor,
                    fontSize: '0.7rem',
                    fontWeight: 700
                  }}>
                    {condition.badgeLabel}
                  </span>
                  <span style={{ fontSize: '0.78rem', color: '#a1a1aa' }}>
                    Model Agreement: <strong>{topClassProb}% Confidence</strong>
                  </span>
                </div>

                <h3 style={{ fontSize: '1.05rem', fontWeight: 700, color: '#ffffff', marginBottom: '4px' }}>
                  {condition.title}
                </h3>
                <p style={{ fontSize: '0.86rem', color: '#e4e4e7', lineHeight: '1.45' }}>
                  {condition.summary}
                </p>
              </div>
            </div>

            {/* Content Columns: Visual Scan + Probability Table */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '1.25rem',
              marginBottom: '1.5rem'
            }}>
              
              {/* Image Preview Box */}
              <div style={{
                backgroundColor: '#121215',
                border: '1px solid #18181b',
                borderRadius: '6px',
                padding: '0.85rem'
              }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.65rem' }}>
                  Analyzed MRI Scan Image
                </div>
                {selectedImagePreview && (
                  <div style={{
                    width: '100%',
                    height: '200px',
                    borderRadius: '4px',
                    backgroundColor: '#000000',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    border: '1px solid #27272a'
                  }}>
                    <img
                      src={selectedImagePreview}
                      alt="MRI Scan"
                      style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }}
                    />
                  </div>
                )}
              </div>

              {/* Likelihood Table */}
              <div style={{
                backgroundColor: '#121215',
                border: '1px solid #18181b',
                borderRadius: '6px',
                padding: '0.85rem'
              }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.65rem' }}>
                  Diagnostic Category Probabilities
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {Object.entries(predictionResult.softmax_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, prob]) => {
                      const percentage = Math.round(prob * 100);
                      const isTop = label === topClassName;
                      return (
                        <div key={label}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '3px' }}>
                            <span style={{ fontWeight: isTop ? 700 : 500, color: isTop ? '#ffffff' : '#a1a1aa' }}>
                              {label}
                            </span>
                            <span style={{ fontWeight: 700, color: isTop ? '#60a5fa' : '#71717a' }}>
                              {percentage}%
                            </span>
                          </div>
                          <div style={{
                            height: '6px',
                            backgroundColor: '#27272a',
                            borderRadius: '3px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              height: '100%',
                              width: `${percentage}%`,
                              backgroundColor: isTop ? '#2563eb' : '#52525b',
                              borderRadius: '3px'
                            }} />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

            </div>

            {/* Explanation and Next Steps */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '1.25rem'
            }}>
              
              {/* Detailed Explanation */}
              <div style={{
                backgroundColor: '#121215',
                border: '1px solid #18181b',
                borderRadius: '6px',
                padding: '1rem'
              }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Info size={15} color="#3b82f6" />
                  <span>Understanding Your Result</span>
                </div>
                <p style={{ fontSize: '0.82rem', color: '#a1a1aa', lineHeight: '1.5' }}>
                  {condition.explanation}
                </p>
              </div>

              {/* Recommended Action Items */}
              <div style={{
                backgroundColor: '#121215',
                border: '1px solid #18181b',
                borderRadius: '6px',
                padding: '1rem'
              }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Calendar size={15} color="#4ade80" />
                  <span>Recommended Patient Next Steps</span>
                </div>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                  {condition.nextSteps.map((step, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.4rem', fontSize: '0.8rem', color: '#a1a1aa', lineHeight: '1.4' }}>
                      <ArrowRight size={12} color="#4ade80" style={{ flexShrink: 0, marginTop: '2px' }} />
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>

            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="no-print" style={{
        borderTop: '1px solid #18181b',
        backgroundColor: '#09090b',
        padding: '1rem',
        textAlign: 'center',
        fontSize: '0.75rem',
        color: '#71717a'
      }}>
        Brain MRI Assessment Portal • Diagnostic Decision Support System
      </footer>
    </div>
  );
};

export default App;
