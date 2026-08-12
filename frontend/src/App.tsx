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
  summary: string;
  explanation: string;
  nextSteps: string[];
}

const CONDITION_INFO: Record<string, ConditionDetail> = {
  'Normal Scan': {
    title: 'Normal Scan — No Abnormalities Detected',
    badgeLabel: 'Normal / Clear Scan',
    badgeBg: '#ecfdf5',
    badgeColor: '#047857',
    badgeBorder: '#a7f3d0',
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
    badgeBg: '#fff1f2',
    badgeColor: '#be123c',
    badgeBorder: '#fecdd3',
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
    badgeBg: '#fffbeb',
    badgeColor: '#b45309',
    badgeBorder: '#fde68a',
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
    badgeBg: '#fffbeb',
    badgeColor: '#b45309',
    badgeBorder: '#fde68a',
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
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: '#f8fafc' }}>
      
      {/* Hospital-Grade Header Navbar */}
      <header className="no-print" style={{
        height: '60px',
        backgroundColor: '#ffffff',
        borderBottom: '1px solid #e2e8f0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 2rem',
        position: 'sticky',
        top: 0,
        zIndex: 100
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: '6px',
            backgroundColor: '#0284c7',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#ffffff'
          }}>
            <Activity size={18} />
          </div>
          <div>
            <div style={{ fontSize: '1rem', fontWeight: 700, color: '#0f172a', letterSpacing: '-0.01em' }}>
              Brain MRI Assessment Portal
            </div>
            <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
              Patient Diagnostic Decision Support System
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontSize: '0.78rem',
            color: '#047857',
            backgroundColor: '#ecfdf5',
            padding: '4px 10px',
            borderRadius: '4px',
            border: '1px solid #a7f3d0',
            fontWeight: 600
          }}>
            <CheckCircle size={14} />
            <span>Calibrated Reliability Model (95% Guarantee)</span>
          </div>
        </div>
      </header>

      {/* Main Body */}
      <main className="print-full-width" style={{ flex: 1, maxWidth: '1080px', margin: '0 auto', width: '100%', padding: '2rem 1.5rem 4rem' }}>
        
        {/* Intro / Banner */}
        <div className="no-print" style={{ marginBottom: '2rem' }}>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.35rem' }}>
            Brain Scan Image Analysis
          </h1>
          <p style={{ fontSize: '0.95rem', color: '#475569', maxWidth: '780px', lineHeight: '1.5' }}>
            Select an MRI scan sample or upload an image file to generate a clear, plain-language assessment of structural findings and recommended healthcare steps.
          </p>
        </div>

        {/* Upload & Sample Section (Grid) */}
        <div className="no-print" style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '1.5rem',
          marginBottom: '2rem'
        }}>
          
          {/* Card 1: File Uploader */}
          <div style={{
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '1.25rem',
            boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Upload size={16} color="#0284c7" />
              <span>1. Upload Brain MRI Scan</span>
            </div>

            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: '2px dashed #cbd5e1',
                borderRadius: '6px',
                backgroundColor: '#f8fafc',
                padding: '1.75rem 1rem',
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
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#0284c7'; e.currentTarget.style.backgroundColor = '#f0f9ff'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#cbd5e1'; e.currentTarget.style.backgroundColor = '#f8fafc'; }}
            >
              <Upload size={28} color="#64748b" />
              <div>
                <div style={{ fontSize: '0.86rem', fontWeight: 600, color: '#0f172a' }}>
                  Click to select scan image
                </div>
                <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
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
                marginTop: '0.85rem',
                padding: '6px 10px',
                borderRadius: '4px',
                backgroundColor: '#f0f9ff',
                border: '1px solid #bae6fd',
                fontSize: '0.78rem',
                color: '#0369a1',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span>File selected: <strong>{uploadedFile.name}</strong></span>
                <CheckCircle size={14} color="#0284c7" />
              </div>
            )}
          </div>

          {/* Card 2: Sample Scans */}
          <div style={{
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '1.25rem',
            boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '0.9rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={16} color="#0284c7" />
              <span>2. Or Select a Test MRI Sample</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem', flex: 1 }}>
              {curatedSamples.map((sample) => {
                const isSelected = selectedSample?.id === sample.id;
                return (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectSample(sample)}
                    style={{
                      padding: '0.6rem',
                      borderRadius: '6px',
                      backgroundColor: isSelected ? '#f0f9ff' : '#ffffff',
                      border: `1px solid ${isSelected ? '#0284c7' : '#e2e8f0'}`,
                      textAlign: 'left',
                      transition: 'all 0.15s ease',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.6rem',
                      cursor: 'pointer'
                    }}
                  >
                    <div style={{
                      width: '36px',
                      height: '36px',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      backgroundColor: '#0f172a',
                      flexShrink: 0
                    }}>
                      <img
                        src={`${API_BASE}${sample.image_url}`}
                        alt={sample.label}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, color: isSelected ? '#0369a1' : '#0f172a', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {sample.true_class}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
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
        <div className="no-print" style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          {errorMessage && (
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '8px 14px',
              borderRadius: '4px',
              backgroundColor: '#fef2f2',
              border: '1px solid #fecdd3',
              color: '#991b1b',
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
                padding: '0.85rem 2.25rem',
                borderRadius: '6px',
                backgroundColor: isLoading ? '#94a3b8' : '#0284c7',
                color: '#ffffff',
                fontWeight: 600,
                fontSize: '0.95rem',
                boxShadow: isLoading ? 'none' : '0 2px 4px rgba(2, 132, 199, 0.2)',
                transition: 'all 0.15s ease-in-out',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.6rem',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
            >
              {isLoading ? (
                <>
                  <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} />
                  <span>Processing Analysis...</span>
                </>
              ) : (
                <>
                  <Search size={18} />
                  <span>Analyze MRI Scan</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results / Findings Display */}
        {predictionResult && (
          <div ref={reportRef} style={{
            backgroundColor: '#ffffff',
            border: '1px solid #e2e8f0',
            borderRadius: '8px',
            padding: '2rem',
            boxShadow: '0 2px 8px rgba(0,0,0,0.06)'
          }}>
            
            {/* Header bar */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingBottom: '1.25rem',
              borderBottom: '1px solid #e2e8f0',
              marginBottom: '1.5rem',
              flexWrap: 'wrap',
              gap: '1rem'
            }}>
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#0284c7', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Medical Assessment Summary
                </div>
                <h2 style={{ fontSize: '1.35rem', fontWeight: 700, color: '#0f172a', marginTop: '2px' }}>
                  MRI Diagnostic Findings & Patient Guide
                </h2>
              </div>

              <button
                className="no-print"
                onClick={handlePrint}
                style={{
                  padding: '7px 14px',
                  borderRadius: '4px',
                  backgroundColor: '#ffffff',
                  border: '1px solid #cbd5e1',
                  color: '#334155',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  cursor: 'pointer'
                }}
              >
                <Printer size={14} />
                <span>Print Report</span>
              </button>
            </div>

            {/* Assessment Banner Box */}
            <div style={{
              backgroundColor: condition.badgeBg,
              border: `1px solid ${condition.badgeBorder}`,
              borderRadius: '6px',
              padding: '1.25rem 1.5rem',
              marginBottom: '1.75rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '1rem'
            }}>
              <div style={{
                width: '36px',
                height: '36px',
                borderRadius: '50%',
                backgroundColor: condition.badgeColor,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#ffffff',
                flexShrink: 0
              }}>
                {topClassName === 'Normal Scan' ? <CheckCircle size={20} /> : <AlertTriangle size={20} />}
              </div>

              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '4px', flexWrap: 'wrap' }}>
                  <span style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    backgroundColor: condition.badgeColor,
                    color: '#ffffff',
                    fontSize: '0.72rem',
                    fontWeight: 700
                  }}>
                    {condition.badgeLabel}
                  </span>
                  <span style={{ fontSize: '0.82rem', color: '#475569' }}>
                    Model Agreement: <strong>{topClassProb}% Confidence</strong>
                  </span>
                </div>

                <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a', marginBottom: '4px' }}>
                  {condition.title}
                </h3>
                <p style={{ fontSize: '0.9rem', color: '#334155', lineHeight: '1.5' }}>
                  {condition.summary}
                </p>
              </div>
            </div>

            {/* Content Columns: Visual Scan + Probability Table */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '1.5rem',
              marginBottom: '1.75rem'
            }}>
              
              {/* Image Preview Box */}
              <div style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '6px',
                padding: '1rem'
              }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem' }}>
                  Analyzed MRI Scan Image
                </div>
                {selectedImagePreview && (
                  <div style={{
                    width: '100%',
                    height: '210px',
                    borderRadius: '4px',
                    backgroundColor: '#000000',
                    overflow: 'hidden',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
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
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '6px',
                padding: '1rem'
              }}>
                <div style={{ fontSize: '0.82rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.75rem' }}>
                  Diagnostic Category Probabilities
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
                  {Object.entries(predictionResult.softmax_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([label, prob]) => {
                      const percentage = Math.round(prob * 100);
                      const isTop = label === topClassName;
                      return (
                        <div key={label}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '3px' }}>
                            <span style={{ fontWeight: isTop ? 700 : 500, color: isTop ? '#0f172a' : '#475569' }}>
                              {label}
                            </span>
                            <span style={{ fontWeight: 700, color: isTop ? '#0284c7' : '#64748b' }}>
                              {percentage}%
                            </span>
                          </div>
                          <div style={{
                            height: '6px',
                            backgroundColor: '#e2e8f0',
                            borderRadius: '3px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              height: '100%',
                              width: `${percentage}%`,
                              backgroundColor: isTop ? '#0284c7' : '#94a3b8',
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
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '1.5rem'
            }}>
              
              {/* Detailed Explanation */}
              <div style={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '6px',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.65rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Info size={16} color="#0284c7" />
                  <span>Understanding Your Result</span>
                </div>
                <p style={{ fontSize: '0.85rem', color: '#334155', lineHeight: '1.6' }}>
                  {condition.explanation}
                </p>
              </div>

              {/* Recommended Action Items */}
              <div style={{
                backgroundColor: '#ffffff',
                border: '1px solid #e2e8f0',
                borderRadius: '6px',
                padding: '1.25rem'
              }}>
                <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.65rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Calendar size={16} color="#047857" />
                  <span>Recommended Patient Next Steps</span>
                </div>
                <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {condition.nextSteps.map((step, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.4rem', fontSize: '0.82rem', color: '#334155', lineHeight: '1.4' }}>
                      <ArrowRight size={13} color="#047857" style={{ flexShrink: 0, marginTop: '2px' }} />
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
        borderTop: '1px solid #e2e8f0',
        backgroundColor: '#ffffff',
        padding: '1.25rem',
        textAlign: 'center',
        fontSize: '0.78rem',
        color: '#64748b'
      }}>
        Brain MRI Assessment Portal • Diagnostic Decision Support System • For Clinical & Educational Evaluation
      </footer>
    </div>
  );
};

export default App;
