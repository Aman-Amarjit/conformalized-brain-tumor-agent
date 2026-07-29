import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Database, Terminal } from 'lucide-react';
import { MRIDiagnosticCanvas } from './components/MRIDiagnosticCanvas';
import { ConformalSetBreakdown } from './components/ConformalSetBreakdown';
import { CoverageQuantileTuner } from './components/CoverageQuantileTuner';
import { TriageAbstentionBanner } from './components/TriageAbstentionBanner';
import { DiagnosticAuditTrail, type AuditItem } from './components/DiagnosticAuditTrail';

const API_BASE = 'http://localhost:8000';

interface CuratedSample {
  id: string;
  filename: string;
  type: string;
  label: string;
  true_class: string;
  expected_set: string[];
  image_url: string;
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

export const App: React.FC = () => {
  const [curatedSamples, setCuratedSamples] = useState<CuratedSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<CuratedSample | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [selectedImagePreview, setSelectedImagePreview] = useState<string | null>(null);

  const [alpha, setAlpha] = useState<number>(0.05);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [auditHistory, setAuditHistory] = useState<AuditItem[]>([]);

  // Fetch Curated Samples & Audit History on mount
  useEffect(() => {
    fetchCuratedSamples();
    fetchAuditHistory();
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
      console.warn("FastAPI backend not reached yet, using local fallback state.");
    }
  };

  const fetchAuditHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/history`);
      if (res.ok) {
        const data = await res.json();
        setAuditHistory(data);
      }
    } catch (e) {
      // Quiet catch
    }
  };

  const handleSelectSample = (sample: CuratedSample) => {
    setSelectedSample(sample);
    setUploadedFile(null);
    setSelectedImagePreview(`${API_BASE}${sample.image_url}`);
  };

  const handleFileUpload = (file: File) => {
    setUploadedFile(file);
    setSelectedSample(null);
    const reader = new FileReader();
    reader.onload = (e) => {
      setSelectedImagePreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);
  };

  const runInference = async () => {
    setIsLoading(true);
    try {
      let url = `${API_BASE}/api/predict?alpha=${alpha}`;
      let formData = new FormData();

      if (selectedSample) {
        url += `&sample_id=${selectedSample.id}`;
      } else if (uploadedFile) {
        formData.append('file', uploadedFile);
      }

      const options: RequestInit = {
        method: 'POST',
      };
      if (uploadedFile) {
        options.body = formData;
      }

      const res = await fetch(url, options);
      if (res.ok) {
        const data: PredictResponse = await res.json();
        setPredictionResult(data);
        fetchAuditHistory();
      }
    } catch (err) {
      console.error("Error during inference:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // Re-run prediction when alpha slider moves if result exists
  const handleAlphaChange = (newAlpha: number) => {
    setAlpha(newAlpha);
  };

  useEffect(() => {
    if (predictionResult) {
      runInference();
    }
  }, [alpha]);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', background: '#000000' }}>
      {/* Top PACS Workstation Header Bar - Pure Black */}
      <header
        style={{
          height: '52px',
          borderBottom: '1px solid #172233',
          background: '#000000',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 1.25rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ background: '#06b6d4', padding: '5px', borderRadius: '3px', display: 'flex' }}>
            <Activity size={18} color="#000000" />
          </div>
          <div>
            <h1 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc', letterSpacing: '0.02em' }} className="mono">
              CLINICAL DECISION SUPPORT // CONFORMAL BRAIN TUMOR ENGINE
            </h1>
            <span style={{ fontSize: '0.65rem', color: '#8493a8', display: 'block', marginTop: '-2px' }} className="mono">
              STATISTICAL PREDICTION SETS & SAFETY ABSTENTION TRIAGE
            </span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }} className="mono">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.7rem', color: '#8493a8', background: '#000000', padding: '3px 8px', borderRadius: '3px', border: '1px solid #172233' }}>
            <Terminal size={12} color="#06b6d4" />
            <span>RAD-CON-04</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.7rem', color: '#8493a8', background: '#000000', padding: '3px 8px', borderRadius: '3px', border: '1px solid #172233' }}>
            <Database size={12} color="#10b981" />
            <span>KAGGLE / SONALI PRIYADARSHINI</span>
          </div>

          <div className="pacs-badge pacs-badge-info">
            <ShieldCheck size={12} />
            <span>MAPIE CALIBRATED</span>
          </div>
        </div>
      </header>

      {/* Main Clinical Diagnostic Console - Pure Black */}
      <main
        style={{
          flex: 1,
          maxHeight: 'calc(100vh - 52px)',
          overflowY: 'auto',
          padding: '1.25rem',
          display: 'grid',
          gridTemplateColumns: '400px 1fr',
          gap: '1.25rem',
          background: '#000000',
        }}
      >
        {/* Left Column: DICOM Canvas & Conformal Tuner Bar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <MRIDiagnosticCanvas
            selectedImage={selectedImagePreview}
            curatedSamples={curatedSamples}
            selectedSampleId={selectedSample?.id || null}
            onSelectSample={handleSelectSample}
            onFileUpload={handleFileUpload}
            isLoading={isLoading}
            onRunInference={runInference}
          />

          <CoverageQuantileTuner
            alpha={alpha}
            onAlphaChange={handleAlphaChange}
            quantileApplied={predictionResult?.quantile_applied}
          />
        </div>

        {/* Right Column: Triage Banner, Conformal Breakdown Matrix, RIS Log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {predictionResult && (
            <TriageAbstentionBanner
              isConfident={predictionResult.is_confident}
              triageMessage={predictionResult.triage_message}
              predictionSet={predictionResult.prediction_set}
            />
          )}

          <ConformalSetBreakdown result={predictionResult} />

          <DiagnosticAuditTrail history={auditHistory} />
        </div>
      </main>
    </div>
  );
};

export default App;
