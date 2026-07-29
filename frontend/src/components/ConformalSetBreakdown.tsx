import React, { useState } from 'react';
import { Activity, ShieldCheck, Check, AlertTriangle, Info, Stethoscope, ChevronDown, ChevronUp } from 'lucide-react';

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

interface Props {
  result: PredictResponse | null;
}

export const ConformalSetBreakdown: React.FC<Props> = ({ result }) => {
  const [showDoctorHelp, setShowDoctorHelp] = useState(false);

  if (!result) {
    return (
      <div className="pacs-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: '280px', color: '#52525b', textAlign: 'center', background: '#000000' }}>
        <Activity size={32} color="#27272a" style={{ marginBottom: '0.75rem' }} />
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#a1a1aa' }} className="mono">
          NO CONFORMAL EVALUATION IN MEMORY
        </div>
        <div style={{ fontSize: '0.7rem', marginTop: '0.25rem', color: '#71717a' }}>
          Select an MRI scan from the queue and click Evaluate Conformal Prediction Set.
        </div>
      </div>
    );
  }

  const { prediction_set, is_confident, target_coverage, alpha, quantile_applied, softmax_probabilities, inference_time_ms } = result;
  const threshold_prob = 1.0 - quantile_applied;

  return (
    <div className="pacs-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem', background: '#000000' }}>
      <div className="pacs-header" style={{ display: 'flex', alignItems: 'center' }}>
        <ShieldCheck size={14} color="#10b981" />
        <span>STATISTICALLY GUARANTEED PREDICTION SET // C_α(X)</span>
        <span className={`pacs-badge ${is_confident ? 'pacs-badge-confident' : 'pacs-badge-triage'}`} style={{ marginLeft: 'auto' }}>
          {(target_coverage * 100).toFixed(0)}% STATUTORY GUARANTEE
        </span>
      </div>

      {/* Main Set Output Display */}
      <div
        style={{
          background: '#000000',
          border: is_confident ? '1px solid #059669' : '1px solid #d97706',
          borderRadius: '2px',
          padding: '0.875rem 1rem',
        }}
      >
        <div style={{ fontSize: '0.675rem', fontWeight: 700, color: is_confident ? '#10b981' : '#f59e0b', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.25rem' }} className="mono">
          {is_confident ? 'SINGLE CONFIDENT DIAGNOSIS' : `AMBIGUOUS DIFFERENTIAL SET (${prediction_set.length} LABELS)`}
        </div>

        <div style={{ fontSize: '1.1rem', fontWeight: 700, color: is_confident ? '#10b981' : '#f59e0b', marginBottom: '0.375rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }} className="mono">
          {is_confident ? <Check size={18} /> : <AlertTriangle size={18} />}
          {prediction_set.length > 0 ? prediction_set.join('  //  ') : 'Out of Distribution / Uncertain'}
        </div>

        <div style={{ fontSize: '0.725rem', color: '#a1a1aa', lineHeight: '1.4' }}>
          Coverage Guarantee: <code className="mono" style={{ color: '#06b6d4' }}>P(Y_true ∈ C_α(X)) ≥ {(target_coverage * 100).toFixed(1)}%</code> over calibrated population.
        </div>
      </div>

      {/* Physician Clinical Guidance Callout Box */}
      <div
        style={{
          background: '#000000',
          border: '1px solid #1e293b',
          borderRadius: '2px',
          padding: '0.75rem 0.875rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', fontWeight: 700, color: '#06b6d4' }}>
            <Stethoscope size={14} />
            <span>CLINICIAN RECOMMENDATION & INTERPRETATION</span>
          </div>

          <button
            type="button"
            className="pacs-btn"
            style={{ padding: '2px 6px', fontSize: '0.65rem' }}
            onClick={() => setShowDoctorHelp(!showDoctorHelp)}
          >
            <Info size={11} /> {showDoctorHelp ? 'HIDE EXPLAINER' : 'PHYSICIAN GUIDE'}
            {showDoctorHelp ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
          </button>
        </div>

        <div style={{ fontSize: '0.725rem', color: '#d4d4d8', lineHeight: '1.5' }}>
          {is_confident ? (
            <span>
              <strong style={{ color: '#10b981' }}>✓ Clear Scan:</strong> The model identified a single diagnosis (<span className="mono" style={{ color: '#10b981' }}>{prediction_set[0]}</span>) with a 95.0% statistical coverage guarantee. Standard diagnostic reading workflow applies.
            </span>
          ) : (
            <span>
              <strong style={{ color: '#f59e0b' }}>⚠️ Borderline Differential:</strong> The scan presents overlapping anatomical features across candidate diagnoses (<span className="mono" style={{ color: '#f59e0b' }}>{prediction_set.join(', ')}</span>). The model abstained from a single guess to prevent misclassification risk. <strong>Secondary expert reading by a radiologist is strongly recommended.</strong>
            </span>
          )}
        </div>

        {/* Expandable Plain-English Medical Glossary */}
        {showDoctorHelp && (
          <div
            style={{
              marginTop: '0.25rem',
              paddingTop: '0.5rem',
              borderTop: '1px solid #18181b',
              fontSize: '0.675rem',
              color: '#a1a1aa',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.375rem',
            }}
          >
            <div>
              <strong style={{ color: '#f4f4f5' }}>1. What is a Conformal Prediction Set?</strong>
              <p style={{ margin: '2px 0 0 0', color: '#71717a' }}>
                Instead of forcing an overconfident single guess when uncertain, the AI outputs a differential list guaranteed to contain the true pathology 95% of the time.
              </p>
            </div>
            <div>
              <strong style={{ color: '#f4f4f5' }}>2. What is Non-Conformity Score ($S_i$)?</strong>
              <p style={{ margin: '2px 0 0 0', color: '#71717a' }}>
                Measures how unusual a diagnosis appears to the model ($1 - P_i$). Lower scores indicate higher probability.
              </p>
            </div>
            <div>
              <strong style={{ color: '#f4f4f5' }}>3. What is Margin ($\Delta S$)?</strong>
              <p style={{ margin: '2px 0 0 0', color: '#71717a' }}>
                Green negative values mean the disease comfortably falls within the 95% safety threshold.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Non-Conformity Threshold Line & Class Metrics Grid */}
      <div style={{ background: '#000000' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.675rem', color: '#a1a1aa', marginBottom: '0.375rem' }} className="mono">
          <span>SCORE CUTOFF (q̂_{alpha} = {quantile_applied.toFixed(4)})</span>
          <span>MIN PROBABILITY CUTOFF: {(threshold_prob * 100).toFixed(1)}%</span>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', background: '#000000' }}>
          {Object.entries(softmax_probabilities).map(([clsName, probVal]) => {
            const inSet = prediction_set.includes(clsName);
            const score_i = 1.0 - probVal;
            const delta = score_i - quantile_applied;

            return (
              <div
                key={clsName}
                style={{
                  background: '#000000',
                  border: '1px solid #18181b',
                  borderRadius: '2px',
                  padding: '0.45rem 0.625rem',
                  fontSize: '0.725rem',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }} className="mono">
                  <span style={{ fontWeight: inSet ? 700 : 400, color: inSet ? '#f4f4f5' : '#52525b' }}>
                    {clsName} {inSet && <span style={{ color: is_confident ? '#10b981' : '#f59e0b' }}>[IN SET]</span>}
                  </span>
                  <div style={{ display: 'flex', gap: '0.875rem', color: '#a1a1aa' }}>
                    <span>Prob: <strong style={{ color: inSet ? '#06b6d4' : '#52525b' }}>{(probVal * 100).toFixed(1)}%</strong></span>
                    <span>S_i: <strong>{score_i.toFixed(4)}</strong></span>
                    <span>ΔS: <strong style={{ color: delta <= 0 ? '#10b981' : '#f59e0b' }}>{delta > 0 ? `+${delta.toFixed(4)}` : delta.toFixed(4)}</strong></span>
                  </div>
                </div>

                <div style={{ width: '100%', height: '4px', background: '#000000', border: '1px solid #18181b', borderRadius: '2px', overflow: 'hidden', position: 'relative' }}>
                  <div
                    style={{
                      width: `${probVal * 100}%`,
                      height: '100%',
                      background: inSet ? (is_confident ? '#10b981' : '#f59e0b') : '#27272a',
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Latency Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.675rem', color: '#71717a', borderTop: '1px solid #18181b', paddingTop: '0.5rem', background: '#000000' }} className="mono">
        <span>LATENCY: {inference_time_ms} ms</span>
        <span>CALIBRATION: SPLIT-CONFORMAL LAC</span>
      </div>
    </div>
  );
};
