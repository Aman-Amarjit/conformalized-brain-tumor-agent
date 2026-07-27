import React from 'react';
import { AlertOctagon, CheckCircle2, UserCheck } from 'lucide-react';

interface Props {
  isConfident: boolean;
  triageMessage: string;
  predictionSet: string[];
}

export const TriageAbstentionBanner: React.FC<Props> = ({ isConfident, triageMessage, predictionSet }) => {
  if (isConfident) {
    return (
      <div
        style={{
          background: '#000000',
          border: '1px solid #059669',
          borderRadius: '2px',
          padding: '0.75rem 1rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
        }}
      >
        <CheckCircle2 size={20} color="#10b981" style={{ flexShrink: 0 }} />
        <div>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#10b981' }} className="mono">
            AUTONOMOUS DIAGNOSTIC CLEARANCE // SINGLE CONFIDENT LABEL
          </div>
          <div style={{ fontSize: '0.725rem', color: '#f4f4f5', marginTop: '2px' }}>
            Scan satisfies 95% statistical coverage criteria with single classification <strong>{predictionSet[0]}</strong>.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        background: '#000000',
        border: '1px solid #d97706',
        borderLeft: '4px solid #f59e0b',
        borderRadius: '2px',
        padding: '0.875rem 1rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
      }}
    >
      <AlertOctagon size={22} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }} className="mono">
            <UserCheck size={14} /> CLINICAL ESCALATION — MANDATORY RADIOLOGIST REVIEW
          </div>
          <span className="pacs-badge pacs-badge-triage">ABSTENTION ESCALATED</span>
        </div>
        
        <p style={{ fontSize: '0.75rem', color: '#f4f4f5', marginTop: '0.25rem', lineHeight: '1.4' }}>
          {triageMessage}
        </p>

        <div style={{ marginTop: '0.375rem', background: '#000000', padding: '0.375rem 0.5rem', borderRadius: '2px', border: '1px solid #d97706', fontSize: '0.7rem', color: '#a1a1aa' }} className="mono">
          SAFETY PROTOCOL: Model abstained from a single point prediction to prevent diagnostic risk. Candidate differential set: <strong>{predictionSet.join(' // ')}</strong>.
        </div>
      </div>
    </div>
  );
};
