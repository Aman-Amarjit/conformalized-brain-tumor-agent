import React from 'react';
import { History, FileText } from 'lucide-react';

export interface AuditItem {
  timestamp: string;
  sample_name: string;
  prediction_set: string[];
  is_confident: boolean;
  target_coverage: number;
  abstention_triage_flag: boolean;
  triage_message: string;
}

interface Props {
  history: AuditItem[];
}

export const DiagnosticAuditTrail: React.FC<Props> = ({ history }) => {
  if (history.length === 0) return null;

  return (
    <div className="pacs-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem', background: '#000000' }}>
      <div className="pacs-header" style={{ borderBottom: 'none', paddingBottom: 0, marginBottom: 0, background: '#000000' }}>
        <History size={14} color="#06b6d4" />
        <span>RIS SESSION AUDIT LOG // HISTORICAL STUDIES</span>
        <span className="mono" style={{ marginLeft: 'auto', fontSize: '0.675rem', color: '#71717a' }}>
          {history.length} EVALUATIONS
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', maxHeight: '190px', overflowY: 'auto', background: '#000000' }}>
        {history.map((item, idx) => (
          <div
            key={idx}
            style={{
              background: '#000000',
              border: '1px solid #18181b',
              borderRadius: '2px',
              padding: '0.45rem 0.625rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '0.725rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={13} color="#71717a" />
              <div>
                <div style={{ fontWeight: 600, color: '#f4f4f5' }} className="mono">{item.sample_name}</div>
                <div style={{ color: '#71717a', fontSize: '0.65rem' }} className="mono">TIME: {item.timestamp}</div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.875rem' }}>
              <div style={{ fontSize: '0.725rem', color: item.is_confident ? '#10b981' : '#f59e0b', fontWeight: 600 }} className="mono">
                {item.prediction_set.join(' / ')}
              </div>

              <span className={`pacs-badge ${item.is_confident ? 'pacs-badge-confident' : 'pacs-badge-triage'}`}>
                {item.is_confident ? 'AUTONOMOUS CLEAR' : 'TRIAGE ESCALATED'}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
