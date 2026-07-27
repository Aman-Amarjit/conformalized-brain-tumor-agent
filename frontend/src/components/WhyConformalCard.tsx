import React from 'react';
import { Activity, ShieldAlert, CheckCircle } from 'lucide-react';

export const WhyConformalCard: React.FC = () => {
  return (
    <div className="med-card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Activity size={18} color="#38bdf8" />
        <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#f8fafc' }}>
          Why Conformal Prediction Beats Standard Classifiers
        </h3>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.875rem' }}>
        {/* Legacy Standard Classifier Card */}
        <div style={{ background: '#090d14', border: '1px solid #334155', borderRadius: '8px', padding: '0.875rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.8rem', fontWeight: 700, color: '#ef4444', marginBottom: '0.375rem' }}>
            <ShieldAlert size={15} /> Legacy Softmax Classifier
          </div>
          <ul style={{ fontSize: '0.75rem', color: '#94a3b8', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <li>Outputs a single rigid label regardless of uncertainty.</li>
            <li>Softmax probabilities are poorly calibrated and often 99% confident on incorrect or out-of-distribution scans.</li>
            <li>No mathematical guarantee on error rates in clinical practice.</li>
          </ul>
        </div>

        {/* Conformalized Agent Card */}
        <div style={{ background: '#090d14', border: '1px solid #059669', borderRadius: '8px', padding: '0.875rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.8rem', fontWeight: 700, color: '#10b981', marginBottom: '0.375rem' }}>
            <CheckCircle size={15} /> Conformalized Diagnostic Agent
          </div>
          <ul style={{ fontSize: '0.75rem', color: '#cbd5e1', paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <li>Outputs calibrated <strong>prediction sets</strong> instead of point predictions.</li>
            <li>Guarantees that true diagnosis is in the set with <strong>95%+ statistical probability</strong>.</li>
            <li>Automatically escalates ambiguous scans to human radiologists when set size exceeds 1.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
