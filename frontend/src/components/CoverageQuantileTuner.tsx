import React from 'react';
import { Sliders, HelpCircle } from 'lucide-react';

interface Props {
  alpha: number;
  onAlphaChange: (newAlpha: number) => void;
  quantileApplied?: number;
}

export const CoverageQuantileTuner: React.FC<Props> = ({ alpha, onAlphaChange, quantileApplied }) => {
  const targetCoveragePct = ((1.0 - alpha) * 100).toFixed(0);

  const handleSlider = (e: React.ChangeEvent<HTMLInputElement>) => {
    const covValue = parseFloat(e.target.value);
    const newAlpha = parseFloat((1.0 - covValue / 100.0).toFixed(2));
    onAlphaChange(newAlpha);
  };

  return (
    <div className="pacs-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.625rem', background: '#000000' }}>
      <div className="pacs-header" style={{ borderBottom: 'none', paddingBottom: 0, marginBottom: 0, background: '#000000' }}>
        <Sliders size={14} color="#06b6d4" />
        <span>CONFORMAL RISK PARAMETER // (1 - α)</span>
        <span className="mono" style={{ marginLeft: 'auto', fontSize: '0.8rem', fontWeight: 700, color: '#06b6d4' }}>
          {targetCoveragePct}% TARGET COVERAGE
        </span>
      </div>

      <div style={{ background: '#000000' }}>
        <input
          type="range"
          min="80"
          max="99"
          step="1"
          value={targetCoveragePct}
          onChange={handleSlider}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#71717a', marginTop: '0.25rem' }} className="mono">
          <span>80% [HIGH PRECISION]</span>
          <span>95% [CLINICAL STD]</span>
          <span>99% [MAX SAFETY]</span>
        </div>
      </div>

      <div style={{ background: '#000000', border: '1px solid #18181b', borderRadius: '2px', padding: '0.5rem 0.625rem', fontSize: '0.7rem', color: '#a1a1aa', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <HelpCircle size={14} color="#06b6d4" style={{ flexShrink: 0 }} />
        <div>
          Target coverage <span className="mono" style={{ color: '#06b6d4' }}>1-α = {targetCoveragePct}%</span> sets cutoff threshold {quantileApplied ? <span className="mono" style={{ color: '#06b6d4' }}>q̂ = {quantileApplied.toFixed(4)}</span> : ''}. Lower α expands prediction set $C_\alpha(X)$ to preserve statutory bounds.
        </div>
      </div>
    </div>
  );
};
