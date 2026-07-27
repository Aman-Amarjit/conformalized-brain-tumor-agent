import React, { useState, useRef } from 'react';
import { Upload, Eye, RefreshCcw, Sun, Sliders, Play } from 'lucide-react';

interface CuratedSample {
  id: string;
  filename: string;
  type: string;
  label: string;
  true_class: string;
  expected_set: string[];
  image_url: string;
}

interface Props {
  selectedImage: string | null;
  curatedSamples: CuratedSample[];
  selectedSampleId: string | null;
  onSelectSample: (sample: CuratedSample) => void;
  onFileUpload: (file: File) => void;
  isLoading: boolean;
  onRunInference: () => void;
}

export const MRIDiagnosticCanvas: React.FC<Props> = ({
  selectedImage,
  curatedSamples,
  selectedSampleId,
  onSelectSample,
  onFileUpload,
  isLoading,
  onRunInference,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isInverted, setIsInverted] = useState(false);
  const [contrastHigh, setContrastHigh] = useState(false);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(true);
  };

  const handleDragLeave = () => {
    setIsDraggingOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  // Compute valid CSS filter string
  const getImageFilter = () => {
    const filters: string[] = [];
    if (isInverted) filters.push('invert(100%)');
    if (contrastHigh) filters.push('contrast(180%) brightness(110%)');
    return filters.length > 0 ? filters.join(' ') : 'none';
  };

  return (
    <div className="pacs-panel" style={{ display: 'flex', flexDirection: 'column', gap: '0.875rem', background: '#000000' }}>
      <div className="pacs-header">
        <Sliders size={14} color="#06b6d4" />
        <span>PACS VIEWPORT // MRI BRAIN AXIAL T2-WEIGHTED</span>
        <span className="pacs-badge pacs-badge-info" style={{ marginLeft: 'auto' }}>DICOM Ready</span>
      </div>

      {/* Viewport Control Strip */}
      <div style={{ display: 'flex', gap: '0.5rem', background: '#000000', padding: '0.375rem 0.5rem', borderRadius: '2px', border: '1px solid #18181b' }}>
        <button
          type="button"
          className="pacs-btn"
          style={{
            padding: '0.25rem 0.5rem',
            fontSize: '0.7rem',
            borderColor: isInverted ? '#06b6d4' : '#27272a',
            color: isInverted ? '#06b6d4' : '#f4f4f5',
          }}
          onClick={(e) => {
            e.stopPropagation();
            setIsInverted((prev) => !prev);
          }}
        >
          <Eye size={12} color={isInverted ? '#06b6d4' : '#f4f4f5'} />
          {isInverted ? 'INVERTED (LUT ON)' : 'INVERT LUT'}
        </button>

        <button
          type="button"
          className="pacs-btn"
          style={{
            padding: '0.25rem 0.5rem',
            fontSize: '0.7rem',
            borderColor: contrastHigh ? '#06b6d4' : '#27272a',
            color: contrastHigh ? '#06b6d4' : '#f4f4f5',
          }}
          onClick={(e) => {
            e.stopPropagation();
            setContrastHigh((prev) => !prev);
          }}
        >
          <Sun size={12} color={contrastHigh ? '#06b6d4' : '#f4f4f5'} />
          {contrastHigh ? 'HIGH CONTRAST ON' : 'HIGH CONTRAST'}
        </button>

        <button
          type="button"
          className="pacs-btn"
          style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem', marginLeft: 'auto' }}
          onClick={(e) => {
            e.stopPropagation();
            setIsInverted(false);
            setContrastHigh(false);
          }}
        >
          <RefreshCcw size={12} /> RESET
        </button>
      </div>

      {/* Main Image Viewport with Drag-and-Drop + DICOM Overlay */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          width: '100%',
          height: '270px',
          background: '#000000',
          border: isDraggingOver ? '2px dashed #06b6d4' : '1px solid #18181b',
          borderRadius: '2px',
          position: 'relative',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
        }}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*"
          style={{ display: 'none' }}
        />

        {/* Anatomical Orientation Markers */}
        <div className="dicom-hud" style={{ top: '8px', left: '50%', transform: 'translateX(-50%)', color: '#71717a' }}>A</div>
        <div className="dicom-hud" style={{ bottom: '8px', left: '50%', transform: 'translateX(-50%)', color: '#71717a' }}>P</div>
        <div className="dicom-hud" style={{ top: '50%', left: '8px', transform: 'translateY(-50%)', color: '#71717a' }}>R</div>
        <div className="dicom-hud" style={{ top: '50%', right: '8px', transform: 'translateY(-50%)', color: '#71717a' }}>L</div>

        {/* DICOM HUD Metadata Top-Left */}
        <div className="dicom-hud" style={{ top: '8px', left: '20px' }}>
          <div>PATIENT ID: MR-2026-904</div>
          <div>AXIAL T2-FLAIR</div>
          <div>TR: 550ms // TE: 14ms</div>
        </div>

        {/* DICOM HUD Metadata Top-Right */}
        <div className="dicom-hud" style={{ top: '8px', right: '20px', textAlign: 'right' }}>
          <div>MATRIX: 128x128</div>
          <div>FOV: 220 mm</div>
          <div>W: 400 L: 180</div>
        </div>

        {selectedImage ? (
          <img
            src={selectedImage}
            alt="MRI Brain Slice"
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              background: '#000000',
              filter: getImageFilter(),
              transition: 'filter 0.15s ease',
            }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: '1rem', color: '#52525b', background: '#000000' }}>
            <Upload size={28} color="#06b6d4" style={{ marginBottom: '0.375rem' }} />
            <div style={{ fontSize: '0.75rem', color: '#a1a1aa', fontWeight: 600 }} className="mono">
              IMPORT DICOM SAMPLES
            </div>
          </div>
        )}
      </div>

      {/* RIS Study Queue */}
      <div style={{ background: '#000000' }}>
        <div style={{ fontSize: '0.675rem', fontWeight: 700, color: '#71717a', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.375rem' }} className="mono">
          RIS STUDY QUEUE (SELECT SAMPLE TO INSPECT)
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem', background: '#000000' }}>
          {curatedSamples.map((sample, idx) => {
            const isSelected = selectedSampleId === sample.id;
            const isConfident = sample.type === 'confident';
            return (
              <div
                key={sample.id}
                onClick={() => {
                  setIsInverted(false);
                  setContrastHigh(false);
                  onSelectSample(sample);
                }}
                style={{
                  background: '#000000',
                  border: isSelected
                    ? isConfident
                      ? '1px solid #10b981'
                      : '1px solid #f59e0b'
                    : '1px solid #18181b',
                  borderRadius: '2px',
                  padding: '0.5rem 0.625rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  cursor: 'pointer',
                  fontSize: '0.725rem',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span className="mono" style={{ color: '#06b6d4', fontWeight: 700 }}>
                    #{idx + 101}
                  </span>
                  <div>
                    <div style={{ fontWeight: 600, color: '#f4f4f5' }} className="mono">
                      {sample.label}
                    </div>
                    <div style={{ fontSize: '0.65rem', color: '#71717a' }}>
                      Ground Truth: {sample.true_class}
                    </div>
                  </div>
                </div>

                <span className={`pacs-badge ${isConfident ? 'pacs-badge-confident' : 'pacs-badge-triage'}`}>
                  {isConfident ? 'CONFIDENT' : 'ABSTAIN / TRIAGE'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Execute Button */}
      <button
        className="pacs-btn pacs-btn-primary"
        onClick={onRunInference}
        disabled={isLoading || !selectedImage}
        style={{ width: '100%', justifyContent: 'center', padding: '0.625rem', opacity: isLoading || !selectedImage ? 0.6 : 1 }}
      >
        <Play size={14} />
        {isLoading ? 'CALCULATING CONFORMAL QUANTILE...' : 'EVALUATE CONFORMAL PREDICTION SET'}
      </button>
    </div>
  );
};
