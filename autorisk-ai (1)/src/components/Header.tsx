import React from 'react';
import { SampleProfileId } from '../types';

interface HeaderProps {
  selectedSampleId: SampleProfileId | null;
  onSelectSample: (sampleId: SampleProfileId) => void;
  onResetAll: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  selectedSampleId,
  onSelectSample,
  onResetAll,
}) => {
  return (
    <header className="h-16 bg-white border-b border-slate-200 px-4 sm:px-8 flex items-center justify-between gap-8 shrink-0">
      {/* Zone 1: Single-Line Contextual Title */}
      <div className="flex items-center gap-2 text-sm whitespace-nowrap shrink-0">
        <span className="text-slate-500 font-medium hidden sm:inline">
          Commercial &amp; Personal Auto
        </span>
        <span className="text-slate-300 hidden sm:inline" aria-hidden="true">
          /
        </span>
        <span className="text-slate-900 font-semibold">Risk Workbench</span>
      </div>

      {/* Zone 2: Clean Single-Line Quick Sample Links */}
      <nav
        className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600"
        aria-label="Sample Risk Profiles"
      >
        <button
          type="button"
          onClick={() => onSelectSample('low')}
          className={`transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            selectedSampleId === 'low'
              ? 'text-blue-600 font-semibold underline underline-offset-4'
              : 'hover:text-slate-900'
          }`}
        >
          Low Risk Sample
        </button>
        <button
          type="button"
          onClick={() => onSelectSample('medium')}
          className={`transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            selectedSampleId === 'medium'
              ? 'text-blue-600 font-semibold underline underline-offset-4'
              : 'hover:text-slate-900'
          }`}
        >
          Medium Risk Sample
        </button>
        <button
          type="button"
          onClick={() => onSelectSample('high')}
          className={`transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
            selectedSampleId === 'high'
              ? 'text-blue-600 font-semibold underline underline-offset-4'
              : 'hover:text-slate-900'
          }`}
        >
          High Risk Sample
        </button>
      </nav>

      {/* Zone 3: Single Primary Action */}
      <div className="flex items-center shrink-0">
        <button
          type="button"
          onClick={onResetAll}
          className="px-4 py-2 text-xs font-semibold text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          New Application
        </button>
      </div>
    </header>
  );
};
