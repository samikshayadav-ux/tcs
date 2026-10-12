import React from 'react';
import { Shield, LayoutDashboard, UserCheck, RotateCcw } from 'lucide-react';
import { SampleProfileId } from '../types';
import { SAMPLE_PROFILE_LIST } from '../data/mockData';

interface SidebarProps {
  selectedSampleId: SampleProfileId | null;
  onSelectSample: (sampleId: SampleProfileId) => void;
  onResetAll: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  selectedSampleId,
  onSelectSample,
  onResetAll,
}) => {
  return (
    <aside className="w-full lg:w-64 shrink-0 bg-[#0B1528] text-slate-100 flex flex-col justify-between border-b lg:border-b-0 lg:border-r border-slate-800 select-none">
      <div>
        {/* Brand Header */}
        <div className="h-16 px-6 flex items-center justify-between border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-md bg-blue-600 flex items-center justify-center text-white shrink-0">
              <Shield className="w-4 h-4" />
            </div>
            <span className="text-base font-bold tracking-tight text-white whitespace-nowrap">
              AutoRisk AI
            </span>
          </div>
          <span className="text-xs font-mono-tabular text-slate-400 lg:hidden">
            Underwriting Console
          </span>
        </div>

        {/* Active Workspace Indicator */}
        <div className="p-4 pb-2">
          <p className="px-3 mb-2 text-xs font-medium text-slate-400 tracking-wide">
            Workspace
          </p>
          <div className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-semibold bg-blue-600 text-white whitespace-nowrap">
            <LayoutDashboard className="w-4 h-4 shrink-0" />
            <span>Risk Workbench</span>
          </div>
        </div>

        {/* Sample Profiles Quick Fill */}
        <div className="p-4">
          <p className="px-3 mb-2 text-xs font-medium text-slate-400 tracking-wide">
            Sample Applicant Profiles
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 gap-1.5">
            {SAMPLE_PROFILE_LIST.map((profile) => {
              const isSelected = selectedSampleId === profile.id;
              return (
                <button
                  key={profile.id}
                  type="button"
                  onClick={() => onSelectSample(profile.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg transition-colors cursor-pointer ${
                    isSelected
                      ? 'bg-slate-800 text-white border border-blue-500/60'
                      : 'text-slate-300 hover:bg-slate-800/70 hover:text-white border border-transparent'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold flex items-center gap-2 whitespace-nowrap">
                      <UserCheck className="w-3.5 h-3.5 text-blue-400 shrink-0" />
                      {profile.label}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1 font-mono-tabular">
                    {profile.values.applicantName} · {profile.shortTag}
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Bottom Reset Action & Underwriter Context */}
      <div className="p-4 border-t border-slate-800/80 space-y-3">
        <button
          type="button"
          onClick={onResetAll}
          className="w-full py-2 px-3 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700/80 text-xs font-medium text-slate-200 flex items-center justify-center gap-2 transition-colors whitespace-nowrap cursor-pointer"
        >
          <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
          <span>Reset Workbench</span>
        </button>

        <div className="px-2 flex items-center justify-between text-xs text-slate-400">
          <span>Underwriter Desk</span>
          <span className="font-mono-tabular text-slate-300">Region US-W</span>
        </div>
      </div>
    </aside>
  );
};
