import React from 'react';
import { AlertTriangle, CheckCircle2, ClipboardList, ShieldAlert } from 'lucide-react';
import { MockRiskAssessment, SampleProfileId } from '../types';
import { SAMPLE_PROFILE_LIST } from '../data/mockData';

interface RiskResultsPanelProps {
  assessment: MockRiskAssessment | null;
  onClearAssessment: () => void;
  onSelectSample: (sampleId: SampleProfileId) => void;
}

export const RiskResultsPanel: React.FC<RiskResultsPanelProps> = ({
  assessment,
  onClearAssessment,
  onSelectSample,
}) => {
  const getCategoryColorClass = (category: MockRiskAssessment['riskCategory']) => {
    if (category === 'Low') return 'text-emerald-700';
    if (category === 'Medium') return 'text-amber-700';
    return 'text-red-700';
  };

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 sm:p-6 flex flex-col justify-between h-full">
      <div className="flex items-center justify-between gap-4 pb-5 border-b border-slate-200">
        <div>
          <h2 className="text-base font-semibold text-slate-900">
            Risk Assessment Results
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Deterministic demo scoring and underwriting summary.
          </p>
        </div>
        {assessment && (
          <button
            type="button"
            onClick={onClearAssessment}
            className="px-3 py-1.5 text-xs font-medium text-slate-600 hover:text-slate-900 border border-slate-200 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Clear Result
          </button>
        )}
      </div>

      {!assessment ? (
        /* Empty Risk Results Panel State */
        <div className="my-auto py-8 flex flex-col items-center text-center">
          <div className="w-11 h-11 rounded-full bg-blue-50 text-blue-600 flex items-center justify-center mb-3">
            <ShieldAlert className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-slate-900">
            No Active Risk Evaluation
          </h3>
          <p className="text-xs text-slate-500 max-w-sm mt-1.5 leading-relaxed">
            Fill out the application form or pick one of the three sample profiles, then click <strong className="font-semibold text-slate-700">Analyze Risk</strong> to calculate an illustrative 0–100 score and underwriting summary.
          </p>

          {/* Unpopulated metric slots preview */}
          <div className="w-full mt-6 pt-5 border-t border-slate-100 grid grid-cols-3 gap-3 text-left">
            <div>
              <p className="text-xs text-slate-400">Demo Score</p>
              <p className="text-sm font-mono-tabular font-semibold text-slate-400 mt-1">
                -- / 100
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-400">Category</p>
              <p className="text-sm font-mono-tabular font-semibold text-slate-400 mt-1">
                Low / Med / High
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-400">Est. Premium</p>
              <p className="text-sm font-mono-tabular font-semibold text-slate-400 mt-1">
                $-- / yr
              </p>
            </div>
          </div>

          <div className="w-full mt-6 pt-5 border-t border-slate-100">
            <p className="text-xs font-medium text-slate-600 mb-2.5">
              Quick-Fill Form with a Sample Profile:
            </p>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {SAMPLE_PROFILE_LIST.map((profile) => (
                <button
                  key={profile.id}
                  type="button"
                  onClick={() => onSelectSample(profile.id)}
                  className="px-3 py-1.5 text-xs font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors whitespace-nowrap cursor-pointer"
                >
                  {profile.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Populated Deterministic Risk Assessment State */
        <div className="mt-4 space-y-5 my-auto">
          {/* Demo Estimate Disclaimer Notice */}
          <div className="px-3.5 py-2.5 rounded-lg bg-slate-50 border border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
            <span>
              <strong className="font-semibold text-slate-800">Demo Estimate:</strong> Illustrative deterministic score only, not a validated insurance prediction.
            </span>
            <span className="font-mono-tabular text-slate-400 shrink-0">
              {assessment.referenceId}
            </span>
          </div>

          {/* Score, Category, and Illustrative Premium */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 py-4 border-y border-slate-100">
            <div>
              <p className="text-xs text-slate-500">Demo Risk Score</p>
              <p className="text-2xl font-bold text-slate-900 font-mono-tabular mt-1">
                {assessment.riskIndexScore}
                <span className="text-xs font-normal text-slate-500"> / 100</span>
              </p>
            </div>

            <div>
              <p className="text-xs text-slate-500">Risk Category</p>
              <p
                className={`text-lg font-bold mt-1 ${getCategoryColorClass(
                  assessment.riskCategory
                )}`}
              >
                {assessment.riskCategory} Risk
              </p>
              <p className="text-xs text-slate-400 font-mono-tabular">
                Band {assessment.categoryRangeLabel}
              </p>
            </div>

            <div>
              <p className="text-xs text-slate-500">Demo Premium Est.</p>
              <p className="text-xl font-bold text-blue-600 font-mono-tabular mt-1">
                ${assessment.estimatedAnnualPremium.toLocaleString()}
                <span className="text-xs font-normal text-slate-500">/yr</span>
              </p>
            </div>
          </div>

          {/* Main Contributing Factors */}
          <div>
            <h3 className="text-xs font-semibold text-slate-900 mb-2">
              Main Contributing Factors
            </h3>
            <ul className="space-y-1.5 text-xs text-slate-700">
              {assessment.contributingFactors.map((item, idx) => (
                <li
                  key={idx}
                  className="flex items-baseline justify-between gap-3 py-1 border-b border-slate-100 last:border-b-0"
                >
                  <span className="text-slate-700">{item.factor}</span>
                  <span className="font-mono-tabular font-medium text-slate-900 shrink-0">
                    {item.impact}
                  </span>
                </li>
              ))}
            </ul>
          </div>

          {/* Underwriting Summary Section */}
          <div className="pt-4 border-t border-slate-200 space-y-4">
            <div className="flex items-center gap-2">
              <ClipboardList className="w-4 h-4 text-blue-600 shrink-0" />
              <h3 className="text-xs font-semibold text-slate-900">
                Underwriting Summary
              </h3>
            </div>

            <p className="text-xs text-slate-600 leading-relaxed">
              {assessment.underwritingSummary}
            </p>

            {/* Recommended Review Actions */}
            <div>
              <p className="text-xs font-semibold text-slate-900 mb-1.5">
                Recommended Review Actions
              </p>
              <ul className="space-y-1.5 text-xs text-slate-600">
                {assessment.recommendedActions.map((action, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <CheckCircle2 className="w-3.5 h-3.5 text-blue-600 shrink-0 mt-0.5" />
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Flagged Missing Information */}
            <div>
              <p className="text-xs font-semibold text-slate-900 mb-1.5">
                Flagged Missing Information
              </p>
              <ul className="space-y-1.5 text-xs text-amber-800">
                {assessment.missingInformationFlags.map((flag, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
                    <span>{flag}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
