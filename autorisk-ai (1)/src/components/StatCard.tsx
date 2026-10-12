import React from 'react';
import { StatMetric } from '../types';

interface StatCardProps {
  metric: StatMetric;
}

export const StatCard: React.FC<StatCardProps> = ({ metric }) => {
  const trendColorClass =
    metric.trend === 'positive'
      ? 'text-blue-700'
      : metric.trend === 'attention'
      ? 'text-amber-700'
      : 'text-slate-700';

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between">
      <div>
        <p className="text-xs font-medium text-slate-500">{metric.label}</p>
        <p className="mt-2 text-2xl font-bold tracking-tight text-slate-900 font-mono-tabular">
          {metric.value}
        </p>
      </div>

      {/* Unboxed clean metadata with middle-dot separator */}
      <div className="mt-4 pt-3 border-t border-slate-100 flex items-center gap-1.5 text-xs text-slate-500">
        <span className={`font-medium tabular-nums ${trendColorClass}`}>
          {metric.change}
        </span>
        <span aria-hidden="true">·</span>
        <span className="truncate">{metric.context}</span>
      </div>
    </div>
  );
};
